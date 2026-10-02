#!/usr/bin/env python3
"""
alns.py — a resumable ALNS for the PC-MT-EVRP-RT-D.

Wraps the flat `while iter_alns <= 20000` loop from main_ALNS.ipynb cell 2 in a
class so the search can be paused, have customers added to its pool, and then
continue from the incumbent it already found. The loop body is logically
unchanged; only the surrounding structure differs.

    solver = ALNS(instance, pool=[...], seed=0)
    solver.run(5000)                       # first segment
    solver.add_customers([17, 42])         # expand the pool mid-search
    solver.run(3000)                       # continue from the incumbent
    print(solver.global_best_val, solver.served_customers())

State carried across `run()` calls: the incumbent solution and its objective,
the global best, adaptive operator weights, the annealing temperature, the
failure counter, and the iteration counter. Nothing is reset between segments,
so a segmented run and one long run differ only in the pool changes made in
between.

Verifying against the notebook
------------------------------
`python alns.py --instance-type RC --instance-index 30 --iterations 20000 --seed 0`
mirrors one notebook experiment. With the same pool the objective should match
main_ALNS.ipynb. Because the notebook draws its pool with an unseeded
`random.sample`, pass the same pool explicitly (`--pool-file`) when comparing,
or seed the notebook to match.

Known quirks preserved from the notebook (deliberately, for fidelity)
---------------------------------------------------------------------
* `zone_removal` is disabled: its branch condition is `<= -1 * weights_cumm[2]`,
  which is never true for a non-negative roulette draw. The mass falls through
  to the next branch (`random_trips`). Disabled by design, per the author.
* `objective_function_global` charges penalties for every customer in `C` that
  is not served, including customers that were never in the pool.
* The depot-repair block runs `route.insert(len(route)-1, 0)` rather than
  appending, matching the notebook.
"""

import argparse
import copy
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import alns_core
from alns_core import (
    generate_non_repeating_integers,
    greedy_insert_claude,
    largest_trips,
    objective_function_global,
    random_trips,
    seq_removal,
    smallest_trips,
    worst_distance_nodes,
    zone_removal,
)


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    """
    Everything about one problem instance, derived from the reference JSON.

    Built by `Instance.from_json`, which reproduces the parameter setup in
    main_ALNS.ipynb cell 2 exactly, including its two known quirks (see
    `vehicle_load_capacity_recorded`).
    """
    instance_id: str
    instance_type: str
    instance_index: int
    n: int                                  # number of customers (depot excluded)
    C: List[int]                            # customer ids, 1..n
    locations: Dict[int, Tuple[float, float]]
    locations_df: pd.DataFrame
    customer_loads: Dict[int, float]        # dict form, depot removed
    customer_loads_array: np.ndarray        # array form passed to the operators
    time_windows: Dict[int, Tuple[float, float]]
    release_time: Dict[int, float]
    pen_val: Dict[int, float]
    reward_val: Dict[int, float]
    traveling_time: Dict[Tuple[int, int], float]
    Q: List[float]                          # per-vehicle load capacity
    b_v: float                              # battery capacity (already scaled)
    eta: float
    recharging_rate: float
    f_recharging: float
    big_Q: float
    v: int

    @property
    def vehicle_load_capacity_actual(self) -> float:
        """
        The per-vehicle capacity the search actually enforces: big_Q / v.

        NOTE: the notebook records `big_Q / 3` in its results while setting
        `Q = [big_Q / 4] * v`. The recorded figure is wrong; this property
        returns the value the solver really uses.
        """
        return self.Q[0]


    @classmethod
    def from_json(
        cls,
        reference_dir: Path,
        instance_type: str,
        instance_index: int,
        battery_multiplier: float = 1.5,
        num_vehicles: int = 4,
    load_basis: str = "per_vehicle",
        eta: float = 0.7,
        recharging_rate: float = 1.5,
        f_recharging: float = 0.0,
    ) -> "Instance":
        files = {"R": "R_instances.json", "C": "C_instances.json", "RC": "RC_instances.json"}
        with open(Path(reference_dir) / files[instance_type], "r") as f:
            instances = json.load(f)
        inst = instances[instance_index]

        n = inst["num_nodes"] - 1
        big_Q = inst["vehicle_capacity"]
        b_v = inst["battery_capacity"] * battery_multiplier
        v = num_vehicles
        # The JSON's vehicle_capacity is the per-vehicle load capacity.
        # main_ALNS.ipynb divided it by the fleet size, which was wrong;
        # "fleet" is kept only to reproduce those older runs.
        base_q = big_Q if load_basis == "per_vehicle" else big_Q / num_vehicles
        Q = [base_q] * v

        locations, customer_loads, time_windows = {}, {}, {}
        release_time, pen_val, reward_val = {}, {}, {}
        for node in inst["nodes"]:
            nid = node["node_id"]
            locations[nid] = (node["x"], node["y"])
            customer_loads[nid] = node["demand"]
            time_windows[nid] = (node["release_time"], node["deadline"])
            release_time[nid] = node["release_time"]
            if not node["is_depot"]:
                pen_val[nid] = node["penalty"]
                reward_val[nid] = node["prize"]
        del customer_loads[0]

        traveling_time = {}
        for i, (x1, y1) in locations.items():
            for j, (x2, y2) in locations.items():
                traveling_time[(i, j)] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        locations_df = pd.DataFrame.from_dict(locations, orient="index", columns=["X", "Y"])
        locations_df["node_id"] = locations_df.index

        return cls(
            instance_id=inst["instance_id"],
            instance_type=instance_type,
            instance_index=instance_index,
            n=n,
            C=list(range(1, n + 1)),
            locations=locations,
            locations_df=locations_df,
            customer_loads=customer_loads,
            customer_loads_array=np.array(list(customer_loads.values())),
            time_windows=time_windows,
            release_time=release_time,
            pen_val=pen_val,
            reward_val=reward_val,
            traveling_time=traveling_time,
            Q=Q,
            b_v=b_v,
            eta=eta,
            recharging_rate=recharging_rate,
            f_recharging=f_recharging,
            big_Q=big_Q,
            v=v,
        )

    def bind(self) -> None:
        """Install this instance's constants as alns_core module globals."""
        alns_core.bind_instance(
            C=self.C,
            traveling_time=self.traveling_time,
            customer_loads=self.customer_loads,
            time_windows=self.time_windows,
            release_time=self.release_time,
            eta=self.eta,
            recharging_rate=self.recharging_rate,
            f_recharging=self.f_recharging,
        )


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------
@dataclass
class ALNSConfig:
    """Search parameters. Defaults reproduce main_ALNS.ipynb cell 2."""
    num_nodes_remove: int = 10        # nodes removed per destroy step
    radius: int = 20                  # zone_removal radius (operator is disabled)
    num_trips: int = 2                # trips removed by smallest_trips
    sigma: Tuple[int, int, int] = (5, 3, 1)   # new best, improving, accepted-worse
    initial_temperature: float = 20000.0
    cooling_rate: float = 0.999
    learning_rate: float = 0.2
    weight_update_interval: int = 750
    restart_after_failures: int = 2000
    route_restriction: float = 99999999999
    track_history: bool = True        # keep per-iteration objective traces


OPERATORS = [
    "seq_removal", "worst_distance_nodes", "zone_removal",
    "random_trips", "largest_trips", "smallest_trips", "random",
]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class ALNS:
    """
    Resumable ALNS over an explicit customer pool.

    `pool` is the set of customers the search may serve. Customers outside it
    are never inserted, though `objective_function_global` still charges their
    penalties, matching the notebook.
    """

    def __init__(
        self,
        instance: Instance,
        pool: Sequence[int],
        config: Optional[ALNSConfig] = None,
        seed: Optional[int] = None,
    ):
        self.instance = instance
        self.config = config or ALNSConfig()
        instance.bind()

        # The operators in alns_core draw from the GLOBAL `random` module, so a
        # private Random() instance would control only part of the stream. This
        # solver therefore owns a saved global-RNG state, swapped in for the
        # duration of each run() call and swapped back out afterwards. That
        # keeps every draw — the solver's and the operators' — on one
        # reproducible per-solver stream, while leaving the caller's global
        # random state untouched so two solvers can be interleaved safely.
        self.seed = seed
        _caller_state = random.getstate()
        random.seed(seed)
        self._rng_state = random.getstate()

        self.pool: List[int] = sorted(set(pool))
        self.excluded: List[int] = sorted(set(instance.C) - set(self.pool))
        self._refresh_locations_df()

        # ---- initial solution: greedy insertion over the pool --------------
        self.current_sol: Dict[Tuple[int, int], List[int]] = {}
        try:
            for customer in self.pool:
                self.current_sol = greedy_insert_claude(
                    self.current_sol, customer, instance.customer_loads_array,
                    instance.traveling_time, instance.Q, instance.b_v,
                )
        finally:
            self._rng_state = random.getstate()
            random.setstate(_caller_state)
        self.initial_sol = copy.deepcopy(self.current_sol)

        # ---- search state (persists across run() calls) ---------------------
        self.current_objective = objective_function_global(
            self.current_sol, instance.pen_val, instance.reward_val
        )
        self.global_best_val = self.current_objective
        self.global_best_sol = copy.deepcopy(self.current_sol)

        self.weights: List[float] = [1 / 7] * 7
        self._recompute_cumulative_weights()
        self.temperature = self.config.initial_temperature
        self.performance = {op: 0 for op in OPERATORS}
        self.number_calls = {op: 0 for op in OPERATORS}
        self.failed_iters = 0
        self.iteration = 0

        # ---- diagnostics ----------------------------------------------------
        self.obj_profile: List[float] = []
        self.best_so_far: List[float] = []
        self.pool_events: List[Dict] = []   # every pool change, with its iteration
        self.cpu_time = 0.0

    # ---- helpers -----------------------------------------------------------
    def _recompute_cumulative_weights(self) -> None:
        self.weights_cumm = list(itertools.accumulate(self.weights))
        self.weights_cumm[-1] = 1

    def _refresh_locations_df(self) -> None:
        """zone_removal reads this; keep it in step with the pool."""
        df = self.instance.locations_df
        self.locations_df = df[df["node_id"].isin([0] + self.pool)]

    # ---- pool expansion ----------------------------------------------------
    def add_customers(self, customers: Sequence[int], insert: bool = True) -> List[int]:
        """
        Add customers to the pool mid-search.

        Only ids in the instance and not already pooled are accepted; the
        accepted list is returned. With `insert=True` each new customer is
        greedily inserted into the incumbent, so the objective reflects them
        immediately; otherwise they stay unserved (and penalised) until an
        operator picks them up.

        Search state — global best, weights, temperature, iteration count — is
        untouched, so the search continues rather than restarting.
        """
        inst = self.instance
        accepted = [c for c in sorted(set(customers))
                    if c in set(inst.C) and c not in set(self.pool)]
        if not accepted:
            return []

        self.pool = sorted(set(self.pool) | set(accepted))
        self.excluded = sorted(set(inst.C) - set(self.pool))
        self._refresh_locations_df()

        if insert:
            caller_state = random.getstate()
            random.setstate(self._rng_state)
            try:
                for customer in accepted:
                    self.current_sol = greedy_insert_claude(
                        self.current_sol, customer, inst.customer_loads_array,
                        inst.traveling_time, inst.Q, inst.b_v,
                    )
            finally:
                self._rng_state = random.getstate()
                random.setstate(caller_state)

        # Re-evaluate: the penalty term changes as soon as a customer is served.
        self.current_objective = objective_function_global(
            self.current_sol, inst.pen_val, inst.reward_val
        )
        if self.current_objective < self.global_best_val:
            self.global_best_val = self.current_objective
            self.global_best_sol = copy.deepcopy(self.current_sol)

        self.pool_events.append({
            "iteration": self.iteration,
            "added": accepted,
            "pool_size": len(self.pool),
            "objective_after": self.current_objective,
        })
        return accepted

    # ---- the search --------------------------------------------------------
    def run(self, n_iterations: Optional[int] = None,
            time_limit: Optional[float] = None) -> float:
        """
        Advance the search, returning the global best value.

        Stops at whichever limit is reached first:
          n_iterations -- run this many more iterations
          time_limit   -- run for this many more seconds of CPU time

        At least one must be given. The budget is measured in CPU time
        (`time.process_time`), not wall clock, so a run gets the same amount of
        actual work whether or not the machine is oversubscribed — which keeps
        time-budgeted arms comparable when many workers share a machine.

        Safe to call repeatedly; each call resumes from the state left by the
        previous one.
        """
        if n_iterations is None and time_limit is None:
            raise ValueError("run() needs n_iterations, time_limit, or both")

        inst = self.instance
        cfg = self.config
        inst.bind()   # in case another instance was bound since the last call

        start_cpu = time.process_time()
        target = None if n_iterations is None else self.iteration + n_iterations
        deadline = None if time_limit is None else start_cpu + time_limit

        caller_state = random.getstate()
        random.setstate(self._rng_state)
        try:
            self._loop(target, deadline)
        finally:
            self._rng_state = random.getstate()
            random.setstate(caller_state)

        self.cpu_time += time.process_time() - start_cpu
        return self.global_best_val

    def _loop(self, target: Optional[int], deadline: Optional[float]) -> None:
        """The search loop itself. Assumes the solver's RNG stream is installed."""
        inst = self.instance
        cfg = self.config

        # Checking the clock every iteration costs ~50 ns against an iteration
        # cost of ~25 ms, so it is free in practice and keeps the stop precise.
        while True:
            if target is not None and self.iteration >= target:
                break
            if deadline is not None and time.process_time() >= deadline:
                break
            roullete_value = random.random()

            # NOTE: branch 2 (zone_removal) tests `<= -1 * weights_cumm[2]`,
            # which never fires. Kept to match the notebook.
            if roullete_value <= self.weights_cumm[0]:
                selected = 0
                nodes_remove = seq_removal(self.current_sol, cfg.num_nodes_remove)
            elif roullete_value <= self.weights_cumm[1]:
                selected = 1
                nodes_remove = worst_distance_nodes(
                    self.current_sol, inst.traveling_time, cfg.num_nodes_remove)
            elif roullete_value <= -1 * self.weights_cumm[2]:
                selected = 2
                nodes_remove = zone_removal(cfg.radius, self.locations_df)
            elif roullete_value <= self.weights_cumm[3]:
                selected = 3
                nodes_remove = random_trips(self.current_sol, 1)
            elif roullete_value <= self.weights_cumm[4]:
                selected = 4
                nodes_remove = largest_trips(self.current_sol, inst.traveling_time, 1)
            elif roullete_value <= self.weights_cumm[5]:
                selected = 5
                nodes_remove = smallest_trips(
                    self.current_sol, inst.traveling_time, cfg.num_trips)
            else:
                selected = 6
                # Draws from the pool, so expansion widens this operator's reach.
                nodes_remove = random.sample(self.pool, cfg.num_nodes_remove)

            # ---- destroy ----------------------------------------------------
            partially_destroyed_sol = copy.deepcopy(self.current_sol)
            for k, route in partially_destroyed_sol.items():
                common_nodes = [node for node in nodes_remove if node in route]
                if common_nodes:
                    route = [node for node in route if node not in common_nodes]
                    partially_destroyed_sol[k] = route

            remove_keys = []
            for k, route in partially_destroyed_sol.items():
                if len(route) == 2 and route[0] == 0 and route[1] == 0:
                    remove_keys.append(k)
                if route[0] != 0:
                    route.insert(0, 0)
                if route[-1] != 0:
                    route.insert(len(route) - 1, 0)
            for k in remove_keys:
                del partially_destroyed_sol[k]

            # ---- repair -----------------------------------------------------
            for customer in nodes_remove:
                partially_destroyed_sol = greedy_insert_claude(
                    partially_destroyed_sol, customer, inst.customer_loads_array,
                    inst.traveling_time, inst.Q, inst.b_v,
                )

            new_objective = objective_function_global(
                partially_destroyed_sol, inst.pen_val, inst.reward_val)

            # ---- accept -----------------------------------------------------
            if new_objective < self.current_objective:
                self.current_sol = copy.deepcopy(partially_destroyed_sol)
                self.current_objective = new_objective
                self.performance[OPERATORS[selected]] = max(
                    cfg.sigma[1], self.performance[OPERATORS[selected]])
                if new_objective < self.global_best_val:
                    self.global_best_val = new_objective
                    self.global_best_sol = copy.deepcopy(self.current_sol)
                    self.performance[OPERATORS[selected]] = cfg.sigma[0]
                    self.failed_iters = 0
                else:
                    self.failed_iters += 1
            else:
                gap = abs(new_objective - self.current_objective)
                if random.random() <= math.exp(-gap / self.temperature):
                    self.current_sol = copy.deepcopy(partially_destroyed_sol)
                    self.current_objective = new_objective
                    self.performance[OPERATORS[selected]] = max(
                        cfg.sigma[2], self.performance[OPERATORS[selected]])

            # ---- bookkeeping -------------------------------------------------
            self.temperature *= cfg.cooling_rate
            if cfg.track_history:
                self.obj_profile.append(new_objective)
                self.best_so_far.append(self.global_best_val)
            self.iteration += 1
            self.number_calls[OPERATORS[selected]] += 1

            if self.failed_iters == cfg.restart_after_failures:
                self.current_sol = copy.deepcopy(self.global_best_sol)

            if self.iteration % cfg.weight_update_interval == 0:
                dummy = [
                    cfg.learning_rate * self.performance[OPERATORS[i]]
                    + (1 - cfg.learning_rate) * self.weights[i]
                    for i in range(len(self.weights))
                ]
                total = sum(dummy)
                self.weights = [w / total for w in dummy]
                self._recompute_cumulative_weights()
                for op in OPERATORS:
                    self.number_calls[op] = 0
                    self.performance[op] = 0

    # ---- results -----------------------------------------------------------
    def served_customers(self) -> List[int]:
        """Customers visited in the global best solution."""
        served = []
        for route in self.global_best_sol.values():
            served.extend(route)
        return sorted({c for c in served if c != 0})

    def unserved_from_pool(self) -> List[int]:
        """Pool members the best solution leaves unvisited — the insertion candidates."""
        return sorted(set(self.pool) - set(self.served_customers()))

    def feasibility_check(self) -> bool:
        """True when no customer outside the pool appears in the best solution."""
        served = set(self.served_customers())
        return not any(c in served for c in self.excluded)

    def summary(self) -> Dict:
        """Result record, keyed to match the notebook's output fields."""
        inst = self.instance
        served = self.served_customers()
        return {
            "instance_id": inst.instance_id,
            "instance_type": inst.instance_type,
            "instance_index": inst.instance_index,
            "objective_value": self.global_best_val,
            "iterations": self.iteration,
            "cpu_time": self.cpu_time,
            "pool_size": len(self.pool),
            "num_served": len(served),
            "served_customers": served,
            "customers_not_selected_first_place": self.excluded,
            "customers_not_visited_from_selected_pool": self.unserved_from_pool(),
            "battery_capacity": inst.b_v,
            "vehicle_load_capacity": inst.vehicle_load_capacity_actual,
            "feasibility_prize": self.feasibility_check(),
            "pool_events": self.pool_events,
            "best_solution": [[list(k), v] for k, v in self.global_best_sol.items()],
        }


# ---------------------------------------------------------------------------
# CLI — a single run, for checking against the notebook
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the resumable ALNS once (used to verify against main_ALNS.ipynb).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reference-dir", type=Path, default=Path("data generation"))
    p.add_argument("--instance-type", choices=["R", "C", "RC"], default="RC")
    p.add_argument("--instance-index", type=int, default=30)
    p.add_argument("--pool-size", type=int, default=85,
                   help="Random pool size, when --pool-file is not given.")
    p.add_argument("--pool-file", type=Path, default=None,
                   help="JSON file holding a list of customer ids; use this to "
                        "reproduce a specific notebook pool exactly.")
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=Path, default=None, help="Write the summary JSON here.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {args.reference_dir.resolve()}",
              file=sys.stderr)
        return 1

    instance = Instance.from_json(args.reference_dir, args.instance_type, args.instance_index)
    print(f"Instance {instance.instance_id}  "
          f"({instance.n} customers, {instance.v} vehicles, "
          f"Q = {instance.vehicle_load_capacity_actual:g}, b_v = {instance.b_v:g})")

    if args.pool_file is not None:
        pool = json.loads(args.pool_file.read_text())
        print(f"Pool: {len(pool)} customers from {args.pool_file}")
    else:
        rng = random.Random(args.seed)
        pool = sorted(rng.sample(instance.C, args.pool_size))
        print(f"Pool: {len(pool)} customers sampled at random (seed = {args.seed})")

    solver = ALNS(instance, pool, seed=args.seed)
    print(f"Initial objective: {solver.current_objective:.4f}")

    solver.run(args.iterations)

    s = solver.summary()
    print(f"\nAfter {s['iterations']:,} iterations:")
    print(f"  objective     = {s['objective_value']:.4f}")
    print(f"  served        = {s['num_served']} / {s['pool_size']} pooled")
    print(f"  unserved pool = {len(s['customers_not_visited_from_selected_pool'])}")
    print(f"  feasible      = {s['feasibility_prize']}")
    print(f"  cpu time      = {s['cpu_time']:.1f} s")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(s, indent=2))
        print(f"\nSummary written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
