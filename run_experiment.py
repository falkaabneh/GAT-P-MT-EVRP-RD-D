#!/usr/bin/env python3
"""
run_experiment.py — the six-arm pool-selection comparison.

Tests whether the GAT surrogate picks better customer pools than random or
greedy selection, by running ALNS from six different starting pools under an
identical iteration budget.

Arms
----
    B    random      uniformly at random
    C1   prize       highest prize
    C2   penalty     highest penalty
    C3   tw_spread   windows tiling the horizon
    C4   cluster     taken cluster by cluster, keeping clusters whole
    C5   density     best (prize + penalty) per unit detour
    D    surrogate   best of ~500 candidates, ranked by the GAT

Every arm selects the same number of customers, so the comparison is apples to
apples: all face the same structural handicap from the objective charging
penalties for customers that were never pooled, and differences reflect
selection strategy alone.

Design
------
The unit of analysis is a *block*: one (instance, battery setting, seed)
triple. Every arm runs within a block on the same instance, under the same
battery setting, with the same ALNS seed, so instance difficulty and search
luck cancel when arms are compared as paired differences rather than as raw
means. Per-run noise is roughly 4% of the objective, so paired differences —
not raw averages — are what the analysis reports.

Battery settings
----------------
"loose" uses the battery capacity recorded in the reference JSON; "tight"
scales it by --tight-multiplier (0.85 by default), matching how the training
labels were generated. Battery is the binding resource in this multi-trip
problem — load capacity barely constrains anything, since the vehicle reloads
at the depot — so the tight setting is where far fewer customers are servable
and pool composition should matter most.

The pools for arms B, C1, C2 and C3 are identical across battery settings, so
the two settings form a paired comparison. Arm D may differ between them: the
candidate set is the same, but the surrogate conditions on battery capacity and
so can rank the candidates differently.

Arm D's candidates include the exact pools used by every other arm (B, C1-C5),
so D can only lose through surrogate misranking. The remainder mixes randomized
greedy across all five criteria, perturbed neighbourhoods of the arm pools, and
uniform random draws, so the surrogate can also reach pools that no single
heuristic would produce.

Budget
------
Under `--time-limit` every arm gets the same CPU seconds. Use `--iterations`
instead for an equal-iteration budget. Pass `--iterations 0` alongside
`--time-limit` to make the budget purely time-based; otherwise whichever limit
is reached first applies.

Execution
---------
Two phases. Phase 1 builds every pool and runs the surrogate (GPU, serial).
Phase 2 dispatches the ALNS runs across worker processes (CPU-bound, trivially
parallel). Results stream to a JSONL file as they finish, so an interrupted
sweep can be resumed with --resume.

Usage
-----
    # The full design: 30 instances x 10 seeds x 6 arms = 1800 runs
    python run_experiment.py --checkpoint models/t2ur1ua6/best.pt --workers 50

    # A quick smoke test first — always do this before committing hours of CPU
    python run_experiment.py --checkpoint models/t2ur1ua6/best.pt \\
        --instances RC_001 RC_002 --seeds 2 --iterations 500 --workers 4 \\
        --output-dir results/smoke

    # Resume an interrupted sweep
    python run_experiment.py --checkpoint models/t2ur1ua6/best.pt --workers 50 --resume

    # Analyse without re-running
    python run_experiment.py --analyse-only --output-dir results/arm_comparison \\
        --baseline C4_cluster
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool as ProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from alns import ALNS, Instance
import pools as poolgen


ARMS = ["B_random", "C1_prize", "C2_penalty", "C3_tw_spread",
        "C4_cluster", "C5_density", "D_surrogate"]

DEFAULT_INSTANCES = (
    [f"R_{i:03d}" for i in range(1, 11)]
    + [f"C_{i:03d}" for i in range(1, 11)]
    + [f"RC_{i:03d}" for i in range(1, 11)]
)


# ---------------------------------------------------------------------------
# Instance lookup
# ---------------------------------------------------------------------------
def battery_multiplier(battery_setting: str, tight_multiplier: float) -> float:
    """1.0 for the loose setting, `tight_multiplier` for tight."""
    return 1.0 if battery_setting == "loose" else tight_multiplier


def resolve_instance(reference_dir: Path, instance_id: str,
                     battery_setting: str = "loose",
                     tight_multiplier: float = 0.85) -> Instance:
    """
    Load an instance by its id (e.g. "RC_031").

    Looks the id up in the reference file rather than assuming index == id - 1,
    so a reordered reference file cannot silently shift which instance runs.
    """
    instance_type = instance_id.rsplit("_", 1)[0]
    files = {"R": "R_instances.json", "C": "C_instances.json", "RC": "RC_instances.json"}
    if instance_type not in files:
        raise ValueError(f"cannot parse a type from instance id {instance_id!r}")

    path = Path(reference_dir) / files[instance_type]
    with open(path, "r") as f:
        records = json.load(f)

    for idx, rec in enumerate(records):
        if rec["instance_id"] == instance_id:
            return Instance.from_json(
                reference_dir, instance_type, idx,
                battery_multiplier=battery_multiplier(battery_setting, tight_multiplier),
            )
    raise KeyError(f"{instance_id} not found in {path}")


# ---------------------------------------------------------------------------
# Phase 1 — pools
# ---------------------------------------------------------------------------
@dataclass
class BlockPools:
    """The pools for one (instance, battery, seed) block, plus arm D diagnostics."""
    instance_id: str
    instance_type: str
    instance_index: int
    seed: int
    battery_setting: str
    pools: Dict[str, List[int]]
    surrogate_predicted: Optional[float] = None   # D's predicted objective
    surrogate_origin: Optional[str] = None        # which generator produced D's pool
    n_candidates: int = 0
    screening_seconds: float = 0.0


def build_block_pools(
    instances_by_battery: Dict[str, Instance],
    seed: int,
    predictor,
    pool_size: int,
    n_candidates: int,
    rcl_k: int,
    vehicle_load_for_surrogate: float,
    n_clusters: int = poolgen.DEFAULT_N_CLUSTERS,
) -> Dict[str, BlockPools]:
    """
    Build the pools for one (instance, seed) block, for every battery setting.

    Pool generation is battery-independent — the criteria use prize, penalty and
    time windows — so arms B, C1, C2 and C3 get identical pools across settings,
    making the two batteries a paired comparison. The candidate set is likewise
    built once and shared; only the surrogate's ranking of it varies, since the
    model conditions on battery capacity. Scoring the shared candidate set twice
    also halves the graph-construction work compared with rebuilding it.

    The pool RNG is derived from the block seed but kept separate from the ALNS
    seed, so two arms never accidentally share a random draw.
    """
    base = next(iter(instances_by_battery.values()))   # geometry is shared
    rng = random.Random((seed + 1) * 1_000_003 + base.instance_index)

    common: Dict[str, List[int]] = {
        "B_random":     poolgen.random_pool(base, pool_size, rng),
        "C1_prize":     poolgen.greedy_prize(base, pool_size),
        "C2_penalty":   poolgen.greedy_penalty(base, pool_size),
        "C3_tw_spread": poolgen.greedy_tw_spread(base, pool_size),
        "C4_cluster":   poolgen.greedy_cluster(base, pool_size, n_clusters),
        "C5_density":   poolgen.greedy_value_density(base, pool_size),
    }

    t0 = time.perf_counter()
    candidates, origins = poolgen.build_candidate_set(
        base,
        n_candidates=n_candidates,
        size=pool_size,
        k=rcl_k,
        rng=rng,
        include_pools=common,
    )
    build_seconds = time.perf_counter() - t0

    out: Dict[str, BlockPools] = {}
    for battery, inst in instances_by_battery.items():
        t1 = time.perf_counter()
        scored = predictor.score_subsets([
            (inst.instance_id, cand, inst.b_v, vehicle_load_for_surrogate)
            for cand in candidates
        ])
        best = int(np.argmin(scored))   # minimisation: most negative predicted objective
        pools = dict(common)
        pools["D_surrogate"] = candidates[best]

        out[battery] = BlockPools(
            instance_id=inst.instance_id,
            instance_type=inst.instance_type,
            instance_index=inst.instance_index,
            seed=seed,
            battery_setting=battery,
            pools=pools,
            surrogate_predicted=float(scored[best]),
            surrogate_origin=origins[best],
            n_candidates=len(candidates),
            # Candidate construction is shared, so charge it to each setting once.
            screening_seconds=build_seconds + (time.perf_counter() - t1),
        )
    return out


# ---------------------------------------------------------------------------
# Phase 2 — ALNS worker
# ---------------------------------------------------------------------------
def _run_one(task: dict) -> dict:
    """
    Run one arm of one block, never raising. Executed in a worker process.

    Any exception is caught and returned as a record carrying an "error" key.
    imap_unordered re-raises a worker exception in the parent and tears the
    whole pool down, so an unguarded failure in run 1,575 of 4,200 would end
    the sweep; catching here lets the remaining runs finish and leaves the
    failures to be retried on the next --resume.

    The Instance is rebuilt here rather than pickled: construction takes
    milliseconds and it keeps `alns_core`'s module-level instance binding local
    to this process, which is what makes parallel execution safe.
    """
    try:
        return _run_one_inner(task)
    except Exception as e:
        import traceback
        return {
            "run_id": task["run_id"],
            "instance_id": task.get("instance_id"),
            "seed": task.get("seed"),
            "battery_setting": task.get("battery_setting", "loose"),
            "arm": task.get("arm"),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _run_one_inner(task: dict) -> dict:
    """The actual run. Wrapped by _run_one so exceptions cannot kill the pool."""
    instance = Instance.from_json(
        Path(task["reference_dir"]), task["instance_type"], task["instance_index"],
        battery_multiplier=battery_multiplier(task.get("battery_setting", "loose"),
                                              task.get("tight_multiplier", 0.85)),
    )
    solver = ALNS(instance, task["pool"], seed=task["seed"])
    initial = solver.current_objective
    t0 = time.perf_counter()

    solver.run(n_iterations=task.get("iterations"),
               time_limit=task.get("time_limit"))
    wall = time.perf_counter() - t0

    s = solver.summary()
    return {
        "run_id":            task["run_id"],
        "instance_id":       task["instance_id"],
        "instance_type":     task["instance_type"],
        "seed":               task["seed"],
        "battery_setting":   task.get("battery_setting", "loose"),
        "battery_capacity":  instance.b_v,
        "arm":               task["arm"],
        "pool_size":         len(task["pool"]),
        "iterations":        s["iterations"],
        "initial_objective": initial,
        "objective":         s["objective_value"],
        "num_served":        s["num_served"],
        "unserved_in_pool":  len(s["customers_not_visited_from_selected_pool"]),
        "feasible":          s["feasibility_prize"],
        "cpu_seconds":       s["cpu_time"],
        "wall_seconds":      wall,
        # Arm D pays for its own screening; other arms have no selection cost.
        "screening_seconds": task.get("screening_seconds", 0.0),
        "total_seconds":     wall + task.get("screening_seconds", 0.0),
        "surrogate_predicted": task.get("surrogate_predicted"),
        "surrogate_origin":    task.get("surrogate_origin"),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse(records: List[dict], baseline: str = "B_random") -> Dict:
    """
    Paired analysis across blocks.

    Blocks missing any arm are skipped, so every reported difference rests on
    the same set of (instance, seed) pairs.
    """
    # Records written before battery settings existed carry no field; they were
    # all loose, so default accordingly and keep older results comparable.
    by_block: Dict[Tuple[str, str, int], Dict[str, dict]] = defaultdict(dict)
    for r in records:
        if "error" in r or "objective" not in r:
            continue
        key = (r["instance_id"], r.get("battery_setting", "loose"), r["seed"])
        by_block[key][r["arm"]] = r

    complete = {b: v for b, v in by_block.items() if all(a in v for a in ARMS)}
    incomplete = len(by_block) - len(complete)

    out = {
        "n_blocks": len(complete),
        "n_incomplete_blocks": incomplete,
        "baseline": baseline,
        "per_arm": {},
        "paired_vs_baseline": {},
        "per_arm_by_type": {},
        "per_arm_by_battery": {},
        "paired_vs_baseline_by_battery": {},
        "win_matrix": {},
    }
    if not complete:
        return out

    for arm in ARMS:
        vals = np.array([v[arm]["objective"] for v in complete.values()])
        served = np.array([v[arm]["num_served"] for v in complete.values()])
        secs = np.array([v[arm]["total_seconds"] for v in complete.values()])
        out["per_arm"][arm] = {
            "mean_objective":  float(vals.mean()),
            "std_objective":   float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "sem_objective":   float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "best_objective":  float(vals.min()),
            "worst_objective": float(vals.max()),
            "mean_served":     float(served.mean()),
            "mean_total_seconds": float(secs.mean()),
        }

    # Paired differences: negative favours the arm, since this is a minimisation.
    for arm in ARMS:
        if arm == baseline:
            continue
        d = np.array([v[arm]["objective"] - v[baseline]["objective"]
                      for v in complete.values()])
        n = len(d)
        sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out["paired_vs_baseline"][arm] = {
            "mean_difference": float(d.mean()),
            "sem_difference":  sem,
            "ci95_low":        float(d.mean() - 1.96 * sem),
            "ci95_high":       float(d.mean() + 1.96 * sem),
            "wins":            int((d < 0).sum()),
            "losses":          int((d > 0).sum()),
            "ties":            int((d == 0).sum()),
            "n":               n,
        }

    for arm in ARMS:
        by_type: Dict[str, List[float]] = defaultdict(list)
        for v in complete.values():
            by_type[v[arm]["instance_type"]].append(v[arm]["objective"])
        out["per_arm_by_type"][arm] = {
            t: {"n": len(vs), "mean_objective": float(np.mean(vs))}
            for t, vs in sorted(by_type.items())
        }

    batteries = sorted({k[1] for k in complete})
    for arm in ARMS:
        out["per_arm_by_battery"][arm] = {}
        for b in batteries:
            vals = np.array([v[arm]["objective"] for k, v in complete.items() if k[1] == b])
            served = np.array([v[arm]["num_served"] for k, v in complete.items() if k[1] == b])
            out["per_arm_by_battery"][arm][b] = {
                "n": len(vals),
                "mean_objective": float(vals.mean()),
                "mean_served": float(served.mean()),
            }

    for b in batteries:
        out["paired_vs_baseline_by_battery"][b] = {}
        for arm in ARMS:
            if arm == baseline:
                continue
            d = np.array([v[arm]["objective"] - v[baseline]["objective"]
                          for k, v in complete.items() if k[1] == b])
            if not len(d):
                continue
            sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            out["paired_vs_baseline_by_battery"][b][arm] = {
                "mean_difference": float(d.mean()),
                "sem_difference": sem,
                "ci95_low": float(d.mean() - 1.96 * sem),
                "ci95_high": float(d.mean() + 1.96 * sem),
                "wins": int((d < 0).sum()),
                "losses": int((d > 0).sum()),
                "n": len(d),
            }

    for a in ARMS:
        out["win_matrix"][a] = {}
        for b in ARMS:
            if a == b:
                out["win_matrix"][a][b] = None
            else:
                out["win_matrix"][a][b] = int(
                    sum(v[a]["objective"] < v[b]["objective"] for v in complete.values())
                )
    return out


def print_analysis(a: Dict) -> None:
    if not a["n_blocks"]:
        print("No complete blocks to analyse.")
        return

    print(f"\n{'=' * 78}")
    print(f"RESULTS — {a['n_blocks']} complete blocks"
          + (f" ({a['n_incomplete_blocks']} incomplete, skipped)"
             if a["n_incomplete_blocks"] else ""))
    print("=" * 78)

    print(f"\nPer arm (objective is minimised, so lower is better)\n")
    print(f"  {'arm':<18} {'mean':>11} {'sem':>8} {'best':>11} "
          f"{'served':>8} {'sec/run':>9}")
    print(f"  {'-'*18} {'-'*11} {'-'*8} {'-'*11} {'-'*8} {'-'*9}")
    for arm in ARMS:
        m = a["per_arm"][arm]
        print(f"  {arm:<18} {m['mean_objective']:>11.2f} {m['sem_objective']:>8.2f} "
              f"{m['best_objective']:>11.2f} "
              f"{m['mean_served']:>8.1f} {m['mean_total_seconds']:>9.1f}")

    print(f"\nPaired differences vs {a['baseline']} "
          f"(negative = better than baseline)\n")
    print(f"  {'arm':<18} {'mean diff':>11} {'95% CI':>22} {'W-L-T':>12} {'sig':>4}")
    print(f"  {'-'*18} {'-'*11} {'-'*22} {'-'*12} {'-'*4}")
    for arm, d in a["paired_vs_baseline"].items():
        ci = f"[{d['ci95_low']:>8.2f}, {d['ci95_high']:>8.2f}]"
        wlt = f"{d['wins']}-{d['losses']}-{d['ties']}"
        # Starred when the 95% CI lies entirely on one side of zero.
        sig = "" if d["ci95_low"] <= 0 <= d["ci95_high"] else "*"
        print(f"  {arm:<18} {d['mean_difference']:>11.2f} {ci:>22} {wlt:>12} {sig:>4}")
    print("\n  * 95% CI excludes zero.")

    if len(a.get("per_arm_by_battery", {}).get(ARMS[0], {})) > 1:
        bats = sorted(a["per_arm_by_battery"][ARMS[0]].keys())
        print(f"\nMean objective by battery setting\n")
        print("  " + f"{'arm':<18}" + "".join(f"{b:>14}" for b in bats))
        print("  " + "-" * 18 + "".join("-" * 14 for _ in bats))
        for arm in ARMS:
            row = f"  {arm:<18}"
            for b in bats:
                m = a["per_arm_by_battery"][arm].get(b)
                row += f"{m['mean_objective']:>14.2f}" if m else f"{'-':>14}"
            print(row)

        print(f"\nMean customers served by battery setting\n")
        print("  " + f"{'arm':<18}" + "".join(f"{b:>14}" for b in bats))
        print("  " + "-" * 18 + "".join("-" * 14 for _ in bats))
        for arm in ARMS:
            row = f"  {arm:<18}"
            for b in bats:
                m = a["per_arm_by_battery"][arm].get(b)
                row += f"{m['mean_served']:>14.1f}" if m else f"{'-':>14}"
            print(row)

        for b in bats:
            print(f"\nPaired differences vs {a['baseline']}, battery = {b}\n")
            print(f"  {'arm':<18} {'mean diff':>11} {'95% CI':>22} {'W-L':>10} {'sig':>4}")
            print(f"  {'-'*18} {'-'*11} {'-'*22} {'-'*10} {'-'*4}")
            for arm, d in a["paired_vs_baseline_by_battery"][b].items():
                ci = f"[{d['ci95_low']:>8.2f}, {d['ci95_high']:>8.2f}]"
                sig = "" if d["ci95_low"] <= 0 <= d["ci95_high"] else "*"
                print(f"  {arm:<18} {d['mean_difference']:>11.2f} {ci:>22} "
                      f"{str(d['wins'])+'-'+str(d['losses']):>10} {sig:>4}")

    print(f"\nMean objective by instance type\n")
    types = sorted({t for arm in ARMS for t in a["per_arm_by_type"][arm]})
    print("  " + f"{'arm':<18}" + "".join(f"{t:>12}" for t in types))
    print("  " + "-" * 18 + "".join("-" * 12 for _ in types))
    for arm in ARMS:
        row = f"  {arm:<18}"
        for t in types:
            v = a["per_arm_by_type"][arm].get(t)
            row += f"{v['mean_objective']:>12.2f}" if v else f"{'-':>12}"
        print(row)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "run_id", "instance_id", "instance_type", "seed", "arm", "pool_size",
    "iterations", "initial_objective", "objective", "num_served",
    "unserved_in_pool", "feasible", "cpu_seconds", "wall_seconds",
    "screening_seconds", "total_seconds", "surrogate_predicted", "surrogate_origin",
    "battery_setting", "battery_capacity",
]


def write_csv(records: List[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def load_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # tolerate a truncated final line from an interrupted run
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Six-arm comparison of customer-pool selection strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="GAT checkpoint for arm D. Required unless --analyse-only.")
    p.add_argument("--reference-dir", type=Path, default=Path("data generation"))
    p.add_argument("--prepared-config", type=Path, default=Path("prepared_data_pool/config.json"),
                   help="config.json from the data_prep run that matches the checkpoint.")
    p.add_argument("--output-dir", type=Path, default=Path("results/arm_comparison"))

    p.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES,
                   help="Instance ids to run.")
    p.add_argument("--seeds", type=int, default=10, help="Seeds per instance (0..seeds-1).")
    p.add_argument("--battery-settings", nargs="+", default=["loose"],
                   choices=["loose", "tight"],
                   help="Battery regimes to run. 'loose' is the capacity in the "
                        "reference JSON; 'tight' scales it by --tight-multiplier.")
    p.add_argument("--tight-multiplier", type=float, default=0.85,
                   help="Battery scale for the tight setting. Must match the value "
                        "used by alns_label_generator.py.")
    p.add_argument("--iterations", type=int, default=20000,
                   help="Iteration budget per run. Ignored when --time-limit is set.")
    p.add_argument("--time-limit", type=float, default=None,
                   help="CPU-seconds budget per run. Overrides --iterations and is "
                        "the fair budget when arms differ in pool size.")
    p.add_argument("--pool-size", type=int, default=poolgen.DEFAULT_POOL_SIZE)
    p.add_argument("--candidates", type=int, default=500, help="Candidate pools for arm D.")
    p.add_argument("--rcl-k", type=int, default=poolgen.DEFAULT_RCL_SIZE,
                   help="Restricted candidate list size for randomized greedy.")
    p.add_argument("--n-clusters", type=int, default=poolgen.DEFAULT_N_CLUSTERS,
                   help="Clusters used by the C4 cluster-aware strategy.")
    p.add_argument("--vehicle-load", type=float, default=50.0,
                   help="vehicle_load_capacity fed to the surrogate. Must match the "
                        "capacity the ALNS enforces (big_Q / num_vehicles).")

    p.add_argument("--workers", type=int, default=8, help="Parallel ALNS processes.")
    p.add_argument("--resume", action="store_true",
                   help="Skip runs already present in results.jsonl.")
    p.add_argument("--analyse-only", action="store_true",
                   help="Re-analyse an existing results.jsonl without running anything.")
    p.add_argument("--baseline", default="B_random", choices=ARMS,
                   help="Arm the paired differences are taken against.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "results.jsonl"

    # ---- analyse-only -----------------------------------------------------
    if args.analyse_only:
        records = load_jsonl(jsonl_path)
        if not records:
            print(f"ERROR: no results at {jsonl_path.resolve()}", file=sys.stderr)
            return 1
        print(f"Loaded {len(records):,} runs from {jsonl_path}")
        a = analyse(records, baseline=args.baseline)
        print_analysis(a)
        (args.output_dir / "analysis.json").write_text(json.dumps(a, indent=2))
        write_csv(records, args.output_dir / "results.csv")
        return 0

    if args.checkpoint is None:
        print("ERROR: --checkpoint is required (or pass --analyse-only)", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint.resolve()}", file=sys.stderr)
        return 1

    done = set()
    existing: List[dict] = []
    if args.resume:
        existing = load_jsonl(jsonl_path)
        done = {r["run_id"] for r in existing}
        print(f"Resuming: {len(done):,} runs already complete")

    # ---- phase 1: pools and surrogate screening ---------------------------
    from predict import Predictor   # imported late so --analyse-only needs no torch

    print(f"Loading surrogate from {args.checkpoint} ...")
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir,
        prepared_config=args.prepared_config if args.prepared_config.is_file() else None,
        node_set="pool",
    )
    print(f"  device={predictor.device}  pe_dim={predictor.pe_dim}  pe_base={predictor.pe_base}")

    budget = (f"{args.time_limit:g} CPU-seconds" if args.time_limit is not None
              else f"{args.iterations:,} iterations")
    print(f"\nBudget per run: {budget}")
    print(f"Battery settings: {', '.join(args.battery_settings)}"
          + (f" (tight = x{args.tight_multiplier})" if "tight" in args.battery_settings else ""))
    print(f"Pool size: {args.pool_size} customers per arm")

    print(f"\nPhase 1: building pools for {len(args.instances)} instances "
          f"x {len(args.battery_settings)} battery x {args.seeds} seeds ...")
    t0 = time.perf_counter()
    tasks: List[dict] = []
    pool_log: List[dict] = []

    for instance_id in args.instances:
        try:
            instances_by_battery = {
                b: resolve_instance(args.reference_dir, instance_id, b, args.tight_multiplier)
                for b in args.battery_settings
            }
        except (KeyError, FileNotFoundError) as e:
            print(f"  SKIP {instance_id}: {e}", file=sys.stderr)
            continue

        for seed in range(args.seeds):
            blocks = build_block_pools(
                instances_by_battery, seed, predictor,
                pool_size=args.pool_size,
                n_candidates=args.candidates,
                rcl_k=args.rcl_k,
                vehicle_load_for_surrogate=args.vehicle_load,
                n_clusters=args.n_clusters,
            )
            for battery, bp in blocks.items():
                pool_log.append({
                    "instance_id": bp.instance_id, "seed": bp.seed,
                    "battery_setting": battery,
                    "battery_capacity": instances_by_battery[battery].b_v,
                    "n_candidates": bp.n_candidates,
                    "surrogate_predicted": bp.surrogate_predicted,
                    "surrogate_origin": bp.surrogate_origin,
                    "screening_seconds": bp.screening_seconds,
                    "pools": bp.pools,
                })
                for arm in ARMS:
                    # Loose runs keep the original id shape so results collected
                    # before battery settings existed still match on --resume.
                    run_id = (f"{instance_id}|s{seed}|{arm}" if battery == "loose"
                              else f"{instance_id}|{battery}|s{seed}|{arm}")
                    if run_id in done:
                        continue
                    uses_surrogate = arm == "D_surrogate"
                    tasks.append({
                        "run_id": run_id,
                        "reference_dir": str(args.reference_dir),
                        "instance_id": instance_id,
                        "instance_type": bp.instance_type,
                        "instance_index": bp.instance_index,
                        "seed": seed,
                        "battery_setting": battery,
                        "tight_multiplier": args.tight_multiplier,
                        "arm": arm,
                        "pool": bp.pools[arm],
                        "iterations": args.iterations,
                        "time_limit": args.time_limit,
                        "screening_seconds": bp.screening_seconds if uses_surrogate else 0.0,
                        "surrogate_predicted": bp.surrogate_predicted if uses_surrogate else None,
                        "surrogate_origin": bp.surrogate_origin if uses_surrogate else None,
                    })
        print(f"  {instance_id}: {args.seeds} seeds x {len(args.battery_settings)} battery")

    (args.output_dir / "pools.json").write_text(json.dumps(pool_log, indent=2))
    print(f"Phase 1 done in {time.perf_counter() - t0:.1f}s — {len(tasks):,} runs queued")

    origins = defaultdict(lambda: defaultdict(int))
    for e in pool_log:
        origins[e["battery_setting"]][e["surrogate_origin"]] += 1
    for b, counts in origins.items():
        print(f"  arm D picks ({b} battery):", dict(counts))

    if not tasks:
        print("Nothing to run.")
    else:
        # ---- phase 2: parallel ALNS ---------------------------------------
        per_run = (args.time_limit if args.time_limit is not None
                   else args.iterations * 0.025)
        est = len(tasks) * per_run / max(args.workers, 1) / 60
        print(f"\nPhase 2: {len(tasks):,} ALNS runs on {args.workers} workers "
              f"(rough estimate {est:.0f} min)")
        t1 = time.perf_counter()
        completed = 0
        failed = 0
        errors_path = args.output_dir / "errors.jsonl"
        with open(jsonl_path, "a") as sink, open(errors_path, "a") as err_sink:
            with ProcessPool(processes=args.workers) as pool:
                for rec in pool.imap_unordered(_run_one, tasks, chunksize=1):
                    if "error" in rec:
                        # Not written to results.jsonl, so --resume retries it.
                        err_sink.write(json.dumps(rec) + "\n")
                        err_sink.flush()
                        failed += 1
                        if failed <= 5:
                            print(f"  FAILED {rec['run_id']}: {rec['error']}", flush=True)
                        elif failed == 6:
                            print("  (further failures logged to errors.jsonl only)", flush=True)
                        continue
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()   # stream results so an interrupt loses at most one run
                    completed += 1
                    attempted = completed + failed
                    if attempted % max(1, len(tasks) // 40) == 0 or attempted == len(tasks):
                        el = time.perf_counter() - t1
                        rate = attempted / el
                        eta = (len(tasks) - attempted) / rate / 60 if rate > 0 else 0
                        print(f"  {attempted:>5}/{len(tasks)}  "
                              f"({100*attempted/len(tasks):5.1f}%)  "
                              f"elapsed {el/60:6.1f} min  eta {eta:6.1f} min", flush=True)
        print(f"Phase 2 done in {(time.perf_counter() - t1)/60:.1f} min"
              + (f" — {failed} run(s) failed, see errors.jsonl; "
                 f"re-run with --resume to retry them" if failed else ""))

    # ---- analysis ---------------------------------------------------------
    records = load_jsonl(jsonl_path)
    a = analyse(records, baseline=args.baseline)
    print_analysis(a)

    (args.output_dir / "analysis.json").write_text(json.dumps(a, indent=2))
    write_csv(records, args.output_dir / "results.csv")
    print(f"\nWritten to {args.output_dir.resolve()}:")
    print("  results.jsonl   one record per run")
    print("  results.csv     the same, flat")
    print("  pools.json      every pool used, plus arm D screening detail")
    print("  analysis.json   paired differences and summaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
