#!/usr/bin/env python3
"""
alns_label_generator.py — build the supervised training set in parallel.

Replaces the serial loop in main_ALNS.ipynb. For every reference instance it
runs ALNS over a sampled customer pool under each combination of battery and
load settings, and writes records in exactly the schema `build_graphs.py`
expects.

Design
------
    100 instances (30 R, 30 C, 40 RC)
      x 2 battery settings  (loose = as in the JSON, tight = x0.85)
      x 2 load settings     (loose = as in the JSON, tight = x0.85)
      x 50 pool samples     (a different 85% of customers each time)
    = 20,000 records

Output
------
One JSON file per instance, `labels_<instance_id>.json`, holding that
instance's 200 records. Sharding this way means a crash costs at most one
instance, and `--resume` skips shards that are already complete.

IMPORTANT: no output filename ends in "battery". `build_graphs.py` forces
`vehicle_load_capacity = 50.0` for any file whose stem ends that way — a
workaround for a bug in the old generator. This generator writes the correct
capacity into every record, so that override must not fire.

Reproducibility
---------------
Every run's pool sample and ALNS seed derive deterministically from
(instance_id, battery setting, load setting, sample index) and the global
`--seed`. Re-running reproduces the dataset exactly, and `--resume` after an
interruption produces the same records the uninterrupted run would have.

Usage
-----
    # Check the plan and the time estimate without running anything
    python alns_label_generator.py --dry-run

    # Small smoke test first
    python alns_label_generator.py --instances R_001 R_002 --samples 3 \\
        --iterations 500 --workers 4 --output-dir training_data_v3_smoke

    # The real thing
    python alns_label_generator.py --workers 50 --output-dir "training data v3"

    # Resume after an interruption
    python alns_label_generator.py --workers 50 --output-dir "training data v3" --resume

Then feed the result to the graph builder:

    python build_graphs.py --node-set pool -t "training data v3" -o processed_graphs_v3
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool as ProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from alns import ALNS, Instance


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_DIR = Path("data generation")
DEFAULT_OUTPUT_DIR = Path("training data v3")
DEFAULT_SAMPLES = 35
DEFAULT_POOL_FRACTION = 0.70
DEFAULT_ITERATIONS = 20000
DEFAULT_TIGHT_MULTIPLIER = 0.85
DEFAULT_NUM_VEHICLES = 4
DEFAULT_SEED = 20260101

SETTINGS = ("loose", "tight")
REFERENCE_FILES = {"R": "R_instances.json", "C": "C_instances.json", "RC": "RC_instances.json"}


# ---------------------------------------------------------------------------
# Instance catalogue
# ---------------------------------------------------------------------------
def catalogue(reference_dir: Path) -> List[Tuple[str, str, int]]:
    """List every available instance as (instance_id, type, index in its file)."""
    out: List[Tuple[str, str, int]] = []
    for itype, fname in REFERENCE_FILES.items():
        path = Path(reference_dir) / fname
        if not path.is_file():
            continue
        with open(path, "r") as f:
            records = json.load(f)
        for idx, rec in enumerate(records):
            out.append((rec["instance_id"], itype, idx))
    return out


def build_instance(
    reference_dir: Path,
    instance_type: str,
    instance_index: int,
    battery_setting: str,
    load_setting: str,
    tight_multiplier: float,
    num_vehicles: int,
    load_basis: str,
) -> Instance:
    """
    Load an instance and apply the battery and load settings.

    "loose" uses the value recorded in the reference JSON; "tight" scales it by
    `tight_multiplier`.

    `load_basis` decides how the JSON's `vehicle_capacity` is read:
      fleet        -- it is the total across the fleet, so each vehicle gets
                      capacity / num_vehicles (this is what main_ALNS.ipynb did)
      per_vehicle  -- it is already the per-vehicle capacity
    """
    battery_multiplier = 1.0 if battery_setting == "loose" else tight_multiplier
    load_multiplier = 1.0 if load_setting == "loose" else tight_multiplier

    inst = Instance.from_json(
        reference_dir, instance_type, instance_index,
        battery_multiplier=battery_multiplier,
        num_vehicles=num_vehicles,
        load_basis=load_basis,
    )
    inst.Q = [q * load_multiplier for q in inst.Q]
    return inst


def derive_seed(global_seed: int, instance_id: str, battery: str,
                load: str, sample: int) -> int:
    """
    A stable seed for one run.

    Built from the run's identity rather than a counter, so a run always gets
    the same seed no matter what order the sweep executes in or where it
    resumes from.
    """
    key = f"{global_seed}|{instance_id}|{battery}|{load}|{sample}"
    return int.from_bytes(
        __import__("hashlib").blake2b(key.encode(), digest_size=4).digest(), "big"
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(task: dict) -> dict:
    """
    Run one ALNS job and return a record in build_graphs.py's schema.

    Executed in a worker process. The Instance is rebuilt here rather than
    pickled: it is cheap to construct, and it keeps `alns_core`'s module-level
    instance binding local to this process, which is what makes parallel
    execution safe.
    """
    inst = build_instance(
        Path(task["reference_dir"]), task["instance_type"], task["instance_index"],
        task["battery_setting"], task["load_setting"],
        task["tight_multiplier"], task["num_vehicles"], task["load_basis"],
    )

    # Sample the pool on its own RNG so it does not consume from the search stream.
    pool_rng = random.Random(task["seed"] ^ 0x5EED)
    pool = sorted(pool_rng.sample(list(inst.C), task["pool_size"]))

    solver = ALNS(inst, pool, seed=task["seed"])
    solver.run(n_iterations=task["iterations"])

    served = set(solver.served_customers())
    pool_set = set(pool)

    return {
        # --- fields build_graphs.py reads -------------------------------
        "instance_id": inst.instance_id,
        "instance_type": inst.instance_type,
        "objective_value": solver.global_best_val,
        "customers_not_selected_first_place": sorted(set(inst.C) - pool_set),
        "customers_not_visited_from_selected_pool": sorted(pool_set - served),
        "battery_capacity": inst.b_v,
        "vehicle_load_capacity": inst.Q[0],
        # --- carried for compatibility with the old schema ---------------
        "experiment_id": task["experiment_id"],
        "instance_index": task["instance_index"],
        "nodes_execluded": [],
        "feasibility_prize": solver.feasibility_check(),
        "best solution": [[list(k), v] for k, v in solver.global_best_sol.items()],
        # --- provenance, useful for slicing the dataset later -------------
        "battery_setting": task["battery_setting"],
        "load_setting": task["load_setting"],
        "sample_index": task["sample_index"],
        "pool": pool,
        "pool_size": len(pool),
        "num_served": len(served),
        "iterations": solver.iteration,
        "cpu_seconds": solver.cpu_time,
        "seed": task["seed"],
    }


# ---------------------------------------------------------------------------
# Sharded output
# ---------------------------------------------------------------------------
def shard_path(output_dir: Path, instance_id: str) -> Path:
    # NB: must not end in "battery" — see the module docstring.
    return Path(output_dir) / f"labels_{instance_id}.json"


def shard_is_complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "r") as f:
            return len(json.load(f)) >= expected
    except (json.JSONDecodeError, OSError):
        return False   # treat a truncated shard as missing and redo it


def write_shard(path: Path, records: List[dict]) -> None:
    """Write via a temp file and rename, so a crash cannot leave a half-file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ALNS training labels in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--instances", nargs="+", default=None,
                   help="Instance ids to process. Default: every instance found.")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                   help="Pool samples per (instance, battery, load) combination.")
    p.add_argument("--pool-fraction", type=float, default=DEFAULT_POOL_FRACTION,
                   help="Fraction of customers placed in each sampled pool.")
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument("--tight-multiplier", type=float, default=DEFAULT_TIGHT_MULTIPLIER,
                   help="Scale applied to battery and load in the tight settings.")
    p.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_VEHICLES)
    p.add_argument("--load-basis", choices=["fleet", "per_vehicle"], default="per_vehicle",
                   help="Whether the JSON's vehicle_capacity is a fleet total "
                        "(divided by --num-vehicles, as main_ALNS.ipynb did) or "
                        "already per vehicle.")	
    p.add_argument("--battery-settings", nargs="+", default=list(SETTINGS),
                   choices=list(SETTINGS))
    p.add_argument("--load-settings", nargs="+", default=list(SETTINGS),
                   choices=list(SETTINGS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--resume", action="store_true",
                   help="Skip instances whose shard is already complete.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and the time estimate, then exit.")
    p.add_argument("--sec-per-run", type=float, default=None,
                   help="Seconds per run used for the estimate. Measured from the "
                        "first completed runs when omitted.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {args.reference_dir.resolve()}",
              file=sys.stderr)
        return 1

    all_instances = catalogue(args.reference_dir)
    if not all_instances:
        print(f"ERROR: no reference files in {args.reference_dir.resolve()}", file=sys.stderr)
        return 1

    if args.instances:
        wanted = set(args.instances)
        selected = [t for t in all_instances if t[0] in wanted]
        missing = wanted - {t[0] for t in selected}
        if missing:
            print(f"ERROR: not found in the reference files: {sorted(missing)}",
                  file=sys.stderr)
            return 1
    else:
        selected = all_instances

    per_instance = len(args.battery_settings) * len(args.load_settings) * args.samples
    total = len(selected) * per_instance

    by_type: Dict[str, int] = {}
    for _, t, _ in selected:
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Instances     : {len(selected)}  ({', '.join(f'{v} {k}' for k, v in sorted(by_type.items()))})")
    print(f"Per instance  : {len(args.battery_settings)} battery x "
          f"{len(args.load_settings)} load x {args.samples} samples = {per_instance}")
    print(f"Total records : {total:,}")
    print(f"Iterations    : {args.iterations:,} per run")
    print(f"Pool size     : {args.pool_fraction:.0%} of customers")
    print(f"Load basis    : {args.load_basis}"
          + (f" (capacity / {args.num_vehicles} per vehicle)" if args.load_basis == "fleet" else ""))
    print(f"Tight setting : x{args.tight_multiplier}")
    print(f"Output        : {args.output_dir.resolve()}")

    # 0.025 s/iteration is measured from the existing runs (20k iterations ~ 500 s).
    sec = args.sec_per_run if args.sec_per_run else args.iterations * 0.025
    cpu_hours = total * sec / 3600
    print(f"\nEstimate at {sec:.0f}s per run: {cpu_hours:,.0f} CPU-hours, "
          f"about {cpu_hours / max(args.workers, 1):,.1f} h on {args.workers} workers")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    # ---- build the task list ------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[dict] = []
    skipped_instances = 0

    for instance_id, itype, idx in selected:
        if args.resume and shard_is_complete(shard_path(args.output_dir, instance_id),
                                             per_instance):
            skipped_instances += 1
            continue
        experiment_id = 0
        for battery in args.battery_settings:
            for load in args.load_settings:
                for sample in range(args.samples):
                    tasks.append({
                        "reference_dir": str(args.reference_dir),
                        "instance_id": instance_id,
                        "instance_type": itype,
                        "instance_index": idx,
                        "battery_setting": battery,
                        "load_setting": load,
                        "sample_index": sample,
                        "experiment_id": experiment_id,
                        "pool_size": max(1, round(args.pool_fraction * 100)),
                        "iterations": args.iterations,
                        "tight_multiplier": args.tight_multiplier,
                        "num_vehicles": args.num_vehicles,
                        "load_basis": args.load_basis,
                        "seed": derive_seed(args.seed, instance_id, battery, load, sample),
                    })
                    experiment_id += 1

    if args.resume:
        print(f"\nResuming: {skipped_instances} instance shards already complete")
    if not tasks:
        print("Nothing to do.")
        return 0

    # Pool size depends on the instance's customer count; fix it up per task.
    counts = {i: None for i, _, _ in selected}
    for t in tasks:
        if counts[t["instance_id"]] is None:
            inst = Instance.from_json(args.reference_dir, t["instance_type"],
                                      t["instance_index"])
            counts[t["instance_id"]] = inst.n
        t["pool_size"] = max(1, round(args.pool_fraction * counts[t["instance_id"]]))

    print(f"\nRunning {len(tasks):,} jobs on {args.workers} workers ...")

    # ---- execute -------------------------------------------------------
    buffers: Dict[str, List[dict]] = {i: [] for i, _, _ in selected}
    completed = 0
    shards_written = 0
    t0 = time.perf_counter()
    report_every = max(1, len(tasks) // 200)

    with ProcessPool(processes=args.workers) as pool:
        for rec in pool.imap_unordered(_run_one, tasks, chunksize=1):
            buffers[rec["instance_id"]].append(rec)
            completed += 1

            # Flush a shard as soon as its instance is finished, so progress
            # survives an interruption and memory stays bounded.
            if len(buffers[rec["instance_id"]]) >= per_instance:
                recs = sorted(buffers.pop(rec["instance_id"]),
                              key=lambda r: r["experiment_id"])
                write_shard(shard_path(args.output_dir, rec["instance_id"]), recs)
                buffers[rec["instance_id"]] = []
                shards_written += 1

            if completed % report_every == 0 or completed == len(tasks):
                el = time.perf_counter() - t0
                rate = completed / el
                eta = (len(tasks) - completed) / rate / 3600 if rate > 0 else 0
                print(f"  {completed:>7,}/{len(tasks):,} ({100*completed/len(tasks):5.1f}%)  "
                      f"shards {shards_written:>3}  elapsed {el/3600:5.2f} h  "
                      f"eta {eta:5.2f} h", flush=True)

    # Flush anything left over (a partially completed instance).
    for instance_id, recs in buffers.items():
        if recs:
            write_shard(shard_path(args.output_dir, instance_id),
                        sorted(recs, key=lambda r: r["experiment_id"]))
            shards_written += 1

    elapsed_h = (time.perf_counter() - t0) / 3600
    wall_per_run = elapsed_h * 3600 / max(completed, 1)
    cpu_per_run = wall_per_run * args.workers
    print(f"\nDone: {completed:,} records in {shards_written} shards, {elapsed_h:.2f} h wall clock")
    print(f"Throughput: {completed / max(elapsed_h, 1e-9):,.0f} runs/hour  "
          f"({cpu_per_run:.0f}s CPU per run across {args.workers} workers)")
    print(f"Pass --sec-per-run {cpu_per_run:.0f} to get an accurate estimate next time.")
    print(f"\nNext:\n  python build_graphs.py --node-set pool "
          f"-t \"{args.output_dir}\" -o processed_graphs_v3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
