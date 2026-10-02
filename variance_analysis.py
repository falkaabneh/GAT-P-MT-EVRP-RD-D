#!/usr/bin/env python3
"""
variance_analysis.py — how much does the customer pool matter, and how noisy is
the ALNS that labels it?

Two analyses that answer complementary questions, and a third that combines them.

  pool       Across many different random pools for the same instance, how much
             does the objective move? Read straight from the generated labels —
             each (instance, battery, load) cell already holds many independent
             pool samples, so this costs no compute.

  stability  For ONE fixed pool, run the ALNS repeatedly with different seeds
             and plot the convergence band. Shows whether a single 20,000
             iteration run is a reliable label or a lottery ticket.

  both       Runs the two and decomposes the variance:

                 Var(across pools) = Var(pool choice) + Var(ALNS noise)

             The pool analysis measures the left side; the stability analysis
             measures the last term directly. The difference is the share of
             objective variance attributable to which customers were pooled —
             which is precisely what a pool-selection surrogate can exploit.

Usage
-----
    # Pool variation, one instance per topology, from existing labels
    python variance_analysis.py --analysis pool \\
        --labels "training data v4" --output-dir figures/variance

    # ALNS stability: 15 seeds on one fixed pool
    python variance_analysis.py --analysis stability \\
        --instance R_001 --repeats 15 --iterations 20000 --workers 15 \\
        --output-dir figures/variance

    # Both, with the decomposition
    python variance_analysis.py --analysis both \\
        --labels "training data v4" --instance R_001 --repeats 15 \\
        --workers 15 --output-dir figures/variance

Outputs
-------
    fig_pool_variation.{pdf,png}   objective across pools, one panel per topology
    fig_alns_stability.{pdf,png}   convergence band + final-value spread
    variance_summary.csv           every statistic behind the figures
    stability_traces.npz           raw per-iteration traces, for re-plotting
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool as ProcessPool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alns import ALNS, Instance


COL_WIDTH, FULL_WIDTH = 3.4, 7.0
TYPE_ORDER = ["C", "R", "RC"]
TYPE_COLORS = {"C": "#0072B2", "R": "#E69F00", "RC": "#009E73"}
REFERENCE_FILES = {"R": "R_instances.json", "C": "C_instances.json", "RC": "RC_instances.json"}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


# ---------------------------------------------------------------------------
# Analysis 1 — variation across pools, from the labels
# ---------------------------------------------------------------------------
def load_labels(labels_dir: Path) -> List[dict]:
    files = sorted(Path(labels_dir).glob("*.json")) + sorted(Path(labels_dir).glob("*.JSON"))
    records: List[dict] = []
    for fp in dict.fromkeys(files):
        try:
            with open(fp) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: {fp.name} is not valid JSON, skipping", file=sys.stderr)
            continue
        records.extend(data if isinstance(data, list) else [data])
    print(f"  loaded {len(records):,} label records from {labels_dir}")
    return records


def pick_instances(records: List[dict], requested: Optional[Sequence[str]]) -> Dict[str, str]:
    """One instance id per topology — the caller's choice, else the first found."""
    if requested:
        chosen = {}
        by_id = {r["instance_id"]: r["instance_type"] for r in records}
        for iid in requested:
            if iid not in by_id:
                raise KeyError(f"{iid} not present in the labels")
            chosen[by_id[iid]] = iid
        return chosen
    out: Dict[str, str] = {}
    for r in records:
        out.setdefault(r["instance_type"], r["instance_id"])
    return out


def pool_variation(records: List[dict], instances: Dict[str, str],
                   battery: str, load: str, out: Path) -> List[dict]:
    """
    Objective across independent random pools, one panel per topology.

    Held at a single (battery, load) setting so the spread reflects pool choice
    and ALNS noise only, not the capacity regime.
    """
    panels = [t for t in TYPE_ORDER if t in instances]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 * len(panels) + 0.6, 2.7),
                             squeeze=False)
    axes = axes[0]
    rows = []

    for ax, topo in zip(axes, panels):
        iid = instances[topo]
        vals = np.array([
            r["objective_value"] for r in records
            if r["instance_id"] == iid
            and r.get("battery_setting", battery) == battery
            and r.get("load_setting", load) == load
        ], dtype=float)
        if not len(vals):
            ax.axis("off")
            continue

        ax.hist(vals, bins=min(20, max(6, len(vals) // 2)),
                color=TYPE_COLORS.get(topo, "#0072B2"),
                alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(vals.mean(), color="#D55E00", linewidth=1.3,
                   label=f"mean {vals.mean():.0f}")
        ax.axvline(vals.min(), color="0.3", linestyle=":", linewidth=1.0)
        ax.axvline(vals.max(), color="0.3", linestyle=":", linewidth=1.0)

        spread = 100.0 * (vals.max() - vals.min()) / abs(vals.mean()) if vals.mean() else np.nan
        ax.set_title(f"{topo} — {iid}  (n={len(vals)})", loc="left", fontsize=9)
        ax.set_xlabel("Objective value")
        if ax is axes[0]:
            ax.set_ylabel("Pools")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)

        rows.append({
            "analysis": "pool", "topology": topo, "instance_id": iid,
            "battery": battery, "load": load, "n": len(vals),
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std(ddof=1)), 2) if len(vals) > 1 else 0.0,
            "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2),
            "range": round(float(vals.max() - vals.min()), 2),
            "range_pct_of_mean": round(spread, 2),
        })

    fig.tight_layout()
    save(fig, out, "fig_pool_variation")

    print(f"\n  Objective across independent random pools "
          f"(battery={battery}, load={load})\n")
    print(f"    {'topo':<5} {'instance':<10} {'n':>4} {'mean':>10} {'std':>9} "
          f"{'min':>10} {'max':>10} {'range':>9} {'% of mean':>10}")
    print(f"    {'-'*5} {'-'*10} {'-'*4} {'-'*10} {'-'*9} {'-'*10} {'-'*10} {'-'*9} {'-'*10}")
    for r in rows:
        print(f"    {r['topology']:<5} {r['instance_id']:<10} {r['n']:>4} "
              f"{r['mean']:>10.1f} {r['std']:>9.1f} {r['min']:>10.1f} "
              f"{r['max']:>10.1f} {r['range']:>9.1f} {r['range_pct_of_mean']:>9.1f}%")
    return rows


# ---------------------------------------------------------------------------
# Analysis 2 — ALNS stability at a fixed pool
# ---------------------------------------------------------------------------
def resolve_instance(reference_dir: Path, instance_id: str,
                     battery_setting: str, tight_multiplier: float) -> Instance:
    itype = instance_id.rsplit("_", 1)[0]
    path = Path(reference_dir) / REFERENCE_FILES[itype]
    with open(path) as f:
        records = json.load(f)
    for idx, rec in enumerate(records):
        if rec["instance_id"] == instance_id:
            return Instance.from_json(
                reference_dir, itype, idx,
                battery_multiplier=1.0 if battery_setting == "loose" else tight_multiplier)
    raise KeyError(f"{instance_id} not found in {path}")


def _stability_worker(task: dict) -> dict:
    """One ALNS replication. Rebuilds the instance so the worker is self-contained."""
    inst = resolve_instance(Path(task["reference_dir"]), task["instance_id"],
                            task["battery_setting"], task["tight_multiplier"])
    solver = ALNS(inst, task["pool"], seed=task["seed"])
    solver.run(n_iterations=task["iterations"])
    return {
        "seed": task["seed"],
        "final": solver.global_best_val,
        "served": len(solver.served_customers()),
        # best_so_far is recorded per iteration by ALNSConfig.track_history.
        "trace": np.asarray(solver.best_so_far, dtype=np.float32),
    }


def alns_stability(reference_dir: Path, instance_id: str, pool: Sequence[int],
                   repeats: int, iterations: int, workers: int,
                   battery_setting: str, tight_multiplier: float,
                   out: Path) -> Tuple[List[dict], np.ndarray]:
    """
    Run the ALNS `repeats` times on one fixed pool and plot the convergence band.

    Every replication sees the same instance and the same customers; only the
    random seed differs. The spread at the final iteration is therefore pure
    ALNS noise, which is the quantity the label set inherits.
    """
    tasks = [{
        "reference_dir": str(reference_dir), "instance_id": instance_id,
        "battery_setting": battery_setting, "tight_multiplier": tight_multiplier,
        "pool": list(pool), "iterations": iterations, "seed": s,
    } for s in range(repeats)]

    print(f"  running {repeats} replications of {instance_id} "
          f"(pool of {len(pool)}, {iterations:,} iterations) on {workers} workers ...")
    results: List[dict] = []
    with ProcessPool(processes=workers) as p:
        for i, rec in enumerate(p.imap_unordered(_stability_worker, tasks), 1):
            results.append(rec)
            print(f"    {i}/{repeats} done  final={rec['final']:.2f}", flush=True)

    results.sort(key=lambda r: r["seed"])
    n = min(len(r["trace"]) for r in results)
    traces = np.vstack([r["trace"][:n] for r in results])

    mean = traces.mean(axis=0)
    sem = traces.std(axis=0, ddof=1) / np.sqrt(len(traces)) if len(traces) > 1 else np.zeros(n)
    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    # Thin individual traces behind the band, so the reader sees the raw runs.
    for tr in traces:
        axes[0].plot(x, tr, color="0.75", linewidth=0.4, alpha=0.7, zorder=1)
    axes[0].fill_between(x, lo, hi, color="#0072B2", alpha=0.30, linewidth=0, zorder=2,
                         label="95% CI of the mean")
    axes[0].plot(x, mean, color="#0072B2", linewidth=1.4, zorder=3,
                 label=f"mean of {len(traces)} runs")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best objective so far")
    axes[0].set_xlim(1, n)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.3)
    axes[0].set_title("(a)", loc="left", fontweight="bold", fontsize=9, pad=4)

    finals = np.array([r["final"] for r in results], dtype=float)
    axes[1].axhline(finals.mean(), color="#D55E00", linewidth=1.2, zorder=1,
                    label=f"mean {finals.mean():.1f}")
    axes[1].scatter(np.arange(len(finals)), finals, s=22, color="#0072B2",
                    edgecolors="black", linewidths=0.4, zorder=3)
    axes[1].set_xlabel("Replication")
    axes[1].set_ylabel("Final objective")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle=":", alpha=0.3)
    axes[1].set_title("(b)", loc="left", fontweight="bold", fontsize=9, pad=4)

    fig.tight_layout()
    save(fig, out, "fig_alns_stability")

    np.savez_compressed(out / "stability_traces.npz", traces=traces,
                        finals=finals, pool=np.array(pool))
    print("  wrote stability_traces.npz")

    cv = 100.0 * finals.std(ddof=1) / abs(finals.mean()) if len(finals) > 1 and finals.mean() else np.nan
    rows = [{
        "analysis": "stability", "topology": instance_id.rsplit("_", 1)[0],
        "instance_id": instance_id, "battery": battery_setting, "load": "loose",
        "n": len(finals),
        "mean": round(float(finals.mean()), 2),
        "std": round(float(finals.std(ddof=1)), 2) if len(finals) > 1 else 0.0,
        "min": round(float(finals.min()), 2), "max": round(float(finals.max()), 2),
        "range": round(float(finals.max() - finals.min()), 2),
        "range_pct_of_mean": round(100.0 * (finals.max() - finals.min()) / abs(finals.mean()), 2)
                              if finals.mean() else np.nan,
        "cv_pct": round(cv, 3),
    }]

    print(f"\n  ALNS variability at a fixed pool ({len(finals)} replications)\n")
    print(f"    mean   {finals.mean():12.2f}")
    print(f"    std    {finals.std(ddof=1):12.2f}")
    print(f"    range  {finals.max() - finals.min():12.2f}  "
          f"({rows[0]['range_pct_of_mean']:.2f}% of the mean)")
    print(f"    CV     {cv:12.3f}%")
    return rows, finals


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------
def decompose(pool_rows: List[dict], stability_rows: List[dict]) -> Optional[dict]:
    """
    Split the across-pool variance into a pool-choice part and an ALNS part.

        Var(across pools) = Var(pool choice) + Var(ALNS noise)

    Each label is one ALNS run on one pool, so the observed across-pool variance
    already contains the run-to-run noise measured by the stability analysis.
    Subtracting leaves the part attributable to which customers were pooled.

    Only valid when both analyses used the same instance and battery setting.
    """
    s = stability_rows[0]
    match = [r for r in pool_rows
             if r["instance_id"] == s["instance_id"] and r["battery"] == s["battery"]]
    if not match:
        return None
    p = match[0]

    var_total = p["std"] ** 2
    var_alns = s["std"] ** 2
    var_pool = max(0.0, var_total - var_alns)
    share = 100.0 * var_pool / var_total if var_total > 0 else np.nan

    print(f"\n{'=' * 70}")
    print(f"VARIANCE DECOMPOSITION — {p['instance_id']}, battery {p['battery']}")
    print("=" * 70)
    print(f"  across {p['n']} random pools      std = {p['std']:10.2f}   var = {var_total:12.1f}")
    print(f"  ALNS noise at a fixed pool    std = {s['std']:10.2f}   var = {var_alns:12.1f}")
    print(f"  attributable to pool choice   std = {np.sqrt(var_pool):10.2f}   var = {var_pool:12.1f}")
    print(f"\n  Pool choice accounts for {share:.1f}% of the objective variance;")
    print(f"  ALNS run-to-run noise accounts for {100 - share:.1f}%.")
    if share < 50:
        print("\n  NOTE: noise dominates here, so a single run per pool is a weak label.")
    return {
        "analysis": "decomposition", "topology": p["topology"],
        "instance_id": p["instance_id"], "battery": p["battery"], "load": p["load"],
        "n": p["n"],
        "std_across_pools": p["std"], "std_alns_noise": s["std"],
        "std_pool_choice": round(float(np.sqrt(var_pool)), 2),
        "pool_choice_share_pct": round(float(share), 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Pool-choice variation and ALNS stability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--analysis", choices=["pool", "stability", "both"], default="both")
    p.add_argument("--labels", type=Path, default=Path("training data v4"),
                   help="Label directory, for the pool analysis.")
    p.add_argument("--reference-dir", type=Path, default=Path("data generation"))
    p.add_argument("--output-dir", type=Path, default=Path("figures/variance"))

    p.add_argument("--pool-instances", nargs="*", default=None,
                   help="Instance ids for the pool analysis, one per topology. "
                        "Defaults to the first of each found in the labels.")
    p.add_argument("--battery", choices=["loose", "tight"], default="loose")
    p.add_argument("--load", choices=["loose", "tight"], default="loose")

    p.add_argument("--instance", default="R_001", help="Instance for the stability analysis.")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--pool-size", type=int, default=None,
                   help="Pool size for the stability run. Defaults to the size "
                        "recorded in the labels, else 85.")
    p.add_argument("--tight-multiplier", type=float, default=0.85)
    p.add_argument("--workers", type=int, default=15)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    rows: List[dict] = []
    pool_rows: List[dict] = []
    stability_rows: List[dict] = []

    records: List[dict] = []
    if args.analysis in ("pool", "both"):
        print("Analysis 1 — variation across pools")
        if not args.labels.is_dir():
            print(f"ERROR: labels directory not found: {args.labels.resolve()}", file=sys.stderr)
            return 1
        records = load_labels(args.labels)
        if not records:
            print("ERROR: no label records found", file=sys.stderr)
            return 1
        try:
            instances = pick_instances(records, args.pool_instances)
        except KeyError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        pool_rows = pool_variation(records, instances, args.battery, args.load,
                                   args.output_dir)
        rows.extend(pool_rows)

    if args.analysis in ("stability", "both"):
        print(f"\nAnalysis 2 — ALNS stability at a fixed pool")
        # Reuse a pool from the labels when available, so the stability run is
        # on exactly the kind of pool the label set contains.
        pool = None
        if not records and args.labels.is_dir():
            records = load_labels(args.labels)
        for r in records:
            if (r["instance_id"] == args.instance
                    and r.get("battery_setting", args.battery) == args.battery
                    and "pool" in r):
                pool = r["pool"]
                print(f"  using a pool of {len(pool)} taken from the labels")
                break
        if pool is None:
            import random
            inst = resolve_instance(args.reference_dir, args.instance,
                                    args.battery, args.tight_multiplier)
            size = args.pool_size or 85
            pool = sorted(random.Random(0).sample(list(inst.C), size))
            print(f"  no label pool found; sampled {size} customers (seed 0)")

        stability_rows, _ = alns_stability(
            args.reference_dir, args.instance, pool, args.repeats, args.iterations,
            args.workers, args.battery, args.tight_multiplier, args.output_dir)
        rows.extend(stability_rows)

    if pool_rows and stability_rows:
        d = decompose(pool_rows, stability_rows)
        if d:
            rows.append(d)
        else:
            print("\n  (decomposition skipped: the two analyses used different "
                  "instances or battery settings)")

    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(args.output_dir / "variance_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote variance_summary.csv")
    print(f"\nAll outputs in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
