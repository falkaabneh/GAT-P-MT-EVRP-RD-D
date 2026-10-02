#!/usr/bin/env python3
"""
arm_figures.py — publication figures for the pool-selection arm comparison.

Complements paper_figures.py, which covers surrogate regression accuracy. This
script covers the experiment: how the nine arms compare, how the comparison
shifts with the battery setting, and how it differs between the two feature
sets.

Reads the results written by run_experiment.py — results.csv when it is
present, otherwise results.jsonl. Pass one --run for a single model, or two to
overlay the feature ablation.

By default the full-pool arm (A) and the growing arm (E) are excluded, so the
comparison is apples to apples: every remaining arm selects the same number of
customers and faces the same structural handicap from the objective charging
penalties for customers that were never pooled. Pass --exclude-arms with no
values to keep them.

    python arm_figures.py \\
        --run results/arms_6f_full_added_heuristics "6-feature" \\
        --run results/arms_8f_full_added_heuristics "8-feature" \\
        --baseline B_random --compare-arm C4_cluster \\
        --output-dir paper_figures/arms

Figures written (PDF for the manuscript, PNG for previews):
    fig_arm_gap.*         mean gap to the baseline, per arm, with 95% CIs
    fig_battery.*         gap to the baseline against battery setting
    fig_served_vs_obj.*   mean served customers against mean objective
    fig_by_topology.*     gap to the baseline per topology
    fig_head_to_head.*    per-block difference between D and --compare-arm
Also writes arm_summary.csv and pairwise_vs_surrogate.csv, the latter holding
the paired difference between the surrogate and every other arm with exact
confidence intervals computed per block.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COL_WIDTH, FULL_WIDTH = 3.4, 7.0
RUN_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
TYPE_ORDER = ["C", "R", "RC"]
BATTERY_ORDER = ["loose", "tight"]


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


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def label_panels(axes, tags=("(a)", "(b)", "(c)", "(d)")) -> None:
    """Left-aligned titles: matplotlib lays these out around the axes, so they
    cannot collide with a long rotated y-label."""
    for ax, tag in zip(np.atleast_1d(axes), tags):
        ax.set_title(tag, loc="left", fontweight="bold", fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# Columns that must be numbers. A CSV gives everything back as text, so these
# are coerced on read; a JSONL already holds the right types and is untouched.
_INT_FIELDS = ("seed", "pool_size", "iterations", "num_served",
               "unserved_in_pool", "final_pool_size", "n_insertions")
_FLOAT_FIELDS = ("objective", "initial_objective", "cpu_seconds", "wall_seconds",
                 "screening_seconds", "total_seconds", "surrogate_predicted",
                 "battery_capacity")


def _coerce(row: dict) -> Optional[dict]:
    """Turn one CSV row into the shape the rest of this script expects."""
    out = dict(row)
    for k in _INT_FIELDS:
        v = out.get(k)
        if v in (None, ""):
            out.pop(k, None)
            continue
        try:
            out[k] = int(float(v))
        except ValueError:
            out.pop(k, None)
    for k in _FLOAT_FIELDS:
        v = out.get(k)
        if v in (None, ""):
            out.pop(k, None)
            continue
        try:
            out[k] = float(v)
        except ValueError:
            out.pop(k, None)
    # A row with no usable objective cannot take part in any comparison.
    return out if "objective" in out and "seed" in out else None


def read_records(run_dir: Path, fmt: str = "auto") -> List[dict]:
    """
    Load the per-run records, from results.csv or results.jsonl.

    Both are written by run_experiment.py and carry the same fields; the CSV is
    the friendlier one to open by hand, so it is preferred when present.
    """
    run_dir = Path(run_dir)
    csv_path, jsonl_path = run_dir / "results.csv", run_dir / "results.jsonl"

    if fmt == "csv" or (fmt == "auto" and csv_path.is_file()):
        if not csv_path.is_file():
            raise FileNotFoundError(f"no results.csv in {run_dir}")
        with open(csv_path, newline="") as f:
            rows = [_coerce(r) for r in csv.DictReader(f)]
        records = [r for r in rows if r is not None]
        print(f"  {run_dir}: read {len(records):,} rows from results.csv")
        return records

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"no results.csv or results.jsonl in {run_dir}")
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"  {run_dir}: read {len(records):,} rows from results.jsonl")
    return records


def load_blocks(run_dir: Path, exclude: Tuple[str, ...] = (),
                fmt: str = "auto") -> Dict[Tuple[str, str, int], Dict[str, dict]]:
    """
    Group records into (instance, battery, seed) blocks, dropping incomplete ones.

    Arms in `exclude` are removed before the completeness check, so excluding an
    arm never discards a block that is otherwise complete.
    """
    blocks: Dict[Tuple[str, str, int], Dict[str, dict]] = defaultdict(dict)
    for r in read_records(run_dir, fmt):
        if "error" in r or "objective" not in r:
            continue
        if r["arm"] in exclude:
            continue
        # Records written before battery settings existed were all loose.
        key = (r["instance_id"], r.get("battery_setting") or "loose", r["seed"])
        blocks[key][r["arm"]] = r

    arms = {a for v in blocks.values() for a in v}
    complete = {k: v for k, v in blocks.items() if len(v) == len(arms)}
    print(f"  {run_dir}: {len(complete)} complete blocks, {len(arms)} arms")
    return complete


def arm_order(blocks) -> List[str]:
    """Arms in their canonical order, keeping any unexpected ones at the end."""
    canonical = ["A_full", "B_random", "C1_prize", "C2_penalty", "C3_tw_spread",
                 "C4_cluster", "C5_density", "D_surrogate", "E_surrogate_grow"]
    present = {a for v in blocks.values() for a in v}
    return [a for a in canonical if a in present] + sorted(present - set(canonical))


def gaps(blocks, arm: str, baseline: str, battery=None, topology=None) -> np.ndarray:
    """Per-block objective difference between `arm` and `baseline`."""
    out = []
    for (iid, bat, seed), v in blocks.items():
        if battery is not None and bat != battery:
            continue
        if topology is not None and v[arm]["instance_type"] != topology:
            continue
        out.append(v[arm]["objective"] - v[baseline]["objective"])
    return np.array(out)


def mean_ci(x: np.ndarray) -> Tuple[float, float]:
    """Mean and 95% half-width."""
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_arm_gap(runs, baseline: str, out: Path) -> None:
    """Primary figure: mean gap to the baseline per arm, grouped by run."""
    arms = [a for a in arm_order(runs[0][1]) if a != baseline]
    x = np.arange(len(arms))
    n = len(runs)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))
    peak = 0.0
    for i, (label, blocks) in enumerate(runs):
        means, errs = [], []
        for a in arms:
            m, e = mean_ci(gaps(blocks, a, baseline))
            means.append(m); errs.append(e)
        peak = max(peak, max(np.array(means) + np.array(errs)))
        ax.bar(x + (i - (n - 1) / 2) * w, means, w * 0.9, yerr=errs,
               capsize=2.5, error_kw=dict(elinewidth=0.8, capthick=0.8),
               color=RUN_COLORS[i % len(RUN_COLORS)], edgecolor="black",
               linewidth=0.4, label=label if n > 1 else None)

    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_ylim(top=peak * 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels([a.split("_", 1)[0] for a in arms])
    ax.set_xlabel("Arm")
    ax.set_ylabel(f"Mean gap to {baseline.split('_')[0]}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if n > 1:
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=n, bbox_to_anchor=(0.5, 1.06))
    save(fig, out, "fig_arm_gap")


def fig_battery(runs, baseline: str, out: Path, focus=("C1_prize", "C5_density",
                                                       "C4_cluster", "D_surrogate")) -> None:
    """Gap to the baseline against battery setting, one panel per run."""
    fig, axes = plt.subplots(1, len(runs), figsize=(COL_WIDTH * len(runs) + 0.6, 2.7),
                             squeeze=False, sharey=True)
    axes = axes[0]
    xs = np.arange(len(BATTERY_ORDER))

    for ax, (label, blocks) in zip(axes, runs):
        present = [a for a in focus if a in arm_order(blocks)]
        for j, arm in enumerate(present):
            ys, es = [], []
            for bat in BATTERY_ORDER:
                m, e = mean_ci(gaps(blocks, arm, baseline, battery=bat))
                ys.append(m); es.append(e)
            ax.errorbar(xs, ys, yerr=es, marker="o", markersize=4, capsize=2.5,
                        elinewidth=0.8, capthick=0.8,
                        color=RUN_COLORS[j % len(RUN_COLORS)],
                        label=arm.split("_", 1)[0])
        ax.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(BATTERY_ORDER)
        ax.set_xlabel("Battery setting")
        ax.set_xlim(-0.3, len(BATTERY_ORDER) - 0.7)
        ax.grid(True, linestyle=":", alpha=0.3)
        if len(runs) > 1:
            ax.set_title(label, loc="center")
    axes[0].set_ylabel(f"Mean gap to {baseline.split('_')[0]}")
    axes[-1].legend(loc="upper right", ncol=2)

    fig.tight_layout()
    save(fig, out, "fig_battery")


def fig_served_vs_obj(runs, out: Path) -> None:
    """Mean served customers against mean objective, one marker per arm."""
    fig, axes = plt.subplots(1, len(runs), figsize=(COL_WIDTH * len(runs) + 0.6, 2.9),
                             squeeze=False)
    axes = axes[0]
    for ax, (label, blocks) in zip(axes, runs):
        arms = arm_order(blocks)
        points = [(arm,
                   float(np.mean([v[arm]["num_served"] for v in blocks.values()])),
                   float(np.mean([v[arm]["objective"] for v in blocks.values()])))
                  for arm in arms]
        xs = [p[1] for p in points]; ys = [p[2] for p in points]
        xr = (max(xs) - min(xs)) or 1.0
        yr = (max(ys) - min(ys)) or 1.0

        placed: List[Tuple[float, float]] = []
        for arm, served, obj in points:
            highlight = arm.startswith(("D_", "A_"))
            ax.scatter(served, obj, s=40 if highlight else 26,
                       c="#D55E00" if highlight else "#0072B2",
                       marker="D" if highlight else "o",
                       zorder=3, edgecolors="black", linewidths=0.4)
            # Arms that tie sit on top of one another (A and E in particular).
            # Try a small ring of offsets and take the first that does not
            # collide with a label already placed, so every label stays close
            # to its own marker rather than drifting away from it.
            candidates = [(5, 3), (5, -9), (-16, 3), (-16, -9), (5, 11), (-16, 11)]
            dx, dy = candidates[0]
            for cx, cy in candidates:
                lx = served + cx * xr / 260
                ly = obj + cy * yr / 150
                if not any(abs(lx - px) / xr < 0.05 and abs(ly - py) / yr < 0.05
                           for px, py in placed):
                    dx, dy = cx, cy
                    break
            ax.annotate(arm.split("_", 1)[0], (served, obj),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7)
            placed.append((served + dx * xr / 260, obj + dy * yr / 150))
        ax.set_xlabel("Mean customers served")
        if ax is axes[0]:
            ax.set_ylabel("Mean objective")
        ax.grid(True, linestyle=":", alpha=0.3)
        if len(runs) > 1:
            ax.set_title(label, loc="center")
    fig.tight_layout()
    save(fig, out, "fig_served_vs_obj")


def fig_by_topology(runs, baseline: str, out: Path,
                    focus=("C1_prize", "C4_cluster", "D_surrogate")) -> None:
    """Gap to the baseline per topology, grouped bars."""
    label, blocks = runs[-1]          # the last run, typically the better model
    present = [a for a in focus if a in arm_order(blocks)]
    x = np.arange(len(TYPE_ORDER))
    w = 0.8 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(COL_WIDTH + 0.6, 2.6))
    for i, arm in enumerate(present):
        means, errs = [], []
        for t in TYPE_ORDER:
            m, e = mean_ci(gaps(blocks, arm, baseline, topology=t))
            means.append(m); errs.append(e)
        ax.bar(x + (i - (len(present) - 1) / 2) * w, means, w * 0.9, yerr=errs,
               capsize=2.5, error_kw=dict(elinewidth=0.8, capthick=0.8),
               color=RUN_COLORS[i % len(RUN_COLORS)], edgecolor="black",
               linewidth=0.4, label=arm.split("_", 1)[0])
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TYPE_ORDER)
    ax.set_xlabel("Instance topology")
    ax.set_ylabel(f"Mean gap to {baseline.split('_')[0]}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title=label if len(runs) > 1 else None, title_fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig_by_topology")


def fig_head_to_head(runs, arm: str, other: str, out: Path) -> None:
    """
    Distribution of the per-block difference between two arms.

    A mean of a few percent can hide a wide spread; this shows it. Negative
    values favour `arm`.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(COL_WIDTH * len(runs) + 0.6, 2.6),
                             squeeze=False, sharey=True)
    axes = axes[0]
    for ax, (label, blocks) in zip(axes, runs):
        present = arm_order(blocks)
        if arm not in present or other not in present:
            ax.axis("off")
            continue
        d = gaps(blocks, arm, other)
        m, e = mean_ci(d)
        ax.hist(d, bins=45, color="#0072B2", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="0.25", linestyle="--", linewidth=0.8)
        ax.axvline(m, color="#D55E00", linewidth=1.2,
                   label=f"mean {m:+.1f}\n95% CI ±{e:.1f}")
        ax.set_xlabel(f"{arm.split('_')[0]} − {other.split('_')[0]} objective")
        if ax is axes[0]:
            ax.set_ylabel("Blocks")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.3)
        if len(runs) > 1:
            ax.set_title(label, loc="center")
    fig.tight_layout()
    save(fig, out, "fig_head_to_head")


# ---------------------------------------------------------------------------
# Pairwise comparison against the surrogate
# ---------------------------------------------------------------------------
def pairwise_vs(runs, focus: str, out: Path,
                batteries=(None,) + tuple(BATTERY_ORDER),
                topologies=(None,) + tuple(TYPE_ORDER)) -> None:
    """
    Paired difference between `focus` and every other arm, with exact CIs.

    These cannot be derived from each arm's gap to a common baseline without the
    covariances, so they are computed per block here. Negative favours `focus`.
    The relative column divides by the other arm's mean objective, which is what
    a "the surrogate is X% better" claim refers to.
    """
    rows = []
    for label, blocks in runs:
        if focus not in arm_order(blocks):
            continue
        for arm in arm_order(blocks):
            if arm == focus:
                continue
            for bat in batteries:
                for topo in topologies:
                    if bat is not None and topo is not None:
                        continue      # one slice at a time keeps the table readable
                    d = gaps(blocks, focus, arm, battery=bat, topology=topo)
                    if not len(d):
                        continue
                    m, e = mean_ci(d)
                    other = np.mean([
                        v[arm]["objective"] for (i, b, s), v in blocks.items()
                        if (bat is None or b == bat)
                        and (topo is None or v[arm]["instance_type"] == topo)
                    ])
                    rows.append({
                        "run": label, "focus": focus, "vs": arm,
                        "battery": bat or "all", "topology": topo or "all",
                        "n_blocks": len(d),
                        "mean_difference": round(m, 2), "ci95": round(e, 2),
                        "relative_pct": round(-100.0 * m / other, 2) if other else np.nan,
                        "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
                        "significant": bool(m + e < 0 or m - e > 0),
                    })

    with open(out / "pairwise_vs_surrogate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  wrote pairwise_vs_surrogate.csv")

    for label, blocks in runs:
        overall = [r for r in rows if r["run"] == label
                   and r["battery"] == "all" and r["topology"] == "all"]
        if not overall:
            continue
        print(f"\n  {label} — {focus} vs each arm "
              f"(negative favours {focus.split('_')[0]})\n")
        print(f"    {'vs':<14} {'mean diff':>11} {'95% CI':>10} {'rel %':>8} "
              f"{'W-L':>12} {'sig':>4}")
        print(f"    {'-'*14} {'-'*11} {'-'*10} {'-'*8} {'-'*12} {'-'*4}")
        for r in sorted(overall, key=lambda r: r["mean_difference"]):
            wl = f"{r['wins']}-{r['losses']}"
            print(f"    {r['vs']:<14} {r['mean_difference']:>11.2f} "
                  f"{'±' + format(r['ci95'], '.2f'):>10} {r['relative_pct']:>7.2f}% "
                  f"{wl:>12} {'*' if r['significant'] else '':>4}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def write_summary(runs, baseline: str, out: Path) -> None:
    rows = []
    for label, blocks in runs:
        for arm in arm_order(blocks):
            for bat in [None] + BATTERY_ORDER:
                d = gaps(blocks, arm, baseline, battery=bat)
                if not len(d):
                    continue
                m, e = mean_ci(d)
                obj = [v[arm]["objective"] for (i, b, s), v in blocks.items()
                       if bat is None or b == bat]
                served = [v[arm]["num_served"] for (i, b, s), v in blocks.items()
                          if bat is None or b == bat]
                rows.append({
                    "run": label, "arm": arm, "battery": bat or "all",
                    "n_blocks": len(d),
                    "mean_objective": round(float(np.mean(obj)), 2),
                    "mean_served": round(float(np.mean(served)), 2),
                    "gap_mean": round(m, 2), "gap_ci95": round(e, 2),
                    "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
                    "ties": int((d == 0).sum()),
                })
    with open(out / "arm_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  wrote arm_summary.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Publication figures for the pool-selection arm comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", action="append", nargs=2, required=True,
                   metavar=("RESULTS_DIR", "LABEL"),
                   help="A results directory and its label. Repeat to compare runs.")
    p.add_argument("--baseline", default="B_random",
                   help="Arm the figures take differences against.")
    p.add_argument("--exclude-arms", nargs="*",
                   default=["A_full", "E_surrogate_grow"],
                   help="Arms to drop before analysis. The default keeps only the "
                        "fixed-size pools, so every arm is directly comparable. "
                        "Pass with no values to keep every arm.")
    p.add_argument("--focus-arm", default="D_surrogate",
                   help="Arm compared pairwise against all others.")
    p.add_argument("--format", choices=["auto", "csv", "jsonl"], default="auto",
                   help="Which results file to read. 'auto' prefers results.csv "
                        "and falls back to results.jsonl.")
    p.add_argument("--compare-arm", default="C4_cluster",
                   help="Arm compared head-to-head against D_surrogate.")
    p.add_argument("--output-dir", type=Path, default=Path("paper_figures/arms"))
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    exclude = tuple(args.exclude_arms or ())
    if exclude:
        print(f"Excluding arms: {', '.join(exclude)}")
    print("Loading runs:")
    runs = []
    for d, label in args.run:
        try:
            runs.append((label, load_blocks(Path(d), exclude=exclude, fmt=args.format)))
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    print(f"\nFigures -> {args.output_dir.resolve()}")
    fig_arm_gap(runs, args.baseline, args.output_dir)
    fig_battery(runs, args.baseline, args.output_dir)
    fig_served_vs_obj(runs, args.output_dir)
    fig_by_topology(runs, args.baseline, args.output_dir)
    fig_head_to_head(runs, args.focus_arm, args.compare_arm, args.output_dir)
    write_summary(runs, args.baseline, args.output_dir)
    pairwise_vs(runs, args.focus_arm, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
