#!/usr/bin/env python3
"""
inspect_labels.py — distributions of the generated ALNS training labels.

Reads the shards written by alns_label_generator.py and reports how served
counts and objective values are distributed, split by network topology
(C / R / RC) and by the four battery x load settings.

Mainly a design check: the target was roughly 80-85% of pooled customers served
by the own fleet, with the rest going to a third party. The printed table shows
whether each setting actually lands in that band, and the histograms show
whether the four settings produce genuinely different regimes or collapse onto
each other.

    python inspect_labels.py --input "training data v3"
    python inspect_labels.py --input "training data v3" --output-dir figures/labels
    python inspect_labels.py --input "training data v3" --no-split-settings

Writes fig_served.{pdf,png}, fig_objective.{pdf,png} and label_summary.csv.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TYPE_ORDER = ["C", "R", "RC"]
SETTING_ORDER = [("loose", "loose"), ("loose", "tight"),
                 ("tight", "loose"), ("tight", "tight")]
# Colourblind-safe (Wong 2011), and distinguishable in greyscale.
SETTING_COLORS = {
    ("loose", "loose"): "#0072B2",
    ("loose", "tight"): "#E69F00",
    ("tight", "loose"): "#009E73",
    ("tight", "tight"): "#CC79A7",
}
TARGET_BAND = (0.80, 0.85)   # the served-fraction band the instances aim for


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "legend.frameon": False, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def load_records(input_dir: Path) -> List[dict]:
    files = sorted(Path(input_dir).glob("*.json")) + sorted(Path(input_dir).glob("*.JSON"))
    files = list(dict.fromkeys(files))
    records: List[dict] = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: {fp.name} is not valid JSON, skipping", file=sys.stderr)
            continue
        records.extend(data if isinstance(data, list) else [data])
    print(f"Loaded {len(records):,} records from {len(files)} shard(s) in {input_dir}")
    return records


def served_fraction(r: dict) -> float:
    """Served customers as a fraction of the pool handed to the ALNS."""
    pool = r.get("pool_size")
    if not pool:
        pool = len(r.get("customers_not_visited_from_selected_pool", [])) + r["num_served"]
    return r["num_served"] / pool if pool else float("nan")


def summarize(records: List[dict], split_settings: bool) -> List[dict]:
    """One summary row per (topology, setting) group, plus per-topology totals."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in records:
        key = (r["instance_type"],
               (r.get("battery_setting"), r.get("load_setting")) if split_settings else ("all", "all"))
        groups[key].append(r)
        groups[(r["instance_type"], ("ALL", "ALL"))].append(r)

    rows = []
    for itype in TYPE_ORDER:
        settings = (SETTING_ORDER + [("ALL", "ALL")]) if split_settings else [("all", "all"), ("ALL", "ALL")]
        for setting in settings:
            rs = groups.get((itype, setting))
            if not rs:
                continue
            served = np.array([r["num_served"] for r in rs], dtype=float)
            frac = np.array([served_fraction(r) for r in rs], dtype=float)
            obj = np.array([r["objective_value"] for r in rs], dtype=float)
            rows.append({
                "topology": itype,
                "battery": setting[0], "load": setting[1],
                "n": len(rs),
                "served_mean": served.mean(), "served_std": served.std(ddof=1) if len(rs) > 1 else 0.0,
                "served_min": served.min(), "served_max": served.max(),
                "served_frac_mean": frac.mean(),
                "obj_mean": obj.mean(), "obj_std": obj.std(ddof=1) if len(rs) > 1 else 0.0,
                "obj_min": obj.min(), "obj_max": obj.max(),
            })
    return rows


def print_summary(rows: List[dict]) -> None:
    print(f"\n{'=' * 96}")
    print("SERVED CUSTOMERS AND OBJECTIVE BY TOPOLOGY AND SETTING")
    print("=" * 96)
    print(f"\n  {'topo':<5} {'battery':<8} {'load':<8} {'n':>6} "
          f"{'served':>14} {'% of pool':>10} {'objective':>20}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*14} {'-'*10} {'-'*20}")
    for row in rows:
        total = row["battery"] == "ALL"
        served = f"{row['served_mean']:6.1f} +-{row['served_std']:5.1f}"
        pct = f"{100*row['served_frac_mean']:9.1f}%"
        obj = f"{row['obj_mean']:10.1f} +-{row['obj_std']:7.1f}"
        flag = "" if total else ("  <- in band" if TARGET_BAND[0] <= row["served_frac_mean"] <= TARGET_BAND[1] else "")
        line = (f"  {row['topology']:<5} {row['battery']:<8} {row['load']:<8} {row['n']:>6,} "
                f"{served:>14} {pct:>10} {obj:>20}{flag}")
        print(("  " + "-" * 92) if total else "", end="" if not total else "\n")
        print(line)
    print(f"\n  Target served fraction: {TARGET_BAND[0]:.0%}-{TARGET_BAND[1]:.0%} of the pool.")


def make_histograms(records: List[dict], output_dir: Path, split_settings: bool,
                    bins: int) -> None:
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_type[r["instance_type"]].append(r)
    present = [t for t in TYPE_ORDER if t in by_type]
    if not present:
        print("No records to plot.", file=sys.stderr)
        return

    for metric, key_fn, xlabel, stem in (
        ("served", lambda r: r["num_served"], "Customers served by own fleet", "fig_served"),
        ("objective", lambda r: r["objective_value"], "Objective value", "fig_objective"),
    ):
        fig, axes = plt.subplots(1, len(present), figsize=(2.6 * len(present) + 0.8, 2.6),
                                 squeeze=False)
        axes = axes[0]
        # A shared range per metric keeps the panels directly comparable.
        allv = np.array([key_fn(r) for r in records], dtype=float)
        lo, hi = float(allv.min()), float(allv.max())
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        edges = np.linspace(lo - pad, hi + pad, bins + 1)

        for ax, itype in zip(axes, present):
            rs = by_type[itype]
            if split_settings:
                for setting in SETTING_ORDER:
                    vals = [key_fn(r) for r in rs
                            if (r.get("battery_setting"), r.get("load_setting")) == setting]
                    if not vals:
                        continue
                    ax.hist(vals, bins=edges, histtype="step", linewidth=1.1,
                            color=SETTING_COLORS[setting],
                            label=f"bat {setting[0]} / load {setting[1]}")
            else:
                ax.hist([key_fn(r) for r in rs], bins=edges, color="#0072B2",
                        alpha=0.8, edgecolor="white", linewidth=0.3)

            ax.set_title(f"{itype}  (n = {len(rs):,})", loc="left", fontsize=9)
            ax.set_xlabel(xlabel)
            if ax is axes[0]:
                ax.set_ylabel("Count")
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.set_axisbelow(True)

        if split_settings:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4,
                       bbox_to_anchor=(0.5, 1.13))

        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(output_dir / f"{stem}.{ext}")
        plt.close(fig)
        print(f"  wrote {stem}.pdf / {stem}.png")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Histograms and summary statistics for the generated training labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=Path("training data v3"),
                   help="Directory holding the label shards.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write figures and the CSV. Defaults to <input>/analysis.")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--no-split-settings", dest="split_settings", action="store_false",
                   help="Pool the four battery/load settings into one histogram per topology.")
    p.set_defaults(split_settings=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.is_dir():
        print(f"ERROR: input directory not found: {args.input.resolve()}", file=sys.stderr)
        return 1

    records = load_records(args.input)
    if not records:
        print("ERROR: no records found", file=sys.stderr)
        return 1

    missing = [k for k in ("instance_type", "num_served", "objective_value")
               if k not in records[0]]
    if missing:
        print(f"ERROR: records are missing {missing}", file=sys.stderr)
        return 1

    has_settings = "battery_setting" in records[0]
    if args.split_settings and not has_settings:
        print("NOTE: records carry no battery_setting field; pooling all settings.")
        args.split_settings = False

    output_dir = args.output_dir or (args.input / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize(records, args.split_settings)
    print_summary(rows)

    set_style()
    print(f"\nFigures -> {output_dir.resolve()}")
    make_histograms(records, output_dir, args.split_settings, args.bins)

    csv_path = output_dir / "label_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote label_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
