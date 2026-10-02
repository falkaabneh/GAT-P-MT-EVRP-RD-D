#!/usr/bin/env python3
"""
paper_figures.py — publication-quality figures and LaTeX tables.

Produces vector (PDF) and raster (PNG) versions of every figure, plus
booktabs-style LaTeX tables and a metrics.json dump. Figures are sized for
standard two-column journal layouts (3.4 in single column, 7.0 in full width)
with serif fonts and 8-9 pt text so they need no rescaling in LaTeX.

Reuses the inference and metric code from evaluate.py, so the numbers here are
identical to those reported by `python evaluate.py`.

Single model:

    python paper_figures.py \\
        --model models/t2ur1ua6/best.pt prepared_data_pool "Pool" \\
        --output_dir paper_figures

Comparing two models (each --model takes CHECKPOINT DATA_DIR LABEL):

    python paper_figures.py \\
        --model models/dpn7elbs/best.pt prepared_data      "Served-set" \\
        --model models/t2ur1ua6/best.pt prepared_data_pool "Candidate pool" \\
        --output_dir paper_figures

Training curves require a history export from Weights & Biases. On the run
page, open the charts panel, choose Export -> CSV, save it, then pass:

        --history_csv wandb_history.csv

Figures written:
    fig_parity.{pdf,png}          predicted vs actual, per instance type
    fig_residuals.{pdf,png}       residual distribution + residual vs predicted
    fig_error_by_type.{pdf,png}   MAE and RMSE grouped by C / R / RC
    fig_error_vs_size.{pdf,png}   binned MAE vs number of nodes
    fig_training_curves.{pdf,png} loss and MAE against epoch (needs --history_csv)

Tables written:
    table_main_results.tex        per-split MAE / RMSE / R^2
    table_by_type.tex             per-instance-type breakdown
    metrics.json                  every number in machine-readable form
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from data_prep import load_prepared_data
from evaluate import load_checkpoint, predict, compute_metrics


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
# Widths in inches for a standard two-column journal page.
COL_WIDTH = 3.4      # single column
FULL_WIDTH = 7.0     # spans both columns

# Colourblind-safe (Wong 2011). Keeps its distinctions in greyscale print.
TYPE_COLORS = {
    "C":  "#0072B2",   # blue
    "R":  "#E69F00",   # orange
    "RC": "#009E73",   # green
}
TYPE_MARKERS = {"C": "o", "R": "s", "RC": "^"}
TYPE_ORDER = ["C", "R", "RC"]
MODEL_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def set_publication_style() -> None:
    """Serif fonts, small text, thin lines — matches a LaTeX article."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset":  "dejavuserif",
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.frameon":    False,
        "axes.linewidth":    0.8,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.3,
        "lines.linewidth":   1.2,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":      42,   # embed TrueType, not Type 3 (many journals require this)
        "ps.fonttype":       42,
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write both a vector PDF (for the manuscript) and a PNG (for previews)."""
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


def label_panels(axes, tags: Tuple[str, ...] = ("(a)", "(b)", "(c)", "(d)")) -> None:
    """
    Put (a), (b), ... above the left edge of each panel.

    Uses a left-aligned title rather than a free-floating text box: matplotlib
    lays titles out around the axes, so the tag can never land on top of a
    rotated y-axis label however long that label is.
    """
    for ax, tag in zip(axes, tags):
        if ax.axison:
            ax.set_title(tag, loc="left", fontweight="bold", fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------
class ModelResults:
    """Predictions and metrics for one model across the splits it was asked for."""

    def __init__(self, label: str, checkpoint: Path, data_dir: Path,
                 device: torch.device, splits: List[str], batch_size: int):
        self.label = label
        self.checkpoint = Path(checkpoint)
        self.data_dir = Path(data_dir)

        train_set, val_set, test_set, _ = load_prepared_data(self.data_dir)
        in_channels = train_set[0].x.shape[1]
        model, ckpt, stats = load_checkpoint(self.checkpoint, device, in_channels)

        self.best_epoch = ckpt.get("epoch")
        self.n_parameters = sum(p.numel() for p in model.parameters())
        self.config = ckpt.get("config", {})

        available = {"train": train_set, "val": val_set, "test": test_set}
        self.predictions: Dict[str, dict] = {}
        self.metrics: Dict[str, dict] = {}

        for name in splits:
            split = available[name]
            if not split:
                continue
            loader = DataLoader(split, batch_size=batch_size, shuffle=False)
            res = predict(model, loader, device, stats)
            self.predictions[name] = res
            self.metrics[name] = compute_metrics(
                res["preds"], res["targets"], res["types"]
            )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_parity(models: List[ModelResults], split: str, output_dir: Path) -> None:
    """
    Predicted vs actual, one panel per model, points coloured by instance type.
    The headline figure: a tight diagonal is the whole story.
    """
    n = len(models)
    width = COL_WIDTH if n == 1 else FULL_WIDTH
    fig, axes = plt.subplots(1, n, figsize=(width, width / n + 0.3), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, models):
        res = m.predictions[split]
        preds, targets = res["preds"], res["targets"]
        types = np.array(res["types"])

        for t in TYPE_ORDER:
            mask = types == t
            if not mask.any():
                continue
            ax.scatter(targets[mask], preds[mask],
                       s=2.5, alpha=0.35, linewidths=0,
                       c=TYPE_COLORS[t], marker=TYPE_MARKERS[t], label=t, rasterized=True)

        lo = float(min(targets.min(), preds.min()))
        hi = float(max(targets.max(), preds.max()))
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="0.25", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")

        mt = m.metrics[split]
        ax.text(0.04, 0.96,
                f"$R^2 = {mt['r2']:.4f}$\nMAE $= {mt['mae']:.1f}$\nRMSE $= {mt['rmse']:.1f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.7", linewidth=0.5, alpha=0.9))

        ax.set_xlabel("Actual objective value")
        if ax is axes[0]:
            ax.set_ylabel("Predicted objective value")
        if len(models) > 1:
            ax.set_title(m.label)
        ax.grid(True, linestyle=":")

    handles = [plt.Line2D([], [], marker=TYPE_MARKERS[t], color=TYPE_COLORS[t],
                          linestyle="", markersize=4, label=t) for t in TYPE_ORDER]
    axes[0].legend(handles=handles, loc="lower right", title="Instance type",
                   title_fontsize=8, handletextpad=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir, "fig_parity")


def figure_residuals(models: List[ModelResults], split: str, output_dir: Path) -> None:
    """Left: residual distribution. Right: residual vs predicted (heteroscedasticity)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    for i, m in enumerate(models):
        res = m.predictions[split]
        residuals = res["preds"] - res["targets"]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        label = m.label if len(models) > 1 else None

        axes[0].hist(residuals, bins=60, histtype="step", linewidth=1.0,
                     color=color, label=label, density=True)
        axes[1].scatter(res["preds"], residuals, s=2, alpha=0.25, linewidths=0,
                        c=color, label=label, rasterized=True)

    axes[0].axvline(0, color="0.25", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Residual (predicted $-$ actual)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, linestyle=":")

    axes[1].axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Predicted objective value")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, linestyle=":")

    if len(models) > 1:
        axes[0].legend(loc="upper right")

    label_panels(axes)

    fig.tight_layout()
    save_figure(fig, output_dir, "fig_residuals")


def figure_error_by_type(models: List[ModelResults], split: str, output_dir: Path) -> None:
    """Grouped bars: MAE and RMSE per instance type."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))
    x = np.arange(len(TYPE_ORDER))
    n = len(models)
    bar_w = 0.8 / n

    for metric_name, ax in zip(("mae", "rmse"), axes):
        peak = 0.0
        for i, m in enumerate(models):
            per_type = m.metrics[split]["per_type"]
            values = [per_type.get(t, {}).get(metric_name, np.nan) for t in TYPE_ORDER]
            peak = max(peak, np.nanmax(values))
            offset = (i - (n - 1) / 2) * bar_w
            bars = ax.bar(x + offset, values, bar_w * 0.9,
                          color=MODEL_COLORS[i % len(MODEL_COLORS)],
                          edgecolor="black", linewidth=0.4,
                          label=m.label if n > 1 else None)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)

        # Bars start at zero (no truncated axis); headroom keeps the value
        # labels and the shared legend clear of the tallest bar.
        ax.set_ylim(0, peak * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(TYPE_ORDER)
        ax.set_xlabel("Instance type")
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, axis="y", linestyle=":")
        ax.set_axisbelow(True)

    label_panels(axes)

    fig.tight_layout()
    if n > 1:
        # One legend above both panels, so neither can collide with the bars.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(n, 4),
                   bbox_to_anchor=(0.5, 1.06), frameon=False)
    save_figure(fig, output_dir, "fig_error_by_type")


def figure_error_vs_size(models: List[ModelResults], split: str,
                         output_dir: Path, n_bins: int = 6) -> None:
    """
    Mean absolute error against graph size, binned by quantile so each point
    rests on a comparable number of graphs. Error bars are +/- 1 standard error.
    """
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))

    for i, m in enumerate(models):
        res = m.predictions[split]
        sizes = np.array(res["num_nodes"], dtype=float)
        abs_err = np.abs(res["preds"] - res["targets"])

        edges = np.quantile(sizes, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        centers, means, ses = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sizes >= lo) & (sizes <= hi if hi == edges[-1] else sizes < hi)
            if mask.sum() < 2:
                continue
            centers.append(sizes[mask].mean())
            means.append(abs_err[mask].mean())
            ses.append(abs_err[mask].std(ddof=1) / np.sqrt(mask.sum()))

        ax.errorbar(centers, means, yerr=ses, marker="o", markersize=3.5,
                    capsize=2, capthick=0.8, elinewidth=0.8,
                    color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    label=m.label if len(models) > 1 else None)

    ax.set_xlabel("Number of nodes (depot $+$ customers)")
    ax.set_ylabel("Mean absolute error")
    ax.grid(True, linestyle=":")
    if len(models) > 1:
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig_error_vs_size")


def figure_training_curves(history_csv: Path, output_dir: Path) -> None:
    """
    Loss and validation MAE against epoch, from a W&B history CSV export.
    Column names are matched loosely, since W&B prefixes them by run name when
    several runs are exported together.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  skipping training curves: pandas is not installed")
        return

    df = pd.read_csv(history_csv)

    def find(*needles: str) -> Optional[str]:
        for col in df.columns:
            low = col.lower()
            if all(nd in low for nd in needles):
                return col
        return None

    epoch_col = find("epoch") or df.columns[0]
    train_col = find("train", "loss")
    val_col   = find("val", "loss")
    mae_col   = find("val", "mae")

    if train_col is None and val_col is None:
        print(f"  skipping training curves: no loss columns found in {history_csv}")
        print(f"    available columns: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4))

    if train_col:
        axes[0].plot(df[epoch_col], df[train_col], color=MODEL_COLORS[0], label="Train")
    if val_col:
        axes[0].plot(df[epoch_col], df[val_col], color=MODEL_COLORS[1], label="Validation")
        best_idx = df[val_col].idxmin()
        best_epoch = df[epoch_col][best_idx]
        axes[0].axvline(best_epoch, color="0.4", linestyle=":", linewidth=0.8)
        # Flip the label to the left of the marker when the best epoch sits in
        # the right-hand half, so the text stays inside the axes.
        late = best_epoch > 0.6 * df[epoch_col].max()
        axes[0].annotate(f"best epoch {int(best_epoch)}",
                         xy=(best_epoch, df[val_col][best_idx]),
                         xytext=(-6 if late else 6, 14),
                         textcoords="offset points", fontsize=7,
                         ha="right" if late else "left")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", linestyle=":")
    axes[0].legend(loc="upper right")

    if mae_col:
        axes[1].plot(df[epoch_col], df[mae_col], color=MODEL_COLORS[1])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation MAE")
        axes[1].grid(True, linestyle=":")
    else:
        axes[1].axis("off")

    label_panels(axes)

    fig.tight_layout()
    save_figure(fig, output_dir, "fig_training_curves")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def table_main_results(models: List[ModelResults], splits: List[str],
                       output_dir: Path) -> None:
    """Per-split MAE / RMSE / R^2, one block per model."""
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Surrogate accuracy on the training, validation and test splits. "
        r"MAE and RMSE are in objective-function units.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Split & $n$ & MAE & RMSE & $R^2$ \\",
        r"\midrule",
    ]
    for m_i, m in enumerate(models):
        for s_i, s in enumerate(splits):
            if s not in m.metrics:
                continue
            mt = m.metrics[s]
            name = m.label if s_i == 0 else ""
            lines.append(
                f"{name} & {s.capitalize()} & {mt['n']:,} & "
                f"{mt['mae']:.2f} & {mt['rmse']:.2f} & {mt['r2']:.4f} \\\\"
            )
        if m_i < len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (output_dir / "table_main_results.tex").write_text("\n".join(lines) + "\n")
    print("  wrote table_main_results.tex")


def table_by_type(models: List[ModelResults], split: str, output_dir: Path) -> None:
    """Per-instance-type MAE and RMSE on one split."""
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Test-set accuracy by instance type ({split} split). "
        r"C denotes clustered, R random, and RC mixed customer geometries.}}",
        r"\label{tab:by-type}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Type & $n$ & MAE & RMSE \\",
        r"\midrule",
    ]
    for m_i, m in enumerate(models):
        per_type = m.metrics[split]["per_type"]
        for t_i, t in enumerate(TYPE_ORDER):
            if t not in per_type:
                continue
            pt = per_type[t]
            name = m.label if t_i == 0 else ""
            lines.append(
                f"{name} & {t} & {pt['n']:,} & {pt['mae']:.2f} & {pt['rmse']:.2f} \\\\"
            )
        if m_i < len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (output_dir / "table_by_type.tex").write_text("\n".join(lines) + "\n")
    print("  wrote table_by_type.tex")


def dump_metrics(models: List[ModelResults], output_dir: Path) -> None:
    payload = {
        m.label: {
            "checkpoint":   str(m.checkpoint),
            "data_dir":     str(m.data_dir),
            "best_epoch":   m.best_epoch,
            "n_parameters": m.n_parameters,
            "config":       m.config,
            "metrics":      m.metrics,
        }
        for m in models
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("  wrote metrics.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication-quality figures and LaTeX tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", action="append", nargs=3, required=True,
                   metavar=("CHECKPOINT", "DATA_DIR", "LABEL"),
                   help="A model to include. Repeat the flag to compare several.")
    p.add_argument("--output_dir", type=Path, default=Path("paper_figures"))
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                   default=["train", "val", "test"],
                   help="Splits to evaluate for the tables.")
    p.add_argument("--figure_split", choices=["train", "val", "test"], default="test",
                   help="Split the figures are drawn from.")
    p.add_argument("--history_csv", type=Path, default=None,
                   help="W&B history CSV export, for the training-curve figure.")
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.figure_split not in args.splits:
        args.splits.append(args.figure_split)

    models: List[ModelResults] = []
    for checkpoint, data_dir, label in args.model:
        ckpt_path = Path(checkpoint)
        data_path = Path(data_dir)
        if not ckpt_path.is_file():
            print(f"ERROR: checkpoint not found: {ckpt_path.resolve()}", file=sys.stderr)
            return 1
        if not data_path.is_dir():
            print(f"ERROR: data directory not found: {data_path.resolve()}", file=sys.stderr)
            return 1

        print(f"\nEvaluating '{label}'")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  data:       {data_path}")
        m = ModelResults(label, ckpt_path, data_path, device, args.splits, args.batch_size)
        models.append(m)
        for s in args.splits:
            if s in m.metrics:
                mt = m.metrics[s]
                print(f"    {s:5s}  MAE = {mt['mae']:7.2f}   "
                      f"RMSE = {mt['rmse']:7.2f}   R^2 = {mt['r2']:.4f}")

    print(f"\nWriting figures ({args.figure_split} split) to {args.output_dir}")
    figure_parity(models, args.figure_split, args.output_dir)
    figure_residuals(models, args.figure_split, args.output_dir)
    figure_error_by_type(models, args.figure_split, args.output_dir)
    figure_error_vs_size(models, args.figure_split, args.output_dir)
    if args.history_csv is not None:
        if args.history_csv.is_file():
            figure_training_curves(args.history_csv, args.output_dir)
        else:
            print(f"  skipping training curves: {args.history_csv} not found")

    print("\nWriting tables")
    table_main_results(models, args.splits, args.output_dir)
    table_by_type(models, args.figure_split, args.output_dir)
    dump_metrics(models, args.output_dir)

    print(f"\nAll outputs in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
