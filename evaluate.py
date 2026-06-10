#!/usr/bin/env python3
"""
evaluate.py — Phase D of the GAT pipeline.

Load a trained checkpoint and produce a comprehensive evaluation report:

  - Per-split (train / val / test) metrics: MAE, RMSE, R^2, residual statistics
  - Per-instance-type breakdown (C / R / RC)
  - Diagnostic plots:
      * pred_vs_actual_<split>.png    -- scatter of predictions vs targets
      * residuals_<split>.png         -- residual histogram, residual vs predicted,
                                         and per-type box plots
      * error_vs_size_<split>.png     -- absolute error vs number of nodes
  - predictions_<split>.csv  -- one row per graph for offline analysis
  - metrics.json             -- numeric summary of all evaluated splits

Designed to be CLI-callable and importable from notebooks for interactive
exploration.

Usage:

    python evaluate.py --checkpoint models/<run_id>/best.pt
    python evaluate.py --checkpoint models/<run_id>/best.pt --splits train val test
    python evaluate.py --checkpoint models/<run_id>/best.pt --output_dir results/run_<run_id>
    python evaluate.py --help

From Python:

    from evaluate import load_checkpoint, predict, compute_metrics
    model, ckpt, stats = load_checkpoint("models/<run_id>/best.pt", device, in_channels)
    results = predict(model, loader, device, stats)
    metrics = compute_metrics(results["preds"], results["targets"], results["types"])
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from data_prep import NormStats, load_prepared_data
from model import GATSurrogate


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("prepared_data")
DEFAULT_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    in_channels: int,
) -> Tuple[GATSurrogate, dict, NormStats]:
    """
    Load a checkpoint and reconstruct the GATSurrogate model from its saved
    config. Returns (model in eval mode, raw checkpoint dict, NormStats).
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]

    model = GATSurrogate(
        in_channels=in_channels,
        edge_in_channels=2,
        hidden_dim=cfg["hidden_dim"],
        edge_hidden_dim=cfg["edge_hidden_dim"],
        num_gat_layers=cfg["num_gat_layers"],
        heads=cfg["heads"],
        concat_heads=(cfg["head_aggr"] == "concat"),
        dropout=cfg["dropout"],
        head_dropout=cfg["head_dropout"],
        head_depth=cfg["head_depth"],
        use_residual=True,
        use_global_conditioning=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats = NormStats.from_dict(ckpt["stats"])
    return model, ckpt, stats


# ---------------------------------------------------------------------------
# Inference + metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: GATSurrogate,
    loader: DataLoader,
    device: torch.device,
    stats: NormStats,
) -> Dict:
    """
    Run inference on a loader. Returns a dict of numpy arrays with both
    normalized and original-unit predictions and targets, plus per-graph
    metadata (instance_type, number of nodes).
    """
    model.eval()
    preds_n, targets_n, types, n_nodes = [], [], [], []

    for batch in loader:
        p = model(batch.to(device)).squeeze(-1).cpu()
        preds_n.append(p)
        targets_n.append(batch.y.cpu())
        for g in batch.to_data_list():
            types.append(g.instance_type)
            n_nodes.append(g.x.shape[0])

    preds_n = torch.cat(preds_n).numpy() if preds_n else np.array([])
    targets_n = torch.cat(targets_n).numpy() if targets_n else np.array([])

    preds_o   = preds_n   * stats.y_std + stats.y_mean
    targets_o = targets_n * stats.y_std + stats.y_mean

    return {
        "preds_normalized":   preds_n,
        "targets_normalized": targets_n,
        "preds":              preds_o,
        "targets":            targets_o,
        "types":              types,
        "num_nodes":          n_nodes,
    }


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    types: Optional[List[str]] = None,
) -> Dict:
    """
    Compute regression metrics in original units, plus a per-instance-type
    breakdown if `types` is provided.
    """
    err = preds - targets
    abs_err = np.abs(err)

    ss_res = float((err ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    out = {
        "n":              int(len(preds)),
        "mae":            float(abs_err.mean()),
        "rmse":           float(np.sqrt((err ** 2).mean())),
        "r2":             float(r2),
        "residual_mean":  float(err.mean()),
        "residual_std":   float(err.std()),
        "residual_min":   float(err.min()) if len(err) else float("nan"),
        "residual_max":   float(err.max()) if len(err) else float("nan"),
        "residual_p5":    float(np.percentile(err, 5))  if len(err) else float("nan"),
        "residual_p95":   float(np.percentile(err, 95)) if len(err) else float("nan"),
    }

    if types is not None:
        types_arr = np.array(types)
        per_type = {}
        for t in sorted(set(types)):
            mask = types_arr == t
            t_err = err[mask]
            per_type[t] = {
                "n":    int(mask.sum()),
                "mae":  float(np.abs(t_err).mean()),
                "rmse": float(np.sqrt((t_err ** 2).mean())),
            }
        out["per_type"] = per_type

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_TYPE_COLORS = {"C": "tab:blue", "R": "tab:orange", "RC": "tab:green"}


def plot_pred_vs_actual(results: Dict, output_path: Path, title: str = "Predicted vs Actual") -> None:
    """Scatter of predicted vs actual values, colored by instance type."""
    preds = results["preds"]
    targets = results["targets"]
    types = np.array(results["types"])

    fig, ax = plt.subplots(figsize=(8, 8))
    for t in ("C", "R", "RC"):
        mask = types == t
        if mask.any():
            ax.scatter(
                targets[mask], preds[mask],
                alpha=0.4, s=10, c=_TYPE_COLORS[t],
                label=f"{t}  (n={mask.sum():,})",
            )
    lo = float(min(targets.min(), preds.min()))
    hi = float(max(targets.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect prediction")

    ax.set_xlabel("Actual objective value")
    ax.set_ylabel("Predicted objective value")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(results: Dict, output_path: Path) -> None:
    """Three-panel residual analysis: histogram, residual vs predicted, box plots by type."""
    preds = results["preds"]
    targets = results["targets"]
    types = np.array(results["types"])
    residuals = preds - targets

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: histogram
    axes[0].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", label="zero")
    axes[0].axvline(residuals.mean(), color="orange", linestyle="--",
                    label=f"mean = {residuals.mean():.1f}")
    axes[0].set_xlabel("Residual (pred - actual)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: residual vs predicted
    axes[1].scatter(preds, residuals, alpha=0.3, s=8, c="steelblue")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted objective value")
    axes[1].set_ylabel("Residual (pred - actual)")
    axes[1].set_title("Residual vs predicted (heteroscedasticity check)")
    axes[1].grid(alpha=0.3)

    # Panel 3: box plot by type
    box_data = [residuals[types == t] for t in ("C", "R", "RC")]
    bp = axes[2].boxplot(box_data, labels=["C", "R", "RC"], patch_artist=True, showfliers=True)
    for patch, t in zip(bp["boxes"], ("C", "R", "RC")):
        patch.set_facecolor(_TYPE_COLORS[t])
        patch.set_alpha(0.5)
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Residual (pred - actual)")
    axes[2].set_title("Residuals by instance type")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_vs_size(results: Dict, output_path: Path) -> None:
    """Absolute error vs number of nodes, colored by instance type."""
    preds = results["preds"]
    targets = results["targets"]
    types = np.array(results["types"])
    sizes = np.array(results["num_nodes"])
    abs_err = np.abs(preds - targets)

    fig, ax = plt.subplots(figsize=(10, 6))
    for t in ("C", "R", "RC"):
        mask = types == t
        if mask.any():
            ax.scatter(
                sizes[mask], abs_err[mask],
                alpha=0.3, s=10, c=_TYPE_COLORS[t],
                label=f"{t}  (n={mask.sum():,})",
            )

    ax.set_xlabel("Number of nodes (depot + served customers)")
    ax.set_ylabel("Absolute error  |pred − actual|")
    ax.set_title("Error vs graph size")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def save_predictions_csv(results: Dict, output_path: Path) -> None:
    """Write one row per graph: type, size, actual, predicted, residual, abs_error."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_type", "num_nodes", "actual", "predicted", "residual", "abs_error"])
        for i in range(len(results["preds"])):
            actual = float(results["targets"][i])
            pred   = float(results["preds"][i])
            writer.writerow([
                results["types"][i],
                int(results["num_nodes"][i]),
                f"{actual:.4f}",
                f"{pred:.4f}",
                f"{pred - actual:.4f}",
                f"{abs(pred - actual):.4f}",
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained GAT checkpoint and produce a comprehensive report "
            "(metrics, plots, CSV predictions)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the checkpoint .pt file (e.g., models/<run_id>/best.pt).")
    p.add_argument("--data_dir",   type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory containing prepared train.pt / val.pt / test.pt / stats.json.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Directory to write evaluation outputs. Default: '<checkpoint_dir>/evaluation/'.")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--splits",     nargs="+", choices=["train", "val", "test"],
                   default=["test"], help="Which splits to evaluate.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / "evaluation"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading prepared data from {args.data_dir} ...")
    train_set, val_set, test_set, _ = load_prepared_data(args.data_dir)
    in_channels = train_set[0].x.shape[1]
    print(f"  detected in_channels = {in_channels}")

    print(f"\nLoading checkpoint from {args.checkpoint} ...")
    model, ckpt, stats = load_checkpoint(args.checkpoint, device, in_channels)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  best epoch:        {ckpt.get('epoch', '?')}")
    val_loss = ckpt.get('val_loss', None)
    if val_loss is not None:
        print(f"  saved val_loss:    {val_loss:.4f}")
    print(f"  model parameters:  {n_params:,}")

    splits = {"train": train_set, "val": val_set, "test": test_set}
    all_metrics: Dict[str, dict] = {}

    for split_name in args.splits:
        split = splits[split_name]
        if not split:
            print(f"\n{split_name}: empty, skipping")
            continue

        print(f"\nEvaluating on {split_name} ({len(split):,} graphs) ...")
        loader = DataLoader(split, batch_size=args.batch_size, shuffle=False)
        results = predict(model, loader, device, stats)
        metrics = compute_metrics(results["preds"], results["targets"], results["types"])
        all_metrics[split_name] = metrics

        print(f"  N      = {metrics['n']:,}")
        print(f"  MAE    = {metrics['mae']:.2f}")
        print(f"  RMSE   = {metrics['rmse']:.2f}")
        print(f"  R^2    = {metrics['r2']:.4f}")
        print(f"  Residual statistics:")
        print(f"    mean       = {metrics['residual_mean']:+.2f}")
        print(f"    std        = {metrics['residual_std']:.2f}")
        print(f"    [P5, P95]  = [{metrics['residual_p5']:+.2f},  {metrics['residual_p95']:+.2f}]")
        print(f"    [min, max] = [{metrics['residual_min']:+.2f},  {metrics['residual_max']:+.2f}]")
        print(f"  By instance type:")
        for t, m in metrics["per_type"].items():
            print(f"    {t:3s}: n = {m['n']:>5d}   MAE = {m['mae']:.2f}   RMSE = {m['rmse']:.2f}")

        plot_pred_vs_actual(
            results,
            args.output_dir / f"pred_vs_actual_{split_name}.png",
            title=f"Predicted vs Actual — {split_name} split",
        )
        plot_residuals(results, args.output_dir / f"residuals_{split_name}.png")
        plot_error_vs_size(results, args.output_dir / f"error_vs_size_{split_name}.png")
        save_predictions_csv(results, args.output_dir / f"predictions_{split_name}.csv")
        print(f"  -> plots and CSV written to {args.output_dir.resolve()}")

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll outputs saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
