#!/usr/bin/env python3
"""
train.py — Phase C of the GAT pipeline.

Trains the GAT surrogate model defined in model.py on the prepared splits
produced by data_prep.py. Uses:
  - MSE loss
  - Adam optimizer with weight decay
  - CosineAnnealingWarmRestarts LR schedule
  - Gradient clipping (max_norm)
  - Early stopping on validation MSE
  - Weights & Biases for logging and sweep support

Reports metrics in BOTH normalized units (loss, for the optimizer / scheduler /
early stopping) and original objective units (MAE, RMSE, R², for human-readable
evaluation), unrolling the y z-scoring stored in prepared_data/stats.json.

After training, loads the best checkpoint and evaluates on the test set,
including a per-instance-type (C / R / RC) breakdown.

Usage — single run (defaults match the agreed Phase C spec):

    python train.py
    python train.py --hidden_dim 256 --num_gat_layers 5 --lr 5e-4

Usage — W&B Bayesian sweep:

    # 1. Create the sweep (returns a SWEEP_ID):
    wandb sweep sweep.yaml

    # 2. Start one or more agents (in separate terminals / on separate GPUs):
    wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

import wandb

# Local imports
from data_prep import NormStats, load_prepared_data
from model import GATSurrogate


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI or by W&B sweep agent)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "prepared_data"
DEFAULT_MODELS_DIR = "models"

# Training
DEFAULT_LR             = 1e-3
DEFAULT_WEIGHT_DECAY   = 1e-4
DEFAULT_BATCH_SIZE     = 64
DEFAULT_EPOCHS         = 110
DEFAULT_GRAD_CLIP      = 1.0
DEFAULT_PATIENCE       = 31

# Scheduler
DEFAULT_T0             = 15      # CosineAnnealingWarmRestarts T_0, in epochs
DEFAULT_T_MULT         = 1       # cycle length multiplier
DEFAULT_ETA_MIN        = 0.0     # minimum LR at cycle bottom

# Model
DEFAULT_HIDDEN_DIM       = 128
DEFAULT_EDGE_HIDDEN_DIM  = 32
DEFAULT_NUM_GAT_LAYERS   = 4
DEFAULT_HEADS            = 4
DEFAULT_HEAD_AGGR        = "concat"   # or "average"
DEFAULT_DROPOUT          = 0.1
DEFAULT_HEAD_DROPOUT     = 0.2
DEFAULT_HEAD_DEPTH       = 2

# W&B
DEFAULT_PROJECT          = "gat-evrp"
DEFAULT_ENTITY           = None        # use the user's default entity


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: GATSurrogate,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Run one epoch of training; return the mean MSE loss over training samples."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)              # [B]
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: GATSurrogate,
    loader: DataLoader,
    device: torch.device,
    stats: NormStats,
    return_per_type: bool = False,
) -> Dict:
    """
    Evaluate on a loader.
    Returns metrics in both normalized space (loss, for optimization) and
    original objective units (mae, rmse, r2, for reporting). If
    return_per_type is True, also includes a per-instance-type breakdown.
    """
    model.eval()
    preds_n, targets_n, types = [], [], []
    for batch in loader:
        p = model(batch.to(device)).squeeze(-1).cpu()
        preds_n.append(p)
        targets_n.append(batch.y.cpu())
        if return_per_type:
            types.extend([g.instance_type for g in batch.to_data_list()])

    if not preds_n:
        nan = float("nan")
        return {"loss": nan, "mae": nan, "rmse": nan, "r2": nan}

    preds_n   = torch.cat(preds_n)
    targets_n = torch.cat(targets_n)

    # Loss in normalized space (what the optimizer / scheduler / early stopping use)
    loss = F.mse_loss(preds_n, targets_n).item()

    # Un-normalize for human-readable metrics
    preds_o   = preds_n   * stats.y_std + stats.y_mean
    targets_o = targets_n * stats.y_std + stats.y_mean
    err = preds_o - targets_o

    mae  = err.abs().mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    ss_res = err.pow(2).sum().item()
    ss_tot = (targets_o - targets_o.mean()).pow(2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    out = {"loss": loss, "mae": mae, "rmse": rmse, "r2": r2}

    if return_per_type:
        per = defaultdict(lambda: {"preds": [], "targets": []})
        for i, t in enumerate(types):
            per[t]["preds"].append(preds_o[i].item())
            per[t]["targets"].append(targets_o[i].item())
        per_out = {}
        for t, d in per.items():
            p = torch.tensor(d["preds"])
            g = torch.tensor(d["targets"])
            e = p - g
            per_out[t] = {
                "n":    len(d["preds"]),
                "mae":  e.abs().mean().item(),
                "rmse": e.pow(2).mean().sqrt().item(),
            }
        out["per_type"] = per_out

    return out


# ---------------------------------------------------------------------------
# CLI (underscored arg names — required for W&B sweep agent compatibility)
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the GAT surrogate model with W&B logging and sweep support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--data_dir",   default=DEFAULT_DATA_DIR,   help="Directory with prepared train/val/test.pt + stats.json.")
    p.add_argument("--models_dir", default=DEFAULT_MODELS_DIR, help="Root directory for checkpoint folders.")

    # Optimizer
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    # Training loop
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--grad_clip",    type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--patience",     type=int,   default=DEFAULT_PATIENCE, help="Early-stopping patience, in epochs.")

    # Scheduler
    p.add_argument("--t0",           type=int,   default=DEFAULT_T0,      help="CosineAnnealingWarmRestarts T_0, in epochs.")
    p.add_argument("--t_mult",       type=int,   default=DEFAULT_T_MULT)
    p.add_argument("--eta_min",      type=float, default=DEFAULT_ETA_MIN)

    # Model
    p.add_argument("--hidden_dim",       type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--edge_hidden_dim",  type=int, default=DEFAULT_EDGE_HIDDEN_DIM)
    p.add_argument("--num_gat_layers",   type=int, default=DEFAULT_NUM_GAT_LAYERS)
    p.add_argument("--heads",            type=int, default=DEFAULT_HEADS)
    p.add_argument("--head_aggr",        choices=["concat", "average"], default=DEFAULT_HEAD_AGGR)
    p.add_argument("--dropout",          type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--head_dropout",     type=float, default=DEFAULT_HEAD_DROPOUT)
    p.add_argument("--head_depth",       type=int,   default=DEFAULT_HEAD_DEPTH)

    # W&B
    p.add_argument("--wandb_project", default=DEFAULT_PROJECT)
    p.add_argument("--wandb_entity",  default=DEFAULT_ENTITY)
    p.add_argument("--wandb_mode",    default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--run_name",      default=None, help="Optional explicit W&B run name.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize W&B. In a sweep, the agent injects its config here, overriding CLI defaults.
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config=vars(args),
    )
    cfg = wandb.config   # source of truth for hyperparameters from this point on

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load prepared splits
    print(f"Loading prepared data from {cfg.data_dir} ...")
    train_set, val_set, test_set, stats = load_prepared_data(Path(cfg.data_dir))
    print(f"  train: {len(train_set):>6,d}   val: {len(val_set):>6,d}   test: {len(test_set):>6,d}")

    if not train_set:
        print("ERROR: empty training set", file=sys.stderr)
        wandb.finish(exit_code=1)
        return 1

    # Loaders
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False) if val_set else None
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False) if test_set else None

    # Model
    in_channels = train_set[0].x.shape[1]
    model = GATSurrogate(
        in_channels=in_channels,
        edge_in_channels=2,
        hidden_dim=cfg.hidden_dim,
        edge_hidden_dim=cfg.edge_hidden_dim,
        num_gat_layers=cfg.num_gat_layers,
        heads=cfg.heads,
        concat_heads=(cfg.head_aggr == "concat"),
        dropout=cfg.dropout,
        head_dropout=cfg.head_dropout,
        head_depth=cfg.head_depth,
        use_residual=True,
        use_global_conditioning=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    wandb.summary["model/n_parameters"] = n_params

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min,
    )

    # Checkpoint location: models/<wandb_run_id>/best.pt
    run_id = wandb.run.id if wandb.run is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(cfg.models_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    # Training loop with early stopping on val_loss
    best_val_loss = float("inf")
    best_epoch    = -1
    patience_ctr  = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip)

        log = {
            "epoch":       epoch,
            "train_loss":  train_loss,
            "lr":          optimizer.param_groups[0]["lr"],
        }
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, stats)
            log.update({
                "val_loss": val_metrics["loss"],
                "val_mae":  val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2":   val_metrics["r2"],
            })

            improved = val_metrics["loss"] < best_val_loss - 1e-6
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch    = epoch
                patience_ctr  = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config":           dict(cfg),
                    "epoch":            epoch,
                    "val_loss":         best_val_loss,
                    "stats":            stats.to_dict(),
                }, best_path)
            else:
                patience_ctr += 1

        scheduler.step()

        msg = f"epoch {epoch:3d}/{cfg.epochs} | train_loss = {train_loss:.4f}"
        if val_metrics is not None:
            msg += (
                f" | val_loss = {val_metrics['loss']:.4f}"
                f" | val_mae = {val_metrics['mae']:.2f}"
                f" | val_rmse = {val_metrics['rmse']:.2f}"
                f" | patience = {patience_ctr}/{cfg.patience}"
            )
        msg += f" | lr = {log['lr']:.2e}"
        print(msg)

        wandb.log(log, step=epoch)

        if val_loader is not None and patience_ctr >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch}, best val_loss = {best_val_loss:.4f})")
            break

    # Final test evaluation using best checkpoint
    if test_loader is not None and best_path.exists():
        print(f"\nLoading best checkpoint from {best_path}")
        ckpt = torch.load(best_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics = evaluate(model, test_loader, device, stats, return_per_type=True)
        print(f"\nTest metrics (best model, epoch {best_epoch}):")
        print(f"  loss (normalized) = {test_metrics['loss']:.4f}")
        print(f"  MAE  (orig units) = {test_metrics['mae']:.2f}")
        print(f"  RMSE (orig units) = {test_metrics['rmse']:.2f}")
        print(f"  R^2               = {test_metrics['r2']:.4f}")
        print(f"\n  Per-instance-type breakdown:")
        for t, m in test_metrics["per_type"].items():
            print(f"    {t:3s}  n = {m['n']:>5d}   MAE = {m['mae']:.2f}   RMSE = {m['rmse']:.2f}")

        flat = {
            "test_loss":      test_metrics["loss"],
            "test_mae":       test_metrics["mae"],
            "test_rmse":      test_metrics["rmse"],
            "test_r2":        test_metrics["r2"],
            "best_val_loss":  best_val_loss,
            "best_epoch":     best_epoch,
        }
        for t, m in test_metrics["per_type"].items():
            flat[f"test_mae/{t}"]  = m["mae"]
            flat[f"test_rmse/{t}"] = m["rmse"]
        for k, v in flat.items():
            wandb.summary[k] = v

    print(f"\nBest checkpoint: {best_path.resolve()}")
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
