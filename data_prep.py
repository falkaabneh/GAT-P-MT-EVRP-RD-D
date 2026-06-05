#!/usr/bin/env python3
"""
data_prep.py — Phase A of the GAT pipeline.

Reads the cached PyG graphs produced by build_graphs.py and:

  1. Splits them into train / val / test (stratified by instance_type by default).
  2. Computes global normalization statistics on the *training* split only.
  3. Normalizes node features:
       - (x, y) coordinates  -> sinusoidal positional encoding (2 * pe_dim dims)
       - demand              -> divided by the graph's vehicle_load_capacity (per-graph)
       - release_time        -> min-max [0, 1] using training-set min/max
       - deadline            -> min-max [0, 1] using training-set min/max
       - time-window width   -> min-max [0, 1] using training-set min/max
  4. Z-scores edge features (distance, distance / battery_capacity) using
     training-set statistics.
  5. Z-scores the regression target y using training-set statistics.
  6. Writes the prepared splits, normalization stats, and run config to disk.

Default I/O layout (relative to the repo root):

    <repo root>/
      processed_graphs/graphs.pt        (input, produced by build_graphs.py)
      prepared_data/                    (output, created by this script)
          train.pt
          val.pt
          test.pt
          stats.json                    (normalization statistics)
          config.json                   (the exact arguments used for this run)

Usage from the shell:

    python data_prep.py                                              # all defaults
    python data_prep.py --split-ratios 0.75 0.15 0.10                # custom split
    python data_prep.py --split-ratios 0.8 0.0 0.2 --no-stratify     # no val, no stratify
    python data_prep.py --pe-dim 32 --pe-base 1000                   # bigger PE
    python data_prep.py --help

Usage from Python (e.g. in a notebook):

    from data_prep import prepare_data, load_prepared_data, NormStats

    # Build everything in-memory:
    graphs = torch.load("processed_graphs/graphs.pt", weights_only=False)
    train, val, test, stats = prepare_data(graphs, split_ratios=(0.8, 0.1, 0.1))

    # Or load the on-disk artifacts produced by a previous CLI run:
    train, val, test, stats = load_prepared_data("prepared_data")
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_PATH = Path("processed_graphs/graphs.pt")
DEFAULT_OUTPUT_DIR = Path("prepared_data")
DEFAULT_SPLIT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)
DEFAULT_PE_DIM = 16
DEFAULT_PE_BASE = 100.0
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Normalization statistics container
# ---------------------------------------------------------------------------
@dataclass
class NormStats:
    """Normalization statistics computed on the training split."""
    # Target
    y_mean: float;   y_std: float
    # Node-feature min-max (release_time, deadline, time-window width)
    rt_min: float;   rt_max: float
    dl_min: float;   dl_max: float
    tw_min: float;   tw_max: float
    # Edge-feature z-score (distance, distance / battery_capacity)
    dist_mean: float;  dist_std: float
    ratio_mean: float; ratio_std: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(**d)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------
def sinusoidal_pe_1d(x: torch.Tensor, dim: int, base: float = DEFAULT_PE_BASE) -> torch.Tensor:
    """
    Sinusoidal positional encoding for a 1D coordinate vector.

    x:    [N] tensor of raw coordinate values
    dim:  encoding dimension (must be a positive even int)
    base: period scale. NLP uses 10000 for token positions, but Solomon
          coordinates live in ~[0, 100], so a smaller base (100 or 1000)
          puts the high-frequency components at a meaningful spatial scale.

    Returns: [N, dim] tensor with the first dim/2 cols being sin(...) and
             the last dim/2 cols being cos(...).
    """
    assert dim % 2 == 0 and dim > 0, f"PE dim must be a positive even int, got {dim}"
    half = dim // 2
    div_term = base ** (torch.arange(half, dtype=torch.float) / half)
    angles = x.unsqueeze(-1) / div_term
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def sinusoidal_pe_2d(coords: torch.Tensor, dim: int, base: float = DEFAULT_PE_BASE) -> torch.Tensor:
    """
    Sinusoidal PE applied independently to each coordinate of 2D positions.

    coords:  [N, 2] tensor (x_coord, y_coord)
    Returns: [N, 2*dim] tensor (PE for x_coord ++ PE for y_coord).
    """
    return torch.cat([
        sinusoidal_pe_1d(coords[:, 0], dim, base),
        sinusoidal_pe_1d(coords[:, 1], dim, base),
    ], dim=-1)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def stratified_split(
    graphs: List[Data],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Split graphs into train/val/test balanced by instance_type."""
    rng = torch.Generator().manual_seed(seed)
    by_type: dict = defaultdict(list)
    for g in graphs:
        by_type[g.instance_type].append(g)

    train, val, test = [], [], []
    for type_graphs in by_type.values():
        n = len(type_graphs)
        perm = torch.randperm(n, generator=rng).tolist()
        shuffled = [type_graphs[i] for i in perm]
        n_train = int(round(n * ratios[0]))
        n_val   = int(round(n * ratios[1]))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    # Shuffle within each split so batches aren't ordered by type
    for split in (train, val, test):
        if split:
            perm = torch.randperm(len(split), generator=rng).tolist()
            split[:] = [split[i] for i in perm]
    return train, val, test


def shuffled_split(
    graphs: List[Data],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Plain shuffled split, ignoring instance_type."""
    rng = torch.Generator().manual_seed(seed)
    n = len(graphs)
    perm = torch.randperm(n, generator=rng).tolist()
    shuffled = [graphs[i] for i in perm]
    n_train = int(round(n * ratios[0]))
    n_val   = int(round(n * ratios[1]))
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# Stats + per-graph normalization
# ---------------------------------------------------------------------------
def compute_stats(graphs: List[Data], eps: float = 1e-8) -> NormStats:
    """Compute global normalization statistics over the given (training) split."""
    if not graphs:
        raise ValueError("compute_stats requires a non-empty list of graphs")

    y      = torch.cat([g.y for g in graphs])
    rts    = torch.cat([g.x[:, 3] for g in graphs])
    dls    = torch.cat([g.x[:, 4] for g in graphs])
    tws    = torch.cat([g.x[:, 5] for g in graphs])
    dists  = torch.cat([g.edge_attr[:, 0] for g in graphs])
    ratios = torch.cat([g.edge_attr[:, 1] for g in graphs])

    return NormStats(
        y_mean=float(y.mean()),   y_std=float(y.std() + eps),
        rt_min=float(rts.min()),  rt_max=float(rts.max()),
        dl_min=float(dls.min()),  dl_max=float(dls.max()),
        tw_min=float(tws.min()),  tw_max=float(tws.max()),
        dist_mean=float(dists.mean()),   dist_std=float(dists.std() + eps),
        ratio_mean=float(ratios.mean()), ratio_std=float(ratios.std() + eps),
    )


def normalize_graph(
    g: Data,
    stats: NormStats,
    pe_dim: int,
    pe_base: float,
) -> Data:
    """
    Return a new Data object with normalized x, edge_attr, and y.

    New node feature layout (dim = 2*pe_dim + 4):
        [ PE_x (pe_dim) | PE_y (pe_dim) | demand_norm | rt_norm | dl_norm | tw_norm ]

    Edge features (dim = 2): [distance_z, range_ratio_z]
    Target y: z-scored.
    """
    coords = g.x[:, :2]
    demand = g.x[:, 2]
    rt     = g.x[:, 3]
    dl     = g.x[:, 4]
    tw     = g.x[:, 5]

    coord_pe = sinusoidal_pe_2d(coords, pe_dim, pe_base)

    vlc = float(g.vehicle_load_capacity.item())
    demand_norm = demand / vlc

    rt_norm = (rt - stats.rt_min) / (stats.rt_max - stats.rt_min)
    dl_norm = (dl - stats.dl_min) / (stats.dl_max - stats.dl_min)
    tw_norm = (tw - stats.tw_min) / (stats.tw_max - stats.tw_min)

    new_x = torch.cat([
        coord_pe,
        demand_norm.unsqueeze(-1),
        rt_norm.unsqueeze(-1),
        dl_norm.unsqueeze(-1),
        tw_norm.unsqueeze(-1),
    ], dim=-1)

    dist_norm  = (g.edge_attr[:, 0] - stats.dist_mean)  / stats.dist_std
    ratio_norm = (g.edge_attr[:, 1] - stats.ratio_mean) / stats.ratio_std
    new_edge_attr = torch.stack([dist_norm, ratio_norm], dim=-1)

    new_y = (g.y - stats.y_mean) / stats.y_std

    out = Data(x=new_x, edge_index=g.edge_index, edge_attr=new_edge_attr, y=new_y)

    # Carry metadata through so we can un-normalize and report per-type later
    out.battery_capacity      = g.battery_capacity
    out.vehicle_load_capacity = g.vehicle_load_capacity
    out.instance_id           = g.instance_id
    out.instance_type         = g.instance_type
    out.nodes_served          = g.nodes_served
    out.original_node_ids     = g.original_node_ids
    return out


# ---------------------------------------------------------------------------
# Public entry point (importable)
# ---------------------------------------------------------------------------
def prepare_data(
    graphs: List[Data],
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    pe_dim: int = DEFAULT_PE_DIM,
    pe_base: float = DEFAULT_PE_BASE,
    stratify: bool = True,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Data], List[Data], List[Data], NormStats]:
    """
    Run the full Phase A pipeline in memory and return:

        train, val, test, stats

    split_ratios: three floats summing to 1. Any one may be 0 to skip that split,
                  e.g. (0.8, 0.0, 0.2) for train/test only.
    pe_dim:       sinusoidal PE size per coordinate (must be even; 8/16/32 typical).
    pe_base:      period scale for the encoding (smaller for coords in [0, 100]).
    stratify:     stratify by instance_type. Set False for pure random shuffle.
    seed:         controls both the split shuffle and the in-split shuffle.
    """
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1, got {sum(split_ratios)}")
    if not (pe_dim > 0 and pe_dim % 2 == 0):
        raise ValueError(f"pe_dim must be a positive even int, got {pe_dim}")

    splitter = stratified_split if stratify else shuffled_split
    train_raw, val_raw, test_raw = splitter(graphs, split_ratios, seed)

    # Stats from training only - prevents val/test leakage
    stats = compute_stats(train_raw)

    train = [normalize_graph(g, stats, pe_dim, pe_base) for g in train_raw]
    val   = [normalize_graph(g, stats, pe_dim, pe_base) for g in val_raw]
    test  = [normalize_graph(g, stats, pe_dim, pe_base) for g in test_raw]
    return train, val, test, stats


# ---------------------------------------------------------------------------
# Disk I/O helpers (importable)
# ---------------------------------------------------------------------------
def save_prepared_data(
    train: List[Data],
    val: List[Data],
    test: List[Data],
    stats: NormStats,
    output_dir: Path,
    config: dict = None,
) -> None:
    """Write train/val/test splits, stats.json, and an optional config.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, output_dir / "train.pt")
    torch.save(val,   output_dir / "val.pt")
    torch.save(test,  output_dir / "test.pt")
    stats.save(output_dir / "stats.json")
    if config is not None:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)


def load_prepared_data(
    output_dir: Path,
) -> Tuple[List[Data], List[Data], List[Data], NormStats]:
    """Load (train, val, test, stats) previously written by save_prepared_data."""
    output_dir = Path(output_dir)
    train = torch.load(output_dir / "train.pt", weights_only=False)
    val   = torch.load(output_dir / "val.pt",   weights_only=False)
    test  = torch.load(output_dir / "test.pt",  weights_only=False)
    stats = NormStats.load(output_dir / "stats.json")
    return train, val, test, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A of the GAT pipeline: split + normalize the PyG graphs "
            "produced by build_graphs.py, and write the prepared dataset to disk."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-path", type=Path, default=DEFAULT_INPUT_PATH,
        help="Path to the .pt file produced by build_graphs.py.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where train.pt / val.pt / test.pt / stats.json / config.json will be written.",
    )
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_SPLIT_RATIOS),
        help="Three floats summing to 1. Use 0 for any split to skip it.",
    )
    parser.add_argument(
        "--pe-dim", type=int, default=DEFAULT_PE_DIM,
        help="Sinusoidal PE dim per coordinate (even integer; 8/16/32 typical).",
    )
    parser.add_argument(
        "--pe-base", type=float, default=DEFAULT_PE_BASE,
        help="Sinusoidal PE base period (smaller for coords in [0, 100]).",
    )
    parser.add_argument(
        "--no-stratify", dest="stratify", action="store_false",
        help="Disable stratification by instance_type (use plain shuffle).",
    )
    parser.set_defaults(stratify=True)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for the shuffle.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input_path.is_file():
        print(f"ERROR: input file not found: {args.input_path.resolve()}", file=sys.stderr)
        print(f"       (run build_graphs.py first)", file=sys.stderr)
        return 1

    print(f"Loading graphs from {args.input_path} ...")
    graphs = torch.load(args.input_path, weights_only=False)
    print(f"  loaded {len(graphs)} graphs")

    split_ratios = tuple(args.split_ratios)

    train, val, test, stats = prepare_data(
        graphs,
        split_ratios=split_ratios,
        pe_dim=args.pe_dim,
        pe_base=args.pe_base,
        stratify=args.stratify,
        seed=args.seed,
    )

    print(f"\nSplit sizes:")
    print(f"  train = {len(train)}")
    print(f"  val   = {len(val)}")
    print(f"  test  = {len(test)}")
    print(f"\nNode feature dim: {2 * args.pe_dim + 4}  (= 2*pe_dim + 4)")
    print(f"Edge feature dim: 2")
    print(f"\nTraining normalization stats:")
    for k, v in stats.to_dict().items():
        print(f"  {k:11s} = {v: .4f}")

    config = {
        "input_path": str(args.input_path),
        "split_ratios": list(split_ratios),
        "pe_dim": args.pe_dim,
        "pe_base": args.pe_base,
        "stratify": args.stratify,
        "seed": args.seed,
        "node_feature_dim": 2 * args.pe_dim + 4,
        "edge_feature_dim": 2,
    }

    save_prepared_data(train, val, test, stats, args.output_dir, config=config)
    print(f"\nSaved prepared data to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
