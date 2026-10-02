#!/usr/bin/env python3
"""
predict.py — Phase E of the GAT pipeline: inference with a trained surrogate.

Loads a trained checkpoint once and scores candidate solutions, either from
raw JSON files (CLI batch mode) or from in-memory customer subsets (in-loop
API mode, e.g. called from inside an ALNS search).

Feature construction is delegated to `build_graphs.build_graph` and
`data_prep.normalize_graph` so that inference features are identical to
training features by construction. Do not reimplement the feature pipeline
here — any divergence would silently corrupt predictions.

IMPORTANT — the vehicle_load_capacity override
----------------------------------------------
build_graphs.py applies a filename-based rule when building the training set:
a JSON file whose stem ends in "battery" has vehicle_load_capacity forced to
50.0 for every record it contains, regardless of the value stored in the JSON.
That value feeds two features — the per-node demand normalization
(demand / vehicle_load_capacity) and the global conditioning scalar — so
inference MUST replicate the rule or predictions on those files are garbage.

`collect_records` below derives the override from the filename and
`score_records` applies it. When calling `score_records` directly with records
loaded some other way, pass `vehicle_load_overrides` yourself.

What the surrogate predicts
---------------------------
Given the depot plus a set of *served* customers (and the battery / load
capacities), it predicts the ALNS objective value for that subset. It does
NOT evaluate a specific route: it estimates what the objective would be once
ALNS has optimized routing for that customer selection. This makes it useful
as a fast evaluator for the customer-selection phase of a search.

CLI usage
---------

    # Score every record in a training-style JSON file
    python predict.py --checkpoint models/<run_id>/best.pt \\
                      --input "training data/some_file.json" \\
                      --reference-dir "data generation" \\
                      --output predictions.csv

    # Score every JSON file in a directory
    python predict.py --checkpoint models/<run_id>/best.pt \\
                      --input "training data" \\
                      --reference-dir "data generation" \\
                      --output predictions.csv

    python predict.py --help

If the input records contain `objective_value`, the CSV also includes the
actual value, the residual, and a summary of MAE / RMSE / R^2 — useful for
validating the surrogate on a held-out file.

Python API usage
----------------

    from predict import Predictor

    predictor = Predictor(
        checkpoint_path="models/<run_id>/best.pt",
        reference_dir="data generation",
        prepared_config="prepared_data/config.json",   # optional but recommended
    )

    # Score a single candidate subset (in-loop)
    obj = predictor.score_subset(
        instance_id="C_012",
        nodes_served=[1, 3, 7, 12, ...],   # customer ids, depot excluded
        battery_capacity=120.0,
        vehicle_load_capacity=50.0,
    )

    # Score many candidates at once (much faster than looping score_subset)
    objs = predictor.score_subsets([
        ("C_012", [1, 3, 7], 120.0, 50.0),
        ("C_012", [1, 3, 7, 9], 120.0, 50.0),
        ...
    ])
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from build_graphs import build_graph, load_reference_instances, NUM_CUSTOMERS
from data_prep import NormStats, normalize_graph, DEFAULT_PE_BASE
from model import GATSurrogate


DEFAULT_BATCH_SIZE = 128
DEFAULT_VEHICLE_LOAD = 50.0

# Filename-based override, mirroring build_graphs.load_all_graphs
BATTERY_SUFFIX = "battery"
BATTERY_OVERRIDE_LOAD = 50.0


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class Predictor:
    """
    A warm, reusable wrapper around a trained GAT surrogate.

    Loads the checkpoint and the reference instance files once at
    construction; subsequent scoring calls reuse them. Instantiate this ONCE
    and hold onto it — constructing a Predictor per call would dominate
    runtime in an in-loop setting.

    Args:
        checkpoint_path:  path to a best.pt written by train.py
        reference_dir:    directory with C_instances.json / R_instances.json /
                          RC_instances.json
        prepared_config:  optional path to prepared_data/config.json. When
                          given, pe_dim and pe_base are read from it, which
                          guarantees they match the training run. When
                          omitted, pe_dim is derived from the model's input
                          layer and pe_base falls back to `pe_base` below.
        pe_base:          fallback PE base period, used only when
                          prepared_config is not supplied.
        device:           torch device; auto-detects CUDA when omitted.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        reference_dir: Path,
        prepared_config: Optional[Path] = None,
        pe_base: float = DEFAULT_PE_BASE,
        device: Optional[torch.device] = None,
        node_set: str = "pool",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_set = node_set

        ckpt = torch.load(Path(checkpoint_path), weights_only=False, map_location=self.device)
        cfg = ckpt["config"]
        self.checkpoint_config = cfg
        self.best_epoch = ckpt.get("epoch")
        self.best_val_loss = ckpt.get("val_loss")

        # Normalization statistics used during training (target + features)
        self.stats = NormStats.from_dict(ckpt["stats"])

        # Recover the node-encoder input width from the saved weights. This is
        # the ground truth for in_channels; the training config does not store it.
        first_weight = ckpt["model_state_dict"]["node_encoder.0.weight"]
        self.in_channels = int(first_weight.shape[1])

        # in_channels = 2 * pe_dim + n_extra, where n_extra is 4 without the
        # prize/penalty features and 6 with them. Both widths can be even, so
        # the checkpoint alone is ambiguous; prepared_data/config.json settles
        # it below and the candidates are recorded here for that check.
        self._pe_dim_candidates = {}
        for n_extra in (4, 6):
            d, rem = divmod(self.in_channels - n_extra, 2)
            if rem == 0 and d > 0:
                self._pe_dim_candidates[n_extra] = d
        if not self._pe_dim_candidates:
            raise ValueError(
                f"Cannot derive pe_dim from in_channels={self.in_channels}; "
                f"expected 2*pe_dim + 4 or 2*pe_dim + 6."
            )
        # Default to the 6-feature reading; config.json overrides when present.
        n_extra_default = 4 if 4 in self._pe_dim_candidates else 6
        self.include_prize_penalty = n_extra_default == 6
        self.pe_dim = self._pe_dim_candidates[n_extra_default]
        self.pe_base = pe_base

        # Prefer the recorded data_prep config when available: it is the only
        # place pe_base is persisted, and it lets us cross-check pe_dim.
        if prepared_config is not None:
            with open(prepared_config, "r") as f:
                prep = json.load(f)
            if "pe_dim" in prep:
                recorded = int(prep["pe_dim"])
                # Accept whichever feature width reconciles the checkpoint's
                # in_channels with the pe_dim this data was prepared at.
                match = [n for n, d in self._pe_dim_candidates.items() if d == recorded]
                if not match:
                    raise ValueError(
                        f"pe_dim mismatch: checkpoint in_channels={self.in_channels} "
                        f"implies pe_dim in {sorted(self._pe_dim_candidates.values())}, "
                        f"but {prepared_config} records {recorded}. The checkpoint and "
                        f"prepared_data are from different runs."
                    )
                self.pe_dim = recorded
                self.include_prize_penalty = match[0] == 6
            self.pe_base = float(prep.get("pe_base", pe_base))

        # Rebuild the architecture and load the trained weights
        self.model = GATSurrogate(
            in_channels=self.in_channels,
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
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.reference_data = load_reference_instances(Path(reference_dir))

    # ---- graph construction ------------------------------------------------
    def _build_normalized_graph(
        self,
        instance_id: str,
        nodes_served: Sequence[int],
        battery_capacity: float,
        vehicle_load_capacity: float,
    ) -> Data:
        """
        Build a normalized PyG Data object for one candidate subset.

        Goes through build_graphs.build_graph by synthesizing the two
        "excluded customers" lists it expects, so the resulting features are
        constructed by exactly the same code path used to build the training
        set.
        """
        served = set(nodes_served)
        excluded = [i for i in range(1, NUM_CUSTOMERS + 1) if i not in served]

        synthetic_record = {
            "instance_id": instance_id,
            "instance_type": self.reference_data[instance_id]["type"],
            "customers_not_selected_first_place": excluded,
            "customers_not_visited_from_selected_pool": [],
            "battery_capacity": float(battery_capacity),
            "vehicle_load_capacity": float(vehicle_load_capacity),
            "objective_value": float("nan"),   # unknown at inference time
        }

        g = build_graph(synthetic_record, self.reference_data,
                        node_set=self.node_set,
                        include_prize_penalty=self.include_prize_penalty)
        return normalize_graph(g, self.stats, self.pe_dim, self.pe_base)

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def _predict_graphs(self, graphs: List[Data], batch_size: int = DEFAULT_BATCH_SIZE) -> List[float]:
        """Run the model over normalized graphs and un-normalize the outputs."""
        if not graphs:
            return []
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        preds_n = []
        for batch in loader:
            preds_n.append(self.model(batch.to(self.device)).squeeze(-1).cpu())
        preds_n = torch.cat(preds_n)
        preds = preds_n * self.stats.y_std + self.stats.y_mean
        return preds.tolist()

    def score_subset(
        self,
        instance_id: str,
        nodes_served: Sequence[int],
        battery_capacity: float,
        vehicle_load_capacity: float = DEFAULT_VEHICLE_LOAD,
    ) -> float:
        """
        Predict the objective value for a single candidate customer subset.

        For scoring many candidates, prefer `score_subsets` — batching is
        substantially faster than repeated single calls.
        """
        g = self._build_normalized_graph(
            instance_id, nodes_served, battery_capacity, vehicle_load_capacity
        )
        return self._predict_graphs([g])[0]

    def score_subsets(
        self,
        candidates: Iterable[Tuple[str, Sequence[int], float, float]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """
        Predict objective values for many candidate subsets at once.

        candidates: iterable of
            (instance_id, nodes_served, battery_capacity, vehicle_load_capacity)

        Returns predictions in the same order as the input.
        """
        graphs = [
            self._build_normalized_graph(iid, served, bat, vlc)
            for iid, served, bat, vlc in candidates
        ]
        return self._predict_graphs(graphs, batch_size=batch_size)

    def score_records(
        self,
        records: Sequence[dict],
        vehicle_load_overrides: Optional[Sequence[Optional[float]]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Score raw training-style JSON records.

        Each record is passed through build_graph unchanged, so the served set
        is derived from `customers_not_selected_first_place` and
        `customers_not_visited_from_selected_pool` exactly as during training.

        vehicle_load_overrides: optional sequence parallel to `records`. A
            non-None entry forces that record's vehicle_load_capacity,
            replicating the filename-based rule applied when the training set
            was built (see the module docstring). Pass the overrides returned
            by `collect_records`; omitting them silently mismatches training
            features for battery-suffixed files.

        Returns (results, skipped) where each result dict contains
        instance_id, instance_type, num_nodes, predicted, and — when the
        record carries an `objective_value` — actual, residual and abs_error.
        """
        if vehicle_load_overrides is None:
            vehicle_load_overrides = [None] * len(records)
        if len(vehicle_load_overrides) != len(records):
            raise ValueError(
                f"vehicle_load_overrides has length {len(vehicle_load_overrides)}, "
                f"expected {len(records)} to match records"
            )

        graphs, metas, skipped = [], [], []

        for idx, (rec, override) in enumerate(zip(records, vehicle_load_overrides)):
            try:
                raw = build_graph(rec, self.reference_data,
                                  vehicle_load_override=override,
                                  node_set=self.node_set,
                                  include_prize_penalty=self.include_prize_penalty)
                graphs.append(normalize_graph(raw, self.stats, self.pe_dim, self.pe_base))
                actual = rec.get("objective_value")
                metas.append({
                    "index":         idx,
                    "instance_id":   raw.instance_id,
                    "instance_type": raw.instance_type,
                    "num_nodes":     int(raw.x.shape[0]),
                    "actual":        float(actual) if actual is not None else None,
                })
            except Exception as e:
                skipped.append({
                    "index":       idx,
                    "instance_id": rec.get("instance_id", "?"),
                    "reason":      str(e),
                })

        preds = self._predict_graphs(graphs, batch_size=batch_size)

        results = []
        for meta, pred in zip(metas, preds):
            row = {
                "instance_id":   meta["instance_id"],
                "instance_type": meta["instance_type"],
                "num_nodes":     meta["num_nodes"],
                "predicted":     pred,
            }
            if meta["actual"] is not None:
                row["actual"] = meta["actual"]
                row["residual"] = pred - meta["actual"]
                row["abs_error"] = abs(pred - meta["actual"])
            results.append(row)

        return results, skipped


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def collect_records(input_path: Path) -> List[Tuple[dict, Optional[float]]]:
    """
    Load JSON records from a file or directory, pairing each with the
    vehicle_load_capacity override implied by its filename.

    Mirrors the rule in build_graphs.load_all_graphs: a filename stem ending
    in "battery" forces vehicle_load_capacity = 50.0. Inference MUST apply the
    same rule or the features will not match how the model was trained.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.json")) + sorted(input_path.glob("*.JSON"))
        files = list(dict.fromkeys(files))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path.resolve()}")

    pairs: List[Tuple[dict, Optional[float]]] = []
    for fp in files:
        override = BATTERY_OVERRIDE_LOAD if fp.stem.lower().endswith(BATTERY_SUFFIX) else None
        if override is not None:
            print(f"  {fp.name}: forcing vehicle_load_capacity = {override}")
        with open(fp, "r") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
        pairs.extend((rec, override) for rec in records)

    print(f"Loaded {len(pairs):,} records from {len(files)} file(s)")
    return pairs


def summarize(results: List[dict]) -> Optional[dict]:
    """Compute MAE / RMSE / R^2 over results that carry ground-truth values."""
    scored = [r for r in results if "actual" in r]
    if not scored:
        return None
    err = torch.tensor([r["residual"] for r in scored])
    actual = torch.tensor([r["actual"] for r in scored])
    ss_res = float((err ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "n":    len(scored),
        "mae":  float(err.abs().mean()),
        "rmse": float(err.pow(2).mean().sqrt()),
        "r2":   1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
    }


def write_csv(results: List[dict], output_path: Path) -> None:
    """Write results to CSV; columns adapt to whether ground truth was present."""
    if not results:
        print("Nothing to write (no successfully scored records).")
        return
    has_actual = "actual" in results[0]
    fields = ["instance_id", "instance_type", "num_nodes", "predicted"]
    if has_actual:
        fields += ["actual", "residual", "abs_error"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score candidate solutions with a trained GAT surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained checkpoint (e.g. models/<run_id>/best.pt).")
    p.add_argument("--input", type=Path, required=True,
                   help="JSON file, or a directory of JSON files, holding records to score.")
    p.add_argument("--reference-dir", type=Path, default=Path("data generation"),
                   help="Directory with C_instances.json / R_instances.json / RC_instances.json.")
    p.add_argument("--prepared-config", type=Path, default=Path("prepared_data/config.json"),
                   help="prepared_data/config.json from the matching data_prep run "
                        "(supplies pe_base and cross-checks pe_dim). Ignored if missing.")
    p.add_argument("--pe-base", type=float, default=DEFAULT_PE_BASE,
                   help="Fallback PE base period, used only when --prepared-config is absent.")
    p.add_argument("--output", type=Path, default=Path("predictions.csv"),
                   help="Where to write the predictions CSV.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--node-set", choices=["pool", "served"], default="pool",
                   help="Must match the semantics the checkpoint was trained with.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint.resolve()}", file=sys.stderr)
        return 1
    if not args.reference_dir.is_dir():
        print(f"ERROR: reference directory not found: {args.reference_dir.resolve()}", file=sys.stderr)
        return 1

    prepared_config = args.prepared_config if args.prepared_config.is_file() else None
    if prepared_config is None:
        print(f"NOTE: {args.prepared_config} not found; "
              f"falling back to --pe-base {args.pe_base}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        reference_dir=args.reference_dir,
        prepared_config=prepared_config,
        pe_base=args.pe_base,
        node_set=args.node_set
    )
    print(f"  device:       {predictor.device}")
    print(f"  best epoch:   {predictor.best_epoch}")
    print(f"  in_channels:  {predictor.in_channels}  (pe_dim = {predictor.pe_dim}, "
          f"pe_base = {predictor.pe_base})")
    print(f"  node_set:     {predictor.node_set}")

    pairs = collect_records(args.input)
    records   = [rec for rec, _ in pairs]
    overrides = [ovr for _, ovr in pairs]

    results, skipped = predictor.score_records(
        records, vehicle_load_overrides=overrides, batch_size=args.batch_size
    )
    print(f"\nScored {len(results):,} records ({len(skipped)} skipped)")
    for s in skipped[:10]:
        print(f"  skipped {s['instance_id']}: {s['reason']}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")

    summary = summarize(results)
    if summary:
        print(f"\nAgainst ground truth ({summary['n']:,} records with objective_value):")
        print(f"  MAE  = {summary['mae']:.2f}")
        print(f"  RMSE = {summary['rmse']:.2f}")
        print(f"  R^2  = {summary['r2']:.4f}")
    else:
        print("\nNo objective_value in the input records — predictions only, no error metrics.")

    write_csv(results, args.output)
    print(f"\nPredictions written to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())