#!/usr/bin/env python3
"""
Build PyTorch Geometric graphs from training data + reference instance files.

Each training JSON contains ALNS solution outcomes for a given graph instance.
For each instance we merge node-level features (x, y, demand, release_time,
deadline, time-window width) from the matching C/R/RC reference file, restrict
the graph to the depot + the customers actually served in the solution, and
attach the ALNS objective value as the graph-level regression target.

The output is a list of `torch_geometric.data.Data` objects, ready for GAT
training.

Directory layout assumed (overridable via CLI):

    <repo root>/
      build_graphs.py
      training data/         # ALNS solution JSONs
      data generation/       # C_instances.json, R_instances.json, RC_instances.json
      processed_graphs/      # created by this script; contains graphs.pt

Usage (from the repo root):

    python build_graphs.py                                  # all defaults
    python build_graphs.py -t "training data" \
                           -r "data generation" \
                           -o processed_graphs \
                           --output-name graphs.pt
    python build_graphs.py --help

TRAINING_DIR is the directory where JSON files exist. Each JSON file holds the
ALNS solution of an instance along with global variables such as vehicle load
capacity and battery range. Each instance also includes the IDs of served
nodes, the objective value, and the reference instance id used to look up the
underlying graph features.

REFERENCE_DIR is the directory where information about the graph instances
themselves lives. Each JSON file (C_instances.json, R_instances.json,
RC_instances.json) holds graph instances keyed by instance_id, where each node
has features such as prize, penalty, (x, y) coordinate, release time, deadline,
etc.

Note: vehicle load capacity and battery capacity are pulled from the training
JSONs (TRAINING_DIR), not from the reference JSONs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_TRAINING_DIR = Path("training data")
DEFAULT_REFERENCE_DIR = Path("data generation")
DEFAULT_OUTPUT_DIR = Path("processed_graphs")
DEFAULT_OUTPUT_NAME = "graphs.pt"

NUM_CUSTOMERS = 100   # customers are nodes 1..100, depot is node 0


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------
def load_reference_instances(reference_dir: Path) -> Dict[str, dict]:
    """Load C/R/RC reference files into a single dict keyed by instance_id."""
    reference: Dict[str, dict] = {}
    for type_name in ("C", "R", "RC"):
        filepath = reference_dir / f"{type_name}_instances.json"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Reference file not found: {filepath.resolve()}\n"
                f"  cwd = {Path.cwd()}"
            )
        with open(filepath, "r") as f:
            instances = json.load(f)
        for inst in instances:
            reference[inst["instance_id"]] = inst
    print(f"Loaded {len(reference)} reference instances from {reference_dir}")
    return reference


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(
    instance: dict,
    reference_data: Dict[str, dict],
    vehicle_load_override: Optional[float] = None,
) -> Data:
    """
    Build a PyG Data object for graph-level regression on objective_value.

    Nodes:     depot (index 0) + served customers (indices 1..K)
    x:         [K+1, 6]  -> [x_coord, y_coord, demand, release_time, deadline, tw_width]
    edges:     complete graph, no self-loops
    edge_attr: [E, 2]    -> [distance, distance / battery_capacity]
    y:         [1]       -> objective_value
    """
    instance_id = instance["instance_id"]
    if instance_id not in reference_data:
        raise KeyError(f"instance_id {instance_id!r} not found in reference data")
    ref = reference_data[instance_id]

    # ---- served customer ids (depot is handled separately) ----------------
    excluded = set(instance["customers_not_selected_first_place"]) | set(
        instance["customers_not_visited_from_selected_pool"]
    )
    nodes_served = [i for i in range(1, NUM_CUSTOMERS + 1) if i not in excluded]
    if not nodes_served:
        raise ValueError(f"{instance_id}: no served customers")

    # Lookup: node_id -> node dict from the reference file
    ref_nodes_by_id = {n["node_id"]: n for n in ref["nodes"]}

    # Graph order: depot first, then served customers in ascending id
    included_ids = [0] + nodes_served

    # ---- node features (K+1 x 6) ------------------------------------------
    features = []
    for nid in included_ids:
        node = ref_nodes_by_id[nid]
        rt = float(node["release_time"])
        dl = float(node["deadline"])
        features.append([
            float(node["x"]),
            float(node["y"]),
            float(node["demand"]),
            rt,
            dl,
            dl - rt,                       # time-window width
        ])
    x = torch.tensor(features, dtype=torch.float)        # [K+1, 6]

    # ---- battery capacity (needed for edge features) ----------------------
    bat_raw = instance.get("battery_capacity")
    if bat_raw is None:
        raise ValueError(f"{instance_id}: battery_capacity missing (needed for edge features)")
    bat = float(bat_raw)

    # ---- complete graph (no self-loops) -----------------------------------
    num_nodes = x.size(0)
    row, col = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)  # [2, E]

    # ---- edge features: [distance, distance / battery_capacity] -----------
    coords = x[:, :2]
    diff = coords[edge_index[0]] - coords[edge_index[1]]
    dist = diff.norm(dim=1, keepdim=True)                # [E, 1]
    range_ratio = dist / bat                             # [E, 1]
    edge_attr = torch.cat([dist, range_ratio], dim=1)    # [E, 2]

    # ---- regression target ------------------------------------------------
    obj = float(instance.get("objective_value", float("nan")))
    y = torch.tensor([obj], dtype=torch.float)           # [1]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # ---- graph-level scalars as tensors (so they batch correctly) ---------
    # The following block handles cases where JSON files do not include the
    # vehicle_load_capacity field (defaults to 50.0) or where the filename
    # suffix `_battery` forces an override of that value to 50.0.
    if vehicle_load_override is not None:
        vlc = vehicle_load_override
    else:
        vlc = float(instance.get("vehicle_load_capacity", 50.0))
    data.battery_capacity      = torch.tensor([bat], dtype=torch.float)
    data.vehicle_load_capacity = torch.tensor([vlc], dtype=torch.float)

    # ---- string / list metadata (not used by the model) -------------------
    data.instance_id       = instance_id
    data.instance_type     = instance["instance_type"]
    data.nodes_served      = nodes_served       # original customer ids
    data.original_node_ids = included_ids       # depot=0 then served, index-aligned with x

    return data


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def load_all_graphs(
    training_dir: Path,
    reference_dir: Path,
) -> List[Data]:
    reference_data = load_reference_instances(reference_dir)

    json_files = sorted(training_dir.glob("*.json")) + sorted(training_dir.glob("*.JSON"))
    json_files = list(dict.fromkeys(json_files))   # dedupe in case both cases match
    print(f"Found {len(json_files)} training JSON files in {training_dir}")

    graphs: List[Data] = []
    skipped = 0
    for filepath in json_files:
        # Filename (without extension) ending in "battery" -> force vehicle_load_capacity = 50.0
        override = 50.0 if filepath.stem.lower().endswith("battery") else None
        if override is not None:
            print(f"  {filepath.name}: forcing vehicle_load_capacity = 50.0")

        with open(filepath, "r") as f:
            instances = json.load(f)
        for instance in instances:
            try:
                graphs.append(build_graph(instance, reference_data, vehicle_load_override=override))
            except Exception as e:
                skipped += 1
                print(f"  skipped {instance.get('instance_id', '?')}: {e}")

    print(f"Built {len(graphs)} graphs ({skipped} skipped)")
    return graphs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build PyTorch Geometric graphs from ALNS solution JSONs and "
            "reference instance files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t", "--training-dir", type=Path, default=DEFAULT_TRAINING_DIR,
        help="Directory containing the training (ALNS solution) JSON files.",
    )
    parser.add_argument(
        "-r", "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
        help="Directory containing C_instances.json, R_instances.json, RC_instances.json.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where the processed dataset will be written (created if missing).",
    )
    parser.add_argument(
        "--output-name", default=DEFAULT_OUTPUT_NAME,
        help="Filename for the saved dataset inside --output-dir.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Validate input directories early with clear errors
    for label, path in (("training", args.training_dir), ("reference", args.reference_dir)):
        if not path.is_dir():
            print(f"ERROR: {label} directory not found: {path.resolve()}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    graphs = load_all_graphs(args.training_dir, args.reference_dir)

    # Quick sanity check on the first graph
    if graphs:
        g = graphs[0]
        print("\nFirst graph summary:")
        print(f"  instance_id      = {g.instance_id}")
        print(f"  instance_type    = {g.instance_type}")
        print(f"  x.shape          = {tuple(g.x.shape)}")
        print(f"  edge_index.shape = {tuple(g.edge_index.shape)}")
        print(f"  edge_attr.shape  = {tuple(g.edge_attr.shape)}")
        print(f"  objective_value  = {g.y.item()}")
        print(f"  battery_capacity = {g.battery_capacity.item()}")
        print(f"  vehicle_load_cap = {g.vehicle_load_capacity.item()}")

    torch.save(graphs, output_path)
    print(f"\nSaved {len(graphs)} graphs to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
