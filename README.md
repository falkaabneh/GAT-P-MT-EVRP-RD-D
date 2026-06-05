# GAT-P-MT-EVRP-RD-D

A Graph Attention Network (GAT) surrogate model for the **Prize-Collecting
Multi-Trip Electric Vehicle Routing Problem with Release Times and Deadlines**.

> Status: data pipeline complete. GAT architecture, training loop, and
> evaluation are in progress.

---

## Overview

The PC-MT-EVRP-RT-D is a complex combinatorial optimization problem with:

- **Customer subset selection** (prize-collecting): a vehicle may skip
  customers, trading off collected prize against unmet-demand penalty.
- **Multi-trip operation**: the vehicle returns to the depot to reload and
  recharge between trips.
- **Heterogeneous time windows**: each customer has a release time and a
  deadline.
- **Battery constraints**: the electric vehicle has a finite driving range
  per trip.
- **Load capacity**: the vehicle has a finite carrying capacity.

This repository builds a GAT-based **surrogate model** that
predicts the objective value of an ALNS solution directly from the graph
representation of the problem instance, enabling rapid evaluation in
search-intensive workflows.

---

## Repository structure

```
GAT-P-MT-EVRP-RD-D/
├── build_graphs.py        # Phase 0: raw JSON  ->  PyG Data objects (graphs.pt)
├── data_prep.py           # Phase A: split + normalize -> prepared_data/
├── requirements.txt
├── README.md
├── .gitignore
│
├── training data/         # ALNS solution JSONs (input)
├── data generation/       # C_instances.json, R_instances.json, RC_instances.json
│
├── processed_graphs/      # build_graphs.py output (gitignored)
│   └── graphs.pt
├── prepared_data/         # data_prep.py output (gitignored)
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   ├── stats.json
│   └── config.json
│
├── notebooks/             # exploratory Jupyter notebooks
│   └── gat_model.ipynb
└── models/                # (future) trained checkpoints (gitignored)
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.4+

Install with:

```bash
pip install -r requirements.txt
```

PyTorch Geometric has optional CUDA-specific dependencies (`torch-scatter`,
`torch-sparse`, `torch-cluster`) that are not needed by the current data
pipeline but become relevant for training. See the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for the wheel matching your CUDA / torch version.

---

## Pipeline

The project is organized as a sequence of scripts, each producing artifacts
consumed by the next.

### 1. Build PyG graphs from raw JSON — `build_graphs.py`

Reads ALNS solution JSONs from `training data/` and reference instance
descriptions from `data generation/`, then builds a list of
`torch_geometric.data.Data` objects, one per ALNS solution.

For each instance:

- **Nodes:** depot (index 0) + the served customers (only nodes actually
  visited in the solution are included; this prevents the GAT from receiving
  the "selection" signal as input).
- **Node features (6 dims):**
  `[x_coord, y_coord, demand, release_time, deadline, time_window_width]`.
- **Edges:** complete graph (every node connected to every other), no
  self-loops.
- **Edge features (2 dims):**
  `[Euclidean distance, distance / battery_capacity]`.
- **Target `y`:** ALNS objective value (used as the regression label).
- **Graph-level scalars:** `battery_capacity`, `vehicle_load_capacity`.

Usage:

```bash
python build_graphs.py                                  # all defaults
python build_graphs.py -t "training data" \
                       -r "data generation" \
                       -o processed_graphs \
                       --output-name graphs.pt
python build_graphs.py --help
```

Output: `processed_graphs/graphs.pt`.

### 2. Split + normalize — `data_prep.py`

Reads `processed_graphs/graphs.pt`, splits the data, computes normalization
statistics on the training split only, and writes the prepared dataset to
disk.

**Split:** stratified by `instance_type` (C / R / RC) by default; ratios are
user-configurable.

**Normalizations:**

| Feature | Strategy |
|---|---|
| Target `y` | z-score (training stats) |
| `(x, y)` coordinates | sinusoidal positional encoding (`2 * pe_dim` dims) |
| `demand` | divided by the graph's own `vehicle_load_capacity` (per-graph) |
| `release_time` | min-max `[0, 1]` (training stats) |
| `deadline` | min-max `[0, 1]` (training stats) |
| `time_window_width` | min-max `[0, 1]` (training stats) |
| `distance` (edge) | z-score (training stats) |
| `distance / battery_capacity` (edge) | z-score (training stats) |

The resulting node feature dimension is `2 * pe_dim + 4` (default: 36).

Usage:

```bash
python data_prep.py                                          # all defaults
python data_prep.py --split-ratios 0.75 0.15 0.10            # custom split
python data_prep.py --pe-dim 32 --pe-base 1000               # larger PE
python data_prep.py --split-ratios 0.8 0.0 0.2 --no-stratify # no val, no stratify
python data_prep.py --help
```

Output: `prepared_data/{train,val,test}.pt`, `stats.json`, `config.json`.

The same functions are importable from a notebook:

```python
from data_prep import prepare_data, load_prepared_data, NormStats

# load from a previous CLI run:
train, val, test, stats = load_prepared_data("prepared_data")

# or do it in memory:
import torch
graphs = torch.load("processed_graphs/graphs.pt", weights_only=False)
train, val, test, stats = prepare_data(graphs, split_ratios=(0.8, 0.1, 0.1))
```

### 3. Train the GAT — *(in progress)*

A graph-level regression model on top of the prepared dataset. Architecture
and training script TBD.

---

## End-to-end quickstart

```bash
# clone and install
git clone https://github.com/falkaabneh/GAT-P-MT-EVRP-RD-D.git
cd GAT-P-MT-EVRP-RD-D
pip install -r requirements.txt

# place the raw data
#   - training data/*.json       (ALNS solutions)
#   - data generation/*.json     (C / R / RC reference instances)

# build the dataset
python build_graphs.py
python data_prep.py

# (next phase) train the GAT
# python train_gat.py
```

---

## Notes on the data

- Each ALNS solution JSON in `training data/` contains around 100 solution
  records, each tied to a reference graph via `instance_id`.
- Reference graphs (`C_instances.json`, `R_instances.json`,
  `RC_instances.json` in `data generation/`) hold the physical layout and
  per-customer parameters (`x`, `y`, `demand`, `release_time`, `deadline`,
  `prize`, `penalty`) for each `instance_id`.
- The same `instance_id` may appear multiple times across the training JSONs
  with different `vehicle_load_capacity` and `battery_capacity` values —
  these are intentionally different supervised examples (the optimal customer
  selection depends on the load and battery budget).
- Filenames in `training data/` ending in `battery` force
  `vehicle_load_capacity = 50.0` regardless of what is in the JSON record,
  to compensate for a data-generation gap.

## Roadmap

- [x] Phase 0 — raw JSON to PyG graphs (`build_graphs.py`)
- [x] Phase A — split + feature normalization (`data_prep.py`)
- [ ] Phase B — GAT model definition
- [ ] Phase C — training loop with logging, early stopping, checkpointing
- [ ] Phase D — evaluation (MAE / RMSE / R²; per-`instance_type` breakdown)
- [ ] Phase E — productionize as `train_gat.py` CLI
