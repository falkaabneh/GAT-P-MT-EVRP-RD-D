# GAT-P-MT-EVRP-RD-D

A Graph Attention Network (GAT) surrogate model for the **Prize-Collecting
Multi-Trip Electric Vehicle Routing Problem with Release Times and Deadlines**.

Predicts the ALNS objective value for a given customer selection directly from
a graph representation of the problem instance, at a fraction of the cost of
running the metaheuristic.

**Test-set performance: R² = 0.956, MAE = 135** (target std ≈ 805), with a
train→test R² gap of 0.0013.

---

## Overview

The PC-MT-EVRP-RT-D is a combinatorial optimization problem with:

- **Customer subset selection** (prize-collecting): a vehicle may skip
  customers, trading off collected prize against unmet-demand penalty.
- **Multi-trip operation**: the vehicle returns to the depot to reload and
  recharge between trips.
- **Heterogeneous time windows**: each customer has a release time and a
  deadline.
- **Battery constraints**: the electric vehicle has a finite driving range
  per trip.
- **Load capacity**: the vehicle has a finite carrying capacity.

Solving it exactly or with metaheuristics (ALNS, branch-and-price) is
expensive. This repository trains a GAT-based **surrogate** that estimates the
objective value of a candidate customer selection, enabling rapid evaluation
inside search-intensive workflows.

---

## Repository structure

```
GAT-P-MT-EVRP-RD-D/
├── build_graphs.py        # Phase 0: raw JSON       -> PyG graphs (graphs.pt)
├── data_prep.py           # Phase A: split+normalize -> prepared_data/
├── model.py               # Phase B: GATSurrogate model definition
├── train.py               # Phase C: training loop + W&B logging
├── evaluate.py            # Phase D: metrics, plots, per-graph CSV
├── predict.py             # Phase E: inference CLI + Predictor API
├── sweep.yaml             # W&B Bayesian hyperparameter sweep config
├── requirements.txt
├── README.md
├── .gitignore
│
├── docs/
│   ├── design_decisions.md   # every design choice, with rationale + ablations
│   └── server_runbook.md     # startup sequence for the GPU server
│
├── training data/         # ALNS solution JSONs (input)
├── data generation/       # C_instances.json, R_instances.json, RC_instances.json
│
├── processed_graphs/      # build_graphs.py output      (gitignored)
├── prepared_data/         # data_prep.py output         (gitignored)
├── models/                # training checkpoints        (gitignored)
└── notebooks/             # exploratory work
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA build strongly recommended for training)
- PyTorch Geometric 2.4+
- Weights & Biases

```bash
pip install -r requirements.txt
```

PyTorch Geometric has CUDA-specific optional dependencies
(`torch-scatter`, `torch-sparse`, `torch-cluster`) needed at training time.
Install the wheels matching your torch + CUDA version — see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Verify GPU availability before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Quickstart

```bash
git clone https://github.com/falkaabneh/GAT-P-MT-EVRP-RD-D.git
cd GAT-P-MT-EVRP-RD-D
pip install -r requirements.txt

# Place the raw data:
#   training data/*.json      (ALNS solutions)
#   data generation/*.json    (C / R / RC reference instances)

python build_graphs.py                        # -> processed_graphs/graphs.pt
python data_prep.py --pe-dim 32 --pe-base 1000  # -> prepared_data/
python train.py                               # -> models/<run_id>/best.pt
python evaluate.py --checkpoint models/<run_id>/best.pt --splits train val test
```

Every script accepts `--help` for the full list of options.

---

## Pipeline

### Phase 0 — Build graphs (`build_graphs.py`)

Merges ALNS solution records with reference instance geometry into a list of
`torch_geometric.data.Data` objects, one per solution.

- **Nodes:** depot (index 0) + only the customers actually served. Excluded
  customers are absent from the graph entirely, so the model never sees which
  customers were rejected.
- **Node features (6):** `x, y, demand, release_time, deadline, tw_width`
- **Edges:** complete graph, no self-loops
- **Edge features (2):** Euclidean distance, distance / battery_capacity
- **Target:** ALNS `objective_value`
- **Graph-level scalars:** `battery_capacity`, `vehicle_load_capacity`

```bash
python build_graphs.py -t "training data" -r "data generation" -o processed_graphs
```

> **Data convention:** JSON files whose name ends in `battery` have
> `vehicle_load_capacity` forced to 50.0, overriding the value in the record.
> This compensates for a data-generation gap. Any code consuming the raw JSONs
> must replicate this rule — `predict.py` does.

### Phase A — Split and normalize (`data_prep.py`)

Splits stratified by `instance_type`, then normalizes using **training-split
statistics only**.

| Feature | Strategy |
|---|---|
| Target `y` | z-score |
| `(x, y)` coordinates | sinusoidal positional encoding (`2 × pe_dim` dims) |
| `demand` | divided by the graph's own `vehicle_load_capacity` |
| `release_time`, `deadline`, `tw_width` | min-max to [0, 1] |
| both edge features | z-score |

```bash
python data_prep.py --split-ratios 0.8 0.1 0.1 --pe-dim 32 --pe-base 1000
```

Node feature dimension becomes `2 × pe_dim + 4` (68 with `pe_dim = 32`).
Outputs `train.pt`, `val.pt`, `test.pt`, `stats.json`, `config.json`.

### Phase B — Model (`model.py`)

```
node features ──> MLP encoder ──┐
                                ├──> 4 × [GATv2Conv → GraphNorm → +residual → ReLU → Dropout]
edge features ──> MLP encoder ──┘                     │
                                                      ▼
                                             global mean pooling
                                                      │
                    [battery_capacity, vehicle_load_capacity] ──> concat
                                                      │
                                                      ▼
                                        MLP head (H+2 → H/2 → 1)
```

Defaults: `hidden_dim = 128`, 4 GATv2 layers, 4 attention heads (concatenated),
`edge_hidden_dim = 32`, dropout 0.1 (body) / 0.2 (head). ~186K parameters.

All architecture choices are constructor arguments and CLI flags — see
`docs/design_decisions.md`.

### Phase C — Training (`train.py`)

MSE loss, Adam (`lr = 1e-3`, `weight_decay = 1e-4`), gradient clipping at 1.0,
`CosineAnnealingWarmRestarts` (`T_0 = 15`), early stopping on validation loss
(patience 31), max 110 epochs. Logs to Weights & Biases; saves the best
checkpoint to `models/<wandb_run_id>/best.pt`.

```bash
wandb login                       # one-time
python train.py                   # defaults
python train.py --hidden_dim 256 --num_gat_layers 3 --lr 5e-4
```

**Bayesian hyperparameter sweep:**

```bash
wandb sweep sweep.yaml            # prints a SWEEP_ID
wandb agent --count 50 <ENTITY>/gat-evrp/<SWEEP_ID>
```

### Phase D — Evaluation (`evaluate.py`)

```bash
python evaluate.py --checkpoint models/<run_id>/best.pt --splits train val test
```

Writes to `models/<run_id>/evaluation/`:

- `metrics.json` — MAE, RMSE, R², residual statistics, per-type breakdown
- `pred_vs_actual_<split>.png`, `residuals_<split>.png`, `error_vs_size_<split>.png`
- `predictions_<split>.csv` — one row per graph

### Phase E — Inference (`predict.py`)

Score new candidate solutions, either in batch from JSON or programmatically
from inside a search loop.

**CLI:**

```bash
python predict.py --checkpoint models/<run_id>/best.pt \
                  --input "training data/some_file.json" \
                  --output predictions.csv
```

When the input records carry `objective_value`, the run also reports MAE /
RMSE / R² against ground truth.

**Python API** — construct the `Predictor` once and reuse it; batch your
candidates for speed:

```python
from predict import Predictor

predictor = Predictor(
    checkpoint_path="models/<run_id>/best.pt",
    reference_dir="data generation",
    prepared_config="prepared_data/config.json",
)

objectives = predictor.score_subsets([
    ("C_012", [1, 3, 7],     120.0, 50.0),
    ("C_012", [1, 3, 7, 9],  120.0, 50.0),
])
```

The surrogate predicts the **ALNS-optimized objective for a customer subset**
— it does not evaluate a specific route. This makes it suited to guiding the
selection phase of a search rather than replacing routing.

---

## Results

Model: `hidden_dim = 128`, 4 layers, 4 heads (concat), `pe_dim = 32`,
`pe_base = 1000`. Best epoch 10; early stopped at epoch 41.

| Split | n | R² | MAE | RMSE |
|---|---|---|---|---|
| Train | 16,023 | 0.9571 | 132.37 | 166.51 |
| Val | 2,003 | 0.9564 | 132.71 | 167.68 |
| **Test** | **2,003** | **0.9558** | **134.93** | **170.75** |

Test MAE by instance type:

| Type | n | MAE | RMSE |
|---|---|---|---|
| R (random) | 583 | 123.87 | 157.90 |
| RC (mixed) | 820 | 135.90 | 171.28 |
| C (clustered) | 600 | 144.34 | 181.68 |

**Observations**

- Negligible overfitting: train→test R² drops by 0.0013.
- Clustered (C) instances are consistently the hardest, ~16% higher MAE than
  random (R), across all three splits.
- Error *decreases* with graph size (mean abs error 163.6 for 62–67 node
  graphs vs 124.8 for 81–86). The dataset is skewed toward large graphs, and
  mean pooling is lower-variance with more nodes.
- A Bayesian sweep over architecture and optimizer hyperparameters did not
  beat this configuration; the best sweep run reached R² = 0.9501 with 2.7×
  the parameters.

---

## Notes on the data

- Each JSON in `training data/` holds roughly 100–1000 ALNS solution records,
  linked to a reference graph by `instance_id`.
- Reference graphs in `data generation/` hold the geometry and per-customer
  parameters (`x`, `y`, `demand`, `release_time`, `deadline`, `prize`,
  `penalty`) for each `instance_id`.
- The same `instance_id` appears many times with different
  `vehicle_load_capacity` and `battery_capacity` — these are intentionally
  distinct training examples, since the optimal customer selection depends on
  the load and battery budget.
- Total dataset: 20,029 graphs (C 30.0%, R 29.1%, RC 40.9%).

---

## Roadmap

- [x] Phase 0 — raw JSON to PyG graphs (`build_graphs.py`)
- [x] Phase A — split + feature normalization (`data_prep.py`)
- [x] Phase B — GAT model definition (`model.py`)
- [x] Phase C — training loop, W&B logging, Bayesian sweep (`train.py`)
- [x] Phase D — evaluation, plots, per-type analysis (`evaluate.py`)
- [x] Phase E — inference CLI and API (`predict.py`)
- [ ] Timing benchmark: surrogate vs ALNS evaluation cost
- [ ] Integration into the ALNS selection phase
- [ ] Ablation study (see `docs/design_decisions.md`)

---

## Documentation

- [`docs/design_decisions.md`](docs/design_decisions.md) — every design choice
  with its rationale, the full change log, and a list of open ablations
- [`docs/server_runbook.md`](docs/server_runbook.md) — GPU server startup
  sequence and troubleshooting

## Author

Faisal Alkaabneh — Assistant Professor, American University of Sharjah.
Work conducted at KAIST.

## Citation

To be added once the associated paper is available.
