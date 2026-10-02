# Design decisions — GAT surrogate for PC-MT-EVRP-RT-D

A complete audit of every design choice made across the pipeline, organized
by phase. The **Source** column indicates how the choice was made:

- **USER** — explicit user choice
- **AGREED** — proposed by Claude, user agreed
- **DEFAULT** — picked as a sensible default (worth revisiting)
- **AUTO** — derived automatically from data (not a design choice per se)

The **Configurable** column shows whether the value is locked into code
(hardcoded), a constructor argument on the class, or a CLI flag.

Use the rightmost columns to add your own notes / hypotheses / ablations
as the project evolves.

---

## Headline result

Best model: `models/dpn7elbs/best.pt` (defaults from `train.py`, best epoch 10,
early stopped at epoch 41).

| Split | R² | MAE | RMSE |
|---|---|---|---|
| Train | 0.9571 | 132.37 | 166.51 |
| Val   | 0.9564 | 132.71 | 167.68 |
| Test  | **0.9558** | **134.93** | **170.75** |

Train→test R² gap: 0.0013. No measurable overfitting.

Per-instance-type test MAE: R = 123.87 < RC = 135.90 < C = 144.34.

Context: target `y_std` ≈ 805, `|y_mean|` ≈ 2277, so test MAE corresponds to
~17% of one standard deviation, or ~6% relative error.

---

## Phase 0 — Graph construction (`build_graphs.py`)

Converts raw ALNS solution JSON + reference instance JSON into PyG `Data`
objects.

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| 0.1 | Graph node set | depot + served customers only | USER | Hardcoded | Avoids leaking the selection decision into the GAT input |
| 0.2 | Node ordering | depot at index 0, customers ascending by id | DEFAULT | Hardcoded | Predictable indexing; depot is always row 0 in x |
| 0.3 | Edge structure | Complete graph, no self-loops | AGREED | Hardcoded | Standard for routing GNNs; edge_attr carries geometry |
| 0.4 | Raw node features (6) | x, y, demand, release_time, deadline, tw_width | USER | Hardcoded | Direct from problem definition |
| 0.5 | Raw edge features (2) | Euclidean distance; distance / battery_capacity | USER | Hardcoded | Geometric + physics-aware (driving-range ratio) |
| 0.6 | Regression target | objective_value | USER | Hardcoded | The ALNS objective is the supervised label |
| 0.7 | Graph-level metadata | battery_capacity, vehicle_load_capacity (per graph) | USER | Hardcoded | Used for demand normalization and global conditioning |
| 0.8 | Missing `vehicle_load_capacity` | Default to 50.0 | USER | Code constant | Data generation gap |
| 0.9 | `*_battery` filename rule | Force `vehicle_load_capacity = 50.0` | USER | Code constant | Data generation gap workaround. **See technical debt item 16** — this is implicit state that every downstream consumer must replicate |
| 0.10 | Missing `battery_capacity` | Raise + skip instance | DEFAULT | Hardcoded | Required for edge feature (range ratio) |
| 0.11 | Empty `nodes_served` | Raise + skip instance | DEFAULT | Hardcoded | Degenerate graph (depot only) |

---

## Phase A — Split and normalize (`data_prep.py`)

Splits the dataset and computes/applies normalization.

### Split

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| A.1 | Split ratios | 80 / 10 / 10 (train / val / test) | USER | `--split-ratios` | Standard ML practice with val for early-stopping decisions |
| A.2 | Stratification | By `instance_type` (C / R / RC) | AGREED | `--no-stratify` to disable | Guarantees C/R/RC are present in each split in correct proportions |
| A.3 | Shuffle within splits | Yes | DEFAULT | Hardcoded | Prevents type-ordered batches |
| A.4 | Phase A seed | 42 | AGREED | `--seed` | Reproducible split (not the same as training seed) |

Resulting split sizes: train 16,023 / val 2,003 / test 2,003 (from 20,029 graphs).
Type proportions in every split: C 30.0%, R 29.1%, RC 40.9%.

### Normalization

All statistics computed on **training split only** to avoid val/test leakage.

| # | Feature | Strategy | Source | Configurable | Rationale |
|---|---------|----------|--------|--------------|-----------|
| A.5 | x, y coordinates | Sinusoidal PE, **`pe_dim = 32` per coord, `pe_base = 1000`** | USER (PE) / AGREED (dim, base) | `--pe-dim`, `--pe-base` | Multi-scale spatial encoding. **The values actually used in the reported run are 32 / 1000**, recorded in `prepared_data/config.json`. Earlier drafts of this doc said 16 / 100 — those were the CLI defaults, not what was run |
| A.6 | demand | Divided by per-graph `vehicle_load_capacity` | USER | Hardcoded | Physics-informed; scale-invariant across instances |
| A.7 | release_time | min-max [0,1] with global (training) min/max | USER | Hardcoded | Time-normalized to a uniform scale |
| A.8 | deadline | min-max [0,1] with global (training) min/max | USER | Hardcoded | Time-normalized to a uniform scale |
| A.9 | tw_width | min-max [0,1] with global (training) min/max | USER | Hardcoded | Redundant w.r.t. rt+dl; explicit helps convergence |
| A.10 | edge `distance` | z-score (training mean/std) | USER | Hardcoded | Standard; distribution is right-skewed but acceptable |
| A.11 | edge `range_ratio` | z-score (training mean/std) | USER | Hardcoded | Already partly normalized by the battery div |
| A.12 | Target `y` | z-score (training mean/std) | AGREED | Hardcoded | Stable MSE optimization; un-normalized at evaluation |

**Resulting feature dimensions (as run):**
- Node feature dim: `2 * 32 + 4 = 68`
- Edge feature dim: 2

### Target distribution (informed the loss choice)

| Statistic | Value |
|---|---|
| n | 20,029 |
| min / max | −4677.15 / +495.42 |
| mean / median | −2276.89 / −2349.76 |
| std | 804.96 |
| skewness | +0.535 |
| excess kurtosis | +0.413 |
| IQR outliers (Tukey) | 666 (3.33%) |

Mild right skew, near-Gaussian tails → MSE is appropriate; Huber was
considered and set aside (see ablation 7).

---

## Phase B — Model architecture (`model.py`)

GAT-based graph-level regressor. Parameter count with the reported
configuration: **185,985**.

### Encoders

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| B.1 | Node encoder | 2-layer MLP: `in_channels → hidden_dim → hidden_dim`, ReLU between | AGREED | `--hidden_dim` | Lift raw features to hidden dim before message passing |
| B.2 | Edge encoder | 2-layer MLP: `2 → edge_hidden_dim → edge_hidden_dim`, ReLU between | USER (MLP) / AGREED (dim) | `--edge_hidden_dim` | Edge embedding fed into `GATv2Conv.edge_dim` |
| B.3 | Hidden dimension (node) | 128 | AGREED | `--hidden_dim` | Common middle ground; divisible by heads |
| B.4 | Edge hidden dimension | 32 | AGREED | `--edge_hidden_dim` | Smaller than node hidden; edges carry less info |

### GAT stack

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| B.5 | GAT variant | GATv2Conv | AGREED | Hardcoded | Fixes static-attention issue; supports `edge_dim` natively |
| B.6 | Number of GAT layers | 4 | USER | `--num_gat_layers` | Receptive field; over-smoothing risk past 4-6 |
| B.7 | Attention heads | 4 | USER | `--heads` | Standard for attention diversity |
| B.8 | Head aggregation | Concatenation | USER | `--head_aggr concat / average` | More parameters; preserves per-head info |
| B.9 | Per-layer out_channels | `hidden_dim // heads` (concat) or `hidden_dim` (average) | AGREED | Auto-derived | Keeps post-layer width = `hidden_dim` for clean residuals |
| B.10 | Self-loops in GAT | Disabled (`add_self_loops=False`) | DEFAULT | Hardcoded | Already a complete graph; self-loops redundant |
| B.11 | Normalization between layers | GraphNorm | USER | Hardcoded | Graph-aware; more stable than BatchNorm for variable-size graphs |
| B.12 | Activation | ReLU | USER | Hardcoded | Standard |
| B.13 | Dropout (GAT body) | 0.1 (on attention weights via GATv2Conv + post-activation) | AGREED | `--dropout` | Light regularization |
| B.14 | Residual connections | Yes | USER | Hardcoded toggle | Helps gradient flow in deeper stacks |
| B.15 | Residual pattern | `h = ReLU(h_in + norm(GAT(h_in)))` | CLAUDE (bug fix) | Hardcoded | **Original was `h = h_in + ReLU(norm(GAT(h_in)))`, which caused mode collapse** — see change log |

### Pooling and head

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| B.16 | Pooling | **Mean** (originally sum) | USER chose sum → CLAUDE switched to mean | Hardcoded | Sum pooling made the head's input magnitude scale with K (graphs range ~62–86 nodes), contributing to mode collapse |
| B.17 | Global conditioning | Concat `[battery_capacity, vehicle_load_capacity]` onto pooled embedding | USER | Hardcoded toggle | Per-node features can't see these; required since both were *used to normalize* other features |
| B.18 | Regression head depth | 2 layers | AGREED | `--head_depth` | One hidden layer + final projection |
| B.19 | Regression head width | `H+2 → H/2 → 1` (taper) | AGREED | Auto from `hidden_dim`, `head_depth` | Modest capacity, low overfitting risk |
| B.20 | Head activation | ReLU between Linear layers | AGREED | Hardcoded | Matches body |
| B.21 | Head dropout | 0.2 | AGREED | `--head_dropout` | Slightly higher than body; head has fewer structural priors |
| B.22 | Dimension assertion | `concat_heads=True` requires `hidden_dim % heads == 0` | DEFAULT | Hardcoded check | Lets concat output match `hidden_dim` exactly |

---

## Phase C — Training (`train.py`)

### Loss and optimizer

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| C.1 | Loss function | MSE | USER | Hardcoded | Target distribution is mild (skew 0.54, ex-kurt 0.41); MSE gives smooth gradients |
| C.2 | Optimizer | Adam | USER | Hardcoded | Standard for graph models |
| C.3 | Learning rate | 1e-3 | USER | `--lr` | Adam default |
| C.4 | Weight decay | 1e-4 | USER | `--weight_decay` | Mild L2 regularization |
| C.5 | Batch size | 64 | USER | `--batch_size` | Standard |
| C.6 | Gradient clipping | `max_norm = 1.0` | USER | `--grad_clip` | Stability insurance |

### Schedule and stopping

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| C.7 | LR scheduler | CosineAnnealingWarmRestarts | USER | Hardcoded class | Smooth decay + warm restarts to escape local minima |
| C.8 | `T_0` | 15 epochs | USER | `--t0` | First restart at epoch 15 |
| C.9 | `T_mult` | 1 | DEFAULT | `--t_mult` | Constant-length cycles |
| C.10 | `eta_min` | 0.0 | DEFAULT | `--eta_min` | LR goes to zero at cycle bottom |
| C.11 | Max epochs | 110 | USER | `--epochs` | Reported run early-stopped at 41 |
| C.12 | Early stopping metric | val_loss (normalized MSE) | DEFAULT | Hardcoded | Same space as loss for consistency |
| C.13 | Early stopping patience | 31 epochs | USER | `--patience` | Survives ~2 full warm restart cycles |
| C.14 | Reproducibility seed | None in production; 42 used temporarily during debugging | USER | Manual snippet | High run-to-run variance was masking the architectural bug; seed removed once fixed |

**Observation from the reported run:** best epoch was 10, and neither the
epoch-15 nor epoch-30 warm restart beat it. Training past ~10 epochs did not
help this configuration.

### Device and logging

| # | Choice | Value | Source | Configurable | Rationale |
|---|--------|-------|--------|--------------|-----------|
| C.15 | Device | Auto: CUDA if available, else CPU | DEFAULT | Hardcoded | Standard |
| C.16 | Logging backend | Weights & Biases (`online`) | USER | `--wandb_mode` | Bayesian sweep support, dashboard, run comparison |
| C.17 | Checkpoint criterion | Lowest val_loss | DEFAULT | Hardcoded | Best generalization estimate |
| C.18 | Checkpoint location | `models/<wandb_run_id>/best.pt` | DEFAULT | `--models_dir` | One folder per run; supports parallel sweep agents |
| C.19 | Checkpoint content | state_dict, config dict, epoch, val_loss, NormStats | DEFAULT | Hardcoded | Everything needed to rebuild and evaluate the model. **Note:** does NOT store `pe_dim` / `pe_base` — `predict.py` derives `pe_dim` from the input layer and reads `pe_base` from `prepared_data/config.json` |

---

## Phase D — Evaluation (`evaluate.py`)

| # | Choice | Value | Source | Rationale |
|---|--------|-------|--------|-----------|
| D.1 | Loss reported | Normalized MSE | DEFAULT | Same space as training |
| D.2 | Human-readable metrics | MAE, RMSE, R², residual mean/std/percentiles in original units | DEFAULT | Un-normalized via `stats.y_mean`, `stats.y_std` |
| D.3 | Test breakdown | Per `instance_type` (C / R / RC) | AGREED | Identify type-specific weaknesses |
| D.4 | Evaluation source | Best checkpoint by val_loss | DEFAULT | Standard practice |
| D.5 | Plots | pred-vs-actual scatter; 3-panel residual analysis; error vs graph size | DEFAULT | Standard regression diagnostics |
| D.6 | Per-graph CSV | instance_type, num_nodes, actual, predicted, residual, abs_error | DEFAULT | Enables offline analysis |

### Findings from evaluation

**Error decreases with graph size** (`corr(abs_error, num_nodes) = −0.104`):

| Size bucket (nodes) | n (test) | Mean abs error |
|---|---|---|
| 62–67 | 36 | 163.6 |
| 67–72 | 159 | 159.4 |
| 72–76 | 245 | 149.6 |
| 76–81 | 615 | 136.7 |
| 81–86 | 948 | 124.8 |

Two compounding explanations: (a) the dataset is heavily skewed toward large
graphs, so small graphs are under-represented in training; (b) mean pooling
over more nodes gives a lower-variance graph embedding.

**C instances are over-represented in the error tail.** 8 of the 20 worst test
predictions are C, against C being 30% of the test set. R accounts for only 3.

**Residuals are slightly negative** (test mean −12.90, ~1.6% of `y_std`), i.e.
the model very mildly under-predicts. The bias grows monotonically
train (−2.3) → val (−6.2) → test (−12.9).

---

## Phase E — Inference (`predict.py`)

| # | Choice | Value | Source | Rationale |
|---|--------|-------|--------|-----------|
| E.1 | Input format | Raw training-style JSON (file or directory) + reference dir | USER | Matches how new data arrives |
| E.2 | Usage modes | CLI batch scoring **and** importable `Predictor` class | USER | Batch CSV scoring plus in-loop ALNS use |
| E.3 | Feature construction | Delegated to `build_graphs.build_graph` + `data_prep.normalize_graph` | DEFAULT | Guarantees inference features match training features; no duplicated logic |
| E.4 | Model warm-loading | `Predictor` loads checkpoint + reference data once at construction | DEFAULT | In-loop use would otherwise be dominated by reload cost |
| E.5 | Batched scoring | `score_subsets()` alongside single-candidate `score_subset()` | DEFAULT | Batching is substantially faster on GPU |
| E.6 | `pe_dim` recovery | Derived from `node_encoder.0.weight` shape: `pe_dim = (in_channels − 4) / 2` | DEFAULT | Checkpoint doesn't store it (C.19) |
| E.7 | `pe_base` recovery | Read from `prepared_data/config.json`; CLI fallback | DEFAULT | Only place it's persisted |
| E.8 | `pe_dim` cross-check | Raises if checkpoint-derived value disagrees with config.json | DEFAULT | Catches mismatched checkpoint / prepared_data pairs early |
| E.9 | Battery-suffix override | Replicated from `build_graphs` in `collect_records` | CLAUDE (bug fix) | See change log — omitting this produced R² = −0.62 |

Validation: scoring all training JSONs through `predict.py` reproduces
R² ≈ 0.95, matching `evaluate.py`.

---

## W&B Bayesian sweep (`sweep.yaml`)

| # | Choice | Value | Source | Notes |
|---|--------|-------|--------|-------|
| S.1 | Search method | Bayesian | USER | Sample-efficient |
| S.2 | Optimized metric | val_loss (minimize) | DEFAULT | Same as early stopping |
| S.3 | Early termination | Hyperband, `min_iter = 20` | DEFAULT | Kills unpromising runs after 20 epochs |
| S.4 | Swept: `lr` | log-uniform [1e-5, 1e-2] | DEFAULT | |
| S.5 | Swept: `weight_decay` | log-uniform [1e-6, 1e-2] | DEFAULT | |
| S.6 | Swept: `hidden_dim` | {64, 128, 256} | DEFAULT | |
| S.7 | Swept: `edge_hidden_dim` | {16, 32, 64} | DEFAULT | |
| S.8 | Swept: `num_gat_layers` | {2, 3, 4, 5, 6} | DEFAULT | |
| S.9 | Swept: `heads` | {2, 4, 8} | DEFAULT | |
| S.10 | Swept: `head_aggr` | {concat, average} | DEFAULT | |
| S.11 | Swept: `dropout` | uniform [0.0, 0.4] | DEFAULT | |
| S.12 | Swept: `head_dropout` | uniform [0.0, 0.4] | DEFAULT | |
| S.13 | Swept: `head_depth` | {1, 2, 3} | DEFAULT | |
| S.14 | Swept: `t0` | {10, 15, 20, 30} | DEFAULT | |
| S.15–18 | Fixed | epochs 110, batch_size 64, patience 31, grad_clip 1.0 | USER | |

### Sweep findings

The sweep was stopped early (option A) once it became clear the baseline was
near-optimal. Representative runs:

| Run | Status | hidden_dim | layers | lr | dropout | Best val_loss | Test R² |
|---|---|---|---|---|---|---|---|
| serene-sweep-1 | Completed | 256 | 3 | 5.4e-4 | 0.046 | 0.0529 | 0.9501 |
| fine-sweep-2 | Killed @ ep 21 | 256 | 6 | 6.8e-4 | 0.344 | ~0.12 | — |
| laced-sweep-3 | Killed @ ep 22 | 64 | 2 | 1.1e-5 | 0.167 | ~1.0 (collapsed) | — |
| **baseline (`dpn7elbs`)** | Completed | **128** | **4** | **1e-3** | **0.1** | **0.0435** | **0.9558** |

Interpretation: a substantially different architecture (2× width, 2.7× the
parameters, half the depth, 8 heads) reached essentially the same
generalization — slightly worse. This is evidence the chosen configuration
sits near the achievable ceiling for this model family on this data, rather
than evidence that tuning was skipped.

Learning rate was the single most consequential hyperparameter: `lr = 1.1e-5`
produced a model that never escaped the mean predictor (val_loss ≈ 1.0 with
z-scored targets).

---

## Open questions and ablations to consider

1. **Pooling strategy** — currently mean. Ablate against sum (with size scaling), max, `Set2Set`, or attention-based readout (`GlobalAttention`).
2. **Sinusoidal PE parameters** — run used `pe_dim = 32`, `pe_base = 1000`. Ablate 8 / 16 / 32 and base 100 / 1000, and against raw normalized coordinates or learnable spatial embeddings.
3. **Including `tw_width`** — redundant with `release_time` and `deadline`. Drop and compare convergence.
4. **Edge feature design** — currently 2 dims. Could add angle relative to depot, `(release_time_j − release_time_i)`, or destination demand.
5. **Number of GAT layers** — fixed at 4, but on a complete graph 1 hop already reaches every node. Test 1–2 layers as a baseline; the sweep's 3-layer run performed comparably.
6. **GraphNorm vs alternatives** — ablate LayerNorm, BatchNorm, or no norm.
7. **Loss function** — MSE chosen from the distribution statistics. A Huber ablation (δ ≈ 1.0 on z-scored y) is cheap and would test the moderate right tail.
8. **Target normalization** — z-score. Log-transform is awkward here because `y` is mostly negative and crosses zero.
9. **Including non-served customers as masked nodes** — deliberately avoided to prevent leaking selection information. Worth revisiting as a deliberate experiment with a masked-attention design.
10. **Global conditioning placement** — currently only at pooling. Could broadcast the two scalars onto every node so the GAT layers see them too.
11. **Capacity values as learned embeddings** — instead of raw scalar concat, discretize and embed.
12. **Multi-seed evaluation** — the reported metrics come from a single run. Repeat with several seeds to estimate variance.
13. **Curriculum by graph size** — small graphs first, then large. May address the small-graph weakness (D findings).
14. **Small-graph oversampling or size-weighted loss** — directly targets the size effect found in evaluation.
15. **Multi-task auxiliary heads** — predict total served demand, travel time, or trip count as auxiliary signals sharing the encoder.
16. **Technical debt: filename-encoded data convention (row 0.9)** — the `*_battery` → `vehicle_load_capacity = 50.0` rule is implicit state carried in a string suffix. Every consumer of the raw JSONs must remember to replicate it, and `predict.py` initially did not. Robust fix: correct the source JSONs, or resolve the value once at build time and never re-derive it.
17. **Timing benchmark** — the practical case for a surrogate is speed. Measure ALNS seconds-per-evaluation against GAT milliseconds-per-evaluation to get the headline speedup figure.

---

## Change log

| Date / stage | Change | Reason |
|------|--------|--------|
| Initial | Sum pooling chosen | User preference |
| Phase C debug | Moved ReLU outside the residual branch: `h = ReLU(h_in + norm(GAT(h_in)))` | Original `h = h_in + ReLU(...)` accumulated only non-negative contributions across 4 layers, biasing embeddings positive and degenerating gradient flow. Model collapsed to a constant mean predictor (train_loss = 1.0000, val_loss = 0.9960, unchanged across 20+ epochs) |
| Phase C debug | Switched sum → mean pooling | Variable graph size (62–86 nodes) made sum-pooled magnitudes vary ~1.4×, giving the regression head an ill-conditioned input distribution. Contributed to the same collapse |
| Phase C debug | Added temporary `torch.manual_seed(42)` | Run-to-run variance was masking the architectural bug; removed once fixed |
| Phase C | First successful full run: R² = 0.956, MAE = 135 (test) | Both fixes together took val_loss from 0.996 (mean predictor) to 0.0435 |
| Phase A (server) | `pe_dim` 16 → 32, `pe_base` 100 → 1000 | Run on the server used these values; recorded in `prepared_data/config.json`. This doc previously listed the CLI defaults instead of the values actually used |
| Phase C | Sweep stopped early after 3 runs | Best sweep run (R² = 0.9501) did not beat the baseline (0.9558); further search judged low-value relative to GPU cost |
| Phase E | `predict.py` initially omitted the battery-suffix override | Produced R² = −0.62 on `*_battery` files: the demand normalization used the JSON's 66.6666 instead of the 50.0 the model was trained with, and the global conditioning scalar was likewise wrong. Fixed by replicating the rule in `collect_records` and threading it through `score_records` |
