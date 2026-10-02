# GAT surrogate for customer-pool selection in the PC-MT-EVRP-RT-D

A graph attention network that predicts the ALNS objective value for a candidate
set of customers, used to choose which customers to hand a prize-collecting
multi-trip electric vehicle routing solver before routing begins.

The pipeline runs in five stages. Each writes files the next one reads, so you
can stop and resume anywhere.

```
reference instances → ALNS labels → PyG graphs → normalised splits → model → experiments
```

---

## Setup

```bash
conda create -n gat python=3.11 -y
conda activate gat
pip install -r requirements.txt
wandb login            # or: python -m wandb login <API_KEY>
```

PyTorch Geometric needs a build matching your CUDA version — follow the
[PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
rather than a bare `pip install torch`. Check with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Training wants a GPU; everything else is CPU-bound and benefits from cores.

---

## Directory layout

Inputs you provide, outputs the scripts create:

```
<repo>/
├── data generation/                 INPUT — reference instances
│   ├── R_instances.json
│   ├── C_instances.json
│   └── RC_instances.json
│
├── training data v4/                stage 1 — one labels_<instance_id>.json per instance
├── processed_graphs_v4_8f/          stage 2 — graphs.pt
├── prepared_data_v4_8f/             stage 3 — train/val/test.pt, stats.json, config.json
├── models/<wandb_run_id>/           stage 4 — best.pt  (+ evaluation/ after evaluate.py)
├── results/<experiment_name>/       stage 5 — results.csv, results.jsonl, pools.json, analysis.json
└── figures/, paper_figures/         analysis output
```

The generated directories are large and should stay out of git:

```
training data*/
processed_graphs*/
prepared_data*/
models/
results/
wandb/
*.pt
__pycache__/
```

Name the `processed_graphs` and `prepared_data` directories after the
configuration they hold (`_8f` for the 8-feature variant, `_pool70` for a 70%
pool). You will accumulate several, and the only thing distinguishing them is
the name.

---

## Running the pipeline

### 1. Generate labels

Runs the ALNS over sampled customer pools under every capacity setting.

```bash
python alns_label_generator.py --dry-run --workers 60      # check the plan first
python alns_label_generator.py --workers 60 --pool-fraction 0.85 \
    --load-basis per_vehicle --output-dir "training data v4"
```

100 instances × 2 battery × 2 load × 35 pools = 14,000 runs, roughly two days on
60 cores. Shards are written per instance; `--resume` skips completed ones.

Inspect what came out:

```bash
python inspect_labels.py --input "training data v4"
```

### 2. Build graphs

```bash
python build_graphs.py --node-set pool --include-prize-penalty \
    -t "training data v4" -o processed_graphs_v4_8f
```

`--node-set pool` puts every pooled customer in the graph. The alternative,
`served`, keeps only those the ALNS actually visited — that is an *output* of
the solver, so it cannot be an input at selection time. Drop
`--include-prize-penalty` for the 6-feature variant.

### 3. Split and normalise

```bash
python data_prep.py -i processed_graphs_v4_8f/graphs.pt \
    -o prepared_data_v4_8f --pe-dim 32 --pe-base 1000
```

Stratifies 80/10/10 by topology and computes all statistics on the training
split only. Confirm the printed feature dimension: **68** for 6 features,
**70** for 8.

### 4. Train and evaluate

```bash
python train.py --data_dir prepared_data_v4_8f --run_name v4-8feat
python evaluate.py --checkpoint models/<run_id>/best.pt \
    --data_dir prepared_data_v4_8f --splits train val test
```

`train.py` prints the W&B run id; that becomes the checkpoint folder name. Note
it down. `evaluate.py` writes metrics, diagnostic plots and per-graph
predictions to `models/<run_id>/evaluation/`.

A Bayesian hyperparameter sweep is available but optional:

```bash
wandb sweep sweep.yaml
wandb agent --count 30 <entity>/gat-evrp/<sweep_id>
```

### 5. Compare pool-selection strategies

```bash
python run_experiment.py --checkpoint models/<run_id>/best.pt \
    --prepared-config prepared_data_v4_8f/config.json \
    --vehicle-load 85 --pool-size 85 \
    --battery-settings loose tight --tight-multiplier 0.85 \
    --time-limit 480 --max-pool 100 --insert-count 3 --seeds 10 \
    --baseline A_full --workers 55 --output-dir results/arms_v4_8f
```

Seven selection strategies run on identical instances, seeds and budgets.
Every arm picks the same number of customers, so all face the same structural
handicap and the differences reflect strategy alone.

| Arm | Strategy | Selects by |
|---|---|---|
| `B_random` | Uniform random | A random subset — the null strategy and usual baseline |
| `C1_prize` | Greedy on prize | The highest-prize customers |
| `C2_penalty` | Greedy on penalty | The customers most expensive to leave unserved |
| `C3_tw_spread` | Time-window spread | Customers whose windows tile the planning horizon evenly |
| `C4_cluster` | Cluster-aware | Whole spatial clusters, ranked by mean value density |
| `C5_density` | Value density | Best (prize + penalty) per unit distance from the depot |
| `D_surrogate` | **GAT surrogate** | Lowest predicted objective among ~500 candidate pools |

`D_surrogate` screens a candidate set that contains every other arm's pool, so
it can only lose by misranking. The rest of the set mixes randomised-greedy
variants of all five criteria, perturbed neighbourhoods of the arm pools, and
uniform random draws, letting the surrogate reach pools no single heuristic
would produce.

Results stream as they complete; `--resume` picks up after an interruption.
Re-analyse without re-running, against whichever baseline you want:

```bash
python run_experiment.py --analyse-only --output-dir results/arms_v4_8f \
    --baseline C4_cluster
```

---

## File reference

| File | Role |
|---|---|
| `alns_core.py` | Destroy/repair operators and the objective function |
| `alns.py` | Resumable ALNS, `Instance` construction |
| `alns_label_generator.py` | Parallel label generation |
| `inspect_labels.py` | Label distributions by topology and setting |
| `build_graphs.py` | Raw JSON → PyG `Data` objects |
| `data_prep.py` | Split, normalise, write `stats.json` and `config.json` |
| `model.py` | `GATSurrogate` definition |
| `train.py` | Training loop, W&B logging, checkpointing |
| `evaluate.py` | Test metrics, diagnostic plots, per-graph CSV |
| `predict.py` | Inference CLI and the warm `Predictor` class |
| `pools.py` | The six heuristic strategies and candidate-set generation |
| `run_experiment.py` | Arm-comparison harness and paired analysis |
| `sweep.yaml` | W&B Bayesian sweep configuration |

`docs/design_decisions.md` records every design choice with its rationale and a
change log; `docs/server_runbook.md` is the startup sequence for a fresh
session on the compute server.
