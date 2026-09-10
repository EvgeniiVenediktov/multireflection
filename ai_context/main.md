# Multireflection - project context

Autonomous alignment of a multipass **Herriott cell** (two opposing concave mirrors used to
lengthen the optical path for TDLAS gas sensing). A camera looks at the reflection pattern on a
mirror; a neural net regresses the angular correction of the entrance mirror; two stepper motors
apply it. Whole loop runs on a Raspberry Pi 4.

Result claimed in the paper/README: 100% alignment success over the full actuation range,
1-2 adjustments on average, ~3 s with ResNet-18.

## Project stages (chronological)

1. **Zemax-simulated data + MLP** - feasibility study (paper SS II-B).
2. **Real camera data + MLP (`SimpleFC`) and CNN/ResNet-18** - the deployed system.
   **This is the current focus**: a paper on it is being revised for IEEE Access.
3. **RL (PPO on a GPU ray-trace sim)** - later exploration, still exploratory.
   **RL is NOT a current concern.** Documented here only so it is not re-discovered from scratch.

## Start here for training, evaluation and the cluster

**`train/TRAINING_AND_EVALS.md`** is the handoff document: current best checkpoint, running
jobs, data, the training script's defaults and their change dates, the CRCD job script and
its knobs, the offline evaluation and robustness sweep, known issues and next steps. Read it
before touching `train/`, `utils/eval_*.py` or `cluster/`. The per-module notes below
(`training.md`, `alignment_loop.md`) are more detailed but not more current.

## Module map

| Area | Files | Context file |
|---|---|---|
| Data: collection, preprocessing, eval logs | `data_process/`, `config.py` | [data_sources.md](data_sources.md) |
| Models: architectures, inference wrapper | `app/inference.py`, `train/train_resnet_direct.py` | [model.md](model.md) |
| Supervised training: dataset, loop, augmentation | `train/train_resnet_direct.py` | [training.md](training.md) |
| Deployed closed loop + evaluation harness | `app/`, `utils/graph_eval.py` | [alignment_loop.md](alignment_loop.md) |
| Ray-trace simulator + parameter/stability search | `simulation/` | [simulation.md](simulation.md) |
| RL (PPO, GRU policy, vectorized env) - dormant | `herriott_env.py`, `policy.py`, `train_rl.py`, `config_sampler.py` | [rl.md](rl.md) |

## Repository layout

```
pyproject.toml          dependencies and extras, managed with uv (uv.lock is committed)
requirements.txt        generated from uv.lock, for environments without uv (the Pi)
config.py               global constants for data collection / inference / eval
app/                    on-device: inference, closed-loop alignment, eval sweep
data_process/           data collection, image preprocessing
train/                  supervised training script (train_resnet_direct.py) + two old notebooks
utils/                  offline evaluation sweep, eval-log plotting
simulation/             GPU ray-trace sim, stability search, interactive viewers
graphs/                 figures used by README and paper
hardware_design/        BOM
cluster/                Pitt CRCD job scripts (see "Cluster" below)
spie-archive/           earlier SPIE conference manuscript (frozen)
paper/                  git submodule - IEEE Access revision (see paper/AGENTS.md, INDEX.md)
mf_control/             git submodule - motor + camera driver (MFController), NOT checked out
```

`mf_control` is an external submodule providing `MFController`
(`start`, `capture_image`, `set_tilt_x/y`, `get_x_tilt/get_y_tilt`, `get_frame_position`, `close`).
It is empty in the working tree; anything importing it only runs on the Raspberry Pi.

## Current best checkpoint

`r512_occ05-20img_n10_e96_3854472.pth` (job 3854472, 2026-09-09, W&B run 93o4zgri). Recipe and
offline-eval numbers are in the README under "Current best checkpoint". Trained with cutout
0.05 to 0.20; the script default is now 0.15 to 0.40.

## Cluster (Pitt CRCD)

Reachable from this machine as `ssh crc` (h2p.crc.pitt.edu, user evv13, key in ~/.ssh/crcd;
non-interactive, so commands can be run there from here). Project checkout at
`/ihome/kchen/evv13/multireflection`, dataset and checkpoints at
`/ix1/kchen/evv/multireflection`. GPU jobs run on the `gpu` cluster, `l40s` partition, via
`cluster/train_l40s.slurm`; the environment is built once with `cluster/check_env.sh`. Compute
nodes have outbound HTTPS and W&B credentials come from `~/.netrc` (verified 2026-09-09 on
gpu-n55). The job script passes `--cpus-per-task` to `srun` explicitly: since Slurm 22.05 srun
does not reliably inherit it from sbatch, which would pin the loader workers to one core.
`WANDB_DIR` is set to the run directory on `/ix1`, so no W&B files accumulate on `/ihome`.
The kchen group has no `smp` billing minutes left (jobs pend on AssocGrpBillingMinutes);
use the `gpu` cluster for test jobs too. Resource rationale and storage notes:
`cluster/README.md`.

## Known inconsistencies (verify before trusting)

- `config.py:INFERENCE_MODEL_TYPE = "SimpleFC"` while `INFERENCE_MODEL_FILE_NAME` names a
  resnet18 checkpoint. One of the two is stale.
- `config.py:SIMILARITY_INDEX_THRESHOLD = 0.95`; paper, README and `utils/graph_eval.py` use 0.97.

## Environment

Managed with **uv**; `uv.lock` is committed. `uv sync` gives the core set, `--extra zemax`
the Windows OpticStudio path, `--extra notebooks` Jupyter. Run things with
`uv run python <script>`.

Removed as unnecessary during the uv transition: `scikit-learn` (a single `train_test_split`
call) and `torchsummary`; `pillow` and `pythonnet` moved to `zemax`. The `lmdb` extra
(`lmdb`, `msgpack`, `lz4`, `torchinfo`) went away with the LMDB pipeline on 2026-09-10
(see [training.md](training.md)).
