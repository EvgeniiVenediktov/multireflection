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

## Module map

| Area | Files | Context file |
|---|---|---|
| Data: collection, preprocessing, LMDB, splits, eval logs | `data_process/`, `config.py` | [data_sources.md](data_sources.md) |
| Models: architectures, inference wrapper | `app/inference.py`, `train/cnn_train.py` | [model.md](model.md) |
| Supervised training: datasets, loop, sweeps | `train/`, `model_real.ipynb` | [training.md](training.md) |
| Deployed closed loop + evaluation harness | `app/`, `utils/graph_eval.py` | [alignment_loop.md](alignment_loop.md) |
| Ray-trace simulator + parameter/stability search | `simulation/` | [simulation.md](simulation.md) |
| RL (PPO, GRU policy, vectorized env) - dormant | `herriott_env.py`, `policy.py`, `train_rl.py`, `config_sampler.py` | [rl.md](rl.md) |

## Repository layout

```
pyproject.toml          dependencies and extras, managed with uv (uv.lock is committed)
requirements.txt        generated from uv.lock, for environments without uv (the Pi)
config.py               global constants for data collection / inference / eval
app/                    on-device: inference, closed-loop alignment, eval sweep
data_process/           data collection, image preprocessing, LMDB build
train/                  supervised training script + notebooks + wandb sweep
utils/                  eval-log plotting, misc helpers
simulation/             GPU ray-trace sim, stability search, interactive viewers
graphs/                 figures used by README and paper
hardware_design/        BOM
spie-archive/           earlier SPIE conference manuscript (frozen)
paper/                  git submodule - IEEE Access revision (see paper/AGENTS.md, INDEX.md)
mf_control/             git submodule - motor + camera driver (MFController), NOT checked out
```

`mf_control` is an external submodule providing `MFController`
(`start`, `capture_image`, `set_tilt_x/y`, `get_x_tilt/get_y_tilt`, `get_frame_position`, `close`).
It is empty in the working tree; anything importing it only runs on the Raspberry Pi.

## Known inconsistencies (verify before trusting)

- `config.py:INFERENCE_MODEL_TYPE = "SimpleFC"` while `INFERENCE_MODEL_FILE_NAME` names a
  resnet18 checkpoint. One of the two is stale.
- `config.py:SIMILARITY_INDEX_THRESHOLD = 0.95`; paper, README and `utils/graph_eval.py` use 0.97.
- `train/cnn_train.py` contains a hardcoded W&B API key. Should be rotated / moved to env.
  `train/train_resnet_direct.py` reads `WANDB_API_KEY` from the environment instead.

## Environment

Managed with **uv**; `uv.lock` is committed. `uv sync` gives the core set, `--extra lmdb`
adds the legacy LMDB pipeline, `--extra zemax` the Windows OpticStudio path, `--extra
notebooks` Jupyter. Run things with `uv run python <script>`.

Removed as unnecessary during the uv transition: `scikit-learn` (was pulled in for a single
`train_test_split` call, now a four-line stdlib shuffle in `prepare_lmdb.py`),
`torchsummary` (unmaintained since 2018, replaced by `torchinfo`), and dead `msgpack` /
`lz4` imports in `cnn_train.py`. `lmdb`, `msgpack`, `lz4` and `torchinfo` moved to the
`lmdb` extra; `pillow` and `pythonnet` to `zemax`.
