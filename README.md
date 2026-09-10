# Deep-Learning-Enabled Autonomous Alignment of Multipass Herriott Cells for Gas Detection

<!-- If you have a preprint or publication link, uncomment and update: -->
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX) -->

A lightweight deep learning system that autonomously aligns Herriott optical cavities using camera feedback and low-cost stepper motors. The system runs entirely on a Raspberry Pi 4, achieves 100% alignment success across the full operational range, and converges in fewer than two steps on average — enabling field-deployable gas sensors that can recalibrate themselves without human intervention.

---

## System Overview

<p align="center">
  <img src="graphs/System photos/system_topdown_view.jpg" width="700" alt="Experimental system for optical cavity automatic alignment"/>
</p>

The system consists of a camera, two stepper motors coupled to a kinematic mirror mount, a GRIN lens laser source, a Herriott cell, and a Raspberry Pi 4 that handles both image acquisition and motor control.

<p align="center">
  <img src="graphs/Hardware Software diagrams/hardware+software.jpg" width="500" alt="Hardware and software architecture"/>
</p>

## How It Works

### Herriott Cell

A Herriott cell uses two opposing concave mirrors to bounce a laser beam back and forth, creating a long optical path in a compact volume. This extended path length makes the cell highly sensitive to trace gas concentrations — but also highly sensitive to mirror misalignment.

<p align="center">
  <img src="graphs/Simulation graphs/cavity_scheme.png" width="400" alt="Herriott cell schematic"/>
</p>

### What the Model Sees

A camera captures the reflection pattern on the mirror surface. The image is cropped, resized to 512×512, converted to grayscale, and masked to isolate the lens region. The pattern changes dramatically with alignment state:

<p align="center">
  <img src="graphs/Input Examples/zero/InputExample(0,0).jpg" width="180" alt="Optimal alignment"/>
  &nbsp;&nbsp;
  <img src="graphs/Input Examples/skewed/x1.10_y-0.80.jpg" width="180" alt="Suboptimal alignment"/>
  &nbsp;&nbsp;
  <img src="graphs/Input Examples/scatter/x2.80_y-0.57.jpg" width="180" alt="Misaligned"/>
</p>
<p align="center">
  <em>Left:</em> Optimal &nbsp;&nbsp;|&nbsp;&nbsp; <em>Center:</em> Suboptimal &nbsp;&nbsp;|&nbsp;&nbsp; <em>Right:</em> Misaligned
</p>

### Alignment Loop

The system operates in a closed loop. At each iteration, it captures an image, preprocesses it, and compares it to a reference image of optimal alignment using the Structural Similarity Index (SSIM). If SSIM exceeds 0.97 (corresponding to <0.07° angular error), alignment is complete. Otherwise, a ResNet-18 regression model predicts the angular correction needed, which is translated into stepper motor commands. The process typically converges in 1–2 steps.

## Results

The system was evaluated across the full mirror actuation range (X: −2° to +4°, Y: −2° to +2°) with starting positions spaced 0.5° apart. Results are averaged over five complete runs.

| Metric | MLP | ResNet-18 |
|---|---|---|
| Success rate | 100% | 100% |
| Time to align | 4.97 ± 2.56 s | **3.07 ± 1.58 s** |
| Number of adjustments | 1.76 ± 0.62 | 1.80 ± 0.79 |
| Final SSIM | **0.983 ± 0.002** | 0.977 ± 0.007 |
| Final angular error | **0.019° ± 0.017°** | 0.045° ± 0.021° |
| RAM usage | 1034 MiB | **83 MiB** |
| Inference time | 1.72 s | **1.16 s** |

ResNet-18 was selected for deployment due to its 12× lower memory footprint and faster inference, making it suitable for resource-constrained edge platforms.

<p align="center">
  <img src="graphs/Evaluation/resnet18_eval.png" width="500" alt="Evaluation heatmap showing number of adjustments per starting position"/>
</p>
<p align="center">
  <em>Number of adjustments needed to reach optimal alignment from each starting position. Most of the operational range converges in 1–2 steps.</em>
</p>

## Future Directions

### Int8 Quantized Inference on the Edge

The current bottleneck is not training but on-device inference: 1.16 s per prediction and 83 MiB of RAM on the Raspberry Pi 4. Since the alignment loop converges in 1–2 steps, inference latency dominates the 3.07 s time-to-align almost entirely.

Quantizing ResNet-18 to int8 targets exactly this. The weights drop from ~44 MB to ~11 MB, and ARM CPUs have native NEON int8 paths reachable through ONNX Runtime, TFLite, or PyTorch's qnnpack backend — typically 2–3× faster inference on this class of hardware.

The workflow would be to train in float on the workstation, apply a short quantization-aware fine-tune to recover any accuracy lost to the int8 grid, and export the quantized model for the Pi. QAT matters here because the system's headline result is sub-0.05° angular accuracy, so the regression head is the part worth validating carefully after quantization.

Note that this is a deployment-side change only. Training stays in float: fp16 mixed precision is the useful lever on the training box, while int8 belongs at inference time.

<!-- 
## Citation

If you use this work, please cite:

```bibtex
@article{venediktov2025herriott,
  title   = {Deep-Learning-Enabled Autonomous Alignment of Multipass Herriott Cells for Gas Detection},
  author  = {Venediktov, Evgenii and Zhong, Shuda and Zhang, Guangyin and Splain, Zach and Chauhdry, Majid H. M. and Ikpeazu, Emeka and Mao, Zhi-Hong and Wright, Ruishu F. and Lalam, Nagesh and Chen, Kevin P.},
  year    = {2025}
}
```


## License

*TBD*
 -->

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment from the lockfile:

```bash
uv sync                      # core: training, inference, simulation, analysis
uv sync --all-extras         # everything, including notebooks
```

Run anything through `uv run`, which resolves the environment automatically:

```bash
uv run python train/train_resnet_direct.py --help
uv run python utils/graph_eval.py eval.log 0.97
```

### Extras

| Extra | Adds | Needed for |
|---|---|---|
| `zemax` | `pythonnet`, `pillow` | `data_process/generate_simulated_data.py` (Windows + OpticStudio only) |
| `notebooks` | `ipykernel`, `jupyterlab` | `train/experiments/*.ipynb` |

### PyTorch and CUDA

PyTorch is installed from PyPI, whose Linux x86_64 wheels bundle the CUDA runtime — no extra index is configured. On Windows, if CUDA is not picked up, add PyTorch's own index to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch" }]
torchvision = [{ index = "pytorch" }]
```

On the Raspberry Pi (aarch64) the default PyPI CPU wheels are used, which is what inference needs.

### requirements.txt

`requirements.txt` is generated from `uv.lock` and kept only for environments without uv, such as the Raspberry Pi. Do not edit it by hand — regenerate it with:

```bash
uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

## Training

`train/TRAINING_AND_EVALS.md` is the full handoff on training, offline evaluation and the
cluster workflow, including the current best checkpoint and its recipe.

`train/train_resnet_direct.py` builds the ResNet-18 and writes a plain state dict that loads
directly into `TiltPredictor(model_type="ResNet18")` for deployment.

### `train/train_resnet_direct.py`

Reads the preprocessed JPEGs straight from the collection directory. Workers return uint8
and the `/255` conversion plus every augmentation run batched on the GPU, so nothing carries
four bytes per pixel through the loader. fp16 autocast, TF32 and channels_last are on by
default. Runs are logged to Weights & Biases; export `WANDB_API_KEY`, or pass `--no-wandb`.

```bash
uv run python train/train_resnet_direct.py                    # full dataset
uv run python train/train_resnet_direct.py --data-share 0.25  # a quarter of it
uv run python train/train_resnet_direct.py --step-filter 4    # 0.04 deg grid instead of 0.01
uv run python train/train_resnet_direct.py --help             # every augmentation knob
```

Augmentation defaults, on for every run: brightness 0.4 and contrast 0.1 jitter per image,
Gaussian noise with a per-image sigma uniform in [0, 0.2] (`--noise-min`, `--noise`), and
two occlusion boxes of 0.15 to 0.40 of the image drawn once per batch and applied to every
image in it. Default 98 epochs, 14 full cosine cycles. `--resolution 256` trains on a
256 px bank (the loader resizes anything that is not already that size).
Every run writes its split to `train_names.txt` / `val_names.txt` in the checkpoint
directory; pass them back with `--train-keys-file` / `--val-keys-file` to reuse a split. Affine is implemented but off by default: the label *is* the position of the spot
pattern, so a translation resembles a different mirror tilt.

### Current best checkpoint

`resnet18_l40s_3854472_best_model.pth`, trained on 2026-09-09 by CRCD job 3854472 on one
L40S (6.3 h). W&B run: [resnet18_l40s_3854472](https://wandb.ai/e-venediktov-university-of-pittsburgh/multireflection/runs/93o4zgri).
Best epoch 93 of 96, validation MSE 6.5e-5 on normalized labels. The checkpoint lives at
`/ix1/kchen/evv/multireflection/runs/3854472/` on the cluster and in `saved_models/real/`
locally (not tracked).

| | |
|---|---|
| data | full `dark512`, 0.01 deg grid, 183,040 train / 45,760 val (random 20%, seed 0) |
| batch size, lr | 512, 2.8e-3 (1e-3 at 64, scaled by sqrt of the batch ratio) |
| optimizer | AdamW, weight decay 1e-3 |
| schedule | CosineAnnealingWarmRestarts, T_0 = 7 epochs, eta_min 1e-5, 96 epochs |
| precision | fp16 autocast, TF32, channels_last |
| augmentation | brightness 0.4, contrast 0.1, Gaussian noise 0.1, cutout p 0.5 x 2 boxes of 0.05 to 0.20 of the image; affine off |
| loader | 10 workers, prefetch 4, uint8 to the GPU |

Note the cutout range: the script default has since been raised to 0.15 to 0.40, so a rerun
with defaults is not the same recipe. Reproduce with
`--batch-size 512 --lr 0.0028 --occlusion-min 0.05 --occlusion-max 0.20`.

Offline sweep (`utils/eval_batched.py`, threshold 0.97, cap 10 adjustments):

| start grid | starts | converged | adjustments | final SSIM | final angular error |
|---|---|---|---|---|---|
| 0.1 deg | 2320 | 100% | 1.14 +- 0.35 (max 2) | 0.984 +- 0.007 | 0.027 +- 0.015 deg (max 0.085) |
| 0.05 deg | 9200 | 100% | 1.14 +- 0.35 (max 2) | 0.984 +- 0.007 | 0.027 +- 0.015 deg (max 0.085) |

### Offline evaluation

`utils/eval_batched.py` replays the paper's alignment sweep (Algorithm 2) on the collected
image bank instead of the hardware: every start on a grid is stepped through the closed
loop, with the model run in GPU batches and the dataset used as the position-to-image
lookup. It writes a per-step trace, a log in the hardware format that `utils/graph_eval.py`
reads, a summary and the heatmaps. About twenty seconds for the full 0.1 degree grid.

```bash
uv run python utils/eval_batched.py --checkpoint saved_models/real/<ckpt>.pth
uv run python utils/eval_batched.py --grid-step 0.5 --threshold 0.97   # coarse
uv run python utils/eval_batched.py --starts-file runs/<job>/val_names.txt  # held-out starts
```

Starts are training positions (the validation split is a random 20% of the same folder), so
this measures closed-loop behaviour on seen data. Time-to-align does not exist offline.

Robustness: `--brightness-fixed`, `--contrast-fixed`, `--noise`, `--occlusion-count` and
friends perturb the frames the model sees (brightness, contrast and occlusion boxes are
fixed per trajectory, noise is redrawn per step); the SSIM stop test still uses the clean
frame unless `--perturb-ssim`. `utils/eval_sweep.py` runs a fixed list of such conditions
on one checkpoint and writes a comparison table (`sweep.md`), optionally to W&B.

### On the Pitt CRCD cluster

```bash
rsync -avP dark512.tar.gz crc:/ix1/kchen/evv/multireflection/data/
ssh crc
cd /ihome/kchen/evv13/multireflection && git pull
bash cluster/check_env.sh
sbatch cluster/train_l40s.slurm
```

The job trains, then runs the offline evaluation on the best checkpoint twice, from the
0.1 degree grid and from the run's validation images, and attaches both summaries and
heatmaps to the same W&B run. `START_CKPT`, `EPOCHS` and `EXTRA_ARGS` can be passed with
`sbatch --export` to fine-tune, shorten, or change training flags. Resource choices, storage layout and the staging
rationale are documented in [cluster/README.md](cluster/README.md).

## Repository Structure

| Path | Contents |
|---|---|
| `config.py` | actuation range, model selection, evaluation grid |
| `app/` | runs on the Raspberry Pi: inference, closed-loop alignment, evaluation sweep |
| `data_process/` | data collection, image preprocessing |
| `train/` | training scripts and experiment notebooks |
| `cluster/` | Pitt CRCD job scripts |
| `utils/` | offline evaluation sweep, evaluation-log parsing and plotting |
| `simulation/` | GPU ray-trace simulator, stability search, interactive viewers |
| `herriott_env.py`, `policy.py`, `train_rl.py`, `config_sampler.py` | reinforcement-learning exploration, currently dormant |
| `ai_context/` | condensed per-module notes on the codebase |
| `graphs/` | figures used here and in the paper |
| `hardware_design/` | bill of materials |
| `mf_control/` | motor and camera driver (git submodule) |
| `paper/`, `spie-archive/` | manuscripts (submodule / archived) |

## Acknowledgments

This work was supported by the University of Pittsburgh.