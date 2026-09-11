# Training and evaluation handoff

Written 2026-09-10. The current summary of how the ResNet-18 tilt regressor is trained,
evaluated offline, and run on the Pitt CRCD cluster. Scripts and `git log` are authoritative
where prose disagrees. Per-module detail lives in `ai_context/training.md` and
`ai_context/alignment_loop.md`; this file is the entry point.

## 1. Purpose and status

The model takes a 512x512 grayscale camera image of the Herriott cell spot pattern (or its
256, 128 or 64 px box-averaged version) and regresses the mirror tilt (x, y) in degrees. The
deployed loop moves the mirror by the negated prediction and stops when the SSIM against the
reference image reaches a threshold. Training is supervised regression on the collected image bank.

**Run names** (since 2026-09-10; checkpoints, run directories and W&B display names):
`r<resolution>[_ft]_occ<box edge % min-max><img|batch>_n<noise sigma x100>_e<epochs>[_s<seed>]_<job>`
(`_s<seed>` only when `--seed` is not 0). Split marker after the epochs: `_hold` = trained on the
spatial hold-out TRAIN set (holdout/README.md), evaluated in `holdout_test/`; `_fullctl` = the
full-data control of the hold-out study; no marker = the random 80/20 split (sweeps in `sweep/`).
The W&B display name is built by the training script when the job starts
(`run_display_name`; the split marker comes from `--split-tag hold|fullctl`). Run directories
and checkpoints still start as `runs/<job>/resnet18_l40s_..._best_model.pth` and are renamed after
their dependent eval jobs have finished.
All runs so far use 2 boxes with p 0.5; `img` = boxes drawn per image, `batch` = drawn once
per batch; `-rot90-bri50` = boxes rotated in [-90, 90] deg and white with p 0.5 (absent =
axis-aligned black boxes); `n00-20` = sigma per image in [0, 0.2]; `_ft` = fine-tuned;
`local` = trained on the dev machine. Checkpoints are `<name>.pth` in `runs/<name>/` on the cluster and in
`saved_models/real/` locally. Old names: `resnet18_l40s_<job>_best_model.pth` in `runs/<job>/`.

**Current best 512 px checkpoint:** `r512_occ05-20img_n10_e96_3854472.pth`, CRCD job
3854472, 2026-09-09, W&B run
[r512_occ05-20img_n10_e96_3854472](https://wandb.ai/e-venediktov-university-of-pittsburgh/multireflection/runs/93o4zgri).
Best epoch 93 of 96, val MSE 6.5e-5 on normalized labels. Offline sweep: 100% converged,
1.14 adjustments, angular error 0.027 deg. Recipe in README "Current best checkpoint";
it predates three default changes listed in section 3.

**Runs on 2026-09-10** (in `/ix1/kchen/evv/multireflection/runs/<name>/`):

| run | what | status | best val MSE | offline eval (grid 0.1) |
|---|---|---|---|---|
| r512_ft_occ15-40img_n10_e49_3866737 | fine-tune of r512_occ05-20img_n10_e96_3854472, cutout 0.15-0.40, fixed noise 0.1 | finished | 7.0e-5 | 100%, 1.11 adj, 0.026 deg |
| r512_occ15-40img_n00-20_e96_3866805 | from scratch, noise sigma per image in [0, 0.2], cutout 0.15-0.40 | finished, 6.5 h | 2.4e-5 | val set (45,760 starts): 100%, 1.02 adj, 0.011 deg |
| r256_occ15-40batch_n00-20_e98_3867169 | from scratch on `dark256` at 256 px, defaults of that afternoon (axis-aligned black boxes) | finished | 2.8e-5 | 100%, 1.03 adj, 0.011 deg; val set (45,760 starts): same |
| r64_occ15-40batch_n00-20_e98_local | from scratch on `dark64` at 64 px on the local A2000, defaults of that afternoon, batch 512, lr 2.8e-3, 5 workers; 30 min at 10k img/s | finished | 4.1e-5 | see section 5 |
| r256_occ15-40batch-rot90-bri50_n00-20_e98_3870306 | from scratch on `dark256`, all current defaults including rotated and bright boxes; 88 min at 3.5k img/s | finished | 3.0e-5 | 100%, 1.03 adj, 0.012 deg; val set: 100%, 1.04, 0.011 |
| r128_occ15-40batch-rot90-bri50_n00-20_e98_3870308 | the same on `dark128`; 21 min at 15k img/s | finished | 3.2e-5 | 100%, 1.05 adj, 0.011 deg; val set: 100%, 1.04, 0.011 |
| r64_occ15-40batch-rot90-bri50_n00-20_e98_3870310 | the same on `dark64`; 8 min at 46k img/s | finished | 4.8e-5 | 100%, 1.16 adj, 0.017 deg; val set: 100%, 1.15, 0.017 |

The offline eval column uses each run's own bank (the job's eval stage); section 5 compares
all models on the same 512 px frames.

Section 5 has the comparison and findings: the 512 px failures came from the recipe, bright
boxes in training are what make glare survivable, and 128 px is the strongest of the
rot90-bri50 runs. None of the small-input models has been checked on hardware. Job 3866737
has no `eval_val/`; it was submitted before that stage existed.

## 2. Data

**Image bank `dark512`.** 228,800 grayscale 512x512 JPEGs named `x{X:.2f}_y{Y:.2f}.jpg`,
one per 0.01 deg tilt position, x in [-2.00, 3.71], y in [-2.00, 1.99]. Mean pixel about
8 of 255; only the spots are bright, so linear renderings look black on a monitor. Locally
at `/mnt/h/dark512/`. On the cluster only the archive is used:
`/ix1/kchen/evv/multireflection/data/dark512.tar.gz` (4.5 GB), staged to node-local NVMe by
every job. The extracted `data/dark512/` directory on `/ix1` is a stale partial copy
(93k files) from an aborted resize and can be deleted.

**`dark256`.** Same files resized to 256x256 with `cv2.INTER_AREA`, JPEG quality 95, built
on the login node by streaming `dark512.tar.gz` (`data/make_dark256.py`, `run_dark256.sh`
next to it). `data/dark256.tar.gz`, 1.85 GB, 228,800 files. Local copy extracted from that
archive at `/home/evv/data/dark256/`.

**`dark128`, `dark64`.** Same recipe (INTER_AREA directly from the 512 px JPEG, quality 95),
built locally in one pass over `/mnt/h/dark512` (6.5 min, 5 processes):
`/home/evv/data/dark128/` (947 MB) and `/home/evv/data/dark64/` (902 MB), 228,800 files
each. Archives on the cluster: `data/dark128.tar.gz` (642 MB), `data/dark64.tar.gz` (259 MB).

**Split.** `build_split` in `train/train_resnet_direct.py`: shuffle with `--split-seed`
(default 0), last 20% (`--val-share`) is validation: 183,040 train, 45,760 val. Every run
writes `train_names.txt` and `val_names.txt` into its checkpoint directory; pass them back
with `--train-keys-file` / `--val-keys-file` to reuse a split. `--data-share` and
`--step-filter` (1/2/4 for 0.01/0.02/0.04 deg grids) subsample for ablations.

**Caveat.** Every position in the bank is on the training grid. The validation images are
held out from the loss but the model has seen their neighbours 0.01 deg away. Offline eval
starts, whether the 0.1 deg grid or `val_names.txt`, therefore measure closed-loop
behaviour on seen conditions, not generalisation to a new setup.

**Labels.** Normalized to [0, 1] from `X/Y_TILT_START/STOP` in `config.py` (x span 5.7,
y span 4). `app/inference.py` de-normalizes with the same constants.

## 3. Training script: `train/train_resnet_direct.py`

**Model.** `resnet18(output_dim=2)`, defined in the script byte-identically to
`app/inference.py:ResNet` so checkpoints are plain state dicts that load into
`TiltPredictor(model_type="ResNet18")` with `strict=True`. With `--compile` the checkpoint
is still saved from the underlying module (no `_orig_mod.` prefix).

**Input pipeline.** Workers return uint8; `/255` and every augmentation run batched on the
GPU. fp16 autocast, TF32 and channels_last on by default (`--no-amp` disables the first and
last). `--resolution N` sets a square input size; the loader resizes anything that is not
already that size, so a 256 px bank trains at 256 only with `--resolution 256`.

**Optimization.** AdamW, weight decay 1e-3, lr 1e-3 default (jobs use 2.8e-3 at batch 512,
sqrt scaling from 1e-3 at 64), `CosineAnnealingWarmRestarts(T_0=7, eta_min=1e-5)` stepped
per epoch, MSE on the normalized 2-vector. Default 98 epochs = 14 full cosine cycles. Best
val loss checkpoint only, `<name>_best_model.pth`.

**Augmentation defaults** (`config` dict; every stage is per image on the GPU except
occlusion):

| stage | default | flags | since |
|---|---|---|---|
| brightness jitter | factor in [0.6, 1.4] | `--brightness 0.4` | original |
| contrast jitter | factor in [0.9, 1.1] | `--contrast 0.1` | original |
| Gaussian noise | sigma per image uniform in [0, 0.2] | `--noise-min 0.0 --noise 0.2` | 2026-09-10 (was fixed 0.1) |
| occlusion (cutout) | 2 boxes, each with p 0.5, edges 0.15 to 0.40 of the image, rotated about the centre by an angle in [-90, 90] deg, white (1.0, glare) with p 0.5 else black (dust), drawn once per batch and applied to every image in it | `--occlusion-*`, `--occlusion-angle`, `--occlusion-bright-prob/-value` | boxes per batch and 0.15-0.40 since 2026-09-10 (was per image, 0.05-0.20); rotation and bright boxes since 2026-09-10 |
| affine | off | `--affine-*` | never on; a translation resembles a different tilt |

The run names encode which recipe each run used (section 1). A rerun of
r512_occ05-20img_n10_e96_3854472 with today's defaults is a different recipe; reproduce it with
`--noise-min` unset, `--noise 0.1 --occlusion-min 0.05 --occlusion-max 0.20 --epochs 96`
(per-batch occlusion cannot be switched back without editing `_occlude`).

**Other flags.** `--starting-checkpoint <abs path>` warm-starts the weights (optimizer and
schedule start fresh). `--compile` runs the model through `torch.compile`; on the local
A2000 it gave no throughput gain and cost a slower first epoch; untested on the L40S.
`--no-wandb` for unlogged runs. `--help` lists everything.

**W&B.** Project `multireflection`, entity `e-venediktov-university-of-pittsburgh`.
Auth comes from `~/.netrc` (`wandb login`) locally and on the cluster; the script prints a
warning that `WANDB_API_KEY` is not set even when netrc auth works, ignore it. Logs per
epoch: train/val loss and logs, LR, best loss, throughput, gradient/parameter histograms.
At epoch 0: 15 raw and 15 augmented samples drawn across the training split,
display-stretched (99.5th percentile to white, gamma 0.5) because linear frames render
black; the network still sees linear input. `WANDB_DIR` and `WANDB_RUN_ID` are honoured, the
job script sets both so the eval stages attach to the same run.

## 4. Cluster: Pitt CRCD

**Access.** `ssh crc` from the dev machine (h2p.crc.pitt.edu, user evv13, key in
`~/.ssh/crcd`, non-interactive). Project checkout `/ihome/kchen/evv13/multireflection`
(git pull there after pushing). Data and runs `/ix1/kchen/evv/multireflection/{data,runs}`.
GPU jobs need `--cluster=gpu`; partition `l40s` (19 nodes, 4x L40S 48 GB, 64 cores). The
group's `smp` billing minutes are exhausted, so use the gpu cluster even for test jobs.

**Environment.** `bash cluster/check_env.sh` builds `.venv` from `uv.lock` (uv in
`~/.local/bin`). The job never syncs, so re-run it after any `uv.lock` change. No modules
are loaded: the torch wheels bundle the CUDA 13 runtime and a cuda module would shadow it.
Driver 595.91.07. The wheel has no native sm_89 kernels and runs on the L40S through
forward compatibility (`cluster/gpu_check.py` verifies).

**Job script `cluster/train_l40s.slurm`.** 1 GPU, 12 CPUs, 96 GB, 1 day. Environment knobs
via `sbatch --export=ALL,...`:

| variable | default | meaning |
|---|---|---|
| `DATASET` | `dark512` | stages `data/<DATASET>.tar.gz`, which must extract to `<DATASET>/` |
| `EPOCHS` | 98 | passed as `--epochs` |
| `START_CKPT` | empty | absolute path; passed as `--starting-checkpoint`, run name gets `_ft_` |
| `EXTRA_ARGS` | empty | appended to the training command, word-split, no spaces inside values |

```bash
sbatch cluster/train_l40s.slurm
sbatch --export=ALL,START_CKPT=/ix1/kchen/evv/multireflection/runs/r512_occ05-20img_n10_e96_3854472/r512_occ05-20img_n10_e96_3854472.pth,EPOCHS=49 cluster/train_l40s.slurm
sbatch --export=ALL,DATASET=dark256,EXTRA_ARGS="--resolution 256" cluster/train_l40s.slurm
sbatch --export=ALL,EXTRA_ARGS="--noise-min 0.0 --noise 0.2 --lr 0.001" cluster/train_l40s.slurm
```

End to end: `module purge`; print torch and GPU info; copy the archive to `$SLURM_SCRATCH`
(node-local NVMe, wiped at job end) and extract; export `WANDB_DIR=$RUN_DIR` and
`WANDB_RUN_ID=l40s-<jobid>`; train with batch 512, lr 2.8e-3, 10 loader workers; run
`utils/eval_batched.py` on the best checkpoint from the 0.1 deg grid (`$RUN_DIR/eval/`,
W&B keys `eval/*`) and from `val_names.txt` (`$RUN_DIR/eval_val/`, keys `eval_val/*`);
`crc-job-stats`. `srun` gets `--cpus-per-task` explicitly because Slurm 22.05+ does not
reliably propagate it and the loader workers would be pinned to one core. The job still
names new runs `resnet18_l40s_<job>`; rename them to the section 1 scheme when they finish.

**Sweep job `cluster/eval_sweep_l40s.slurm`.** 1 GPU, 16 CPUs, 64 GB, 3 h. Stages
`dark512`, runs `utils/eval_sweep.py` on `CKPT` with `--model-resolution $MODEL_RES` and
writes `<checkpoint dir>/sweep/`. Knobs: `CKPT` (required), `MODEL_RES` (512), `OUT_DIR`,
`GRID_STEP` (0.1), `CONDITIONS`, `EXTRA_ARGS`.

```bash
R=/ix1/kchen/evv/multireflection/runs
sbatch --export=ALL,CKPT=$R/r256_occ15-40batch_n00-20_e98_3867169/r256_occ15-40batch_n00-20_e98_3867169.pth,MODEL_RES=256 cluster/eval_sweep_l40s.slurm
```

**Runtime.** Staging 20 s (dark256) to 460 s (dark512 on a slow node). 512 px, batch 512:
about 235 s per epoch at 780 img/s, GPU at 100% and 43 of 46 GB, so batch 512 is the ceiling
and a larger batch would not help. 96 epochs took 6.3 h. 256 px: 52 s per epoch at
3500 img/s, 98 epochs in 1.4 h. Both eval stages together take under two minutes.

**Monitor.**

```bash
squeue -M gpu -u evv13
tail -f logs/mfl-resnet18-<jobid>.out       # epoch lines; .err holds tqdm and wandb output
crc-job-stats <jobid>                       # after it ends
```

## 5. Offline evaluation

**`utils/eval_batched.py`** replays the hardware sweep (`app/eval.py`, paper Algorithm 2)
on the image bank, no hardware:

1. Every start position on a grid (`--grid-step`, default 0.1) or from `--starts-file`
   (a list of image names, e.g. a run's `val_names.txt`) is one trajectory; all advance
   together.
2. At each step the image at the current position is looked up; SSIM against
   `x0.00_y0.00.jpg` is computed with skimage and rounded to 2 decimals, exactly as
   `app/inference.py:evaluate_position` does. SSIM at or above `--threshold` (0.97) stops
   the trajectory; `--max-adj` (10) caps it.
3. Otherwise the model predicts (x, y) in a GPU batch (fp16), the prediction is clipped to
   the tilt range and negated, the position is rounded to 0.01 and clamped to the bank.
4. Images and SSIM are memoized per position; SSIM runs in a thread pool.

Outputs in `--out-dir` (default `eval_results/<checkpoint stem>/`, gitignored):
`trace.csv` (every trajectory at every step, t=0 included, with the unrounded `ssim_raw`),
`eval.log` (the hardware log format, readable by `utils/graph_eval.py`), `summary.json`,
and three heatmaps (adjustments, final angular error, final SSIM). `--wandb` resumes the run
in `WANDB_RUN_ID` and writes `<--wandb-prefix>/*` summary keys plus the images. Full 0.1
grid: 2320 starts, about 25 s on the A2000. Time-to-align does not exist offline.

**Perturbations** (added 2026-09-10; with no flags the output is byte-identical to the
version before, verified by diff). They change the frames the model sees, using
`GpuAugment`'s definitions: `--brightness B` (per-trajectory factor in [1-B, 1+B]) or
`--brightness-fixed F`; `--contrast C` / `--contrast-fixed F`; `--noise S` with optional
`--noise-min` (sigma per trajectory, noise values redrawn every step); `--occlusion-count N`
with `--occlusion-min/max` (0.15/0.40) and `--occlusion-prob` (1.0). Brightness, contrast
and box geometry are drawn once per trajectory (seeded by `--perturb-seed` and the
trajectory index) and kept for all its steps, since dust does not move between motor moves.
Frames are quantized to 8 bit. The SSIM stop test uses the clean frame unless
`--perturb-ssim`, which disables memoization and computes SSIM per trajectory and step.

**`utils/eval_sweep.py`** runs 39 conditions on one checkpoint and writes `sweep.csv` /
`sweep.md` (`--conditions` selects a subset, `--wandb` logs a table): clean; noise sigma
0.02/0.05/0.10/0.15/0.20/0.30; brightness factor 0.4/0.6/0.8/1.2/1.4/1.6; contrast 0.9/1.1;
1 or 2 boxes of random edge 0.15-0.40 (`occlusion_1/2`); 1 or 2 boxes of fixed edge
10/20/30/40/50% (`occlusion_<n>x<edge>`, 1 to 25% of the area per box); 1x30, 2x30 and
2x50 boxes white (`_bright`), rotated in [-45, 45] deg (`_rot`) or both (`_rotbright`);
`occlusion_2_mixed` (2 boxes of 0.15-0.40, rotated in [-90, 90], white with p 0.5: the
training occlusion with every box applied); `combined` (the training augmentation before
2026-09-10) and `combined_harsh` (noise 0.2, brightness 0.6, 2 boxes of 30%). Eval flags:
`--occlusion-angle A`, `--occlusion-bright-prob P`, `--occlusion-bright-value V`; the new
draws come after the box geometry, so conditions without them are unchanged.
Seed 0, so every model gets the same per-start draws. Clean-frame SSIM stop test.

**Comparing input sizes.** `eval_batched.py --model-resolution N` perturbs the 512 px frame,
box-averages it by the integer factor 512/N (`avg_pool2d`, the same average as
`cv2.INTER_AREA`) and re-quantizes to 8 bit; the SSIM stop test stays on the clean 512 px
frame. Running every model on `dark512` this way gives identical frames, perturbations and
stop test. `utils/eval_compare.py LABEL=path/sweep.csv ...` merges the per-model tables.
Checked locally on the clean 0.1 grid: r256 on `dark512` with `--model-resolution 256` gives
100% / 1.02 / 0.012 deg against 100% / 1.03 / 0.011 on `dark256`; r64 gives 100% / 1.05 /
0.016 against 100% / 1.10 / 0.014 on `dark64` (remaining differences: JPEG re-encoding of
the small banks, and the stop test on the 512 frame instead of the small one).

Results of 2026-09-10: 8 models x 39 conditions, 2320 starts each on the 0.1 deg grid,
every model on `dark512` frames with `--model-resolution` (`cluster/eval_sweep_l40s.slurm`;
per-model tables in `runs/<name>/sweep/`, the ten rotated/bright rows for the first five
models in `runs/<name>/sweep_occ/`; merged locally in `eval_results/compare_sweep512_all.md`).
Each cell is success / mean adjustments (converged starts) / mean final angular error in deg
(all starts). The earlier 13-condition runs on each model's own bank agree with these rows.

| condition | r512_occ05-20img_n10_e96_3854472 | r512_ft_occ15-40img_n10_e49_3866737 | r512_occ15-40img_n00-20_e96_3866805 | r256_occ15-40batch_n00-20_e98_3867169 | r64_occ15-40batch_n00-20_e98_local | r256_occ15-40batch-rot90-bri50_n00-20_e98_3870306 | r128_occ15-40batch-rot90-bri50_n00-20_e98_3870308 | r64_occ15-40batch-rot90-bri50_n00-20_e98_3870310 |
|---|---|---|---|---|---|---|---|---|
| clean | 100% / 1.14 / 0.027 | 100% / 1.11 / 0.026 | 100% / 1.01 / 0.011 | 100% / 1.02 / 0.012 | 100% / 1.05 / 0.016 | 100% / 1.02 / 0.012 | 100% / 1.03 / 0.012 | 100% / 1.06 / 0.020 |
| noise_0.02 | 100% / 1.11 / 0.026 | 100% / 1.05 / 0.019 | 100% / 1.02 / 0.010 | 100% / 1.02 / 0.011 | 100% / 1.05 / 0.016 | 100% / 1.03 / 0.012 | 100% / 1.03 / 0.012 | 100% / 1.07 / 0.020 |
| noise_0.05 | 100% / 1.06 / 0.023 | 100% / 1.04 / 0.016 | 100% / 1.02 / 0.011 | 100% / 1.03 / 0.011 | 100% / 1.05 / 0.018 | 100% / 1.03 / 0.012 | 100% / 1.04 / 0.013 | 100% / 1.08 / 0.020 |
| noise_0.10 | 100% / 1.03 / 0.018 | 100% / 1.02 / 0.010 | 100% / 1.02 / 0.012 | 100% / 1.03 / 0.013 | 100% / 1.08 / 0.021 | 100% / 1.04 / 0.014 | 100% / 1.04 / 0.014 | 100% / 1.09 / 0.022 |
| noise_0.15 | 100% / 1.48 / 0.030 | 100% / 1.12 / 0.022 | 100% / 1.03 / 0.013 | 100% / 1.06 / 0.018 | 100% / 1.11 / 0.024 | 100% / 1.05 / 0.015 | 100% / 1.05 / 0.015 | 100% / 1.11 / 0.023 |
| noise_0.20 | 100% / 2.13 / 0.046 | 100% / 1.72 / 0.038 | 100% / 1.04 / 0.015 | 100% / 1.13 / 0.024 | 100% / 1.15 / 0.025 | 100% / 1.06 / 0.017 | 100% / 1.06 / 0.017 | 100% / 1.13 / 0.026 |
| noise_0.30 | 17.80% / 4.82 / 1.271 | 0.52% / 1.42 / 2.902 | 100% / 1.11 / 0.025 | 100% / 1.40 / 0.025 | 100% / 1.31 / 0.034 | 100% / 1.14 / 0.025 | 100% / 1.10 / 0.020 | 100% / 1.21 / 0.035 |
| brightness_0.4 | 33.41% / 1.75 / 0.141 | 12.93% / 1.11 / 0.204 | 100% / 1.47 / 0.021 | 100% / 1.23 / 0.026 | 100% / 1.24 / 0.030 | 100% / 1.29 / 0.023 | 100% / 1.23 / 0.024 | 100% / 1.51 / 0.043 |
| brightness_0.6 | 100% / 1.42 / 0.037 | 100% / 1.98 / 0.055 | 100% / 1.02 / 0.012 | 100% / 1.03 / 0.014 | 100% / 1.07 / 0.019 | 100% / 1.04 / 0.017 | 100% / 1.05 / 0.015 | 100% / 1.13 / 0.026 |
| brightness_0.8 | 100% / 1.22 / 0.028 | 100% / 1.22 / 0.033 | 100% / 1.02 / 0.011 | 100% / 1.02 / 0.012 | 100% / 1.05 / 0.016 | 100% / 1.03 / 0.012 | 100% / 1.03 / 0.012 | 100% / 1.07 / 0.021 |
| brightness_1.2 | 100% / 1.11 / 0.027 | 100% / 1.07 / 0.022 | 100% / 1.02 / 0.011 | 100% / 1.02 / 0.012 | 100% / 1.05 / 0.016 | 100% / 1.02 / 0.013 | 100% / 1.03 / 0.012 | 100% / 1.06 / 0.020 |
| brightness_1.4 | 100% / 1.09 / 0.026 | 100% / 1.06 / 0.021 | 100% / 1.02 / 0.012 | 100% / 1.02 / 0.013 | 100% / 1.05 / 0.018 | 100% / 1.03 / 0.013 | 100% / 1.03 / 0.013 | 100% / 1.07 / 0.020 |
| brightness_1.6 | 100% / 1.14 / 0.027 | 100% / 1.07 / 0.021 | 100% / 1.03 / 0.016 | 100% / 1.03 / 0.015 | 100% / 1.07 / 0.021 | 100% / 1.03 / 0.015 | 100% / 1.04 / 0.015 | 100% / 1.08 / 0.022 |
| contrast_0.9 | 100% / 1.09 / 0.024 | 100% / 1.10 / 0.025 | 100% / 1.02 / 0.011 | 100% / 1.02 / 0.011 | 100% / 1.05 / 0.016 | 100% / 1.02 / 0.012 | 100% / 1.03 / 0.012 | 100% / 1.07 / 0.021 |
| contrast_1.1 | 100% / 1.20 / 0.028 | 100% / 1.11 / 0.026 | 100% / 1.02 / 0.011 | 100% / 1.02 / 0.011 | 100% / 1.05 / 0.016 | 100% / 1.02 / 0.013 | 100% / 1.03 / 0.012 | 100% / 1.06 / 0.020 |
| occlusion_1 | 93.02% / 1.34 / 0.042 | 99.66% / 1.16 / 0.030 | 100% / 1.03 / 0.016 | 100% / 1.03 / 0.016 | 100% / 1.08 / 0.020 | 100% / 1.06 / 0.017 | 100% / 1.06 / 0.016 | 100% / 1.12 / 0.024 |
| occlusion_2 | 82.11% / 1.54 / 0.065 | 98.32% / 1.23 / 0.036 | 100% / 1.05 / 0.020 | 100% / 1.06 / 0.019 | 100% / 1.13 / 0.024 | 99.87% / 1.10 / 0.021 | 100% / 1.08 / 0.019 | 99.87% / 1.19 / 0.028 |
| occlusion_1x10 | 100% / 1.15 / 0.028 | 100% / 1.12 / 0.026 | 100% / 1.02 / 0.012 | 100% / 1.02 / 0.012 | 100% / 1.06 / 0.017 | 100% / 1.03 / 0.013 | 100% / 1.03 / 0.013 | 100% / 1.08 / 0.021 |
| occlusion_1x20 | 99.74% / 1.21 / 0.031 | 100% / 1.13 / 0.028 | 100% / 1.02 / 0.014 | 100% / 1.03 / 0.014 | 100% / 1.07 / 0.018 | 100% / 1.04 / 0.015 | 100% / 1.04 / 0.014 | 100% / 1.09 / 0.023 |
| occlusion_1x30 | 92.59% / 1.46 / 0.043 | 99.57% / 1.18 / 0.031 | 100% / 1.03 / 0.016 | 100% / 1.04 / 0.016 | 100% / 1.08 / 0.021 | 100% / 1.06 / 0.018 | 100% / 1.06 / 0.016 | 100% / 1.13 / 0.024 |
| occlusion_1x40 | 66.94% / 1.60 / 0.099 | 96.03% / 1.28 / 0.041 | 100% / 1.06 / 0.020 | 100% / 1.07 / 0.021 | 100% / 1.14 / 0.024 | 99.96% / 1.11 / 0.023 | 100% / 1.09 / 0.019 | 99.83% / 1.20 / 0.028 |
| occlusion_1x50 | 43.53% / 1.91 / 0.218 | 76.94% / 1.39 / 0.084 | 95.95% / 1.20 / 0.034 | 98.71% / 1.17 / 0.028 | 99.31% / 1.26 / 0.029 | 97.07% / 1.21 / 0.032 | 100% / 1.19 / 0.024 | 98.53% / 1.42 / 0.033 |
| occlusion_2x10 | 99.96% / 1.15 / 0.028 | 100% / 1.12 / 0.027 | 100% / 1.02 / 0.012 | 100% / 1.03 / 0.013 | 100% / 1.07 / 0.018 | 100% / 1.03 / 0.014 | 100% / 1.04 / 0.014 | 100% / 1.09 / 0.022 |
| occlusion_2x20 | 98.75% / 1.30 / 0.034 | 100% / 1.15 / 0.030 | 100% / 1.03 / 0.016 | 100% / 1.04 / 0.016 | 100% / 1.09 / 0.021 | 100% / 1.06 / 0.018 | 100% / 1.05 / 0.016 | 100% / 1.13 / 0.025 |
| occlusion_2x30 | 77.89% / 1.73 / 0.071 | 98.02% / 1.25 / 0.038 | 99.91% / 1.06 / 0.021 | 100% / 1.07 / 0.020 | 100% / 1.15 / 0.025 | 99.96% / 1.11 / 0.023 | 100% / 1.10 / 0.020 | 99.96% / 1.22 / 0.028 |
| occlusion_2x40 | 40.00% / 1.98 / 0.192 | 81.64% / 1.43 / 0.073 | 98.02% / 1.20 / 0.032 | 98.66% / 1.19 / 0.029 | 99.48% / 1.30 / 0.031 | 97.16% / 1.27 / 0.033 | 99.66% / 1.22 / 0.026 | 97.72% / 1.40 / 0.038 |
| occlusion_2x50 | 21.12% / 2.66 / 0.379 | 47.46% / 1.76 / 0.193 | 78.41% / 1.54 / 0.088 | 87.11% / 1.44 / 0.064 | 92.16% / 1.55 / 0.051 | 82.41% / 1.52 / 0.076 | 90.99% / 1.49 / 0.049 | 85.22% / 1.79 / 0.079 |
| combined | 89.57% / 1.44 / 0.050 | 100% / 1.06 / 0.019 | 99.91% / 1.07 / 0.020 | 100% / 1.08 / 0.021 | 99.96% / 1.19 / 0.027 | 99.96% / 1.12 / 0.023 | 100% / 1.11 / 0.021 | 99.27% / 1.26 / 0.031 |
| combined_harsh | 20.00% / 3.88 / 0.253 | 45.43% / 3.26 / 0.167 | 99.91% / 1.20 / 0.029 | 99.91% / 1.38 / 0.032 | 83.41% / 1.66 / 0.067 | 99.74% / 1.38 / 0.031 | 99.91% / 1.27 / 0.030 | 63.75% / 1.52 / 0.101 |
| occlusion_1x30_bright | 0.69% / 2.25 / 1.216 | 0.69% / 2.06 / 1.370 | 1.25% / 3.10 / 1.226 | 0.47% / 1.36 / 1.229 | 7.63% / 3.93 / 1.191 | 98.97% / 1.09 / 0.023 | 99.96% / 1.08 / 0.019 | 99.01% / 1.24 / 0.032 |
| occlusion_2x30_bright | 0.30% / 2.57 / 1.677 | 0.56% / 2.08 / 1.811 | 0.78% / 3.56 / 1.659 | 0.43% / 3.90 / 1.829 | 1.64% / 3.92 / 1.848 | 96.85% / 1.21 / 0.034 | 98.66% / 1.16 / 0.027 | 88.79% / 1.46 / 0.054 |
| occlusion_2x50_bright | 0.30% / 2.14 / 2.504 | 0.43% / 3.10 / 2.448 | 0.43% / 2.60 / 2.317 | 0.30% / 2.71 / 2.390 | 0.60% / 1.86 / 2.439 | 31.68% / 2.53 / 0.330 | 70.34% / 1.55 / 0.201 | 15.82% / 3.46 / 0.963 |
| occlusion_1x30_rot | 92.80% / 1.49 / 0.043 | 99.18% / 1.21 / 0.032 | 100% / 1.05 / 0.017 | 100% / 1.04 / 0.016 | 100% / 1.09 / 0.021 | 100% / 1.05 / 0.018 | 100% / 1.06 / 0.016 | 100% / 1.12 / 0.025 |
| occlusion_2x30_rot | 77.41% / 1.73 / 0.071 | 97.16% / 1.31 / 0.040 | 99.66% / 1.12 / 0.023 | 100% / 1.08 / 0.021 | 100% / 1.16 / 0.025 | 99.91% / 1.10 / 0.023 | 100% / 1.08 / 0.021 | 99.78% / 1.20 / 0.028 |
| occlusion_2x50_rot | 21.85% / 2.53 / 0.346 | 49.35% / 1.95 / 0.182 | 77.24% / 1.64 / 0.087 | 86.90% / 1.48 / 0.064 | 92.16% / 1.61 / 0.049 | 83.23% / 1.50 / 0.067 | 92.16% / 1.45 / 0.045 | 86.64% / 1.75 / 0.068 |
| occlusion_1x30_rotbright | 0.34% / 2.00 / 1.437 | 0.82% / 2.58 / 1.372 | 0.73% / 3.53 / 1.522 | 0.69% / 2.69 / 1.380 | 5.22% / 4.18 / 1.329 | 99.96% / 1.07 / 0.020 | 99.96% / 1.06 / 0.018 | 99.35% / 1.20 / 0.030 |
| occlusion_2x30_rotbright | 0.26% / 2.00 / 1.901 | 0.52% / 4.08 / 1.857 | 0.43% / 3.60 / 2.002 | 0.43% / 3.10 / 1.897 | 1.16% / 3.00 / 1.945 | 99.35% / 1.16 / 0.028 | 99.70% / 1.13 / 0.024 | 94.05% / 1.38 / 0.043 |
| occlusion_2x50_rotbright | 0.22% / 1.80 / 2.391 | 0.34% / 2.62 / 2.417 | 0.13% / 2.33 / 2.438 | 0.09% / 0.50 / 2.399 | 0.65% / 2.73 / 2.456 | 34.48% / 2.55 / 0.276 | 74.18% / 1.53 / 0.114 | 20.43% / 3.48 / 0.707 |
| occlusion_2_mixed | 21.51% / 1.58 / 1.107 | 25.26% / 1.28 / 1.055 | 26.12% / 1.20 / 1.170 | 26.25% / 1.14 / 0.972 | 30.82% / 1.59 / 1.032 | 99.35% / 1.12 / 0.023 | 99.87% / 1.09 / 0.021 | 97.93% / 1.25 / 0.035 |

Findings:

- The 512 px failures come from the recipe, not the resolution. r512_occ15-40img_n00-20
  (noise range, 0.15-0.40 boxes) stays at 100% for noise 0.30, brightness 0.4 and
  combined_harsh, like the 256 px run, and has the lowest clean error (0.011 deg); it trails
  256 only on 50% boxes (1 box 96.0 vs 98.7%, 2 boxes 78.4 vs 87.1%). The fixed-noise
  runs (3854472 and its fine-tune) fail at noise 0.30 and brightness 0.4.
- Rotation alone changes nothing: rotated boxes score within about 1% of the same boxes
  axis-aligned, for every model.
- Without bright boxes in training every model fails glare: at most 7.6% success with one
  30% bright box, final errors 1.2 to 2.5 deg.
- The rot90-bri50 runs handle it: one or two 30% bright boxes 89 to 100%, occlusion_2_mixed
  98 to 100%, with dark-box, noise and brightness rows at the level of the earlier 256 px
  run. At 64 px the cost is visible: combined_harsh 63.8% (earlier 64 px run 83.4%), clean
  error 0.020 deg (0.016).
- Among the rot90-bri50 runs, 128 px is best or tied on nearly every row: clean 0.012 deg as
  at 256, noise 0.30 1.10 adjustments / 0.020 deg, 1 box of 50% 100%, 2 boxes of 50% 91.0%
  (256: 82.4%, 64: 85.2%), 2 bright 50% boxes 70.3% (256: 31.7%, 64: 15.8%). One seed each.
- Two 50% boxes, bright or rotated-bright, remain the failure case for every model.

With
`--perturb-ssim`, noise sigma 0.1 alone pins the SSIM near 0.1, so the stop test never
fires although the final error stays small: SSIM is too noise-sensitive to serve as a
convergence test on noisy frames, which is relevant for the hardware too.

## 6. Known issues and open questions

- The `WANDB_API_KEY is not set` warning is printed whenever the variable is absent, even
  though netrc auth works. Cosmetic.
- `config.py`: `INFERENCE_MODEL_TYPE = "SimpleFC"` while the file name is a ResNet-18
  checkpoint; `SIMILARITY_INDEX_THRESHOLD = 0.95` while the paper, README and both eval
  scripts use 0.97. Neither is used by training or the offline eval.
- `GpuAugment` applies brightness and contrast in random order; the eval perturbation uses a
  fixed order (brightness, contrast, noise, occlusion) and quantizes to 8 bit. Small,
  documented differences.
- The 3854472 checkpoint predates the per-batch occlusion, the noise range and the 98-epoch
  default. Section 3 says how to reproduce its recipe.
- All eval starts are seen positions (section 2). There is no held-out set off the
  training grid.
- The 256, 128 and 64 px models have been evaluated only offline. `TiltPredictor` takes
  any of them: it reads the input size from the `r<res>_` checkpoint name (or
  `config.py:INFERENCE_INPUT_RESOLUTION`) and area-resizes the 512 px processed frame to it;
  set `INFERENCE_MODEL_TYPE = "ResNet18"` and the file name in `config.py` to deploy one.
- One training seed per recipe. Differences of a few percent on the hardest occlusion
  conditions may be run-to-run variation.
- `--compile` is untested on the L40S.
- `torch.nn.functional.interpolate(mode="area")` (adaptive average pooling) on CUDA, torch
  2.14, returns the first image for every row when the batch was built as
  `torch.from_numpy(np.stack(images)[:, None])` (channel stride 0; `.contiguous()` and
  `.clone()` keep it). `avg_pool2d` is correct. Use it, or check per-image outputs, before
  any batched area resize.
- `utils/graph_eval.py` writes to `./graphs/eval/<log path minus .log>...png` relative to
  the working directory, so run it from inside the results directory or the nested path
  fails. `eval_batched.py` writes its own heatmaps, so this only matters for the hardware
  log format.
- SSIM as a stop criterion is fragile under noise (section 5).

## 7. Next steps

- Resolution: with the rotated and bright box recipe, 128 px is best or tied on the sweep
  (section 5) with a quarter of 256's pixels. Repeat 128 and 256 with another seed before
  deciding, then check inference time and RAM on the Pi (the README's int8 plan is the other
  lever there).
- Two bright boxes of 50% edge remain the failure case (16 to 74% success).
- Fine-tuning used the full lr 2.8e-3 with the same restart schedule; a lower lr
  (`EXTRA_ARGS="--lr 0.001"`) is untried.
- If runs need to be faster at 512 px, the L40S is compute-bound at batch 512, so the
  options are `--compile` (test on the L40S first) or DDP over the node's 4 GPUs.
- A held-out evaluation set that is not on the collection grid (a fresh collection, or a
  different day's images) is the only way to measure generalisation; the current numbers
  cannot.
