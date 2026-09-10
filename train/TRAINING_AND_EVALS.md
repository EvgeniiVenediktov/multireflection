# Training and evaluation handoff

Written 2026-09-10. The current summary of how the ResNet-18 tilt regressor is trained,
evaluated offline, and run on the Pitt CRCD cluster. Scripts and `git log` are authoritative
where prose disagrees. Per-module detail lives in `ai_context/training.md` and
`ai_context/alignment_loop.md`; this file is the entry point.

## 1. Purpose and status

The model takes a 512x512 (or 256x256) grayscale camera image of the Herriott cell spot
pattern and regresses the mirror tilt (x, y) in degrees. The deployed loop moves the mirror
by the negated prediction and stops when the SSIM against the reference image reaches a
threshold. Training is supervised regression on the collected image bank.

**Current best 512 px checkpoint:** `resnet18_l40s_3854472_best_model.pth`, CRCD job
3854472, 2026-09-09, W&B run
[resnet18_l40s_3854472](https://wandb.ai/e-venediktov-university-of-pittsburgh/multireflection/runs/93o4zgri).
Best epoch 93 of 96, val MSE 6.5e-5 on normalized labels. Offline sweep: 100% converged,
1.14 adjustments, angular error 0.027 deg. Recipe in README "Current best checkpoint";
it predates three default changes listed in section 3.

**Runs on 2026-09-10** (all in `/ix1/kchen/evv/multireflection/runs/<job>/`):

| job | what | status at 17:25 EDT | best val MSE | offline eval (grid 0.1) |
|---|---|---|---|---|
| 3866737 | fine-tune of 3854472, 49 epochs, cutout 0.15-0.40, per-image boxes, fixed noise 0.1 | finished | 7.0e-5 | 100%, 1.11 adj, 0.026 deg |
| 3866805 | from scratch, 96 epochs, noise sigma per image in [0, 0.2], per-image boxes | epoch 57 of 96 | 3.3e-5 so far | pending |
| 3867169 | from scratch on `dark256` at 256 px, 98 epochs, all current defaults | finished | 2.8e-5 | 100%, 1.03 adj, 0.011 deg; val set (45,760 starts): same |

The 256 px run is the standout: 4.5x the throughput (3496 vs 780 img/s), lower val loss,
and less than half the angular error of the 512 px best. It has not been checked on
hardware, and its eval ran against the 256 px bank, so its SSIM values are not directly
comparable to the 512 px ones. Job 3866737 has no `eval_val/`; it was submitted before that
stage existed.

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
next to it). `data/dark256.tar.gz`, 1.85 GB, 228,800 files. No local copy.

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
| occlusion (cutout) | 2 boxes, each with p 0.5, edges 0.15 to 0.40 of the image, fill 0, drawn once per batch and applied to every image in it | `--occlusion-*` | boxes per batch and 0.15-0.40 since 2026-09-10 (was per image, 0.05-0.20) |
| affine | off | `--affine-*` | never on; a translation resembles a different tilt |

Which run used what: 3854472 = fixed noise 0.1, per-image boxes 0.05-0.20, 96 epochs.
3866737 = fixed noise 0.1, per-image boxes 0.15-0.40, 49 epochs. 3866805 = noise range
[0, 0.2], per-image boxes 0.15-0.40, 96 epochs. 3867169 = all current defaults, 98 epochs.
A rerun of 3854472 with today's defaults is a different recipe; reproduce it with
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
sbatch --export=ALL,START_CKPT=/ix1/kchen/evv/multireflection/runs/3854472/resnet18_l40s_3854472_best_model.pth,EPOCHS=49 cluster/train_l40s.slurm
sbatch --export=ALL,DATASET=dark256,EXTRA_ARGS="--resolution 256" cluster/train_l40s.slurm
sbatch --export=ALL,EXTRA_ARGS="--noise-min 0.0 --noise 0.2 --lr 0.001" cluster/train_l40s.slurm
```

End to end: `module purge`; print torch and GPU info; copy the archive to `$SLURM_SCRATCH`
(node-local NVMe, wiped at job end) and extract; export `WANDB_DIR=$RUN_DIR` and
`WANDB_RUN_ID=l40s-<jobid>`; train with batch 512, lr 2.8e-3, 10 loader workers; run
`utils/eval_batched.py` on the best checkpoint from the 0.1 deg grid (`$RUN_DIR/eval/`,
W&B keys `eval/*`) and from `val_names.txt` (`$RUN_DIR/eval_val/`, keys `eval_val/*`);
`crc-job-stats`. `srun` gets `--cpus-per-task` explicitly because Slurm 22.05+ does not
reliably propagate it and the loader workers would be pinned to one core.

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

**`utils/eval_sweep.py`** runs 13 conditions on one checkpoint and writes `sweep.csv` /
`sweep.md` (`--conditions` selects a subset, `--wandb` logs a table). Result for
`resnet18_l40s_3854472`, 2320 starts each, clean-frame SSIM stop test:

| condition | success | adjustments | angular error (deg) |
|---|---|---|---|
| clean | 100% | 1.14 | 0.027 |
| noise 0.05 / 0.10 / 0.20 | 100% / 100% / 100% | 1.05 / 1.03 / 2.14 | 0.023 / 0.018 / 0.047 |
| brightness 0.6 / 0.8 / 1.2 / 1.4 | 100% | 1.42 / 1.22 / 1.11 / 1.09 | 0.037 / 0.028 / 0.027 / 0.026 |
| contrast 0.9 / 1.1 | 100% | 1.09 / 1.20 | 0.024 / 0.028 |
| occlusion 1 box / 2 boxes | 93.0% / 82.1% | 1.34 / 1.53 (max 9) | 0.042 / 0.065 (max 0.79) |
| combined (training augmentation at eval) | 89.4% | 1.47 (max 10) | 0.050 |

Occlusion is what breaks the loop; light noise, brightness and contrast are absorbed. With
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
- The 256 px result has been evaluated only against the 256 px bank and only offline.
  `TiltPredictor` on the Pi still preprocesses to 512; deploying a 256 model needs the
  resize in `config.py` / `app/` changed to match.
- `--compile` is untested on the L40S.
- `utils/graph_eval.py` writes to `./graphs/eval/<log path minus .log>...png` relative to
  the working directory, so run it from inside the results directory or the nested path
  fails. `eval_batched.py` writes its own heatmaps, so this only matters for the hardware
  log format.
- SSIM as a stop criterion is fragile under noise (section 5).

## 7. Next steps

- When 3866805 finishes, compare the four runs on `eval/*` and `eval_val/*` in W&B and
  in `runs/<job>/eval*/summary.json`: does the noise range help, does per-batch occlusion
  (3867169 only, confounded with 256 px) help. Run `utils/eval_sweep.py` on each checkpoint
  for the robustness table; it is the metric that separates them, since clean-frame numbers
  are all near 100%.
- Decide on 256 px. If the offline gain holds up, retrain at 256 with the 3854472 recipe
  for a clean A/B, then check inference time and RAM on the Pi, which is where 256 should
  pay off most (the README's int8 plan is the other lever there).
- Fine-tuning used the full lr 2.8e-3 with the same restart schedule; a lower lr
  (`EXTRA_ARGS="--lr 0.001"`) is untried.
- If runs need to be faster at 512 px, the L40S is compute-bound at batch 512, so the
  options are `--compile` (test on the L40S first) or DDP over the node's 4 GPUs.
- A held-out evaluation set that is not on the collection grid (a fresh collection, or a
  different day's images) is the only way to measure generalisation; the current numbers
  cannot.
