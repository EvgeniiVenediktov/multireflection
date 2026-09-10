# Deployed alignment loop and evaluation

For the offline evaluation and robustness sweep, `train/TRAINING_AND_EVALS.md` is the
current summary; this file holds per-module detail.

All of `app/` runs **on the Raspberry Pi 4** and imports `mf_control.controller.MFController`
(submodule, not checked out here). Nothing in `app/` is runnable on a dev machine without hardware.

## Closed loop - `app/autoalignment.py`

```
load reference image(s) from OPTIMUM_IMAGE_PATH_LIST
connect MFController(image_size=(1920, 1440))
load TiltPredictor(INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE)
loop:
    img  = process_image_from_webcam(controller.capture_image(), 512x512, grayscale)
    ssim = evaluate_position(img, references)
    if ssim >= SIMILARITY_INDEX_THRESHOLD: done
    dx, dy = model.predict([img])[0]              # de-normalized degrees
    x = clip(controller.get_x_tilt() - dx, ...)   # move by the negated prediction
    y = clip(controller.get_y_tilt() - dy, ...)
    controller.set_tilt_x(x); controller.set_tilt_y(y)
```
On any exception the mount is returned to (0, 0) and the controller closed. Positions are appended
to `autoalignment.log`. Converges in 1-2 iterations in practice (paper Alg. 1).

## Manual / stepwise variant - `app/real_life_prediction.py`

Same pipeline but shows the processed frame and asks for confirmation before each move. Use this
for debugging a checkpoint on hardware.

## Evaluation sweep - `app/eval.py`

Paper Alg. 2. Builds a grid of start positions (`EVAL_GRID_STEP = 0.5` deg, clamped to x<=4.0,
y<=2.4), drives the mount to each, then runs the alignment loop with a cap of
`EVAL_MAX_ADJ_NUMBER = 10` adjustments. One `logging.info` line per iteration:

```
origin_x:..,origin_y:..,adj_n:..,pred_x:..,pred_y:..,pos_x:..,pos_y:..,sim_index:..
```

Reported numbers are averaged over five complete runs.

## Offline batched sweep - `utils/eval_batched.py`

Replays the sweep above on the image bank (`/mnt/h/dark512` locally, the staged copy on the
cluster) with no hardware: the folder is the position -> image lookup, every start on a
grid (`--grid-step`, default 0.1) is advanced simultaneously, the model runs in GPU batches,
SSIM is memoized per position in a thread pool. Same update rule as `app/eval.py`
(negated clipped prediction, round to 0.01, cap `--max-adj` 10), threshold 0.97 by
default, SSIM rounded to 2 decimals like `evaluate_position`. Outputs in
`eval_results/<ckpt>/` (gitignored): `trace.csv` (every start at every step t, t=0
included, with `ssim_raw` unrounded), `eval.log` (hardware format, `graph_eval.py` reads
it), `summary.json`, `heatmap_{adjustments,angular_error,final_ssim}.png`. `--starts-file` replaces the grid with the positions listed in a file of image names, e.g.
a run's `val_names.txt`, which is the only held-out evaluation available. `--wandb`
resumes the run in `WANDB_RUN_ID` and writes `<prefix>/*` summary keys plus the images
(`--wandb-prefix`, default `eval`); the L40S job runs both the grid (`eval/`) and the
validation set (`eval_val/`). Starts are training positions, and time-to-align does not exist here.

Result for `resnet18_l40s_3854472` (2026-09-10): 2320 starts, 100% converged,
1.14 +- 0.35 adjustments, final SSIM 0.984 +- 0.007, angular error 0.027 +- 0.015 deg.

### Perturbation evals

`eval_batched.py` can perturb the frames the model sees (same definitions and units as
`GpuAugment`): `--brightness B` / `--brightness-fixed F`, `--contrast C` /
`--contrast-fixed F`, `--noise S` (+ `--noise-min`), `--occlusion-count N` with
`--occlusion-min/max/prob`, `--perturb-seed`. Brightness, contrast and boxes are drawn once
per trajectory and kept for all its steps; noise is redrawn per step; frames are quantized
to 8 bit. The SSIM stop test uses the clean frame unless `--perturb-ssim` (then SSIM is
per (trajectory, step), not memoized; note SSIM is so noise-sensitive that noise 0.1 alone
keeps it near 0.1, so the loop never "converges" even though the final error is small).
`utils/eval_sweep.py` runs 13 conditions (clean, noise 0.05/0.1/0.2, brightness
0.6/0.8/1.2/1.4, contrast 0.9/1.1, occlusion 1/2 boxes, combined) and writes
`sweep.csv` / `sweep.md`; `--conditions` selects a subset, `--wandb` logs a table.

Sweep of `resnet18_l40s_3854472` (2026-09-10, 0.1 grid): noise up to 0.1, any brightness
0.6 to 1.4 and contrast 0.9 to 1.1 stay at 100% success; noise 0.2 doubles the adjustments;
occlusion is what breaks the loop: 93% success with one box, 82% with two, 89% for the
combined training-style augmentation.

## Log analysis - `utils/graph_eval.py`

`python utils/graph_eval.py [eval.log] [threshold=0.97]`. Groups lines by origin, finds the first
iteration whose `sim_index` crosses the threshold, and derives per-origin: number of adjustments,
wall-clock time from first to last adjustment (from log timestamps), final SSIM, and final angular
error (distance of the final position from zero). Produces the interpolated heatmaps in
`graphs/Evaluation/`.

## Relevant `config.py` knobs

`SIMILARITY_INDEX_THRESHOLD`, `OPTIMUM_IMAGE_PATH_LIST` (absolute Pi paths),
`INFERENCE_MODEL_FILE_NAME`, `INFERENCE_MODEL_TYPE`, `X/Y_TILT_START/STOP` (system range -
the physical actuation limits; the separate `DATA_COLLECTION_X/Y_TILT_STOP` only affect dataset
collection and default to the system values),
`DATA_COLLECTION_FINAL_RESOLUTION`, `DATA_COLLECTION_CVT_TO_GRAYSCALE`,
`EVAL_GRID_STEP`, `EVAL_MAX_ADJ_NUMBER`.

Small inconsistencies exist between these values and the published ones - see [main.md](main.md).
