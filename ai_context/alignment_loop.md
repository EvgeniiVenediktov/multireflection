# Deployed alignment loop and evaluation

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
