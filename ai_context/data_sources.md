# Data sources

Two independent data origins: **Zemax-simulated** images (early feasibility) and **real camera**
images (everything deployed). A third, fully synthetic source exists only inside the RL branch
(GPU ray tracer, see [simulation.md](simulation.md)).

## 1. Real camera dataset (primary)

**Collection** - `data_process/collect_real_data.py`
- Drives `MFController` (submodule `mf_control`) over a raster of mirror tilts and saves one
  image per tilt. Serpentine sweep in Y to avoid backlash-free retrace time.
- Ranges/step from `config.py`: `X_TILT_START`/`Y_TILT_START` plus
  `DATA_COLLECTION_X_TILT_STOP`/`DATA_COLLECTION_Y_TILT_STOP` - the data-collection range,
  which defaults to the system `X/Y_TILT_STOP` and can be overridden to sweep a different span,
  `REAL_DATA_COLLECTION_STEP = 0.01` deg, `REAL_DATA_COLLECTION_DELAY = 0.1` s settle.
- Paper's collected range: X in [-2, +3.7] deg, Y in [-2, +2] deg, 0.01 deg step -> **228,000 images**.
- **Label is the filename**: `x{x:.2f}_y{y:.2f}.jpg`. There is no separate label file anywhere in
  the pipeline; every dataset re-parses the name (`imname_to_target`, defined redundantly in
  `data_process/prepare_lmdb.py:54`, `train/cnn_train.py:34`, `train/experiments/wandb_sweep.py`).
- Positions logged to `data_collection_log.txt`.

Dataset variants live in sibling directories / key files by lighting condition:
`dark512` (the one used), `color_light`, `color_mainlight`, `newdirty`. Key files carry a suffix
per variant (`-light.jpg`, `-mainlight.jpg`, `-newdirty.jpg`) written by
`utils/record_filenames.py` and the `*_postfix_keyproc` helpers in `prepare_lmdb.py:332-340`.

**Preprocessing** - `data_process/preprocess_images.py`
`process_image_from_webcam` (`:26`) is the single function shared by collection, live inference and
evaluation, so train/serve preprocessing cannot drift:
1. BGR -> grayscale (`DATA_COLLECTION_CVT_TO_GRAYSCALE`)
2. center-crop to a square of the short side (`to_square`, `:8`)
3. resize to `DATA_COLLECTION_FINAL_RESOLUTION` = 512x512
4. circular mask - zero everything outside the inscribed circle (`apply_circular_mask`, `:49`)
5. Gaussian blur 7x7

`process_single_image` (`:65`) is the LMDB-build variant: resize + circular mask only (no
grayscale conversion, no blur - the stored JPEGs are already grayscale from collection).
Sanity test: `data_process/preprocess_images_test.py`.

**Packing** - `data_process/prepare_lmdb.py`
- `create_lmdb_from_images` (`:65`) reads a folder, runs `process_single_image`, and stores each
  image under its filename key. Two payload formats:
  - **tensor** (current): `torch.save` of a `(1,H,W)` float32 tensor normalized to [0,1].
  - **msgpack** of a uint8 list (legacy; `LMDB_USE_COMPRESSION` / lz4 path).
  Read back with `lmdb_bytes_to_torch_tensor` (`:17`).
- Key list is written alongside the LMDB (`keys.txt`, `dark512.txt`, ...).
- Subsampling by step is done at key level, not by re-collecting:
  `filter_002step` / `filter_004step` / `filter_016step` (`:285-330`) keep tilts whose
  hundredths are divisible by 2/4/16 -> 0.02/0.04/0.16 deg grids.
- `write_split_keys` (`:171`) does a random 80/20 train/val split and writes
  `*_train.txt` / `*_val.txt` **inside the LMDB directory**. The dataset classes read these by name.
- `copy_selected_entries` (`:197`) migrates/reformats between LMDBs (msgpack -> tensor).
- The `__main__` block is a scratchpad of one-off invocations (mostly commented out) plus a
  missing-key diff between a source folder and a destination LMDB. Treat it as history, not API.

Known LMDB paths (external drives, not in repo): `/mnt/h/dark512` (jpg source),
`/mnt/h/black_512_0_001step_tensor.lmdb` (current), `/mnt/e/real_512_0_001step.lmdb`,
`/mnt/e/color.lmdb`. Entry counts recorded in comments: dark 228,000; +light 456,000;
+mainlight 513,200.

## 2. Zemax-simulated dataset (feasibility stage only)

`data_process/generate_simulated_data.py` - Windows-only, drives OpticStudio through the ZOS-API
(`pythonnet`). Tilts the entrance mirror over a grid, runs a non-sequential ray trace, reads the
detector, saves `x{..}_y{..}.jpg` with the same filename-as-label convention.
Produced the 60,000 128x128 images behind paper SS II-B. Not used by the deployed model.

## 3. How data reaches the model

`LMDB key -> label`: parse filename -> normalize to [0,1] with
`x = (x + 2) / 5.7`, `y = (y + 2) / 4` (i.e. X in [-2, 3.7], Y in [-2, 2]).
Done in `train/cnn_train.py:118-127` (in-memory dataset), `:266-271` (lazy dataset) and
`train/experiments/wandb_sweep.py`. Inference reverses it in `app/inference.py:TiltPredictor.predict`
using the system `X_TILT_START/STOP` from `config.py`, while collection uses
`DATA_COLLECTION_X/Y_TILT_STOP`.
The hardcoded 5.7 and 4 correspond to the system range X in [-2, 3.7], Y in [-2, 2]. Changing
`X_TILT_START/STOP` or `Y_TILT_START/STOP` invalidates existing checkpoints, since
`app/inference.py` de-normalizes predictions with those same values.
Details of the Dataset/DataLoader classes and augmentation: [training.md](training.md).

## 4. Evaluation data

Evaluation is **online on hardware**, not on a held-out file set - `app/eval.py`:
- Grid of start positions spaced `EVAL_GRID_STEP = 0.5` deg over the actuation range.
- From each start, run the alignment loop up to `EVAL_MAX_ADJ_NUMBER = 10` adjustments.
- Stop condition: SSIM against `OPTIMUM_IMAGE_PATH_LIST` reference image(s) >= threshold.
- Every iteration appends a CSV-ish line to `eval.log`:
  `origin_x, origin_y, adj_n, pred_x, pred_y, pos_x, pos_y, sim_index`.
- `utils/graph_eval.py` parses `eval.log` and produces the heatmaps/statistics in
  `graphs/Evaluation/` (adjustment count per start position, time to align, final SSIM,
  final angular error). Threshold defaults to 0.97 there.

See [alignment_loop.md](alignment_loop.md) for the loop itself.
