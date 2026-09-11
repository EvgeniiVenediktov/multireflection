# Spatial hold-out split

A random 80/20 split leaks: positions 0.01 deg apart are near-identical frames captured seconds
apart. This pipeline splits by mirror position instead, so VAL and TEST measure accuracy beyond
lookup distance, and it is used unchanged for every resolution and seed.

## Split - `holdout/make_split.py`

```
python holdout/make_split.py --data-dir /home/evv/data/dark64 --out-dir /home/evv/data/holdout_split
```

Integer grid indices ix = round(x * 100), iy = round(y * 100) from the bank listing: 572 x 400 =
228,800 positions, x in [-2.00, 3.71], y in [-2.00, 1.99] (no y = +2.00 row). Chebyshev holes,
clipped at the edges:

- TEST: max(|dx|, |dy|) <= 0.10 deg around each of the 108 Algorithm 2 origins (X in -2.0 ... 3.5,
  Y in -2.0 ... 2.0, step 0.5) except (0, 0): 40,215 positions (17.58%).
- VAL: max(|dx|, |dy|) <= 0.07 deg around the 88 midpoints (X + 0.25, Y + 0.25): 19,800 (8.65%).
- TRAIN: the rest, 168,785 (73.77%). Asserted: TEST and VAL disjoint, every position in exactly
  one set, everything within 0.17 deg of (0, 0) in TRAIN.
- dist_to_train_deg: Euclidean distance to the nearest TRAIN position (distance transform);
  TEST reaches 0.11, VAL 0.08.

Outputs (`/home/evv/data/holdout_split/` locally, `/ix1/kchen/evv/multireflection/data/holdout_split/`
on the cluster): `split.csv` (ix, iy, x_deg, y_deg, split, dist_to_train_deg), `train_names.txt`,
`val_names.txt`, `test_names.txt`, `closed_loop_origins.txt` (the 108 origins; the 12 at y = 2.00
clamped to 1.99; (0, 0) is TRAIN), `counts.json`, `split_map.png` (copy in this directory).

## Training and model selection

Run names carry the split: `r<res>_..._e<epochs>_hold[_s<seed>]_<job>` for hole-trained runs,
`..._e<epochs>_fullctl_<job>` for the full-data control; names without a marker are random-split runs.
The W&B name carries this from the start: the submit scripts pass `--split-tag hold` / `fullctl` to
the training script, which builds the display name (`run_display_name`).

`train/train_resnet_direct.py --train-keys-file train_names.txt --val-keys-file val_names.txt`:
trains on TRAIN, keeps the best-VAL checkpoint and records the selected epoch in
`train_summary.json` and the W&B summary (`best_epoch`). TEST names are never passed to training.

- `holdout/submit_64.sh` (on the cluster checkout): one training seed (0) per resolution on dark<RES> via
  `cluster/train_l40s.slurm`, each followed by `cluster/holdout_test_l40s.slurm` (TEST evaluation
  and closed-loop replay). `RES` and `SEEDS` override.
- `holdout/submit_control.sh`: the full-data control. Same recipe on TRAIN + VAL + TEST for a fixed
  `EPOCHS` equal to the selected epoch of the hole-trained runs, `--save-last`; its closed-loop
  replay is the comparison. Its TEST numbers are not a hold-out result.

Hardware closed loop (Algorithm 2): `app/eval.py` already starts from the same 108 origins
(`EVAL_GRID_STEP = 0.5`); five runs with the hole-trained model and with the control.

## Offline TEST evaluation

`holdout/eval_test.py` evaluates one checkpoint open-loop on every TEST position of the split:
one prediction per TEST image, compared with the position in the file name. TEST frames are
decoded once and reused for all conditions and seeds.

```
python holdout/eval_test.py --checkpoint runs/<name>/<name>.pth --model-resolution 256 \
    --data-dir /mnt/h/dark512 --split /home/evv/data/holdout_split/split.csv
```

- Frames come from the 512 px bank; the perturbation (utils/eval_batched.Perturbation) acts on
  the 512 px frame, which is box-averaged (avg_pool2d) to `--model-resolution`, as in the sweeps.
  A bank already at the model resolution also works (no box average; the perturbation then acts on
  the small frame, so only clean numbers are comparable). The perturbation index of a sample is its
  TEST sample index.
- `--conditions`: names from `utils/eval_sweep.CONDITIONS` (default: clean, noise_0.10/0.20/0.30,
  brightness_0.4/0.6/1.6, occlusion_2x30, occlusion_2x40, occlusion_2x30_rotbright,
  occlusion_2_mixed, combined_harsh). `--perturb-seeds` (default 0,1,2); clean runs once.
- Metrics per condition and seed: n, angular error mean/std/median/p95/max, signed per-axis mean
  and std, per-axis mean absolute error; aggregated over seeds as mean and std across seeds. The
  same per distance-to-TRAIN bucket (0.01 deg steps, 0.01 to 0.11).
- Nearest-neighbour baseline (clean, `--nn`/`--no-nn`): label of the L2-nearest TRAIN image, TEST
  and TRAIN both box-averaged from the bank's frames to `--nn-resolution` (default the model
  resolution) and quantized to 8 bit; float32 search on the GPU in chunks. TRAIN frames are reduced
  while loading (memory N_train x r x r). `--nn-test-samples` defaults to all TEST samples at
  r <= 128 and 5000 otherwise (`--nn-seed` picks the subset). Without `--nn-resolution`, a
  resolution that does not fit in 75 percent of the available memory (Slurm limit aware) is halved
  until it fits; a 512 px model on a 64 GB job gets a 256 px NN. `nn_metrics.csv` also has the
  model's clean error on the same samples.
- `--max-test-samples K`: debug, a fixed random subset (seed 0) of K TEST samples.
- Outputs in `--out-dir` (default `<checkpoint dir>/holdout_test/`): `metrics_seeds.csv`,
  `metrics.csv`, `buckets.csv`, `nn_metrics.csv`, `nn_buckets.csv`, `summary.json` (settings,
  sample sizes, NN resolution and why, timings), `error_vs_distance.png` (model clean, two perturbed
  conditions, NN baseline).

Cluster: `cluster/holdout_test_l40s.slurm` (1 L40S, 16 CPUs, 64 GB) stages dark512 and runs
eval_test.py. Knobs: `CKPT` (required), `MODEL_RES`, `SPLIT`, `OUT_DIR`, `EXTRA_ARGS`. With
`CLOSED_LOOP=1` (default) it then runs the offline closed-loop replay `utils/eval_batched.py
--starts-file $ORIGINS` (default `closed_loop_origins.txt` of the split, 108 origins) once clean and
once per condition in `CL_CONDITIONS` (default `occlusion_2x30,occlusion_2x30_rotbright,noise_0.20`,
perturbation seed 0), writing to `$OUT_DIR/closed_loop/<condition>/`. `holdout/condition_flags.py
NAME` prints a condition's flags.

Local smoke numbers (r64 checkpoint trained on all positions, so only a code check, RTX A2000):
64 px bank, full TEST (40,215): clean 0.0270 deg, NN (64 px, all TEST vs 168,785 TRAIN) 0.0526 deg;
load TEST 4 s, 27k images/s, TRAIN load 17 s, NN search 12 s. On 300 TEST samples the 512 -> 64 path
and the native 64 px bank give the same clean mean error (0.0239 deg). cv2 INTER_AREA and
avg_pool2d + rounding differ by at most one gray level on about 1.4 percent of pixels.
