# Supervised training

See `train/TRAINING_AND_EVALS.md` first; it is the current summary. This file holds
per-module detail.

Entry point: **`train/train_resnet_direct.py`**, the only training path. It reads the
preprocessed JPEGs in the collection folder directly.

## Direct-from-disk training - `train/train_resnet_direct.py`

Reads the JPEGs in `DATA_DIR` (already 512x512 grayscale, masked and blurred by
`collect_real_data.py`) with `DirectImageDataset`; the only per-sample work is the JPEG
decode. Throughput numbers are in the history section above.

- Workers return **uint8**; `/255` and every augmentation run batched on the GPU. The
  network sees the same [0, 1] float32 as before - only the storage dtype changed.
- fp16 autocast, TF32 and channels_last on by default (`--no-amp` to disable).
- `GpuAugment` implements brightness/contrast, affine, Gaussian noise and random occlusion
  **per sample**, occlusion included since 2026-09-11 (a box shared by the batch is cancelled by
  BatchNorm batch statistics, so such models failed on occlusion in eval mode;
  `--occlusion-per-batch` restores it). On 2026-09-10 boxes were drawn once per batch and shared by every
  image in it (earlier runs, r512_occ05-20img_n10_e96_3854472,
  r512_ft_occ15-40img_n10_e49_3866737 and r512_occ15-40img_n00-20_e96_3866805, used per-image
  boxes, hence `img` in their names; later runs have `batch`). Since 2026-09-10 each box is
  also rotated about its centre by an angle in [-90, 90] deg (`--occlusion-angle`) and is
  white (1.0, glare) with p 0.5 instead of black (`--occlusion-bright-prob/-value`);
  `rotated_box_mask` is the shared geometry, and run names carry `-rot90-bri50`. Since 2026-09-11
  the fill is a gray level uniform in [0, 1] per box (`--occlusion-fill-min/-max`, names
  `-fill0-255`, with per-image boxes `_occ15-40img-rot90-fill0-255`); `--occlusion-binary-fill` restores white/black. Default epochs 98 = 14 cosine cycles of 7 (was 96).
  torchvision's v2 transforms sample parameters once per call, so applying
  them to a batched tensor would give every image in the batch the same jitter or the same
  rotation. Affine is wired up but is semantically risky here: the label is the position of
  the spot pattern, so a translation resembles a different mirror tilt. Ablate it.
- Label normalization derives from `X/Y_TILT_START/STOP` in `config.py` (`imname_to_target`
  parses the `x{..}_y{..}.jpg` filename), so it cannot drift from what `app/inference.py`
  de-normalizes with.
- Checkpoints are plain state dicts, verified to load into
  `TiltPredictor(model_type="ResNet18")` with `strict=True`.
- W&B logging is on by default and reads `WANDB_API_KEY` from the environment; no key is
  stored in the repository. `--no-wandb` disables it.
- `--data-share` (fraction of the dataset, default 1.0) and `--step-filter` (1/2/4 = 0.01 /
  0.02 / 0.04 deg grid) make data-density ablations cheap. `--help` lists every knob.
- Noise: sigma drawn per image uniformly in `[--noise-min, --noise]`, defaults 0.0 and 0.2
  since 2026-09-10 (before: fixed 0.1; r512_occ15-40img_n00-20_e96_3866805 was the first run with the range).
  `--noise-min` unset (None) gives a fixed sigma.
- `--resolution N` (default None = `TRAINING_IMAGE_RESOLUTION`, 512): square input size; the
  loader resizes images of any other size. Used with the `dark256` bank (dark512 resized
  with INTER_AREA on the cluster login node, `/ix1/kchen/evv/multireflection/data/dark256.tar.gz`).
- Every run writes `train_names.txt` / `val_names.txt` into `--checkpoint-dir` (split is
  deterministic from `--split-seed`, but the files make it explicit and let
  `utils/eval_batched.py --starts-file` evaluate on held-out positions).

## History: the LMDB pipeline (removed 2026-09-10)

The original pipeline packed the JPEGs into an LMDB of float32 tensors
(`data_process/prepare_lmdb.py`) and trained from it (`train/cnn_train.py`, a jupytext-style
`# %%` script, plus the sweep agent `train/experiments/wandb_sweep.py` and the notebooks
`model_real.ipynb`, `train/experiments/{model_real,model_real copy,train_model,conv,dog,
processing_experiments}.ipynb`). The LMDB archives were deleted in September 2026 and the
code and notebooks were removed from the tree on 2026-09-10; all of it is in git history
before that date. Things worth knowing from that era:

- The deployed checkpoint, `resnet18_001step_BS_avid-sweep-4406_DarkOnly512_lmdb_50bs_0001lr_aug+_best_model.pth`
  (see `config.py`), was produced by it. Naming convention: architecture, data step, sweep
  name, dataset, storage (`lmdb`), batch size, LR, augmentation on.
- Sweeps searched `ConfigCNN` conv stacks; the winning run name (`avid-sweep-4406`) is
  embedded in that checkpoint filename. `ConfigCNN` itself was defined only in the removed
  files.
- Loss `nn.MSELoss` on the normalized 2-vector, `AdamW(lr=1e-3, weight_decay=1e-3)`,
  `CosineAnnealingWarmRestarts(T_0=7, eta_min=1e-5)`, batch 400, 96 epochs; augmentation was
  `ColorJitter(brightness 0.4, contrast 0.1)` then `GaussianNoise(sigma 0.1)` per sample on the
  CPU. `train_resnet_direct.py` keeps the same loss/optimizer family and moves augmentation to
  the GPU.
- Label normalization was hardcoded as `/5.7` and `/4`; the direct script derives it from
  `config.py` instead (see below).
- The LMDB path was also the throughput bottleneck. Measured on an RTX A2000: the float32 LMDB
  delivered 96.5 img/s with 8 workers (its ceiling), while the JPEG folder gives 468 img/s
  cold and 1975 img/s once the 4.83 GB dataset is in page cache. The model itself does
  73 img/s in fp32 and 200 img/s with fp16 autocast + TF32 + channels_last, so the LMDB path
  capped throughput as soon as AMP was enabled.

## Notebooks

`train/experiments/synthetic_model.ipynb` (Zemax synthetic-data model) and
`train/experiments/test.ipynb` (scratch) are the notebooks that remain; historical only.

## Gotchas

- Imports assume the repo root is on `sys.path` (`from preprocess_images import ...`,
  `from config import ...`) - run from the project root.
- Changing the actuation range (`X/Y_TILT_START/STOP` in `config.py`) silently invalidates old
  checkpoints, since both training labels and inference de-normalization derive from it.
