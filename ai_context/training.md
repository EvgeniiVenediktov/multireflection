# Supervised training

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
  **per sample**. torchvision's v2 transforms sample parameters once per call, so applying
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
