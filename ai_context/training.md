# Supervised training

Entry point: **`train/cnn_train.py`** (a jupytext-style `# %%` script; the notebooks in
`train/experiments/` and the root `model_real.ipynb` are earlier snapshots of the same flow).
Everything is driven by the `config` dict at `train/cnn_train.py:53`.

## Config keys that matter

- `data_folder` - path to the LMDB directory (e.g. `/mnt/h/black_512_0_001step_tensor.lmdb`).
- `dataset_type` - `LMDBImageDataset` (lazy) or `InMemoryLMDBImageDataset` (caching).
  Note: the `match` at `:308` compares against `"InMemoryImageDataset"`, which does not equal the
  class name used in the config - the in-memory branch is effectively unreachable as written.
- `dataset_train_keys_fname` / `dataset_val_keys_fname` - key-list files *inside* the LMDB dir,
  produced by `write_split_keys` (see [data_sources.md](data_sources.md)).
- `dataset_config_flatten` - `True` for `SimpleFC`, `False` for CNN/ResNet.
- `batch_size` 400, `lr` 1e-3, `epochs` 96, `lr_scheduler_loop` 7, `use_amp` False.
- Augmentation toggles: `use_jitter_transform` (brightness 0.4 / contrast 0.1),
  `use_noise_transform` (Gaussian sigma 0.1), `use_grayscale_transform`,
  `use_clahegrad_transform`, `use_high_pass_transform`.
- `starting_checkpoint_fname` / `checkpoint_folder` for warm starts.
- The whole dict is logged to W&B as the run config, and the run is named `experiment_name`.

## Datasets

`LMDBImageDataset` (`:201`) - lazy. Reads the key file, parses labels from the keys, opens the
LMDB lazily in each worker (`open_lmdb` on first `__getitem__`, so it is fork-safe), decodes
`torch.save` bytes to a float32 `(1,H,W)` tensor, applies transforms, optionally flattens, then
normalizes the label to [0,1].

`InMemoryLMDBImageDataset` (`:101`) - same, but caches every decoded tensor in `self.images` and
tracks `loaded_indexes`. Labels are precomputed in `__init__`. Opens the LMDB txn in the parent
process, so it does not survive `num_workers > 0` cleanly.

DataLoaders (`:319`): train `shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4,
persistent_workers=True`; val `num_workers=4, shuffle=False`.

## Augmentation

`torchvision.transforms.v2`, composed at `:288`, applied inside `__getitem__`:
`ColorJitter(brightness, contrast)` then `GaussianNoise(sigma)`. Validation gets an empty/None
compose unless grayscale conversion is enabled. Order matters - jitter before noise.

## Loop - `train/cnn_train.py:627` `train()`

- Loss `nn.MSELoss` on the normalized 2-vector; optimizer `AdamW(lr, weight_decay=1e-3)`;
  scheduler `CosineAnnealingWarmRestarts(T_0=lr_scheduler_loop, eta_min=1e-5)`.
- `GradScaler` is instantiated and used, but `autocast(..., enabled=False)` - AMP is wired up and
  disabled; the scaler is a no-op passthrough.
- Per epoch: train pass, val pass under `inference_mode`, `scheduler.step()`,
  checkpoint on best val loss to `./saved_models/real/<experiment_name>_best_model.pth`.
- W&B logs train/val loss, their logs, LR, best loss, and a 0.8/0.2 weighted total.
- `torch.manual_seed(0)` at import; split randomness lives in the key files, not here.

Checkpoint naming convention (see `config.py`):
`resnet18_001step_BS_avid-sweep-4406_DarkOnly512_lmdb_50bs_0001lr_aug+_best_model.pth`
= architecture, data step, sweep name, dataset, storage, batch size, LR, augmentation on.

## Hyperparameter sweeps - `train/experiments/wandb_sweep.py`

Standalone sweep agent. Key difference from `cnn_train.py`: it loads the **entire split onto the
GPU once** (`load_dataset_to_gpu`) and keeps it in a module-level `_DATA_CACHE`, so sweep trials
skip all I/O. Sweeps searched `ConfigCNN` conv stacks; the winning run name (`avid-sweep-4406`)
is embedded in the deployed checkpoint filename.

## Notebooks

`model_real.ipynb` (root) and `train/experiments/*.ipynb` - historical: MLP-era training,
processing experiments, synthetic-data model, dog/conv scratch tests. Useful only as provenance;
`train/cnn_train.py` supersedes them.

## Gotchas

- Imports assume the repo root is on `sys.path` (`from preprocess_images import ...`,
  `from config import ...`) - run from the project root.
- Label normalization constants (`/5.7`, `/4`) are hardcoded in three places; changing the
  actuation range silently invalidates old checkpoints.
- Hardcoded W&B API key at `train/cnn_train.py` (`wandb.login(key=...)`) - rotate it.
