# Models

Task: image (1x512x512, grayscale, circular-masked, [0,1]) -> 2 floats, the normalized
(x_tilt, y_tilt) of the current mirror pose. The alignment controller moves by the *negated*
prediction, so the network effectively predicts the correction.

Architectures are defined **twice**: `app/inference.py` (deployment copy) and
`train/cnn_train.py` (training copy). They are not shared; keep them in sync when editing.

## Deployed candidates (paper Table 1)

| | MLP (`SimpleFC`) | ResNet-18 |
|---|---|---|
| params | 3.3e8 | 1.1e7 |
| RAM on Pi | 1034 MiB | 83 MiB |
| inference | 1.72 s | 1.16 s |
| time to align | 4.97 +/- 2.56 s | 3.07 +/- 1.58 s |
| final SSIM | 0.983 +/- 0.002 | 0.977 +/- 0.007 |
| final angular error | 0.019 +/- 0.017 deg | 0.045 +/- 0.021 deg |

ResNet-18 was selected for deployment (12x smaller memory footprint).

### `SimpleFC` - `app/inference.py:89`, `train/cnn_train.py:570`
Flattened 512*512 -> 1024 -> 256 -> 32 -> 2, each hidden layer BatchNorm1d + ReLU.
Requires `flatten_data=True` in the dataset (or `.flatten()` in `TiltPredictor.predict`).

### `ResNet` / `resnet18` - `app/inference.py:279,345`, `train/cnn_train.py:434,500`
Hand-written ResNet (`BasicBlock` / `Bottleneck`), first conv takes **1 channel**, head is
`Linear(512, 2)`. Factories for 18/34/50/101/152 exist; only 18 is used.
Kaiming init on convs, constant init on BN.

## Experimental architectures (not deployed)

- `ConfigCNN` - `train/cnn_train.py:526`. CNN built from the `conv_config` list of dicts
  (out_channels / kernel_size / stride / padding, huge first kernels: 149, 31, 7) plus a
  2-layer FC head sized from `size_after_conv`. Used for the wandb sweeps.
- `GradientMagnitude` + `GradientSimpleFC` - `app/inference.py:24,67`. Learn on Sobel gradient
  magnitude (Gaussian blur -> Sobel -> per-image min-max normalize) instead of raw pixels.
- `CLAHEGradTransform` - `app/inference.py:175`. OpenCV CLAHE + blur + Sobel preprocessing,
  paired with `SimpleFC` under model type `CLAHEGradSimpleFC`.
- `WideConv` - `app/inference.py:109`. 3 conv layers + global average pool + small FC head.
- `CnnExtractor` - `app/inference.py:133`. 3-channel input, single wide conv then a large FC
  stack. Contains leftover debug prints.

## Inference wrapper - `app/inference.py:350` `TiltPredictor`

- `TiltPredictor(model_fname, model_type)`; `model_type` selects the architecture via a `match`
  (`SimpleFC`, `WideConv`, `GradientSimpleFC`, `CLAHEGradSimpleFC`, `CnnExtractor`, `ResNet18`).
- Loads `./saved_models/<fname>` state dict, `.eval()`, moves to CUDA if available (CPU on the Pi).
- `predict(img, scale_predictions=True)`: optional preprocessing transform, `float()/255`,
  per-type reshaping, forward under `no_grad`, then de-normalizes
  `pred_x * (X_TILT_STOP - X_TILT_START) + X_TILT_START` (same for y).
  Note the `/255` here - LMDB tensors are already normalized at build time, so training and
  inference normalization paths differ; inputs at inference come from OpenCV uint8 images.
- Model choice is driven by `config.py:INFERENCE_MODEL_FILE_NAME` / `INFERENCE_MODEL_TYPE`
  (currently inconsistent - see [main.md](main.md)).

## Stopping criterion (not a model)

`evaluate_position` - `app/inference.py:11`. Max SSIM (`skimage.metrics.structural_similarity`)
between the current processed frame and each reference image in `OPTIMUM_IMAGE_PATH_LIST`,
rounded to 2 decimals. SSIM 0.97 corresponds to ~0.07 deg angular error (paper Appendix B,
`graphs/Evaluation/ssim_vs_distance_*.png`).

The RL policy network is a separate lineage - see [rl.md](rl.md).
