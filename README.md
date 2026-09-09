# Deep-Learning-Enabled Autonomous Alignment of Multipass Herriott Cells for Gas Detection

<!-- If you have a preprint or publication link, uncomment and update: -->
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX) -->

A lightweight deep learning system that autonomously aligns Herriott optical cavities using camera feedback and low-cost stepper motors. The system runs entirely on a Raspberry Pi 4, achieves 100% alignment success across the full operational range, and converges in fewer than two steps on average — enabling field-deployable gas sensors that can recalibrate themselves without human intervention.

---

## System Overview

<p align="center">
  <img src="graphs/System photos/system_topdown_view.jpg" width="700" alt="Experimental system for optical cavity automatic alignment"/>
</p>

The system consists of a camera, two stepper motors coupled to a kinematic mirror mount, a GRIN lens laser source, a Herriott cell, and a Raspberry Pi 4 that handles both image acquisition and motor control.

<p align="center">
  <img src="graphs/Hardware Software diagrams/hardware+software.jpg" width="500" alt="Hardware and software architecture"/>
</p>

## How It Works

### Herriott Cell

A Herriott cell uses two opposing concave mirrors to bounce a laser beam back and forth, creating a long optical path in a compact volume. This extended path length makes the cell highly sensitive to trace gas concentrations — but also highly sensitive to mirror misalignment.

<p align="center">
  <img src="graphs/Simulation graphs/cavity_scheme.png" width="400" alt="Herriott cell schematic"/>
</p>

### What the Model Sees

A camera captures the reflection pattern on the mirror surface. The image is cropped, resized to 512×512, converted to grayscale, and masked to isolate the lens region. The pattern changes dramatically with alignment state:

<p align="center">
  <img src="graphs/Input Examples/zero/InputExample(0,0).jpg" width="180" alt="Optimal alignment"/>
  &nbsp;&nbsp;
  <img src="graphs/Input Examples/skewed/x1.10_y-0.80.jpg" width="180" alt="Suboptimal alignment"/>
  &nbsp;&nbsp;
  <img src="graphs/Input Examples/scatter/x2.80_y-0.57.jpg" width="180" alt="Misaligned"/>
</p>
<p align="center">
  <em>Left:</em> Optimal &nbsp;&nbsp;|&nbsp;&nbsp; <em>Center:</em> Suboptimal &nbsp;&nbsp;|&nbsp;&nbsp; <em>Right:</em> Misaligned
</p>

### Alignment Loop

The system operates in a closed loop. At each iteration, it captures an image, preprocesses it, and compares it to a reference image of optimal alignment using the Structural Similarity Index (SSIM). If SSIM exceeds 0.97 (corresponding to <0.07° angular error), alignment is complete. Otherwise, a ResNet-18 regression model predicts the angular correction needed, which is translated into stepper motor commands. The process typically converges in 1–2 steps.

## Results

The system was evaluated across the full mirror actuation range (X: −2° to +4°, Y: −2° to +2°) with starting positions spaced 0.5° apart. Results are averaged over five complete runs.

| Metric | MLP | ResNet-18 |
|---|---|---|
| Success rate | 100% | 100% |
| Time to align | 4.97 ± 2.56 s | **3.07 ± 1.58 s** |
| Number of adjustments | 1.76 ± 0.62 | 1.80 ± 0.79 |
| Final SSIM | **0.983 ± 0.002** | 0.977 ± 0.007 |
| Final angular error | **0.019° ± 0.017°** | 0.045° ± 0.021° |
| RAM usage | 1034 MiB | **83 MiB** |
| Inference time | 1.72 s | **1.16 s** |

ResNet-18 was selected for deployment due to its 12× lower memory footprint and faster inference, making it suitable for resource-constrained edge platforms.

<p align="center">
  <img src="graphs/Evaluation/resnet18_eval.png" width="500" alt="Evaluation heatmap showing number of adjustments per starting position"/>
</p>
<p align="center">
  <em>Number of adjustments needed to reach optimal alignment from each starting position. Most of the operational range converges in 1–2 steps.</em>
</p>

## Future Directions

### Int8 Quantized Inference on the Edge

The current bottleneck is not training but on-device inference: 1.16 s per prediction and 83 MiB of RAM on the Raspberry Pi 4. Since the alignment loop converges in 1–2 steps, inference latency dominates the 3.07 s time-to-align almost entirely.

Quantizing ResNet-18 to int8 targets exactly this. The weights drop from ~44 MB to ~11 MB, and ARM CPUs have native NEON int8 paths reachable through ONNX Runtime, TFLite, or PyTorch's qnnpack backend — typically 2–3× faster inference on this class of hardware.

The workflow would be to train in float on the workstation, apply a short quantization-aware fine-tune to recover any accuracy lost to the int8 grid, and export the quantized model for the Pi. QAT matters here because the system's headline result is sub-0.05° angular accuracy, so the regression head is the part worth validating carefully after quantization.

Note that this is a deployment-side change only. Training stays in float: fp16 mixed precision is the useful lever on the training box, while int8 belongs at inference time.

<!-- 
## Citation

If you use this work, please cite:

```bibtex
@article{venediktov2025herriott,
  title   = {Deep-Learning-Enabled Autonomous Alignment of Multipass Herriott Cells for Gas Detection},
  author  = {Venediktov, Evgenii and Zhong, Shuda and Zhang, Guangyin and Splain, Zach and Chauhdry, Majid H. M. and Ikpeazu, Emeka and Mao, Zhi-Hong and Wright, Ruishu F. and Lalam, Nagesh and Chen, Kevin P.},
  year    = {2025}
}
```


## Installation

*Coming soon.*

## Usage

*Coming soon.*

## Repository Structure

*Coming soon.*

## License

*TBD*
 -->

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment from the lockfile:

```bash
uv sync                      # core: training, inference, simulation, analysis
uv sync --extra lmdb         # adds the legacy LMDB pipeline
uv sync --all-extras         # everything, including notebooks
```

Run anything through `uv run`, which resolves the environment automatically:

```bash
uv run python train/train_resnet_direct.py --help
uv run python utils/graph_eval.py eval.log 0.97
```

### Extras

| Extra | Adds | Needed for |
|---|---|---|
| `lmdb` | `lmdb`, `msgpack`, `lz4`, `torchinfo` | `data_process/prepare_lmdb.py`, `train/cnn_train.py` |
| `zemax` | `pythonnet`, `pillow` | `data_process/generate_simulated_data.py` (Windows + OpticStudio only) |
| `notebooks` | `ipykernel`, `jupyterlab` | `model_real.ipynb`, `train/experiments/*.ipynb` |

### PyTorch and CUDA

PyTorch is installed from PyPI, whose Linux x86_64 wheels bundle the CUDA runtime — no extra index is configured. On Windows, if CUDA is not picked up, add PyTorch's own index to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch" }]
torchvision = [{ index = "pytorch" }]
```

On the Raspberry Pi (aarch64) the default PyPI CPU wheels are used, which is what inference needs.

### requirements.txt

`requirements.txt` is generated from `uv.lock` and kept only for environments without uv, such as the Raspberry Pi. Do not edit it by hand — regenerate it with:

```bash
uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt
```

## Acknowledgments

This work was supported by the University of Pittsburgh.