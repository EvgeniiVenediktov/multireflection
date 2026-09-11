"""
train_resnet_direct.py - ResNet-18 tilt regression, reading JPEGs straight from disk.

The images in
DATA_DIR are already fully preprocessed by collect_real_data.py (512x512, grayscale,
circular-masked, blurred), so nothing has to happen at load time except a JPEG decode.

Design notes:
  - Workers hand back uint8 tensors. The /255 normalization and all augmentation happen
    on the GPU, batched, right before the forward pass. Storing or transporting float32
    would inflate every image from 256 KiB to 1 MiB for zero extra information.
  - The network still sees exactly the same [0, 1] float32 input it always did.
  - fp16 autocast, TF32 and channels_last are on by default (measured on an RTX A2000,
    torch 2.14 + CUDA 13: 73 img/s fp32 -> 200 img/s with all three). Disable with --no-amp.
  - --compile runs the model through torch.compile (Inductor). Off by default: the first
    epoch is slower while it compiles, so it only pays off on long runs. Checkpoints are
    saved from the underlying module, so they load exactly as uncompiled ones do.
  - Augmentation is brightness/contrast, Gaussian noise and random occlusion, all
    per-sample on the GPU. See GpuAugment for why torchvision's v2 transforms are not
    used. Affine is implemented but off by default: it costs ~7% throughput and teaches
    a translation invariance the task does not have.
  - Label normalization is derived from the tilt range in config.py so it cannot silently
    drift away from what app/inference.py uses to de-normalize predictions.
  - Every run is logged to W&B. Export WANDB_API_KEY, or pass --no-wandb.

Run from the repository root:
    python train/train_resnet_direct.py
    python train/train_resnet_direct.py --data-share 0.25 --name quarter_data
    python train/train_resnet_direct.py --affine-degrees 3 --affine-translate 0.02
    python train/train_resnet_direct.py --occlusion-prob 0.8 --occlusion-count 3
    python train/train_resnet_direct.py --no-augment --no-wandb --epochs 2
    python train/train_resnet_direct.py --help
"""

import os
import sys
import time
import math
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# torchvision.transforms.v2 is intentionally not used: see GpuAugment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    X_TILT_START,
    X_TILT_STOP,
    Y_TILT_START,
    Y_TILT_STOP,
    TRAINING_IMAGE_RESOLUTION,
)

# OpenCV spawns its own threads; with several DataLoader workers that oversubscribes the CPU.
cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config = {
    "experiment_name": "resnet18_direct_dark512",
    "data_dir": "/mnt/h/dark512",

    # Optional key files, one image filename per line. When both are None the directory
    # listing is split deterministically using "split_seed".
    "train_keys_file": None,
    "val_keys_file": None,
    "val_share": 0.2,
    "split_seed": 0,

    # Keep only every Nth hundredth of a degree on both axes. 1 = every image (228,800),
    # 2 = 0.02 deg grid (1/4), 4 = 0.04 deg grid (1/16). Cheap way to run a density ablation.
    "step_filter": 1,

    # Fraction of the (already step-filtered) dataset to use. 1.0 = everything.
    # Applied with split_seed, so the same share is reproducible across runs.
    "data_share": 1.0,
    "resolution": None,           # square input size; None = TRAINING_IMAGE_RESOLUTION from config.py

    # Throughput is flat from bs=16 to bs=128 (153-159 img/s) because the GPU is
    # power-limited, so this is chosen for headroom rather than speed: 3.9 GiB peak on a
    # 12 GiB card. Do not go past 128 - bs=160 needs 10.65 GiB and collapses to 87 img/s.
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.001,
    "epochs": 98,                 # 14 full cosine cycles of 7
    "lr_scheduler_loop": 7,
    "lr_min": 0.00001,

    # Worker count makes no measurable difference (2 workers feed the GPU as well as 12),
    # so this is cores-1 rather than anything tuned. Validation runs once per epoch and
    # gets its own smaller count so its persistent workers do not sit idle competing with
    # the training workers for the whole run.
    "num_workers": 5,
    "val_num_workers": 2,
    "prefetch_factor": 4,

    # Measured on an RTX A2000: fp32 73 img/s, +TF32 98, +AMP 150, +channels_last 200.
    "use_amp": True,
    "use_tf32": True,
    "use_channels_last": True,
    "use_compile": False,

    # GPU-side augmentation, applied after the /255 conversion so the magnitudes below
    # are in [0, 1] units and match the earlier LMDB pipeline (removed 2026-09-10). Set any value to 0 to disable
    # that stage.
    "jitter_brightness": 0.4,
    "jitter_contrast": 0.1,
    "noise_level": 0.2,           # upper bound of the per-image sigma range
    "noise_level_min": 0.0,       # None = fixed sigma noise_level for every image

    # Affine. OFF by default, for two reasons. Semantically, the label IS the position and
    # shape of the spot pattern, so a translation looks to the network exactly like a
    # different mirror tilt - the augmentation teaches an invariance the task does not
    # have. It is also the single most expensive stage: enabling it costs ~7% throughput
    # (174 -> 162 img/s), where occlusion and the photometric stages cost ~1% each.
    # Enable deliberately, with an ablation, e.g. --affine-degrees 3 --affine-translate 0.02
    "affine_degrees": 0.0,        # rotation, +/- degrees
    "affine_translate": 0.0,      # max shift as a fraction of image size
    "affine_scale": 0.0,          # zoom, +/- fraction
    "affine_shear": 0.0,          # shear, +/- degrees

    # Random occlusion (cutout). Models dust or debris on the mirror surface.
    "occlusion_prob": 0.5,        # probability a given box is applied, per batch
    "occlusion_count": 2,         # boxes attempted per batch, shared by every image in it
    "occlusion_min": 0.15,        # box edge as a fraction of image size
    "occlusion_max": 0.40,
    "occlusion_value": 0.0,       # fill value in normalized units
    "occlusion_angle": 90.0,      # each box rotated about its centre by an angle in [-A, A] deg
    "occlusion_bright_prob": 0.5,  # probability a box is a bright patch (glare) instead of dark
    "occlusion_bright_value": 1.0,

    "checkpoint_dir": "./saved_models/real",
    "starting_checkpoint": None,

    "use_wandb": True,
    "wandb_project": "multireflection",
    "wandb_log_samples": 15,      # raw + augmented images logged once at the first epoch

    "seed": 0,
}


# Label normalization. Mirrors app/inference.py:TiltPredictor.predict, which de-normalizes
# with these same config values. Changing the tilt range in config.py invalidates every
# checkpoint trained before the change.
X_SPAN = X_TILT_STOP - X_TILT_START
Y_SPAN = Y_TILT_STOP - Y_TILT_START


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def imname_to_target(name: str) -> tuple[float, float]:
    """Parses image names of format x{x_value}_y{y_value}.jpg"""
    name = name.split(".jpg")[0]
    x, y = name.split("_")
    return float(x[1:]), float(y[1:])


def keep_by_step(name: str, step: int) -> bool:
    """True if the image sits on a grid of `step` hundredths of a degree on both axes."""
    if step <= 1:
        return True
    x, y = imname_to_target(name)
    return round(x * 100) % step == 0 and round(y * 100) % step == 0


def list_images(data_dir: str, step: int) -> list[str]:
    names = []
    with os.scandir(data_dir) as it:
        for entry in it:
            if not entry.name.endswith((".jpg", ".png")):
                continue
            if keep_by_step(entry.name, step):
                names.append(entry.name)
    names.sort()
    return names


def read_keys_file(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def take_share(names: list[str], share: float, seed: int) -> list[str]:
    """Deterministic random subset holding `share` of the list. 1.0 keeps everything."""
    if share >= 1.0:
        return names
    if not 0.0 < share <= 1.0:
        raise ValueError(f"data_share must be in (0, 1], got {share}")
    keep = max(1, int(round(len(names) * share)))
    return random.Random(seed).sample(names, keep)


def build_split(cfg: dict) -> tuple[list[str], list[str]]:
    if cfg["train_keys_file"] and cfg["val_keys_file"]:
        train = read_keys_file(cfg["train_keys_file"])
        val = read_keys_file(cfg["val_keys_file"])
        if cfg["step_filter"] > 1:
            train = [k for k in train if keep_by_step(k, cfg["step_filter"])]
            val = [k for k in val if keep_by_step(k, cfg["step_filter"])]
        # Shrink both sides by the same share so the train/val ratio is preserved.
        train = take_share(train, cfg["data_share"], cfg["split_seed"])
        val = take_share(val, cfg["data_share"], cfg["split_seed"] + 1)
        return train, val

    names = list_images(cfg["data_dir"], cfg["step_filter"])
    names = take_share(names, cfg["data_share"], cfg["split_seed"])
    rng = random.Random(cfg["split_seed"])
    rng.shuffle(names)
    n_val = int(len(names) * cfg["val_share"])
    return names[n_val:], names[:n_val]


class DirectImageDataset(Dataset):
    """
    Reads preprocessed JPEGs from a directory. Returns uint8 image tensors; the caller is
    responsible for the /255 conversion (see to_float_and_augment).

    Labels come from the filename and are normalized to [0, 1] with the config tilt range.
    """

    def __init__(self, data_dir: str, names: list[str], resolution=(512, 512)):
        self.data_dir = data_dir
        self.names = names
        self.resolution = tuple(resolution)

        labels = np.empty((len(names), 2), dtype=np.float32)
        for i, name in enumerate(names):
            x, y = imname_to_target(name)
            labels[i, 0] = (x - X_TILT_START) / X_SPAN
            labels[i, 1] = (y - Y_TILT_START) / Y_SPAN
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        path = os.path.join(self.data_dir, self.names[index])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape != self.resolution:
            img = cv2.resize(img, self.resolution)
        # (H, W) uint8 -> (1, H, W) uint8. No float conversion here: it would quadruple
        # what every worker copies through shared memory for no added information.
        return torch.from_numpy(img).unsqueeze(0), self.labels[index]


def rotated_box_mask(H, W, top, left, box_h, box_w, angle_deg, device):
    """(H, W) bool mask of the box_h x box_w rectangle placed at (top, left), rotated by
    angle_deg about its centre. Pixel centres sit at +0.5, so angle 0 covers exactly rows
    top..top+box_h-1 and columns left..left+box_w-1. utils/eval_batched.py uses the same test."""
    t = math.radians(angle_deg)
    dr = torch.arange(H, device=device, dtype=torch.float32)[:, None] + 0.5 - (top + box_h / 2)
    dc = torch.arange(W, device=device, dtype=torch.float32)[None, :] + 0.5 - (left + box_w / 2)
    u = dc * math.cos(t) + dr * math.sin(t)
    v = dr * math.cos(t) - dc * math.sin(t)
    return (u.abs() <= box_w / 2) & (v.abs() <= box_h / 2)


class GpuAugment:
    """
    Photometric jitter, affine warp, Gaussian noise and random occlusion, applied to a
    normalized float batch on the GPU.

    Photometric, noise and affine stages draw their parameters PER IMAGE. torchvision's v2
    transforms sample once per CALL, so ColorJitter or RandomAffine applied to a batched
    tensor gives every image in the batch the same brightness, or the same rotation. That
    is a silent loss of augmentation diversity relative to the earlier per-sample CPU
    pipeline, which was why these are written out by hand.

    Occlusion is the exception: its boxes are drawn once PER BATCH and applied to every
    image in it (decided 2026-09-10; runs before that used per-image boxes). Each box is
    rotated by a random angle and is dark (dust) or bright (glare); both since 2026-09-10.

    Order: affine -> photometric (brightness/contrast, random order) -> noise -> occlusion.
    Occlusion goes last so the boxes are not blurred away or rescaled by the warp.
    """

    def __init__(self, brightness=0.0, contrast=0.0, noise_sigma=0.0, noise_sigma_min=None,
                 affine_degrees=0.0, affine_translate=0.0, affine_scale=0.0,
                 affine_shear=0.0, occlusion_prob=0.0, occlusion_count=0,
                 occlusion_min=0.05, occlusion_max=0.2, occlusion_value=0.0,
                 occlusion_angle=0.0, occlusion_bright_prob=0.0, occlusion_bright_value=1.0):
        self.brightness = brightness
        self.contrast = contrast
        self.noise_sigma = noise_sigma
        # None: every image gets noise_sigma. Otherwise sigma is drawn per image, uniformly
        # in [noise_sigma_min, noise_sigma], so the network sees clean and noisy frames.
        self.noise_sigma_min = noise_sigma_min
        self.affine_degrees = affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shear = affine_shear
        self.occlusion_prob = occlusion_prob
        self.occlusion_count = occlusion_count
        self.occlusion_min = occlusion_min
        self.occlusion_max = occlusion_max
        self.occlusion_value = occlusion_value
        self.occlusion_angle = occlusion_angle
        self.occlusion_bright_prob = occlusion_bright_prob
        self.occlusion_bright_value = occlusion_bright_value

    @property
    def uses_affine(self) -> bool:
        return any((self.affine_degrees, self.affine_translate,
                    self.affine_scale, self.affine_shear))

    @property
    def uses_occlusion(self) -> bool:
        return self.occlusion_prob > 0 and self.occlusion_count > 0

    # -- photometric --------------------------------------------------------

    def _factors(self, x, amount):
        return torch.empty(
            x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype
        ).uniform_(max(0.0, 1.0 - amount), 1.0 + amount)

    def _brightness(self, x):
        return (x * self._factors(x, self.brightness)).clamp_(0.0, 1.0)

    def _contrast(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - mean) * self._factors(x, self.contrast) + mean).clamp_(0.0, 1.0)

    # -- geometric ----------------------------------------------------------

    def _affine(self, x):
        """Per-sample affine warp via a batch of 2x3 matrices fed to grid_sample.

        The matrix maps output coordinates back to input coordinates, so it is the inverse
        of the visual transform. The parameter distributions are symmetric about identity,
        so sampling the inverse directly is equivalent in distribution and avoids a
        per-sample matrix inversion.
        """
        B = x.shape[0]
        dev, dt = x.device, x.dtype

        def u(amount):
            return torch.empty(B, device=dev, dtype=dt).uniform_(-amount, amount)

        ang = u(self.affine_degrees) * (math.pi / 180.0)
        shear = u(self.affine_shear) * (math.pi / 180.0)
        scale = 1.0 + u(self.affine_scale)
        # grid_sample works in normalized coords spanning [-1, 1], hence the factor 2.
        tx = u(self.affine_translate) * 2.0
        ty = u(self.affine_translate) * 2.0

        cos, sin = torch.cos(ang), torch.sin(ang)
        tan_shear = torch.tan(shear)

        theta = torch.zeros(B, 2, 3, device=dev, dtype=dt)
        theta[:, 0, 0] = scale * cos
        theta[:, 0, 1] = scale * (-sin + cos * tan_shear)
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = scale * sin
        theta[:, 1, 1] = scale * (cos + sin * tan_shear)
        theta[:, 1, 2] = ty

        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        # Zero padding matches the black background outside the circular mask.
        return torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

    def _occlude(self, x):
        """Cutout boxes drawn once per batch and applied to every image in it.

        Each box is placed as an axis-aligned rectangle, rotated about its centre by an
        angle in [-occlusion_angle, occlusion_angle] degrees (corners may leave the frame),
        and filled with occlusion_bright_value with probability occlusion_bright_prob,
        otherwise occlusion_value.
        """
        _, _, H, W = x.shape
        rng = random.Random()  # CPU draws; a handful of scalars per batch
        for _ in range(self.occlusion_count):
            if rng.random() >= self.occlusion_prob:
                continue
            box_h = max(1, min(H, int(rng.uniform(self.occlusion_min, self.occlusion_max) * H)))
            box_w = max(1, min(W, int(rng.uniform(self.occlusion_min, self.occlusion_max) * W)))
            top = rng.randint(0, H - box_h)
            left = rng.randint(0, W - box_w)
            angle = rng.uniform(-self.occlusion_angle, self.occlusion_angle)
            bright = rng.random() < self.occlusion_bright_prob
            value = self.occlusion_bright_value if bright else self.occlusion_value
            if angle == 0.0:
                x[:, :, top:top + box_h, left:left + box_w] = value
            else:
                x.masked_fill_(rotated_box_mask(H, W, top, left, box_h, box_w, angle, x.device), value)
        return x

    # -- pipeline -----------------------------------------------------------

    def __call__(self, x):
        if self.uses_affine:
            x = self._affine(x)

        ops = []
        if self.brightness > 0:
            ops.append(self._brightness)
        if self.contrast > 0:
            ops.append(self._contrast)
        random.shuffle(ops)
        for op in ops:
            x = op(x)

        if self.noise_sigma > 0:
            if self.noise_sigma_min is not None and self.noise_sigma_min < self.noise_sigma:
                sigma = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
                sigma.uniform_(self.noise_sigma_min, self.noise_sigma)
            else:
                sigma = self.noise_sigma
            x = (x + torch.randn_like(x) * sigma).clamp_(0.0, 1.0)

        if self.uses_occlusion:
            x = self._occlude(x)
        return x


def make_gpu_augment(cfg: dict):
    """Augmentation applied on the GPU, on a normalized float batch."""
    aug = GpuAugment(
        brightness=cfg["jitter_brightness"],
        contrast=cfg["jitter_contrast"],
        noise_sigma=cfg["noise_level"],
        noise_sigma_min=cfg["noise_level_min"],
        affine_degrees=cfg["affine_degrees"],
        affine_translate=cfg["affine_translate"],
        affine_scale=cfg["affine_scale"],
        affine_shear=cfg["affine_shear"],
        occlusion_prob=cfg["occlusion_prob"],
        occlusion_count=cfg["occlusion_count"],
        occlusion_min=cfg["occlusion_min"],
        occlusion_max=cfg["occlusion_max"],
        occlusion_value=cfg["occlusion_value"],
        occlusion_angle=cfg["occlusion_angle"],
        occlusion_bright_prob=cfg["occlusion_bright_prob"],
        occlusion_bright_value=cfg["occlusion_bright_value"],
    )
    active = (aug.brightness or aug.contrast or aug.noise_sigma
              or aug.uses_affine or aug.uses_occlusion)
    return aug if active else None


def display_stretch(image: torch.Tensor) -> "np.ndarray":
    """(H,W) float in [0,1] -> uint8 for viewing: 99.5th percentile to white, gamma 0.5.

    Display only. The dataset is mostly black with a few bright spots, so the linear image
    looks empty on a monitor.
    """
    x = image.detach().float().clamp(0, 1)
    hi = torch.quantile(x.flatten(), 0.995).clamp(min=1e-3)
    return (x.div(hi).clamp(0, 1).sqrt() * 255).round().to(torch.uint8).cpu().numpy()


def to_float_and_augment(images, device, augment, channels_last: bool):
    """uint8 (B,1,H,W) on CPU -> normalized float32 batch on the GPU."""
    images = images.to(device, non_blocking=True)
    images = images.float().div_(255.0)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    if augment is not None:
        images = augment(images)
    return images


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Kept byte-identical in structure to app/inference.py:ResNet so checkpoints written here
# load directly into TiltPredictor(model_type="ResNet18"). Edit both or neither.


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_dim=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(output_dim=2):
    """ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], output_dim=output_dim)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def save_model(model: nn.Module, fname: str, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, fname))


def run_epoch(model, loader, criterion, device, cfg, augment,
              optimizer=None, scaler=None, desc="epoch"):
    """One pass. Training when optimizer is given, validation otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0
    n_images = 0
    t_start = time.time()

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in pbar:
            batch_images = images.shape[0]
            images = to_float_and_augment(
                images, device, augment if training else None, cfg["use_channels_last"]
            )
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg["use_amp"]):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1
            n_images += batch_images

            elapsed = time.time() - t_start
            pbar.set_postfix(
                loss=f"{batch_loss:.5f}",
                mean=f"{total_loss / n_batches:.5f}",
                img_s=f"{n_images / elapsed:.0f}" if elapsed > 0 else "-",
                refresh=False,
            )

    pbar.close()
    return total_loss / max(n_batches, 1)


def main(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["use_amp"] and device.type != "cuda":
        print("AMP requested but CUDA is unavailable, disabling")
        cfg["use_amp"] = False
        cfg["use_channels_last"] = False

    if cfg["use_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"device: {device}")
    print(f"label normalization: x in [{X_TILT_START}, {X_TILT_STOP}] span {X_SPAN}, "
          f"y in [{Y_TILT_START}, {Y_TILT_STOP}] span {Y_SPAN}")

    print(f"amp fp16: {cfg['use_amp']}, tf32: {cfg['use_tf32']}, "
          f"channels_last: {cfg['use_channels_last']}")

    train_names, val_names = build_split(cfg)
    print(f"train: {len(train_names)} images, val: {len(val_names)} images "
          f"(step_filter={cfg['step_filter']}, data_share={cfg['data_share']})")
    # Record the split next to the checkpoints, so later evaluations can be restricted to
    # the images this run never trained on (utils/eval_batched.py --starts-file), and so a
    # rerun can reuse it via --train-keys-file / --val-keys-file.
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    for fname, names in (("train_names.txt", train_names), ("val_names.txt", val_names)):
        with open(os.path.join(cfg["checkpoint_dir"], fname), "w") as f:
            f.write("\n".join(names) + "\n")
    print(f"split written to {cfg['checkpoint_dir']}/{{train,val}}_names.txt")
    if not train_names:
        raise RuntimeError(f"No images found under {cfg['data_dir']}")

    # Images that are not already this size are resized by the loader, so a 256 px bank
    # trains at 256 only if --resolution 256 is given, otherwise it is upsampled to 512.
    resolution = (cfg["resolution"], cfg["resolution"]) if cfg["resolution"] else tuple(TRAINING_IMAGE_RESOLUTION)
    print(f"input resolution: {resolution[0]}x{resolution[1]}")
    train_dataset = DirectImageDataset(cfg["data_dir"], train_names, resolution)
    val_dataset = DirectImageDataset(cfg["data_dir"], val_names, resolution)

    def loader_kwargs(workers: int) -> dict:
        kw = dict(
            batch_size=cfg["batch_size"],
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=workers > 0,
        )
        if workers > 0:
            kw["prefetch_factor"] = cfg["prefetch_factor"]
        return kw

    # Separate counts on purpose: sharing one kwargs dict gives the validation loader as
    # many persistent workers as training, and they stay alive competing for cores through
    # every training epoch despite running once per epoch.
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              **loader_kwargs(cfg["num_workers"]))
    val_loader = DataLoader(val_dataset, shuffle=False,
                            **loader_kwargs(cfg["val_num_workers"]))
    print(f"loaders: batch_size={cfg['batch_size']}, "
          f"train workers={cfg['num_workers']}, val workers={cfg['val_num_workers']}")

    model = resnet18(output_dim=2).to(device)
    if cfg["use_channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    if cfg["starting_checkpoint"]:
        state = torch.load(
            os.path.join(cfg["checkpoint_dir"], cfg["starting_checkpoint"]),
            map_location=device,
        )
        model.load_state_dict(state)
        print(f"resumed from {cfg['starting_checkpoint']}")

    # Keep a handle on the plain module: torch.compile wraps it, and the wrapper's
    # state_dict keys carry an _orig_mod. prefix that TiltPredictor would not accept.
    base_model = model
    if cfg["use_compile"]:
        model = torch.compile(model)
        print("torch.compile: on (the first epoch includes compilation time)")

    optimizer = optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, cfg["lr_scheduler_loop"], eta_min=cfg["lr_min"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["use_amp"])
    augment = make_gpu_augment(cfg)
    if augment is None:
        print("augmentation: none")
    else:
        print(f"augmentation: brightness={augment.brightness} contrast={augment.contrast} "
              f"noise={augment.noise_sigma}"
              + (f" (per-image in [{augment.noise_sigma_min}, {augment.noise_sigma}])"
                 if augment.noise_sigma_min is not None else "")
              + f" | affine deg={augment.affine_degrees} "
              f"translate={augment.affine_translate} scale={augment.affine_scale} "
              f"shear={augment.affine_shear} | occlusion p={augment.occlusion_prob} "
              f"n={augment.occlusion_count} size=[{augment.occlusion_min}, "
              f"{augment.occlusion_max}] angle=+-{augment.occlusion_angle} "
              f"bright p={augment.occlusion_bright_prob} value={augment.occlusion_bright_value}")

    run = None
    if cfg["use_wandb"]:
        import wandb

        # Expects WANDB_API_KEY in the environment; no key is stored in this repository.
        if not os.environ.get("WANDB_API_KEY"):
            print("WANDB_API_KEY is not set. Export it, run 'wandb login', "
                  "or pass --no-wandb to train without logging.")
        run = wandb.init(project=cfg["wandb_project"], name=cfg["experiment_name"],
                         config=cfg, resume="allow")
        run.watch(base_model, log="all", log_freq=200)
        run.summary["train_images"] = len(train_dataset)
        run.summary["val_images"] = len(val_dataset)
        run.summary["x_span"] = X_SPAN
        run.summary["y_span"] = Y_SPAN

        # One look at what the network is actually being fed, post-augmentation. The
        # images are dark (mean ~8/255, spots near 255), so the linear image is nearly
        # black on screen. Log it stretched for display; the network still gets linear.
        # Samples are drawn at random across the whole training split so they cover
        # different tilt positions rather than neighbouring files.
        n_samples = min(cfg["wandb_log_samples"], len(train_dataset))
        if n_samples > 0:
            picks = sorted(random.Random(cfg["seed"]).sample(range(len(train_dataset)), n_samples),
                           key=lambda i: train_dataset.names[i])
            raw = torch.stack([train_dataset[i][0] for i in picks])
            shown = to_float_and_augment(raw, device, augment, cfg["use_channels_last"])
            run.log({
                "input_samples": [
                    wandb.Image(display_stretch(raw[k, 0].float().div(255)),
                                caption=train_dataset.names[i])
                    for k, i in enumerate(picks)
                ],
                "augmented_samples": [
                    wandb.Image(display_stretch(shown[k, 0]), caption=train_dataset.names[i])
                    for k, i in enumerate(picks)
                ],
            }, step=0)

    best_loss = float("inf")
    ckpt_name = cfg["experiment_name"] + "_best_model.pth"

    for epoch in range(cfg["epochs"]):
        t_start = time.time()
        last_lr = scheduler.get_last_lr()[0]

        epoch_tag = f"epoch {epoch + 1}/{cfg['epochs']}"
        train_loss = run_epoch(model, train_loader, criterion, device, cfg, augment,
                               optimizer=optimizer, scaler=scaler,
                               desc=f"{epoch_tag} train")
        val_loss = run_epoch(model, val_loader, criterion, device, cfg, augment,
                             desc=f"{epoch_tag}   val")
        scheduler.step()

        elapsed = time.time() - t_start
        throughput = len(train_dataset) / elapsed

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(base_model, ckpt_name, cfg["checkpoint_dir"])

        print(f"Epoch {epoch + 1}/{cfg['epochs']}, Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, Best: {best_loss:.6f}, "
              f"{elapsed:.0f}s ({throughput:.0f} img/s)")

        if run is not None:
            run.log({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "LR": last_lr,
                "best_loss": best_loss,
                "log_train_loss": math.log(train_loss) if train_loss > 0 else 0.0,
                "log_val_loss": math.log(val_loss) if val_loss > 0 else 0.0,
                "epoch_seconds": elapsed,
                "train_img_per_s": throughput,
            }, step=epoch)

    print("Best loss:", best_loss)
    print("Saved to:", os.path.join(cfg["checkpoint_dir"], ckpt_name))
    if run is not None:
        run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train ResNet-18 tilt regression on JPEGs read straight from disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--data-dir", default=config["data_dir"])
    data.add_argument("--data-share", type=float, default=config["data_share"],
                      help="fraction of the dataset to train on, in (0, 1]")
    data.add_argument("--step-filter", type=int, default=config["step_filter"],
                      help="1 = all images, 2 = 0.02 deg grid, 4 = 0.04 deg grid")
    data.add_argument("--val-share", type=float, default=config["val_share"])
    data.add_argument("--split-seed", type=int, default=config["split_seed"])
    data.add_argument("--resolution", type=int, default=config["resolution"],
                      help="square input size in px; images of another size are resized by the loader "
                           "(default: TRAINING_IMAGE_RESOLUTION in config.py)")
    data.add_argument("--train-keys-file", default=config["train_keys_file"])
    data.add_argument("--val-keys-file", default=config["val_keys_file"])

    opt = p.add_argument_group("optimization")
    opt.add_argument("--name", default=config["experiment_name"])
    opt.add_argument("--epochs", type=int, default=config["epochs"])
    opt.add_argument("--batch-size", type=int, default=config["batch_size"])
    opt.add_argument("--lr", type=float, default=config["lr"])
    opt.add_argument("--weight-decay", type=float, default=config["weight_decay"])
    opt.add_argument("--num-workers", type=int, default=config["num_workers"])
    opt.add_argument("--val-num-workers", type=int, default=config["val_num_workers"],
                     help="validation runs once per epoch, so it needs fewer")
    opt.add_argument("--checkpoint-dir", default=config["checkpoint_dir"],
                     help="where best_model.pth is written; use persistent storage on a cluster")
    opt.add_argument("--starting-checkpoint", default=config["starting_checkpoint"])
    opt.add_argument("--seed", type=int, default=config["seed"],
                     help="torch/random/numpy seed: weight init, shuffling, photometric and noise draws "
                          "(the data split has its own --split-seed; occlusion boxes are unseeded)")
    opt.add_argument("--no-amp", action="store_true",
                     help="disable fp16 autocast and channels_last (both on by default)")
    opt.add_argument("--compile", action="store_true",
                     help="torch.compile the model; first epoch is slower while it compiles")

    aug = p.add_argument_group("augmentation (0 disables a stage)")
    aug.add_argument("--brightness", type=float, default=config["jitter_brightness"])
    aug.add_argument("--contrast", type=float, default=config["jitter_contrast"])
    aug.add_argument("--noise", type=float, default=config["noise_level"],
                     help="Gaussian noise sigma in [0, 1] units (the upper bound if --noise-min is set)")
    aug.add_argument("--noise-min", type=float, default=config["noise_level_min"],
                     help="sample the noise sigma per image uniformly in [NOISE_MIN, --noise]")
    aug.add_argument("--affine-degrees", type=float, default=config["affine_degrees"])
    aug.add_argument("--affine-translate", type=float, default=config["affine_translate"],
                     help="max shift as a fraction of image size")
    aug.add_argument("--affine-scale", type=float, default=config["affine_scale"])
    aug.add_argument("--affine-shear", type=float, default=config["affine_shear"])
    aug.add_argument("--occlusion-prob", type=float, default=config["occlusion_prob"])
    aug.add_argument("--occlusion-count", type=int, default=config["occlusion_count"])
    aug.add_argument("--occlusion-min", type=float, default=config["occlusion_min"])
    aug.add_argument("--occlusion-max", type=float, default=config["occlusion_max"])
    aug.add_argument("--occlusion-angle", type=float, default=config["occlusion_angle"],
                     help="box rotation drawn in [-A, A] degrees; 0 keeps boxes axis-aligned")
    aug.add_argument("--occlusion-bright-prob", type=float, default=config["occlusion_bright_prob"],
                     help="probability a box is filled with --occlusion-bright-value instead of black")
    aug.add_argument("--occlusion-bright-value", type=float, default=config["occlusion_bright_value"])
    aug.add_argument("--no-augment", action="store_true",
                     help="turn every augmentation stage off at once")

    log = p.add_argument_group("logging")
    log.add_argument("--wandb-project", default=config["wandb_project"])
    log.add_argument("--no-wandb", action="store_true",
                     help="run without W&B logging (on by default)")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    config.update({
        "data_dir": args.data_dir,
        "data_share": args.data_share,
        "step_filter": args.step_filter,
        "val_share": args.val_share,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "resolution": args.resolution,
        "train_keys_file": args.train_keys_file,
        "val_keys_file": args.val_keys_file,

        "experiment_name": args.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "val_num_workers": args.val_num_workers,
        "checkpoint_dir": args.checkpoint_dir,
        "starting_checkpoint": args.starting_checkpoint,

        "jitter_brightness": args.brightness,
        "jitter_contrast": args.contrast,
        "noise_level": args.noise,
        "noise_level_min": args.noise_min,
        "affine_degrees": args.affine_degrees,
        "affine_translate": args.affine_translate,
        "affine_scale": args.affine_scale,
        "affine_shear": args.affine_shear,
        "occlusion_prob": args.occlusion_prob,
        "occlusion_count": args.occlusion_count,
        "occlusion_min": args.occlusion_min,
        "occlusion_max": args.occlusion_max,
        "occlusion_angle": args.occlusion_angle,
        "occlusion_bright_prob": args.occlusion_bright_prob,
        "occlusion_bright_value": args.occlusion_bright_value,

        "wandb_project": args.wandb_project,
        "use_wandb": not args.no_wandb,
        "use_compile": args.compile,
    })

    if args.no_amp:
        config["use_amp"] = False
        config["use_channels_last"] = False

    if args.no_augment:
        for key in ("jitter_brightness", "jitter_contrast", "noise_level",
                    "affine_degrees", "affine_translate", "affine_scale",
                    "affine_shear", "occlusion_prob"):
            config[key] = 0.0

    main(config)
