"""Offline, batched replica of the hardware alignment sweep in app/eval.py.

The collected image bank (one 512x512 grayscale JPEG per 0.01 degree tilt
position, named x{X:.2f}_y{Y:.2f}.jpg) is used as a lookup table from mirror
position to camera image, so the closed loop

    image = bank[pos]; sim = SSIM(image, bank[0, 0])
    stop if sim >= threshold or adj_n >= max_adj
    pos = clamp(round(pos - clip(model(image)), 2))

can be simulated for every start on a grid without touching the hardware.
All starts are advanced simultaneously and the model is run in GPU batches;
SSIM is memoized by position and computed in a thread pool.

Caveats:
  * The start grid positions are training positions: the validation split
    was a random 20 percent of the same image folder, so this measures the
    closed-loop behaviour on seen data, not generalisation.
  * Time-to-align does not exist offline; only the number of adjustments,
    final SSIM and final angular error are meaningful.
  * Positions are clamped to the range actually present in the image bank.

Outputs (in --out-dir): trace.csv (one row per trajectory per step, t=0
included), eval.log (same line format as app/eval.py, readable by
utils/graph_eval.py), summary.json, and three heatmaps over the start grid:
number of adjustments (the paper's evaluation figure), final angular error and
final SSIM.

Perturbations (all off by default) distort the image the MODEL sees, to measure
how robust the closed loop is to conditions it may meet on the hardware. They use
the same definitions and units as GpuAugment in train/train_resnet_direct.py
(float images in [0, 1]; brightness/contrast are multiplicative factors; noise
is additive Gaussian, clamped; occlusion boxes are filled with 0.0). Conditions
that belong to the physical setup persist over a trajectory: the brightness
factor, the contrast factor, the noise sigma and the occlusion boxes are drawn
once per trajectory from (--perturb-seed, trajectory index) and re-applied
identically at every step, only the noise values are re-drawn per step. The
perturbed frame is quantized to 8 bit like a camera frame. By default the SSIM
stop test still uses the clean image, so it measures whether the model brings
the mirror to true alignment; --perturb-ssim feeds the perturbed frame to the
stop test as well (then SSIM is computed per step instead of memoized).

With --wandb the summary is written into a W&B run and the three output files
are uploaded to it. The run is the one named by WANDB_RUN_ID, so a training job
that exports that variable before training gets its evaluation on the same run
as its loss curves (this is what cluster/train_l40s.slurm does). Without
WANDB_RUN_ID a new run is created.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from skimage.metrics import structural_similarity
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import X_TILT_START, X_TILT_STOP, Y_TILT_START, Y_TILT_STOP, EVAL_MAX_ADJ_NUMBER  # noqa: E402
from app.inference import resnet18, evaluate_position  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "saved_models" / "real" / "r512_occ05-20img_n10_e96_3854472.pth"
DEFAULT_DATA_DIR = "/mnt/h/dark512"
PERTURB_KEYS = ("brightness", "brightness_fixed", "contrast", "contrast_fixed", "noise", "noise_min",
                "occlusion_count", "occlusion_min", "occlusion_max", "occlusion_prob", "perturb_ssim",
                "perturb_seed")


def clip(v, minv, maxv):
    return min(max(v, minv), maxv)


class ImageBank:
    """Position -> image lookup backed by the collected JPEG folder."""

    def __init__(self, data_dir, workers):
        self.data_dir = Path(data_dir)
        self.files = {}
        for name in os.listdir(self.data_dir):
            if not (name.startswith("x") and name.endswith(".jpg")):
                continue
            stem = name[:-4]
            try:
                xs, ys = stem.split("_y")
                x = round(float(xs[1:]), 2)
                y = round(float(ys), 2)
            except ValueError:
                continue
            self.files[(x, y)] = name
        if not self.files:
            raise RuntimeError(f"no images found in {self.data_dir}")
        xs = sorted({k[0] for k in self.files})
        ys = sorted({k[1] for k in self.files})
        self.x_min, self.x_max = xs[0], xs[-1]
        self.y_min, self.y_max = ys[0], ys[-1]
        self.images = {}
        self.ssim = {}
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.reference = self._decode((0.0, 0.0))
        self.images[(0.0, 0.0)] = self.reference

    def clamp(self, x, y):
        return clip(x, self.x_min, self.x_max), clip(y, self.y_min, self.y_max)

    def _decode(self, pos):
        name = self.files.get(pos)
        if name is None:
            raise KeyError(f"no image for position {pos}")
        img = cv2.imread(str(self.data_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"failed to read {name}")
        return img

    def _decode_and_ssim(self, pos):
        img = self.images.get(pos)
        if img is None:
            img = self._decode(pos)
        raw = float(structural_similarity(img, self.reference))
        return pos, img, raw

    def fetch(self, positions, with_ssim=True):
        """Ensure images (and, with with_ssim, SSIM) are cached for every position in the list."""
        if not with_ssim:
            missing = sorted({p for p in positions if p not in self.images})
            for pos, img in zip(missing, self.pool.map(self._decode, missing)):
                self.images[pos] = img
            return
        missing = sorted({p for p in positions if p not in self.ssim})
        if not missing:
            return
        for pos, img, raw in self.pool.map(self._decode_and_ssim, missing):
            self.images[pos] = img
            self.ssim[pos] = raw

    def ssim_of(self, images):
        """Raw SSIM of each uint8 image against the reference, in the thread pool."""
        return list(self.pool.map(lambda im: float(structural_similarity(im, self.reference)), images))

    def close(self):
        self.pool.shutdown()


class Perturbation:
    """Per-trajectory image perturbations, applied on the GPU to a float batch in [0, 1].

    Brightness factor, contrast factor, noise sigma and occlusion boxes are drawn once
    per trajectory from numpy's default_rng seeded with (seed, trajectory index); the
    noise values come from a torch generator seeded with `seed` and are re-drawn every
    time apply() is called. Order: brightness -> contrast -> noise -> occlusion -> 8 bit
    quantization (GpuAugment shuffles brightness/contrast; here the order is fixed).
    """

    def __init__(self, args, n, image_shape, device):
        self.device = device
        self.seed = args.perturb_seed
        self.brightness = args.brightness
        self.brightness_fixed = args.brightness_fixed
        self.contrast = args.contrast
        self.contrast_fixed = args.contrast_fixed
        self.noise = args.noise or 0.0
        self.noise_min = args.noise_min if self.noise > 0 else None
        self.occlusion_count = args.occlusion_count
        self.occlusion_min = args.occlusion_min
        self.occlusion_max = args.occlusion_max
        self.occlusion_prob = args.occlusion_prob
        if self.occlusion_prob is None:
            self.occlusion_prob = 1.0 if self.occlusion_count > 0 else 0.0
        self.occlusion_angle = args.occlusion_angle
        self.occlusion_bright_prob = args.occlusion_bright_prob
        self.occlusion_bright_value = args.occlusion_bright_value
        self.ssim = bool(args.perturb_ssim)
        self.use_brightness = bool(self.brightness) or self.brightness_fixed is not None
        self.use_contrast = bool(self.contrast) or self.contrast_fixed is not None
        self.use_noise = self.noise > 0
        self.use_occlusion = self.occlusion_count > 0 and self.occlusion_prob > 0
        self.enabled = self.use_brightness or self.use_contrast or self.use_noise or self.use_occlusion

        H, W = image_shape
        K = max(self.occlusion_count, 0)
        bright = np.ones(n)
        contr = np.ones(n)
        sigma = np.full(n, self.noise)
        boxes = np.zeros((n, max(K, 1), 4), dtype=np.int64)  # top, left, h, w; h = w = 0: not applied
        angles = np.zeros((n, max(K, 1)))  # degrees, rotation about the box centre
        fills = np.zeros((n, max(K, 1)))  # 0 = dark box, occlusion_bright_value = bright box
        b_amp = self.brightness or 0.0
        c_amp = self.contrast or 0.0
        for i in range(n):
            rng = np.random.default_rng([self.seed, i])
            # Every draw happens regardless of which flags are on, so e.g. the boxes of a
            # trajectory do not change when brightness is switched on.
            b = rng.uniform(max(0.0, 1.0 - b_amp), 1.0 + b_amp)
            c = rng.uniform(max(0.0, 1.0 - c_amp), 1.0 + c_amp)
            s = rng.uniform(self.noise_min, self.noise) if self.noise_min is not None else self.noise
            bright[i] = self.brightness_fixed if self.brightness_fixed is not None else b
            contr[i] = self.contrast_fixed if self.contrast_fixed is not None else c
            sigma[i] = s
            for k in range(K):
                applied = rng.random() < self.occlusion_prob
                box_h = max(1, min(H, int(rng.uniform(self.occlusion_min, self.occlusion_max) * H)))
                box_w = max(1, min(W, int(rng.uniform(self.occlusion_min, self.occlusion_max) * W)))
                top = int(rng.integers(0, H - box_h + 1))
                left = int(rng.integers(0, W - box_w + 1))
                if applied:
                    boxes[i, k] = (top, left, box_h, box_w)
            # Drawn after all boxes, so the geometry above is what it was before these existed
            for k in range(K):
                angles[i, k] = rng.uniform(-1.0, 1.0) * self.occlusion_angle
                if rng.random() < self.occlusion_bright_prob:
                    fills[i, k] = self.occlusion_bright_value
        self.bright_np, self.contr_np, self.sigma_np, self.boxes_np = bright, contr, sigma, boxes
        self.angles = torch.tensor(angles, dtype=torch.float32, device=device)
        self.fills = torch.tensor(fills, dtype=torch.float32, device=device)
        self.bright = torch.tensor(bright, dtype=torch.float32, device=device)
        self.contr = torch.tensor(contr, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.boxes = torch.tensor(boxes, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(self.seed)

    def settings(self):
        return {
            "brightness": self.brightness,
            "brightness_fixed": self.brightness_fixed,
            "contrast": self.contrast,
            "contrast_fixed": self.contrast_fixed,
            "noise": self.noise if self.use_noise else None,
            "noise_min": self.noise_min,
            "occlusion_count": self.occlusion_count,
            "occlusion_min": self.occlusion_min if self.use_occlusion else None,
            "occlusion_max": self.occlusion_max if self.use_occlusion else None,
            "occlusion_prob": self.occlusion_prob if self.use_occlusion else None,
            **({"occlusion_angle": self.occlusion_angle, "occlusion_bright_prob": self.occlusion_bright_prob,
                "occlusion_bright_value": self.occlusion_bright_value}
               if self.use_occlusion and (self.occlusion_angle or self.occlusion_bright_prob) else {}),
            "perturb_ssim": self.ssim,
            "perturb_seed": self.seed,
        }

    def describe(self):
        parts = []
        if self.brightness_fixed is not None:
            parts.append(f"brightness factor {self.brightness_fixed}")
        elif self.brightness is not None:
            parts.append(f"brightness factor in [{max(0.0, 1 - self.brightness):g}, {1 + self.brightness:g}] per trajectory")
        if self.contrast_fixed is not None:
            parts.append(f"contrast factor {self.contrast_fixed}")
        elif self.contrast is not None:
            parts.append(f"contrast factor in [{max(0.0, 1 - self.contrast):g}, {1 + self.contrast:g}] per trajectory")
        if self.use_noise:
            if self.noise_min is not None:
                parts.append(f"noise sigma in [{self.noise_min:g}, {self.noise:g}] per trajectory, re-drawn per step")
            else:
                parts.append(f"noise sigma {self.noise:g}, re-drawn per step")
        if self.use_occlusion:
            parts.append(f"{self.occlusion_count} occlusion box(es) of edge [{self.occlusion_min:g}, {self.occlusion_max:g}] "
                         f"with prob {self.occlusion_prob:g} per trajectory"
                         + (f", rotated in [-{self.occlusion_angle:g}, {self.occlusion_angle:g}] deg"
                            if self.occlusion_angle else "")
                         + (f", bright ({self.occlusion_bright_value:g}) with prob {self.occlusion_bright_prob:g}"
                            if self.occlusion_bright_prob else ""))
        if not parts:
            return "none"
        return "; ".join(parts) + f"; seed {self.seed}; SSIM on {'perturbed' if self.ssim else 'clean'} image"

    def apply(self, x, idx):
        """x: float (B, 1, H, W) in [0, 1] on self.device; idx: trajectory index per row."""
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        if self.use_brightness:
            x = (x * self.bright[idx].view(-1, 1, 1, 1)).clamp_(0.0, 1.0)
        if self.use_contrast:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - mean) * self.contr[idx].view(-1, 1, 1, 1) + mean).clamp_(0.0, 1.0)
        if self.use_noise:
            noise = torch.randn(x.shape, generator=self.gen, device=self.device, dtype=x.dtype)
            x = (x + noise * self.sigma[idx].view(-1, 1, 1, 1)).clamp_(0.0, 1.0)
        if self.use_occlusion and not (self.occlusion_angle or self.occlusion_bright_prob):
            H, W = x.shape[-2:]
            box = self.boxes[idx]  # (B, K, 4)
            top, left, bh, bw = box[..., 0, None], box[..., 1, None], box[..., 2, None], box[..., 3, None]
            rows = torch.arange(H, device=self.device)[None, None, :]
            cols = torch.arange(W, device=self.device)[None, None, :]
            rmask = (rows >= top) & (rows < top + bh)  # (B, K, H)
            cmask = (cols >= left) & (cols < left + bw)  # (B, K, W)
            mask = (rmask[..., :, None] & cmask[..., None, :]).any(dim=1)  # (B, H, W)
            x = x.masked_fill(mask[:, None], 0.0)
        elif self.use_occlusion:
            # Rotated and/or bright boxes, one at a time (a later box paints over an earlier one).
            # Pixel centres at +0.5, so angle 0 covers exactly the axis-aligned box above.
            H, W = x.shape[-2:]
            rows = torch.arange(H, device=self.device, dtype=torch.float32).view(1, H, 1) + 0.5
            cols = torch.arange(W, device=self.device, dtype=torch.float32).view(1, 1, W) + 0.5
            for k in range(self.boxes.shape[1]):
                top, left, bh, bw = (self.boxes[idx, k, j].float().view(-1, 1, 1) for j in range(4))
                t = torch.deg2rad(self.angles[idx, k]).view(-1, 1, 1)
                dr = rows - (top + bh / 2)
                dc = cols - (left + bw / 2)
                u = dc * torch.cos(t) + dr * torch.sin(t)
                v = dr * torch.cos(t) - dc * torch.sin(t)
                mask = (u.abs() <= bw / 2) & (v.abs() <= bh / 2) & (bh > 0)  # (B, H, W)
                x = torch.where(mask[:, None], self.fills[idx, k].view(-1, 1, 1, 1), x)
        # A camera frame is 8 bit: quantize so the model and the SSIM test see the same frame
        return x.mul_(255.0).round_().div_(255.0)

    @torch.inference_mode()
    def render(self, images, idx, batch_size):
        """Perturbed uint8 frames (list of (H, W) arrays) for uint8 images of the trajectories idx."""
        out = []
        for s in range(0, len(images), batch_size):
            chunk = images[s:s + batch_size]
            x = torch.from_numpy(np.stack(chunk)[:, None]).to(self.device).float().div_(255.0)
            x = self.apply(x, idx[s:s + batch_size])
            out.extend(x.mul_(255.0).round_().to(torch.uint8).cpu().numpy()[:, 0])
        return out


class BatchedPredictor:
    def __init__(self, checkpoint, batch_size, fp16, perturb=None, model_resolution=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.batch_size = batch_size
        self.model_resolution = model_resolution
        self.model = resnet18(output_dim=2)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.perturb = perturb
        self.images_seen = 0
        self.seconds = 0.0

    @torch.inference_mode()
    def predict(self, images, idx=None):
        """images: list of uint8 (H, W) arrays -> float32 array (N, 2) in degrees.

        idx: trajectory index per image; when given and a perturbation is configured,
        it is applied to the batch on the GPU before the model. With model_resolution the
        (perturbed) frame is then area-downscaled to that size and quantized to 8 bit.
        """
        out = np.empty((len(images), 2), dtype=np.float32)
        for s in range(0, len(images), self.batch_size):
            chunk = images[s:s + self.batch_size]
            t0 = time.perf_counter()
            x = torch.from_numpy(np.stack(chunk)[:, None]).to(self.device, non_blocking=True)
            x = x.float().div_(255.0)
            if self.perturb is not None and idx is not None:
                x = self.perturb.apply(x, idx[s:s + self.batch_size])
            if self.model_resolution and x.shape[-1] != self.model_resolution:
                # The perturbation acts on the camera frame; the device resizes it to the model input.
                # A box average over an integer factor is cv2.INTER_AREA, the resize the banks were
                # built with. Not interpolate(mode="area"): on this batch (channel stride 0) the
                # adaptive pooling kernel returns the first image for every row (torch 2.14, CUDA).
                x = torch.nn.functional.avg_pool2d(x, x.shape[-1] // self.model_resolution)
                x = x.mul_(255.0).round_().div_(255.0)
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                y = self.model(x)
            y = y.float().cpu().numpy()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.seconds += time.perf_counter() - t0
            self.images_seen += len(chunk)
            out[s:s + len(chunk)] = y
        out[:, 0] = out[:, 0] * (X_TILT_STOP - X_TILT_START) + X_TILT_START
        out[:, 1] = out[:, 1] * (Y_TILT_STOP - Y_TILT_START) + Y_TILT_START
        return out


def read_starts_file(path, bank):
    """Start positions from a file of image names (x{X}_y{Y}.jpg) or 'x y' pairs."""
    starts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".jpg"):
                xs, ys = line[:-4].split("_y")
                pos = (round(float(xs[1:]), 2), round(float(ys), 2))
            else:
                x, y = line.replace(",", " ").split()
                pos = (round(float(x), 2), round(float(y), 2))
            if pos in bank.files:
                starts.append(pos)
    if not starts:
        raise RuntimeError(f"no usable start positions in {path}")
    return sorted(set(starts))


def build_grid(bank, step):
    xs = np.arange(X_TILT_START, bank.x_max + step / 2, step)
    ys = np.arange(Y_TILT_START, bank.y_max + step / 2, step)
    xs = [round(float(v), 2) for v in xs if round(float(v), 2) <= bank.x_max]
    ys = [round(float(v), 2) for v in ys if round(float(v), 2) <= bank.y_max]
    return [(x, y) for x in xs for y in ys]


def log_line(ox, oy, adj_n, xp, yp, x, y, sim):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return (f"{ts} - INFO - origin_x:{ox},origin_y:{oy},adj_n:{adj_n},"
            f"pred_x:{xp},pred_y:{yp},pos_x:{x},pos_y:{y},sim_index:{sim}\n")


def run(args):
    workers = args.workers or os.cpu_count() or 1
    t_start = time.perf_counter()
    bank = ImageBank(args.data_dir, workers)
    print(f"Image bank: {len(bank.files)} images, x in [{bank.x_min}, {bank.x_max}], "
          f"y in [{bank.y_min}, {bank.y_max}]")
    if args.starts_file:
        grid = read_starts_file(args.starts_file, bank)
        starts_desc = f"Starts: {len(grid)} (from {args.starts_file})"
    else:
        grid = build_grid(bank, args.grid_step)
        starts_desc = f"Starts: {len(grid)} (grid step {args.grid_step})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = bank.reference.shape
    if args.model_resolution and (H != W or H % args.model_resolution):
        raise SystemExit(f"--model-resolution {args.model_resolution} must divide the square bank size {H}x{W}")
    perturb = Perturbation(args, len(grid), bank.reference.shape, device)
    if not perturb.enabled:
        perturb = None
    predictor = BatchedPredictor(args.checkpoint, args.batch_size, not args.no_fp16, perturb, args.model_resolution)
    print(f"Model: {args.checkpoint} on {predictor.device}, fp16={predictor.fp16}"
          + (f", input {args.model_resolution} px" if args.model_resolution else ""))
    print(starts_desc)
    print(f"Perturbation: {perturb.describe() if perturb else 'none'}")
    perturb_ssim = perturb is not None and perturb.ssim

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "eval_results" / Path(args.checkpoint).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trajectory state, indexed by start
    n = len(grid)
    pos = list(grid)
    adj = [0] * n
    done = [False] * n
    converged = [False] * n
    final_ssim = [0.0] * n
    trace = []  # rows for trace.csv
    log_lines = []

    active = list(range(n))
    with tqdm(desc="steps", unit="step") as bar:
        while active:
            bar.set_postfix(active=len(active))
            # 1. images + SSIM for the current positions
            bank.fetch([pos[i] for i in active], with_ssim=not perturb_ssim)
            if perturb_ssim:
                # 2. the frame of this step is perturbed once and shared by the SSIM test
                # and the model, so SSIM is per (trajectory, step) and cannot be memoized
                frames = perturb.render([bank.images[pos[i]] for i in active], active, args.batch_size)
                frame_of = dict(zip(active, frames))
                raw_of = dict(zip(active, bank.ssim_of(frames)))
            else:
                raw_of = {i: bank.ssim[pos[i]] for i in active}
            if bar.n == 0 and not perturb_ssim:
                # One-time check that our rounding matches the hardware metric
                for p in (pos[active[0]], pos[active[-1]]):
                    assert evaluate_position(bank.images[p], [bank.reference]) == round(bank.ssim[p], 2), p
            still = []
            images = []
            for i in active:
                raw = raw_of[i]
                sim = round(raw, 2)
                if sim >= args.threshold:
                    done[i] = True
                    converged[i] = True
                elif adj[i] >= args.max_adj:
                    done[i] = True
                if done[i]:
                    final_ssim[i] = sim
                    trace.append([*grid[i], adj[i], *pos[i], sim, raw, "", "", "", "", int(converged[i])])
                else:
                    still.append(i)
                    images.append(frame_of[i] if perturb_ssim else bank.images[pos[i]])
            active = still
            if not active:
                break
            # 3. predict for the whole active batch (frames already perturbed when perturb_ssim)
            preds = predictor.predict(images, None if perturb_ssim else active)
            # 4. move
            for k, i in enumerate(active):
                px, py = float(preds[k, 0]), float(preds[k, 1])
                x_pred = -clip(px, X_TILT_START, X_TILT_STOP)
                y_pred = -clip(py, Y_TILT_START, Y_TILT_STOP)
                x = round(pos[i][0] + x_pred, 2)
                y = round(pos[i][1] + y_pred, 2)
                x, y = bank.clamp(x, y)
                raw = raw_of[i]
                sim = round(raw, 2)
                trace.append([*grid[i], adj[i], *pos[i], sim, raw, x_pred, y_pred, x, y, 0])
                pos[i] = (x, y)
                adj[i] += 1
                log_lines.append(log_line(grid[i][0], grid[i][1], adj[i], x_pred, y_pred, x, y, sim))
            bar.update(1)
    bank.close()
    wall = time.perf_counter() - t_start

    # Outputs
    trace.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(out_dir / "trace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["origin_x", "origin_y", "t", "pos_x", "pos_y", "ssim", "ssim_raw",
                    "pred_x", "pred_y", "next_x", "next_y", "converged"])
        w.writerows(trace)
    with open(out_dir / "eval.log", "w") as f:
        f.writelines(log_lines)

    adj_arr = np.array(adj, dtype=np.float64)
    conv_arr = np.array(converged, dtype=bool)
    ssim_arr = np.array(final_ssim, dtype=np.float64)
    err_arr = np.array([np.hypot(*p) for p in pos], dtype=np.float64)

    def stats(a):
        if not len(a):
            return {"mean": None, "std": None, "max": None}
        return {"mean": float(np.mean(a)), "std": float(np.std(a)), "max": float(np.max(a))}

    perturb_settings = perturb.settings() if perturb else Perturbation(args, 0, bank.reference.shape, device).settings()

    summary = {
        "n_starts": n,
        "n_converged": int(conv_arr.sum()),
        "success_rate": float(conv_arr.mean()),
        "adjustments_converged": stats(adj_arr[conv_arr]),
        "adjustments_all": stats(adj_arr),
        "final_ssim": stats(ssim_arr),
        "final_angular_error_deg": stats(err_arr),
        "wall_time_s": wall,
        "inference_images": predictor.images_seen,
        "inference_img_per_s": predictor.images_seen / predictor.seconds if predictor.seconds else None,
        "ssim_positions_evaluated": len(bank.ssim),
        "perturbed": perturb is not None,
        "settings": {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "threshold": args.threshold,
            "grid_step": None if args.starts_file else args.grid_step,
            "starts_file": args.starts_file,
            "max_adj": args.max_adj,
            "batch_size": args.batch_size,
            "fp16": predictor.fp16,
            "workers": workers,
            "device": str(predictor.device),
            **({"model_resolution": args.model_resolution} if args.model_resolution else {}),
            **perturb_settings,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    heatmaps = plot_heatmaps(grid, adj_arr, err_arr, ssim_arr, out_dir)

    def fmt(s):
        return f"{s['mean']:.4f} +- {s['std']:.4f}" if s["mean"] is not None else "n/a"

    def spread(a):
        return f"min={np.min(a):.4g} max={np.max(a):.4g} median={np.median(a):.4g}"

    print()
    print(f"{'starts':<28}{n}")
    print(f"{'perturbation':<28}{perturb.describe() if perturb else 'none'}")
    print(f"{'converged':<28}{summary['n_converged']} ({100 * summary['success_rate']:.2f} %)")
    print(f"{'adjustments (converged)':<28}{fmt(summary['adjustments_converged'])}")
    print(f"{'adjustments (all)':<28}{fmt(summary['adjustments_all'])}")
    print(f"{'final SSIM':<28}{fmt(summary['final_ssim'])}")
    print(f"{'final angular error (deg)':<28}{fmt(summary['final_angular_error_deg'])}")
    print(f"{'wall time (s)':<28}{wall:.1f}")
    ips = summary["inference_img_per_s"]
    print(f"{'inference (img/s)':<28}{ips:.1f}" if ips else f"{'inference (img/s)':<28}n/a")
    print(f"{'outputs':<28}{out_dir}")
    print()
    print(f"adjustments       {spread(adj_arr)}")
    print(f"final SSIM        {spread(ssim_arr)}")
    print(f"angular error     {spread(err_arr)}")
    print(f"not converged     {int((~conv_arr).sum())} starts")
    print(f"heatmaps          {', '.join(h.name for h in heatmaps)}")

    if args.wandb:
        log_to_wandb(summary, out_dir, heatmaps, args.wandb_project, args.wandb_prefix)
    return summary


def plot_heatmaps(grid, adj, err, ssim, out_dir):
    """One image per metric over the start grid, in the style of utils/graph_eval.py.

    The starts are a regular grid, so the values are placed directly into a 2D array
    instead of being interpolated.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.array(sorted({p[0] for p in grid}))
    ys = np.array(sorted({p[1] for p in grid}))
    ix = {v: i for i, v in enumerate(xs)}
    iy = {v: i for i, v in enumerate(ys)}
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    def to_grid(values):
        z = np.full((len(ys), len(xs)), np.nan)
        for (x, y), v in zip(grid, values):
            z[iy[y], ix[x]] = v
        return z

    paths = []
    panels = [
        ("heatmap_adjustments.png", adj, "Number of Adjustments", True),
        ("heatmap_angular_error.png", err, "Final angular error (deg)", False),
        ("heatmap_final_ssim.png", ssim, "Final SSIM", False),
    ]
    for name, values, label, integer in panels:
        plt.figure()
        im = plt.imshow(to_grid(values), extent=extent, origin="lower", cmap="viridis", aspect="auto")
        cbar = plt.colorbar(im, label=label)
        if integer:
            cbar.set_ticks(np.arange(int(np.min(values)), int(np.max(values)) + 1))
        plt.xlabel("X origin (deg)")
        plt.ylabel("Y origin (deg)")
        path = out_dir / name
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def log_to_wandb(summary, out_dir, heatmaps, project, prefix="eval"):
    import wandb

    run_id = os.environ.get("WANDB_RUN_ID")
    run = wandb.init(project=project, id=run_id, resume="allow" if run_id else None,
                     name=None if run_id else f"eval_{Path(summary['settings']['checkpoint']).stem}",
                     job_type="eval")
    flat = {
        f"{prefix}/n_starts": summary["n_starts"],
        f"{prefix}/success_rate": summary["success_rate"],
        f"{prefix}/adjustments_mean": summary["adjustments_converged"]["mean"],
        f"{prefix}/adjustments_std": summary["adjustments_converged"]["std"],
        f"{prefix}/final_ssim_mean": summary["final_ssim"]["mean"],
        f"{prefix}/final_ssim_std": summary["final_ssim"]["std"],
        f"{prefix}/final_angular_error_mean": summary["final_angular_error_deg"]["mean"],
        f"{prefix}/final_angular_error_std": summary["final_angular_error_deg"]["std"],
        f"{prefix}/grid_step": summary["settings"]["grid_step"],
        f"{prefix}/threshold": summary["settings"]["threshold"],
        f"{prefix}/perturbed": summary["perturbed"],
    }
    for key in PERTURB_KEYS:
        flat[f"{prefix}/{key}"] = summary["settings"][key]
    run.summary.update(flat)
    run.log({f"{prefix}/{h.stem}": wandb.Image(str(h)) for h in heatmaps})
    for name in ("summary.json", "trace.csv", "eval.log"):
        run.save(str(out_dir / name), base_path=str(out_dir), policy="now")
    print(f"W&B: summary written to run {run.id} ({run.url})")
    run.finish()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--starts-file", default=None,
                   help="start positions from a file of image names, e.g. a run's val_names.txt; overrides --grid-step")
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--max-adj", type=int, default=EVAL_MAX_ADJ_NUMBER)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=None, help="thread pool size for decode+SSIM (default: cpu count)")
    p.add_argument("--out-dir", default=None, help="default: eval_results/<checkpoint stem>/")
    p.add_argument("--no-fp16", action="store_true", help="disable fp16 autocast on CUDA")
    p.add_argument("--model-resolution", type=int, default=None,
                   help="model input size in px: frames (after any perturbation) are area-downscaled to it; "
                        "the SSIM stop test keeps the bank's resolution")
    p.add_argument("--wandb", action="store_true",
                   help="write the summary to W&B; resumes the run in WANDB_RUN_ID if set")
    p.add_argument("--wandb-project", default="multireflection")
    p.add_argument("--wandb-prefix", default="eval", help="key prefix for the W&B summary, e.g. eval_val")
    q = p.add_argument_group("perturbations", "applied to the image the model sees; same units as GpuAugment "
                             "in train/train_resnet_direct.py; all off by default")
    b = q.add_mutually_exclusive_group()
    b.add_argument("--brightness", type=float, default=None,
                   help="per-trajectory brightness factor drawn once in [1-B, 1+B]")
    b.add_argument("--brightness-fixed", type=float, default=None, help="brightness factor F for every trajectory")
    c = q.add_mutually_exclusive_group()
    c.add_argument("--contrast", type=float, default=None,
                   help="per-trajectory contrast factor drawn once in [1-C, 1+C]")
    c.add_argument("--contrast-fixed", type=float, default=None, help="contrast factor F for every trajectory")
    q.add_argument("--noise", type=float, default=None, help="additive Gaussian noise sigma (in [0, 1] units), re-drawn per step")
    q.add_argument("--noise-min", type=float, default=None,
                   help="with --noise S: sigma drawn once per trajectory in [min, S]")
    q.add_argument("--occlusion-count", type=int, default=0, help="boxes per trajectory, geometry drawn once per trajectory")
    q.add_argument("--occlusion-min", type=float, default=0.15, help="box edge as a fraction of the image")
    q.add_argument("--occlusion-max", type=float, default=0.40)
    q.add_argument("--occlusion-prob", type=float, default=None,
                   help="probability each box is applied (default 1.0 when --occlusion-count > 0)")
    q.add_argument("--occlusion-angle", type=float, default=0.0,
                   help="each box is rotated about its centre by an angle drawn in [-A, A] degrees")
    q.add_argument("--occlusion-bright-prob", type=float, default=0.0,
                   help="probability a box is filled with --occlusion-bright-value instead of black")
    q.add_argument("--occlusion-bright-value", type=float, default=1.0, help="bright box fill in [0, 1]")
    q.add_argument("--perturb-ssim", action="store_true",
                   help="also feed the perturbed frame to the SSIM stop test (default: SSIM on the clean image)")
    q.add_argument("--perturb-seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.noise_min is not None and (args.noise is None or args.noise_min >= args.noise):
        p.error("--noise-min needs --noise S with min < S")
    for name in ("brightness", "brightness_fixed", "contrast", "contrast_fixed", "noise"):
        v = getattr(args, name)
        if v is not None and v < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.perturb_ssim and not any((args.brightness, args.brightness_fixed, args.contrast, args.contrast_fixed,
                                      args.noise, args.occlusion_count)):
        p.error("--perturb-ssim needs at least one perturbation")
    return args


if __name__ == "__main__":
    run(parse_args())
