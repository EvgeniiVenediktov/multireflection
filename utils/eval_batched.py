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

DEFAULT_CHECKPOINT = REPO_ROOT / "saved_models" / "real" / "resnet18_l40s_3854472_best_model.pth"
DEFAULT_DATA_DIR = "/mnt/h/dark512"


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

    def fetch(self, positions):
        """Ensure images and SSIM are cached for every position in the list."""
        missing = sorted({p for p in positions if p not in self.ssim})
        if not missing:
            return
        for pos, img, raw in self.pool.map(self._decode_and_ssim, missing):
            self.images[pos] = img
            self.ssim[pos] = raw

    def close(self):
        self.pool.shutdown()


class BatchedPredictor:
    def __init__(self, checkpoint, batch_size, fp16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.batch_size = batch_size
        self.model = resnet18(output_dim=2)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.images_seen = 0
        self.seconds = 0.0

    @torch.inference_mode()
    def predict(self, images):
        """images: list of uint8 (512, 512) arrays -> float32 array (N, 2) in degrees."""
        out = np.empty((len(images), 2), dtype=np.float32)
        for s in range(0, len(images), self.batch_size):
            chunk = images[s:s + self.batch_size]
            t0 = time.perf_counter()
            x = torch.from_numpy(np.stack(chunk)[:, None]).to(self.device, non_blocking=True)
            x = x.float().div_(255.0).contiguous(memory_format=torch.channels_last)
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
    predictor = BatchedPredictor(args.checkpoint, args.batch_size, not args.no_fp16)
    print(f"Model: {args.checkpoint} on {predictor.device}, fp16={predictor.fp16}")

    if args.starts_file:
        grid = read_starts_file(args.starts_file, bank)
        print(f"Starts: {len(grid)} (from {args.starts_file})")
    else:
        grid = build_grid(bank, args.grid_step)
        print(f"Starts: {len(grid)} (grid step {args.grid_step})")

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
            bank.fetch([pos[i] for i in active])
            if bar.n == 0:
                # One-time check that our rounding matches the hardware metric
                for p in (pos[active[0]], pos[active[-1]]):
                    assert evaluate_position(bank.images[p], [bank.reference]) == round(bank.ssim[p], 2), p
            still = []
            images = []
            for i in active:
                raw = bank.ssim[pos[i]]
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
                    images.append(bank.images[pos[i]])
            active = still
            if not active:
                break
            # 3. predict for the whole active batch
            preds = predictor.predict(images)
            # 4. move
            for k, i in enumerate(active):
                px, py = float(preds[k, 0]), float(preds[k, 1])
                x_pred = -clip(px, X_TILT_START, X_TILT_STOP)
                y_pred = -clip(py, Y_TILT_START, Y_TILT_STOP)
                x = round(pos[i][0] + x_pred, 2)
                y = round(pos[i][1] + y_pred, 2)
                x, y = bank.clamp(x, y)
                raw = bank.ssim[pos[i]]
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
        return {"mean": float(np.mean(a)), "std": float(np.std(a))} if len(a) else {"mean": None, "std": None}

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
    }
    run.summary.update(flat)
    run.log({f"{prefix}/{h.stem}": wandb.Image(str(h)) for h in heatmaps})
    for name in ("summary.json", "trace.csv", "eval.log"):
        run.save(str(out_dir / name), base_path=str(out_dir), policy="now")
    print(f"W&B: summary written to run {run.id} ({run.url})")
    run.finish()


def parse_args():
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
    p.add_argument("--wandb", action="store_true",
                   help="write the summary to W&B; resumes the run in WANDB_RUN_ID if set")
    p.add_argument("--wandb-project", default="multireflection")
    p.add_argument("--wandb-prefix", default="eval", help="key prefix for the W&B summary, e.g. eval_val")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
