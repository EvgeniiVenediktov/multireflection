"""Open-loop evaluation of one checkpoint on the TEST positions of the spatial hold-out split.

For every TEST image of the split (square holes around evaluation origins, see
holdout/README.md) the model predicts (x, y) once and the prediction is compared with the
position in the file name; there is no closed loop. TEST frames are decoded once (thread
pool) and reused for every condition and seed.

Frames come from the 512 px bank so that perturbations act on the camera-size frame, as in
utils/eval_sweep.py: utils/eval_batched.BatchedPredictor applies the Perturbation of the
condition, box-averages to --model-resolution and runs the model. The perturbation index of
a sample is its TEST sample index, so a seed gives every sample the same boxes and factors
for every model. A bank already at the model resolution is accepted (the box average is then
a no-op, the perturbation acts on the small frame), e.g. for a quick local check.

Conditions are names from utils/eval_sweep.CONDITIONS; perturbed ones run once per
--perturb-seeds seed, clean runs once.

Nearest-neighbour baseline (clean frames only): each TEST sample (all of them at an NN
resolution <= 128 px, 5000 otherwise, or --nn-test-samples) is assigned the label of the
TRAIN image at the smallest L2 distance, both box-averaged from the bank's frames to
--nn-resolution (default: the model resolution) and quantized to 8 bit. TRAIN frames are
reduced while loading, so memory is N_train x r x r bytes. When --nn-resolution is not
given and that does not fit in memory, the largest resolution that fits is used and the
reason is recorded. The search runs on the GPU in float32 chunks.

Outputs (in --out-dir, default <checkpoint dir>/holdout_test/):
  metrics_seeds.csv   one row per condition and seed
  metrics.csv         one row per condition: mean over seeds, *_std = std over seeds
  buckets.csv         per condition, seed and distance to TRAIN (0.01 deg buckets): n, mean/std error
  nn_metrics.csv      NN baseline and the model (clean) on the same TEST samples
  nn_buckets.csv      the same two, per distance bucket
  summary.json        settings, sample sizes, timings
  error_vs_distance.png
Errors are in degrees; ang_err = Euclidean norm of (pred - true); err_x/err_y = signed
per-axis error (mean and std), abs_err_x/abs_err_y = mean absolute per-axis error.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import eval_batched, eval_sweep  # noqa: E402

DEFAULT_SPLIT = "/home/evv/data/holdout_split/split.csv"
DEFAULT_CONDITIONS = ["clean", "noise_0.10", "noise_0.20", "noise_0.30", "brightness_0.4", "brightness_0.6",
                      "brightness_1.6", "occlusion_2x30", "occlusion_2x40", "occlusion_2x30_rotbright",
                      "occlusion_2_mixed", "combined_harsh"]
PLOT_PREFERRED = ["occlusion_2_mixed", "noise_0.20"]  # perturbed conditions drawn in the plot, if run
MAX_TEST_SEED = 0  # --max-test-samples subset is always drawn with this seed
STAT_KEYS = ["ang_err_mean", "ang_err_std", "ang_err_median", "ang_err_p95", "ang_err_max",
             "err_x_mean", "err_y_mean", "abs_err_x_mean", "abs_err_y_mean", "err_x_std", "err_y_std"]
SEED_COLUMNS = ["condition", "seed", "n", *STAT_KEYS, "wall_s"]
AGG_COLUMNS = ["condition", "seeds", "n", *[c for k in STAT_KEYS for c in (k, f"{k}_std")], "wall_s"]
BUCKET_COLUMNS = ["condition", "seed", "dist_deg", "n", "ang_err_mean", "ang_err_std"]
NN_COLUMNS = ["method", "resolution", "n", *STAT_KEYS]
NN_BUCKET_COLUMNS = ["method", "resolution", "dist_deg", "n", "ang_err_mean", "ang_err_std"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def position(name):
    xs, ys = name[:-4].split("_y")
    return float(xs[1:]), float(ys)


def read_split(path):
    """TEST names, true positions and distances to TRAIN; TRAIN names and positions."""
    path = Path(path)
    splits = {}
    csv_names = {"train": [], "test": []}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["ix"]), int(r["iy"]))
            splits[key] = (r["split"], float(r["dist_to_train_deg"]))
            if r["split"] in csv_names:
                csv_names[r["split"]].append(f"x{r['x_deg']}_y{r['y_deg']}.jpg")
    out = {}
    for part in ("train", "test"):
        names_file = path.parent / f"{part}_names.txt"
        if names_file.exists():
            names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
        else:
            names = csv_names[part]
        pos = np.array([position(n) for n in names], dtype=np.float64)
        keys = np.rint(pos * 100).astype(int)
        info = [splits.get((int(a), int(b))) for a, b in keys]
        bad = [n for n, i in zip(names, info) if i is None or i[0] != part]
        if bad:
            raise RuntimeError(f"{len(bad)} {part} names disagree with {path}, e.g. {bad[0]}")
        out[part] = (names, pos, np.array([i[1] for i in info], dtype=np.float64))
    return out


def box_average(img, size):
    """Integer-factor box average (cv2.INTER_AREA), the reduction the banks and avg_pool2d use."""
    if img.shape[0] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_frames(data_dir, names, workers, size=None, shape=None):
    """uint8 array (N, s, s): frames decoded in a thread pool, box-averaged to size while loading."""
    data_dir = Path(data_dir)
    size = size or shape[0]
    arr = np.empty((len(names), size, size), dtype=np.uint8)

    def load(i):
        img = cv2.imread(str(data_dir / names[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"failed to read {data_dir / names[i]}")
        if shape is not None and img.shape != shape:
            raise ValueError(f"{names[i]}: shape {img.shape}, expected {shape}")
        arr[i] = box_average(img, size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(load, range(len(names))):
            pass
    return arr


def error_stats(pred, true):
    e = pred.astype(np.float64) - true
    a = np.hypot(e[:, 0], e[:, 1])
    return {
        "n": int(len(a)),
        "ang_err_mean": float(a.mean()),
        "ang_err_std": float(a.std()),
        "ang_err_median": float(np.median(a)),
        "ang_err_p95": float(np.percentile(a, 95)),
        "ang_err_max": float(a.max()),
        "err_x_mean": float(e[:, 0].mean()),
        "err_y_mean": float(e[:, 1].mean()),
        "abs_err_x_mean": float(np.abs(e[:, 0]).mean()),
        "abs_err_y_mean": float(np.abs(e[:, 1]).mean()),
        "err_x_std": float(e[:, 0].std()),
        "err_y_std": float(e[:, 1].std()),
    }, a


def bucket_stats(ang, dist):
    keys = np.rint(dist * 100).astype(int)
    rows = []
    for k in np.unique(keys):
        a = ang[keys == k]
        rows.append({"dist_deg": round(k / 100, 2), "n": int(len(a)),
                     "ang_err_mean": float(a.mean()), "ang_err_std": float(a.std())})
    return rows


def aggregate(name, runs):
    row = {"condition": name, "seeds": len(runs), "n": runs[0]["n"]}
    for k in STAT_KEYS:
        v = np.array([r[k] for r in runs], dtype=np.float64)
        row[k] = float(v.mean())
        row[f"{k}_std"] = float(v.std()) if len(v) > 1 else None
    row["wall_s"] = float(sum(r["wall_s"] for r in runs))
    return row


def write_csv(rows, path, columns):
    def fmt(v):
        return "" if v is None else f"{v:.6g}" if isinstance(v, float) else v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c)) for c in columns})


def available_memory():
    """Bytes this process may still allocate: MemAvailable, capped by the Slurm memory limit."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    mb = os.environ.get("SLURM_MEM_PER_NODE")
    if not mb and os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_PER_TASK"):
        mb = int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_PER_TASK"])
    if mb:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        limit = int(mb) * 2 ** 20 - rss
        avail = limit if avail is None else min(avail, limit)
    return avail


def nn_samples_default(resolution, n_test):
    return n_test if resolution <= 128 else min(5000, n_test)


def plan_nn(args, frame_size, n_test_frames, n_test, n_train):
    """NN resolution and TEST sample count; falls back to a smaller resolution if memory is short.

    Frames + NN features + 4 GB must stay under 75 percent of the available memory (page cache of the
    decoded JPEGs, CUDA host allocations and batch copies need the rest)."""
    margin = 4 * 2 ** 30

    def need(r, q):
        return (n_test_frames * frame_size ** 2 + (n_train + q) * r * r + margin) / 0.75

    def samples(r):
        return min(args.nn_test_samples, n_test) if args.nn_test_samples else nn_samples_default(r, n_test)

    avail = available_memory()
    if args.nn_resolution:
        r = args.nn_resolution
        if frame_size % r:
            raise SystemExit(f"--nn-resolution {r} does not divide the bank's {frame_size} px frames")
        reason = "--nn-resolution"
        if avail is not None and need(r, samples(r)) > avail:
            log(f"warning: NN at {r} px needs ~{need(r, samples(r)) / 2 ** 30:.1f} GB, "
                f"~{avail / 2 ** 30:.1f} GB available")
        return r, samples(r), reason, avail
    r = args.model_resolution
    reason = "model resolution"
    while avail is not None and need(r, samples(r)) > avail and r % 2 == 0 and r > 16:
        r //= 2
        reason = (f"model resolution {args.model_resolution} px needs "
                  f"~{need(args.model_resolution, samples(args.model_resolution)) / 2 ** 30:.1f} GB, "
                  f"~{avail / 2 ** 30:.1f} GB available")
    return r, samples(r), reason, avail


@torch.inference_mode()
def nearest_neighbours(train, queries, device):
    """Index into train of the L2-nearest row for every query; uint8 arrays (N, r, r), float32 on device."""
    n_train, n_query = len(train), len(queries)
    dim = train.shape[1] * train.shape[2]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        budget = int(torch.cuda.mem_get_info(device)[0] * 0.6)
    else:
        budget = 4 * 2 ** 30
    # Query chunk (float32) at most half the budget; train chunk: uint8 + float32 + squares = 9 bytes/px;
    # distance matrix 4 bytes per query x train pair.
    qc = max(1, min(n_query, budget // (8 * dim)))
    tc = max(1, min(n_train, (budget - 4 * dim * qc) // (9 * dim + 4 * qc)))
    train_t = torch.from_numpy(train)
    best = np.empty(n_query, dtype=np.int64)
    for qs in range(0, n_query, qc):
        q = torch.from_numpy(queries[qs:qs + qc]).to(device).reshape(-1, dim).float()
        best_d = torch.full((len(q),), float("inf"), device=device)
        best_i = torch.zeros(len(q), dtype=torch.long, device=device)
        for ts in range(0, n_train, tc):
            t = train_t[ts:ts + tc].to(device).reshape(-1, dim).float()
            # ||q - t||^2 = ||q||^2 + ||t||^2 - 2 q.t; ||q||^2 is constant per row, so it is left out
            d = torch.addmm((t * t).sum(1), q, t.T, beta=1.0, alpha=-2.0)
            m, a = d.min(1)
            better = m < best_d
            best_d = torch.where(better, m, best_d)
            best_i = torch.where(better, a + ts, best_i)
            del t, d
        best[qs:qs + len(q)] = best_i.cpu().numpy()
    return best, {"query_chunk": int(qc), "train_chunk": int(tc), "dtype": "float32"}


def plot(seed_buckets, nn_buckets, plot_conditions, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def per_bucket(rows):
        # mean over seeds of the per-seed bucket means
        by = {}
        for r in rows:
            by.setdefault(r["dist_deg"], []).append(r["ang_err_mean"])
        d = sorted(by)
        return d, [float(np.mean(by[k])) for k in d]

    series = []
    if "clean" in seed_buckets:
        series.append(("model, clean", seed_buckets["clean"], "#2a78d6", "-"))
    for name, color in zip(plot_conditions, ("#eb6834", "#1baf7a")):
        series.append((f"model, {name}", seed_buckets[name], color, "-"))
    if nn_buckets:
        series.append(("nearest-neighbour baseline, clean", nn_buckets, "#52514e", "--"))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    allv = []
    for label, rows, color, ls in series:
        d, v = per_bucket(rows)
        allv += v
        ax.plot(d, v, ls, color=color, lw=2, marker="o", ms=5, label=label)
    positive = [v for v in allv if v > 0]
    if positive and max(positive) / min(positive) > 30:
        ax.set_yscale("log")
    ax.set_xlabel("distance of the TEST position to the nearest TRAIN position (deg)")
    ax.set_ylabel("mean angular error (deg)")
    ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
    ax.grid(True, color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b0afab")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv=None):
    names = [c[0] for c in eval_sweep.CONDITIONS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-resolution", type=int, required=True, choices=(64, 128, 256, 512))
    p.add_argument("--data-dir", default=eval_batched.DEFAULT_DATA_DIR,
                   help="image bank; 512 px normally, a bank at the model resolution also works")
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   help="split.csv; train_names.txt/test_names.txt next to it are used when present")
    p.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/holdout_test/")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help="comma-separated names from utils/eval_sweep.CONDITIONS: " + ", ".join(names))
    p.add_argument("--perturb-seeds", default="0,1,2", help="comma-separated; clean runs once")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=None, help="decode threads (default: usable CPUs)")
    p.add_argument("--no-fp16", action="store_true", help="disable fp16 autocast for the model")
    p.add_argument("--max-test-samples", type=int, default=None,
                   help=f"debug: evaluate a random subset (seed {MAX_TEST_SEED}) of this many TEST samples")
    p.add_argument("--nn", action=argparse.BooleanOptionalAction, default=True,
                   help="nearest-neighbour baseline (default on)")
    p.add_argument("--nn-test-samples", type=int, default=None,
                   help="TEST samples for the NN baseline (default: all at NN resolution <= 128, else 5000)")
    p.add_argument("--nn-seed", type=int, default=0, help="seed of the NN TEST subsample")
    p.add_argument("--nn-resolution", type=int, default=None,
                   help="NN comparison resolution (default: the model resolution, reduced if memory is short)")
    args = p.parse_args(argv)
    try:
        args.perturb_seeds = [int(s) for s in args.perturb_seeds.split(",") if s.strip()]
    except ValueError:
        p.error("--perturb-seeds must be comma-separated integers")
    if not args.perturb_seeds:
        p.error("--perturb-seeds needs at least one seed")
    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in names]
    if unknown:
        p.error(f"unknown conditions: {', '.join(unknown)}")
    args.conditions = wanted
    if args.workers is None:
        args.workers = len(os.sched_getaffinity(0))
    return args


def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    timings = {}
    flags_of = dict(eval_sweep.CONDITIONS)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent / "holdout_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = read_split(args.split)
    test_names, test_pos, test_dist = split["test"]
    train_names, train_pos, _ = split["train"]
    n_test_split = len(test_names)
    test_idx = np.arange(n_test_split)
    if args.max_test_samples and args.max_test_samples < n_test_split:
        test_idx = np.sort(np.random.default_rng(MAX_TEST_SEED).choice(n_test_split, args.max_test_samples,
                                                                          replace=False))
    test_names = [test_names[i] for i in test_idx]
    test_pos, test_dist = test_pos[test_idx], test_dist[test_idx]
    n_test = len(test_names)

    first = cv2.imread(str(Path(args.data_dir) / test_names[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise SystemExit(f"cannot read {Path(args.data_dir) / test_names[0]}")
    frame_size = first.shape[0]
    if first.shape[0] != first.shape[1] or frame_size % args.model_resolution:
        raise SystemExit(f"bank frames {first.shape} are not a square multiple of {args.model_resolution} px")

    nn_plan = None
    if args.nn:
        nn_res, nn_q, nn_reason, avail = plan_nn(args, frame_size, n_test, n_test, len(train_names))
        nn_plan = {"resolution": nn_res, "test_samples": nn_q, "resolution_reason": nn_reason,
                   "available_memory_gb": None if avail is None else round(avail / 2 ** 30, 1)}

    t0 = time.perf_counter()
    frames = load_frames(args.data_dir, test_names, args.workers, shape=first.shape)
    timings["load_test_s"] = time.perf_counter() - t0
    log(f"TEST: {n_test} of {n_test_split} frames ({frame_size} px) loaded in {timings['load_test_s']:.1f} s")

    predictor = eval_batched.BatchedPredictor(args.checkpoint, args.batch_size, not args.no_fp16,
                                              perturb=None, model_resolution=args.model_resolution)
    device = predictor.device
    idx = np.arange(n_test)
    seed_rows, rows, bucket_rows = [], [], []
    seed_buckets = {}
    clean_pred = None
    t_eval = time.perf_counter()
    for name in args.conditions:
        flags = flags_of[name]
        runs = []
        for seed in (args.perturb_seeds if flags else args.perturb_seeds[:1]):
            t0 = time.perf_counter()
            if flags:
                pargs = eval_batched.parse_args(flags + ["--perturb-seed", str(seed)])
                predictor.perturb = eval_batched.Perturbation(pargs, n_test, first.shape, device)
            else:
                predictor.perturb = None
            pred = predictor.predict(frames, idx=idx)
            stats, ang = error_stats(pred, test_pos)
            if not flags:
                clean_pred = pred
            wall = time.perf_counter() - t0
            runs.append({"condition": name, "seed": seed, **stats, "wall_s": wall})
            b = [{"condition": name, "seed": seed, **r} for r in bucket_stats(ang, test_dist)]
            bucket_rows += b
            seed_buckets.setdefault(name, []).extend(b)
            log(f"  {name:<26} seed {seed}  mean {stats['ang_err_mean']:.4f} deg  {wall:.1f} s")
        seed_rows += runs
        rows.append(aggregate(name, runs))
        # partial tables survive an interrupted run
        write_csv(seed_rows, out_dir / "metrics_seeds.csv", SEED_COLUMNS)
        write_csv(rows, out_dir / "metrics.csv", AGG_COLUMNS)
        write_csv(bucket_rows, out_dir / "buckets.csv", BUCKET_COLUMNS)
    predictor.perturb = None
    timings["model_eval_s"] = time.perf_counter() - t_eval
    throughput = predictor.images_seen / predictor.seconds if predictor.seconds else None

    nn_rows, nn_bucket_rows = [], []
    nn_info = None
    if args.nn:
        r, q = nn_plan["resolution"], nn_plan["test_samples"]
        sub = np.arange(n_test)
        if q < n_test:
            sub = np.sort(np.random.default_rng(args.nn_seed).choice(n_test, q, replace=False))
        t0 = time.perf_counter()
        train_feat = load_frames(args.data_dir, train_names, args.workers, size=r, shape=first.shape)
        timings["nn_load_train_s"] = time.perf_counter() - t0
        log(f"NN: {len(train_names)} TRAIN frames reduced to {r} px in {timings['nn_load_train_s']:.1f} s")
        test_feat = np.stack([box_average(frames[i], r) for i in sub])
        t0 = time.perf_counter()
        nearest, chunks = nearest_neighbours(train_feat, test_feat, device)
        timings["nn_search_s"] = time.perf_counter() - t0
        del train_feat
        methods = [("nn", train_pos[nearest])]
        if clean_pred is not None:
            methods.append(("model_clean_same_samples", clean_pred[sub]))
        for method, pred in methods:
            stats, ang = error_stats(pred, test_pos[sub])
            nn_rows.append({"method": method, "resolution": r if method == "nn" else args.model_resolution, **stats})
            nn_bucket_rows += [{"method": method, "resolution": nn_rows[-1]["resolution"], **b}
                               for b in bucket_stats(ang, test_dist[sub])]
        write_csv(nn_rows, out_dir / "nn_metrics.csv", NN_COLUMNS)
        write_csv(nn_bucket_rows, out_dir / "nn_buckets.csv", NN_BUCKET_COLUMNS)
        nn_info = {**nn_plan, "test_samples": int(len(sub)), "seed": args.nn_seed,
                   "train_samples": len(train_names), **chunks}

    plot_conditions = [c for c in PLOT_PREFERRED if c in seed_buckets]
    plot_conditions += [c for c in args.conditions if flags_of[c] and c not in plot_conditions]
    plot_conditions = plot_conditions[:2]
    title = (f"{Path(args.checkpoint).stem}\nTEST n={n_test}, model {args.model_resolution} px"
             + (f", NN {nn_info['resolution']} px on n={nn_info['test_samples']}" if nn_info else ""))
    plot(seed_buckets, [b for b in nn_bucket_rows if b["method"] == "nn"], plot_conditions,
         out_dir / "error_vs_distance.png", title)
    timings["total_s"] = time.perf_counter() - t_start

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_resolution": args.model_resolution,
        "data_dir": str(args.data_dir),
        "bank_resolution": frame_size,
        "split": str(Path(args.split).resolve()),
        "test_samples_in_split": n_test_split,
        "test_samples": n_test,
        "max_test_samples": args.max_test_samples,
        "max_test_seed": MAX_TEST_SEED if n_test < n_test_split else None,
        "train_samples_in_split": len(train_names),
        "conditions": {c: flags_of[c] for c in args.conditions},
        "perturb_seeds": args.perturb_seeds,
        "batch_size": args.batch_size,
        "fp16": predictor.fp16,
        "workers": args.workers,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "model_images_per_s": throughput,
        "nn": nn_info,
        "plot_conditions": plot_conditions,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "metrics": rows,
        "nn_metrics": nn_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    clean = next((r for r in rows if r["condition"] == "clean"), None)
    worst = max(rows, key=lambda r: r["ang_err_mean"])
    print(f"{Path(args.checkpoint).stem}: TEST n={n_test}/{n_test_split}, bank {frame_size} px -> model "
          f"{args.model_resolution} px, {len(rows)} conditions, seeds {args.perturb_seeds}")
    if clean:
        print(f"clean: mean {clean['ang_err_mean']:.4f} deg, median {clean['ang_err_median']:.4f}, "
              f"p95 {clean['ang_err_p95']:.4f}, max {clean['ang_err_max']:.4f}")
    if worst is not clean:
        std = worst["ang_err_mean_std"]
        print(f"worst: {worst['condition']} mean {worst['ang_err_mean']:.4f} deg"
              + (f" (std over seeds {std:.4f})" if std is not None else ""))
    if nn_rows:
        same = next((r for r in nn_rows if r["method"] == "model_clean_same_samples"), None)
        print(f"NN baseline ({nn_info['resolution']} px, n={nn_info['test_samples']}): mean "
              f"{nn_rows[0]['ang_err_mean']:.4f} deg"
              + (f"; model clean on the same samples {same['ang_err_mean']:.4f} deg" if same else ""))
    print("timings: " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items()) + f" -> {out_dir}")
    return summary


if __name__ == "__main__":
    main()
