"""Robustness sweep: utils/eval_batched.py over a list of perturbation conditions.

Runs the closed-loop replay (eval_batched.run, imported, not shelled out) for one
checkpoint under each condition and writes one comparison table. Conditions, in order:

    clean
    noise_0.02 .. noise_0.30                 additive Gaussian sigma, re-drawn per step
    brightness_0.4 .. brightness_1.6         fixed brightness factor
    contrast_0.9, contrast_1.1               fixed contrast factor (GpuAugment's 0.1 extremes)
    occlusion_1, occlusion_2                 1 or 2 boxes per trajectory, edge 0.15 to 0.40
    occlusion_1x10 .. occlusion_2x50         1 or 2 boxes of a fixed edge, 10 to 50% of the image
    occlusion_{1x30,2x30,2x50}_bright        the same boxes filled white (saturated) instead of black
    occlusion_{1x30,2x30,2x50}_rot           rotated by an angle in [-45, 45] deg (a square repeats every 90)
    occlusion_{1x30,2x30,2x50}_rotbright     rotated and white
    occlusion_2_mixed                        2 boxes, edge 0.15-0.40, rotated in [-90, 90], white with p 0.5:
                                             the training occlusion (every box applied) at eval
    occlusion_2x30_fill                      2 boxes of 30%, gray fill uniform in [0, 1] per box
    occlusion_{2x30,2x50}_rotfill            rotated in [-45, 45] deg and gray
    occlusion_2_mixedfill                    2 boxes, edge 0.15-0.40, rotated in [-90, 90], gray in [0, 1]:
                                             the gray-fill training occlusion (since 2026-09-11) at eval
    combined                                noise 0.1 + brightness 0.4 + contrast 0.1 +
                                             2 boxes: the training augmentation at eval
    combined_harsh                           noise 0.2 + brightness 0.6 + 2 boxes of edge 30%

Every condition with a perturbation runs once per perturbation seed (--perturb-seeds,
default 0,1,2: different box positions, brightness/contrast draws and noise); clean has
nothing random and runs once. Each run writes its normal eval outputs to
<out-dir>/<condition>/seed<k>/. The sweep writes sweep_seeds.csv (one row per condition and
seed) and sweep.csv / sweep.md (one row per condition: mean over seeds, *_std = standard
deviation over seeds, max = max over seeds). adjustments_* are over converged starts (the
same quantity eval_batched reports to W&B), the angular error columns are over all starts.
To compare models of different input sizes on the same frames, run every one on the 512 px
bank with --model-resolution set to the model's size.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import eval_batched  # noqa: E402

def _boxes(count, lo, hi):
    return ["--occlusion-count", str(count), "--occlusion-min", lo, "--occlusion-max", hi, "--occlusion-prob", "1.0"]


CONDITIONS = [
    ("clean", []),
    *[(f"noise_{s}", ["--noise", s]) for s in ("0.02", "0.05", "0.10", "0.15", "0.20", "0.30")],
    *[(f"brightness_{f}", ["--brightness-fixed", f]) for f in ("0.4", "0.6", "0.8", "1.2", "1.4", "1.6")],
    ("contrast_0.9", ["--contrast-fixed", "0.9"]),
    ("contrast_1.1", ["--contrast-fixed", "1.1"]),
    ("occlusion_1", _boxes(1, "0.15", "0.40")),
    ("occlusion_2", _boxes(2, "0.15", "0.40")),
    *[(f"occlusion_{n}x{e}", _boxes(n, f"0.{e}", f"0.{e}")) for n in (1, 2) for e in ("10", "20", "30", "40", "50")],
    *[(f"occlusion_{n}x{e}_{kind}", _boxes(n, f"0.{e}", f"0.{e}") + flags)
      for kind, flags in (("bright", ["--occlusion-bright-prob", "1.0"]),
                          ("rot", ["--occlusion-angle", "45"]),
                          ("rotbright", ["--occlusion-angle", "45", "--occlusion-bright-prob", "1.0"]))
      for n, e in ((1, "30"), (2, "30"), (2, "50"))],
    ("occlusion_2_mixed", _boxes(2, "0.15", "0.40") + ["--occlusion-angle", "90", "--occlusion-bright-prob", "0.5"]),
    ("occlusion_2x30_fill", _boxes(2, "0.30", "0.30") + ["--occlusion-fill-max", "1.0"]),
    ("occlusion_2x30_rotfill", _boxes(2, "0.30", "0.30") + ["--occlusion-angle", "45", "--occlusion-fill-max", "1.0"]),
    ("occlusion_2x50_rotfill", _boxes(2, "0.50", "0.50") + ["--occlusion-angle", "45", "--occlusion-fill-max", "1.0"]),
    ("occlusion_2_mixedfill", _boxes(2, "0.15", "0.40") + ["--occlusion-angle", "90", "--occlusion-fill-max", "1.0"]),
    ("combined", ["--noise", "0.1", "--brightness", "0.4", "--contrast", "0.1", *_boxes(2, "0.15", "0.40")]),
    ("combined_harsh", ["--noise", "0.2", "--brightness-fixed", "0.6", *_boxes(2, "0.30", "0.30")]),
]

SEED_COLUMNS = ["condition", "seed", "starts", "success_rate", "adjustments_mean", "adjustments_std",
                "adjustments_max", "final_ssim_mean", "final_angular_error_mean", "final_angular_error_std",
                "final_angular_error_max", "wall_s"]
COLUMNS = ["condition", "seeds", "starts", "success_rate", "success_rate_std", "adjustments_mean",
           "adjustments_mean_std", "adjustments_max", "final_ssim_mean", "final_angular_error_mean",
           "final_angular_error_mean_std", "final_angular_error_max", "wall_s"]


def row_from_summary(name, summary):
    adj = summary["adjustments_converged"]
    err = summary["final_angular_error_deg"]
    return {
        "condition": name,
        "starts": summary["n_starts"],
        "success_rate": summary["success_rate"],
        "adjustments_mean": adj["mean"],
        "adjustments_std": adj["std"],
        "adjustments_max": adj["max"],
        "final_ssim_mean": summary["final_ssim"]["mean"],
        "final_angular_error_mean": err["mean"],
        "final_angular_error_std": err["std"],
        "final_angular_error_max": err["max"],
        "wall_s": summary["wall_time_s"],
    }


def aggregate(name, runs):
    """One row for a condition from its per-seed rows: mean, std and max over the seeds."""
    def values(key):
        return np.array([r[key] for r in runs if r[key] is not None], dtype=np.float64)

    def mean(key):
        v = values(key)
        return float(v.mean()) if len(v) else None

    def std(key):
        v = values(key)
        return float(v.std()) if len(v) > 1 else None

    def vmax(key):
        v = values(key)
        return float(v.max()) if len(v) else None

    return {
        "condition": name,
        "seeds": len(runs),
        "starts": runs[0]["starts"],
        "success_rate": mean("success_rate"),
        "success_rate_std": std("success_rate"),
        "adjustments_mean": mean("adjustments_mean"),
        "adjustments_mean_std": std("adjustments_mean"),
        "adjustments_max": vmax("adjustments_max"),
        "final_ssim_mean": mean("final_ssim_mean"),
        "final_angular_error_mean": mean("final_angular_error_mean"),
        "final_angular_error_mean_std": std("final_angular_error_mean"),
        "final_angular_error_max": vmax("final_angular_error_max"),
        "wall_s": float(values("wall_s").sum()),
    }


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_csv(rows, path, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def write_table(rows, seed_rows, out_dir):
    write_csv(seed_rows, out_dir / "sweep_seeds.csv", SEED_COLUMNS)
    write_csv(rows, out_dir / "sweep.csv", COLUMNS)
    md =["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for r in rows:
        md.append("| " + " | ".join(fmt(r[c]) for c in COLUMNS) + " |")
    (out_dir / "sweep.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


def log_to_wandb(rows, checkpoint, project, prefix):
    import wandb

    run_id = os.environ.get("WANDB_RUN_ID")
    run = wandb.init(project=project, id=run_id, resume="allow" if run_id else None,
                     name=None if run_id else f"sweep_{Path(checkpoint).stem}", job_type="eval")
    table = wandb.Table(columns=COLUMNS, data=[[r[c] for c in COLUMNS] for r in rows])
    run.log({f"{prefix}/table": table})
    flat = {}
    for r in rows:
        flat[f"{prefix}/{r['condition']}/success_rate"] = r["success_rate"]
        flat[f"{prefix}/{r['condition']}/final_angular_error_mean"] = r["final_angular_error_mean"]
    run.summary.update(flat)
    print(f"W&B: sweep written to run {run.id} ({run.url})")
    run.finish()


def parse_args():
    names = [c[0] for c in CONDITIONS]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(eval_batched.DEFAULT_CHECKPOINT))
    p.add_argument("--data-dir", default=eval_batched.DEFAULT_DATA_DIR)
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="default: eval_results/<checkpoint stem>_sweep/")
    p.add_argument("--conditions", default=None,
                   help="comma-separated subset of: " + ", ".join(names))
    p.add_argument("--perturb-ssim", action="store_true",
                   help="passed through: SSIM stop test on the perturbed frame (ignored for clean)")
    p.add_argument("--perturb-seeds", default="0,1,2",
                   help="comma-separated perturbation seeds; every perturbed condition runs once per seed")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--model-resolution", type=int, default=None,
                   help="passed through: area-downscale the (perturbed) frames to the model's input size")
    p.add_argument("--wandb", action="store_true", help="log the table to W&B; resumes the run in WANDB_RUN_ID if set")
    p.add_argument("--wandb-project", default="multireflection")
    p.add_argument("--wandb-prefix", default="sweep")
    args = p.parse_args()
    try:
        args.perturb_seeds = [int(s) for s in args.perturb_seeds.split(",") if s.strip()]
    except ValueError:
        p.error("--perturb-seeds must be comma-separated integers")
    if not args.perturb_seeds:
        p.error("--perturb-seeds needs at least one seed")
    if args.conditions:
        wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in names]
        if unknown:
            p.error(f"unknown conditions: {', '.join(unknown)}")
        args.conditions = wanted
    else:
        args.conditions = names
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "eval_results" / f"{Path(args.checkpoint).stem}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = ["--checkpoint", args.checkpoint, "--data-dir", args.data_dir, "--grid-step", str(args.grid_step),
              "--batch-size", str(args.batch_size)]
    if args.workers is not None:
        common += ["--workers", str(args.workers)]
    if args.no_fp16:
        common.append("--no-fp16")
    if args.model_resolution:
        common += ["--model-resolution", str(args.model_resolution)]

    rows, seed_rows = [], []
    t0 = time.perf_counter()
    for name, flags in CONDITIONS:
        if name not in args.conditions:
            continue
        # Without a perturbation nothing is random, so one run stands for every seed
        seeds = args.perturb_seeds if flags else args.perturb_seeds[:1]
        runs = []
        for seed in seeds:
            argv = common + flags + ["--perturb-seed", str(seed), "--out-dir", str(out_dir / name / f"seed{seed}")]
            if args.perturb_ssim and flags:
                argv.append("--perturb-ssim")
            print(f"\n===== condition: {name}, perturbation seed {seed} =====")
            summary = eval_batched.run(eval_batched.parse_args(argv))
            runs.append({"seed": seed, **row_from_summary(name, summary)})
        seed_rows += runs
        rows.append(aggregate(name, runs))
        write_table(rows, seed_rows, out_dir)  # partial tables survive an interrupted sweep
    md = write_table(rows, seed_rows, out_dir)
    print(f"\nsweep of {len(rows)} conditions x seeds {args.perturb_seeds} in {time.perf_counter() - t0:.0f} s -> {out_dir}\n")
    print(md)
    if args.wandb:
        log_to_wandb(rows, args.checkpoint, args.wandb_project, args.wandb_prefix)


if __name__ == "__main__":
    main()
