"""Robustness sweep: utils/eval_batched.py over a list of perturbation conditions.

Runs the closed-loop replay (eval_batched.run, imported, not shelled out) for one
checkpoint under each condition and writes one comparison table. Conditions, in order:

    clean
    noise_0.05, noise_0.10, noise_0.20       additive Gaussian sigma, re-drawn per step
    brightness_0.6 .. brightness_1.4         fixed brightness factor
    contrast_0.9, contrast_1.1               fixed contrast factor (GpuAugment's 0.1 extremes)
    occlusion_1, occlusion_2                 1 or 2 boxes per trajectory, edge 0.15 to 0.40
    combined                                 noise 0.1 + brightness 0.4 + contrast 0.1 +
                                             2 boxes: the training augmentation at eval

Each condition writes its normal eval outputs to <out-dir>/<condition>/; the sweep writes
sweep.csv and sweep.md next to them. adjustments_* are over converged starts (the same
quantity eval_batched reports to W&B), the angular error columns are over all starts.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import eval_batched  # noqa: E402

CONDITIONS = [
    ("clean", []),
    ("noise_0.05", ["--noise", "0.05"]),
    ("noise_0.10", ["--noise", "0.10"]),
    ("noise_0.20", ["--noise", "0.20"]),
    ("brightness_0.6", ["--brightness-fixed", "0.6"]),
    ("brightness_0.8", ["--brightness-fixed", "0.8"]),
    ("brightness_1.2", ["--brightness-fixed", "1.2"]),
    ("brightness_1.4", ["--brightness-fixed", "1.4"]),
    ("contrast_0.9", ["--contrast-fixed", "0.9"]),
    ("contrast_1.1", ["--contrast-fixed", "1.1"]),
    ("occlusion_1", ["--occlusion-count", "1", "--occlusion-min", "0.15", "--occlusion-max", "0.40",
                     "--occlusion-prob", "1.0"]),
    ("occlusion_2", ["--occlusion-count", "2", "--occlusion-min", "0.15", "--occlusion-max", "0.40",
                     "--occlusion-prob", "1.0"]),
    ("combined", ["--noise", "0.1", "--brightness", "0.4", "--contrast", "0.1", "--occlusion-count", "2",
                  "--occlusion-min", "0.15", "--occlusion-max", "0.40", "--occlusion-prob", "1.0"]),
]

COLUMNS = ["condition", "starts", "success_rate", "adjustments_mean", "adjustments_std", "adjustments_max",
           "final_ssim_mean", "final_angular_error_mean", "final_angular_error_std", "final_angular_error_max",
           "wall_s"]


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


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_table(rows, out_dir):
    with open(out_dir / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
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
    p.add_argument("--perturb-seed", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--wandb", action="store_true", help="log the table to W&B; resumes the run in WANDB_RUN_ID if set")
    p.add_argument("--wandb-project", default="multireflection")
    p.add_argument("--wandb-prefix", default="sweep")
    args = p.parse_args()
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
              "--batch-size", str(args.batch_size), "--perturb-seed", str(args.perturb_seed)]
    if args.workers is not None:
        common += ["--workers", str(args.workers)]
    if args.no_fp16:
        common.append("--no-fp16")

    rows = []
    t0 = time.perf_counter()
    for name, flags in CONDITIONS:
        if name not in args.conditions:
            continue
        argv = common + flags + ["--out-dir", str(out_dir / name)]
        if args.perturb_ssim and flags:
            argv.append("--perturb-ssim")
        print(f"\n===== condition: {name} =====")
        summary = eval_batched.run(eval_batched.parse_args(argv))
        rows.append(row_from_summary(name, summary))
        write_table(rows, out_dir)  # partial table survives an interrupted sweep
    md = write_table(rows, out_dir)
    print(f"\nsweep of {len(rows)} conditions in {time.perf_counter() - t0:.0f} s -> {out_dir}\n")
    print(md)
    if args.wandb:
        log_to_wandb(rows, args.checkpoint, args.wandb_project, args.wandb_prefix)


if __name__ == "__main__":
    main()
