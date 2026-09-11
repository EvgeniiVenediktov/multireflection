"""Final mean angular error vs. Gaussian noise level for one model.

Runs utils/eval_batched.py at --noise 0 (once) and at each level in --levels (once per
perturbation seed in --perturb-seeds), the same way utils/eval_sweep.py drives its
noise_* conditions, and plots the closed-loop final angular error against noise sigma.

    python utils/noise_curve.py --checkpoint saved_models/real/<ckpt>.pth --model-resolution 128

--from-sweep reuses an existing utils/eval_sweep.py output (its sweep_seeds.csv) instead of
running anything: the clean and noise_* rows already have exactly this data.

    python utils/noise_curve.py --from-sweep eval_results/<ckpt>_sweep/sweep_seeds.csv --label <name>

Outputs (in --out-dir, default eval_results/<label>_noise_curve/):
  noise_curve_seeds.csv  one row per level and seed (level 0 has one row, seed 0)
  noise_curve.csv         one row per level: mean and std over seeds
  noise_curve.png         final mean angular error (deg, error bars = std over seeds) vs
                           sigma, success rate on a secondary axis
Each run mode level's eval outputs also land in <out-dir>/noise_<level>/seed<k>/, exactly
as utils/eval_sweep.py lays them out for its noise_* conditions.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import eval_batched  # noqa: E402

SEED_COLUMNS = ["level", "seed", "starts", "success_rate", "adjustments_mean",
                "final_angular_error_mean", "final_angular_error_std"]
LEVEL_COLUMNS = ["level", "runs", "starts", "final_angular_error_mean", "final_angular_error_mean_std",
                  "success_rate_mean", "success_rate_std", "adjustments_mean"]


def fmt_level(level):
    return "0" if level == 0 else f"{level:g}"


def row_from_summary(summary):
    return {
        "starts": summary["n_starts"],
        "success_rate": summary["success_rate"],
        "adjustments_mean": summary["adjustments_converged"]["mean"],
        "final_angular_error_mean": summary["final_angular_error_deg"]["mean"],
        "final_angular_error_std": summary["final_angular_error_deg"]["std"],
    }


def run_levels(args, out_dir):
    """Run mode: eval_batched.run for each level x seed, as eval_sweep.py does for noise_*."""
    common = ["--checkpoint", args.checkpoint, "--data-dir", args.data_dir,
              "--batch-size", str(args.batch_size)]
    if args.starts_file:
        common += ["--starts-file", args.starts_file]
    else:
        common += ["--grid-step", str(args.grid_step)]
    if args.workers is not None:
        common += ["--workers", str(args.workers)]
    if args.model_resolution:
        common += ["--model-resolution", str(args.model_resolution)]

    seed_rows = []
    for level in args.levels:
        seeds = args.perturb_seeds if level else args.perturb_seeds[:1]
        for seed in seeds:
            argv = list(common) + ["--perturb-seed", str(seed),
                                    "--out-dir", str(out_dir / f"noise_{fmt_level(level)}" / f"seed{seed}")]
            if level:
                argv += ["--noise", str(level)]
            print(f"\n===== noise level {fmt_level(level)}, perturbation seed {seed} =====")
            summary = eval_batched.run(eval_batched.parse_args(argv))
            seed_rows.append({"level": level, "seed": seed, **row_from_summary(summary)})
    return seed_rows


def load_from_sweep(path):
    """Take the clean and noise_* rows out of an existing sweep_seeds.csv."""
    seed_rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            if cond == "clean":
                level = 0.0
            elif cond.startswith("noise_"):
                level = float(cond[len("noise_"):])
            else:
                continue
            seed_rows.append({
                "level": level,
                "seed": int(row["seed"]),
                "starts": int(row["starts"]),
                "success_rate": float(row["success_rate"]),
                "adjustments_mean": float(row["adjustments_mean"]) if row["adjustments_mean"] not in ("", "None") else None,
                "final_angular_error_mean": float(row["final_angular_error_mean"]),
                "final_angular_error_std": float(row["final_angular_error_std"]),
            })
    if not seed_rows:
        raise SystemExit(f"no clean/noise_* rows found in {path}")
    return seed_rows


def aggregate(seed_rows):
    levels = sorted({r["level"] for r in seed_rows})
    rows = []
    for level in levels:
        runs = [r for r in seed_rows if r["level"] == level]

        def values(key):
            return np.array([r[key] for r in runs if r[key] is not None], dtype=np.float64)

        def mean(key):
            v = values(key)
            return float(v.mean()) if len(v) else None

        def std(key):
            v = values(key)
            return float(v.std()) if len(v) > 1 else None

        rows.append({
            "level": level,
            "runs": len(runs),
            "starts": runs[0]["starts"],
            "final_angular_error_mean": mean("final_angular_error_mean"),
            "final_angular_error_mean_std": std("final_angular_error_mean"),
            "success_rate_mean": mean("success_rate"),
            "success_rate_std": std("success_rate"),
            "adjustments_mean": mean("adjustments_mean"),
        })
    return rows


def write_csv(rows, path, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def plot(rows, label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [r["level"] for r in rows]
    err = [r["final_angular_error_mean"] for r in rows]
    err_std = [r["final_angular_error_mean_std"] or 0.0 for r in rows]
    success = [100 * r["success_rate_mean"] for r in rows]
    run_counts = sorted({r["runs"] for r in rows})
    n_runs = str(run_counts[0]) if len(run_counts) == 1 else f"{run_counts[0]}-{run_counts[-1]}"
    starts = rows[0]["starts"]

    fig, ax1 = plt.subplots()
    ax1.errorbar(levels, err, yerr=err_std, marker="o", color="tab:blue", capsize=3, label="final angular error")
    ax1.set_xlabel("noise sigma")
    ax1.set_ylabel("final angular error (deg)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(levels, success, marker="s", color="tab:orange", label="success rate")
    ax2.set_ylabel("success rate (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 105)

    plt.title(f"{label}: noise robustness ({n_runs} run(s)/level, {starts} starts)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="required unless --from-sweep")
    p.add_argument("--model-resolution", type=int, default=None, choices=(64, 128, 256, 512))
    p.add_argument("--data-dir", default=eval_batched.DEFAULT_DATA_DIR)
    p.add_argument("--levels", default="0,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40",
                   help="comma-separated noise sigmas; 0 (clean) runs once regardless of --perturb-seeds")
    p.add_argument("--perturb-seeds", default="0,1,2", help="comma-separated perturbation seeds")
    p.add_argument("--grid-step", type=float, default=0.1)
    p.add_argument("--starts-file", default=None, help="overrides --grid-step")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--label", default=None, help="default: checkpoint stem")
    p.add_argument("--out-dir", default=None, help="default: eval_results/<label>_noise_curve/")
    p.add_argument("--from-sweep", default=None,
                   help="path to an existing eval_sweep.py sweep_seeds.csv; no GPU run, reuses its clean/noise_* rows")
    args = p.parse_args(argv)

    if args.from_sweep:
        if not args.label:
            p.error("--from-sweep needs --label")
    else:
        if not args.checkpoint:
            p.error("--checkpoint is required unless --from-sweep is given")
        if not args.label:
            args.label = Path(args.checkpoint).stem

    try:
        args.levels = sorted({float(s) for s in args.levels.split(",") if s.strip()})
    except ValueError:
        p.error("--levels must be comma-separated numbers")
    if not args.levels:
        p.error("--levels needs at least one value")

    try:
        args.perturb_seeds = [int(s) for s in args.perturb_seeds.split(",") if s.strip()]
    except ValueError:
        p.error("--perturb-seeds must be comma-separated integers")
    if not args.perturb_seeds:
        p.error("--perturb-seeds needs at least one seed")

    return args


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "eval_results" / f"{args.label}_noise_curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.from_sweep:
        seed_rows = load_from_sweep(args.from_sweep)
        wanted = set(args.levels)
        seed_rows = [r for r in seed_rows if r["level"] in wanted] or seed_rows
    else:
        seed_rows = run_levels(args, out_dir)

    write_csv(seed_rows, out_dir / "noise_curve_seeds.csv", SEED_COLUMNS)
    rows = aggregate(seed_rows)
    write_csv(rows, out_dir / "noise_curve.csv", LEVEL_COLUMNS)
    plot(rows, args.label, out_dir / "noise_curve.png")

    elapsed = time.perf_counter() - t0
    clean_err = next((r["final_angular_error_mean"] for r in rows if r["level"] == 0), None)
    worst = rows[-1]
    print(f"{args.label}: {len(rows)} noise levels, clean err {clean_err:.4f} deg -> "
          f"{fmt_level(worst['level'])} err {worst['final_angular_error_mean']:.4f} deg "
          f"(success {100 * worst['success_rate_mean']:.1f}%) in {elapsed:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
