#!/bin/bash
# bench_accel_l40s.slurm - timing only: ppo.py iterations with ACCEL levels off and on
# (--accel-levels), r9's training config otherwise (camera, 1024 envs x 128 steps, all
# weights trainable), one run after the other on the same node; no evals, no W&B.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/bench_accel_l40s.slurm
#   sbatch --export=ALL,ITERS=24,LEVELS=4096,ACCEL_ARGS="--accel-edits 512" rl/cluster/bench_accel_l40s.slurm
#
# PROJECT_DIR may point to an export of a commit (VENV_PY stays the project venv).
# Output: $RUNS_ROOT/bench_accel/<jobid>/{off,on}/train.csv and bench.md: medians over the
# iterations after SKIP of rollout, update, wall time per iteration, pool refill and
# level update time, plus the level state at the end.

#SBATCH --job-name=mfl-bench-accel
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
VENV_PY="${VENV_PY:-/ihome/kchen/evv13/multireflection/.venv/bin/python}"
OUT="$RUNS_ROOT/bench_accel/${SLURM_JOB_ID:-manual}"
INIT_FROM="${INIT_FROM:-$RUNS_ROOT/4459206/best_snapshot_it2100.pt}"  # r9's init (r8 best)
ORACLES="${ORACLES:-$RUNS_ROOT/4467696}"  # Run dir with this config's eval oracles (copied)
ITERS="${ITERS:-16}"
SKIP="${SKIP:-4}"  # Warm-up iterations left out of the medians (compile, level fill)
LEVELS="${LEVELS:-4096}"
ACCEL_ARGS="${ACCEL_ARGS:-}"  # Further --accel-* flags of the "on" run

module purge
cd "$PROJECT_DIR/rl"
mkdir -p "$OUT"
CACHE="${SLURM_SCRATCH:-$OUT}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date); code $PROJECT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

COMMON=(--setups full --scenarios pool --atlas "$RUNS_ROOT/atlas/atlas_v1.pt"
        --eval-tables "$RUNS_ROOT/eval_sets" --sep-range 20 400 --laser-range -8 8
        --laser-offset-radius 1.5 --init-from "$INIT_FROM" --history-len 32
        --near-best-frac 0.25 --pool-region-mix 0.2,0.2,0.25,0.35 --lr 1e-4 --best-metric region
        --camera --camera-set rotation_p=0 --ent-target-start 0.7 --ent-target-end 0.7
        --no-encoder-freeze --encoder-lr-scale 0.1 --log-path-head --logp-coef 0.5
        --eval-every 0 --no-wandb --total-env-steps $((ITERS * 1024 * 128)))
for mode in off on; do
    RD="$OUT/$mode"
    mkdir -p "$RD"
    cp "$ORACLES"/oracle_*.pt "$RD"/ 2>/dev/null || true
    EXTRA=()
    [ "$mode" = on ] && EXTRA=(--accel-levels "$LEVELS" $ACCEL_ARGS)
    echo "== $mode: ${EXTRA[*]:-(levels off)}"
    srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" ppo.py --run-dir "$RD" \
        "${COMMON[@]}" "${EXTRA[@]}"
done

"$VENV_PY" - "$OUT" "$SKIP" <<'EOF'
import csv, math, os, statistics, sys

out, skip = sys.argv[1], int(sys.argv[2])
cols = ("time_rollout_s", "time_update_s", "wall_s", "pool/refresh_s", "accel/update_s")
res = {}
for mode in ("off", "on"):
    rows = list(csv.DictReader(open(os.path.join(out, mode, "train.csv"))))
    for r0, r1 in zip(rows, rows[1:]):
        r1["wall_s"] = 60 * (float(r1["wall_min"]) - float(r0["wall_min"]))
    use = [r for r in rows[1:] if int(r["iteration"]) > skip]
    med = lambda k: statistics.median(float(r[k]) for r in use) if use and k in use[0] else math.nan
    res[mode] = ({k: med(k) for k in cols}, rows[-1], len(use))
lines = [f"# ACCEL levels timing ({os.path.basename(out)})", "",
         f"Medians over iterations > {skip} ({res['off'][2]} / {res['on'][2]} iterations).", "",
         "| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
for mode in ("off", "on"):
    lines.append(f"| {mode} | " + " | ".join(f"{res[mode][0][k]:.3f}" for k in cols) + " |")
w0, w1 = res["off"][0]["wall_s"], res["on"][0]["wall_s"]
lines += ["", f"wall time per iteration on / off: {w1 / w0:.4f} ({w1 - w0:+.2f} s)"]
last = res["on"][1]
lines.append("level state at the end: " + ", ".join(
    f"{k} {float(last[k]):g}" for k in ("accel/levels", "accel/families", "accel/draw_share",
                                       "accel/candidates", "accel/dropped", "accel/inserted",
                                       "accel/inserted_edits", "accel/evicted", "accel/edits_good",
                                       "accel/score_q50", "plr/level_episodes", "plr/edit_episodes",
                                       "plr/level_shortfall", "plr/edit_shortfall") if k in last))
text = "\n".join(lines) + "\n"
open(os.path.join(out, "bench.md"), "w").write(text)
print(text)
EOF
echo "finished $(date)"
crc-job-stats 2>/dev/null || true
