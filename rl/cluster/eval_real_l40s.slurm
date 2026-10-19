#!/bin/bash
# eval_real_l40s.slurm - policy evaluation on the real tilt raster (eval_real.py eval): episodes
# from random grid tilts, the frame of each new tilt fetched from the real dataset; result =
# the pattern the policy ends on (final frame, spot count), per axis map.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=<ppo best.pt> rl/cluster/eval_real_l40s.slurm
#   sbatch --export=ALL,CKPT=<ppo best.pt>,ENCODER=<pretrain model.pt> rl/cluster/eval_real_l40s.slurm
#
# Settings: CKPT (required), ENCODER (optional), RASTER (the eval_real.py build output,
# default /ix1/kchen/evv/multireflection/data/real_raster), EPISODES 1024, SEED 0, AXIS_MAP all
# (no commas inside --export: use AXIS_MAP=all or a single map exported beforehand).
# Output: $RUNS_ROOT/eval_real/<jobid>/{real_eval.json, episodes_*.npz, final_frames.png}

#SBATCH --job-name=mfl-eval-real
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
OUT="$RUNS_ROOT/eval_real/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
RASTER="${RASTER:-/ix1/kchen/evv/multireflection/data/real_raster}"
CKPT="${CKPT:?CKPT (ppo.py checkpoint) is required}"
ENCODER="${ENCODER:-}"
EPISODES="${EPISODES:-1024}"
SEED="${SEED:-0}"
AXIS_MAP="${AXIS_MAP:-all}"

module purge
cd "$PROJECT_DIR/rl"
mkdir -p "$OUT"
echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date); ckpt $CKPT; encoder ${ENCODER:-(the checkpoint's)}"
ENC_ARGS=()
[ -n "$ENCODER" ] && ENC_ARGS=(--encoder "$ENCODER")
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" "$VENV_PY" -u eval_real.py eval --ckpt "$CKPT" \
    "${ENC_ARGS[@]}" --raster "$RASTER" --episodes "$EPISODES" --seed "$SEED" \
    --axis-map "$AXIS_MAP" --out "$OUT"
echo "finished $(date)"
