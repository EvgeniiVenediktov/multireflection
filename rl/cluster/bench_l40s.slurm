#!/bin/bash
# bench_l40s.slurm - timing only: throughput of ppo.py iterations for several batch layouts
# (bench_ppo.py: frozen / trained encoder, envs x rollout steps, checkpointing, minibatches).
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/bench_l40s.slurm
#   sbatch --export=ALL,CONFIGS="f1024x128 f4096x32" rl/cluster/bench_l40s.slurm
#
# Output: $RUNS_ROOT/bench/<jobid>/bench.md / bench.json

#SBATCH --job-name=mfl-bench
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
OUT="$RUNS_ROOT/bench/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
ENCODER_INIT="${ENCODER_INIT:-$RUNS_ROOT/encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt}"
INIT_FROM="${INIT_FROM:-$RUNS_ROOT/4407578/best_snapshot_it3800.pt}"
CONFIGS="${CONFIGS:-}"

module purge
cd "$PROJECT_DIR/rl"
mkdir -p "$OUT"
CACHE="${SLURM_SCRATCH:-$OUT}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
CFG_ARGS=()
[ -n "$CONFIGS" ] && CFG_ARGS=(--configs $CONFIGS)
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" bench_ppo.py \
    --encoder-init "$ENCODER_INIT" --init-from "$INIT_FROM" --out "$OUT" "${CFG_ARGS[@]}"
rm -rf "$OUT/tmp"
echo "finished $(date)"
crc-job-stats 2>/dev/null || true
