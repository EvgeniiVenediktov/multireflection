#!/bin/bash
# bench_render_l40s.slurm - timing only: frame data paths for encoder training (sim with /
# without per-hit outputs, current targets + randomized renderer, camera renderer eager and
# compiled per chunk size, one 3-channel encoder training step), bench_render.py.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/bench_render_l40s.slurm
#   sbatch --export=ALL,BENCH_ARGS="--batches 1024 --chunks 256" rl/cluster/bench_render_l40s.slurm
#
# Output: $RUNS_ROOT/bench_render/<jobid>/bench_render.md / bench_render.json

#SBATCH --job-name=mfl-bench-render
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
OUT="$RUNS_ROOT/bench_render/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
BENCH_ARGS="${BENCH_ARGS:-}"

module purge
cd "$PROJECT_DIR/rl"
mkdir -p "$OUT"
CACHE="${SLURM_SCRATCH:-$OUT}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" -u bench_render.py --out "$OUT" $BENCH_ARGS
echo "finished $(date)"
crc-job-stats 2>/dev/null || true
