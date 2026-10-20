#!/bin/bash
# probe_path_l40s.slurm - path probe (probe_path.py): frozen agent of a ppo.py checkpoint, rollouts
# in its training env, a new head on the core features trained to predict log P (mm), compared
# with the existing path head (P / P_ref).
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=<ppo checkpoint> rl/cluster/probe_path_l40s.slurm
#
# Output: $RUNS_ROOT/probe_path/<jobid>/{probe_path.json, probe_path.md, head.pt}

#SBATCH --job-name=mfl-probe-path
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
OUT="$RUNS_ROOT/probe_path/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
CKPT="${CKPT:?CKPT (ppo.py checkpoint) is required}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

module purge
cd "$PROJECT_DIR/rl"
mkdir -p "$OUT"
CACHE="${SLURM_SCRATCH:-$OUT}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
echo "job ${SLURM_JOB_ID:-none} on $(hostname), $(date); ckpt $CKPT"
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" -u probe_path.py --ckpt "$CKPT" --out "$OUT" $EXTRA_ARGS
echo "finished $(date)"
