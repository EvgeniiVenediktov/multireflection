#!/bin/bash
# eval_l40s.slurm - Evaluation of a ppo.py checkpoint on seen and unseen scenarios
# (rl/eval_policy.py) on one CRCD L40S. Inference only, no training.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/rl_runs/4407578/best.pt rl/cluster/eval_l40s.slurm
#
# Sets (eval_policy.py --sets; default the run's own sets: full,offgrid,extrap_short,
# extrap_long,set4 for atlas runs, cont first for pool runs (continuous scenarios of the
# run's ranges, D7); add holdout for runs trained with --holdout-frac > 0): full = the training grid (seen),
# offgrid = midpoints between grid points (unseen, interpolation), extrap_short /
# extrap_long = separation 20-49 / 201-350 mm (unseen, extrapolation), set4 = the four
# named setups. Both policies (argmax, sampled), N episodes per set. The off-grid tables
# are built once in TABLES if missing (~1 min on the L40S for the three, estimate from
# 43 s on an RTX A2000) and loaded otherwise; the oracles are cached in the checkpoint's
# run dir (oracle_full.pt / oracle_eval.pt of the run are reused; new: oracle_offgrid.pt,
# oracle_extrap_short.pt, oracle_extrap_long.pt; a few s each). Evaluation: ~4 s per set
# and policy on the L40S (run 4407578's evals), so the job takes a few minutes.
#
# Output: OUT/eval_<run id>_<ckpt stem>.json and .md (run id = the checkpoint's directory
# name, e.g. eval_4407578_best.md).
#
# Environment variables (sbatch --export=ALL,NAME=value,...):
#   CKPT        ppo.py checkpoint (required), e.g. <run dir>/best.pt or latest.pt
#   OUT         report directory (default: the checkpoint's run dir)
#   SETS        default empty = the run's sets (see above)
#   POLICY      default argmax,sampled
#   N           episodes per set (default 4096 = the runs' eval_n, so the cached oracles of
#               the run are reused; another N caches its oracles in OUT/oracles_n<N>)
#   ATLAS       scenario atlas, default /ix1/kchen/evv/multireflection/rl_runs/atlas/atlas_v1.pt
#   TABLES      off-grid tables dir, default /ix1/kchen/evv/multireflection/rl_runs/eval_sets
#   EXTRA_ARGS  appended to eval_policy.py (word-split), e.g. "--tag it1200 --compile"
#
# Several checkpoints, one job each:
#   for c in best latest; do sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/rl_runs/4407578/$c.pt rl/cluster/eval_l40s.slurm; done
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>
#
# Runs .venv/bin/python directly (built by cluster/check_env.sh from uv.lock). Never
# installs anything.

#SBATCH --job-name=mfl-eval
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
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
CKPT="${CKPT:?set CKPT=<ppo.py checkpoint>}"
OUT="${OUT:-$(dirname "$CKPT")}"
SETS="${SETS:-}"
POLICY="${POLICY:-argmax,sampled}"
N="${N:-4096}"
ATLAS="${ATLAS:-$RUNS_ROOT/atlas/atlas_v1.pt}"
TABLES="${TABLES:-$RUNS_ROOT/eval_sets}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# No modules: the venv carries its own CPython and CUDA runtime, and a cuda module would
# shadow it through LD_LIBRARY_PATH. Slurm does not source ~/.bashrc, so purge explicitly.
module purge

cd "$PROJECT_DIR/rl"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at CKPT=$CKPT"; exit 1; }
[ -f "$ATLAS" ] || { echo "no atlas at ATLAS=$ATLAS"; exit 1; }
mkdir -p "$OUT" "$TABLES"

# Triton / inductor kernel caches on node-local scratch
CACHE="${SLURM_SCRATCH:-$OUT}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
$VENV_PY -c "import torch, triton; print('torch       :', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, '| available', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "checkpoint  : $CKPT"
echo "sets        : ${SETS:-run defaults} ($N episodes, policies $POLICY)"
echo "atlas       : $ATLAS"
echo "tables      : $TABLES"
echo "out         : $OUT"
echo "extra args  : $EXTRA_ARGS"
echo "=================================================================="

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" "$VENV_PY" eval_policy.py \
    --ckpt "$CKPT" \
    --sets "$SETS" \
    --policy "$POLICY" \
    --n "$N" \
    --out "$OUT" \
    --atlas "$ATLAS" \
    --tables "$TABLES" \
    $EXTRA_ARGS

echo "=================================================================="
echo "finished    : $(date)"
ls -lh "$OUT"/eval_*.md "$OUT"/eval_*.json 2>/dev/null || true
echo "=================================================================="

crc-job-stats 2>/dev/null || true
