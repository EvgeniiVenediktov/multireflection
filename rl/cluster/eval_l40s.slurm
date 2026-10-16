#!/bin/bash
# eval_l40s.slurm - Evaluation of a ppo.py checkpoint on seen and unseen scenarios
# (rl/eval_policy.py) on one CRCD L40S. Inference only, no training.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/rl_runs/4407578/best.pt rl/cluster/eval_l40s.slurm
#
# Sets (eval_policy.py --sets; default cont,region,set4, the run's own set for --setups all
# / measured runs): cont = 4096 seeded continuous scenarios of the run's ranges (D7; 90 / 10
# good / impossible), region = 4096 good continuous scenarios, 1024 per target-region bin
# (area of the M2 box cells with path >= 0.9 P_max: < 0.02, 0.02-0.1, 0.1-0.5, >= 0.5
# deg^2), set4 = the four named setups (per setup and target-region bin). The grid
# benchmarks of the atlas runs with SETS (e.g. cont,region,set4,full,offgrid,extrap_short,
# extrap_long; add holdout for runs trained with --holdout-frac > 0): full = the training
# grid of the atlas runs, offgrid = midpoints between grid points, extrap_short /
# extrap_long = separation 20-49 / 201-350 mm. Both policies (argmax, sampled), N episodes
# per set. The tables are built once in TABLES if missing (cont ~10 s, region ~35 s on an
# RTX A2000; off-grid ~1 min for the three) and loaded otherwise; the oracles are cached in
# the checkpoint's run dir (oracle_cont.pt / oracle_region.pt / oracle_eval.pt of the run
# are reused; a few s each when new). Evaluation: ~4 s per set and policy on the L40S (run
# 4407578's evals), so the job takes a few minutes.
#
# Output: OUT/eval_<run id>_<ckpt stem>.json and .md (run id = the checkpoint's directory
# name, e.g. eval_4407578_best.md).
#
# Environment variables (sbatch --export=ALL,NAME=value,...):
#   CKPT        ppo.py checkpoint (required), e.g. <run dir>/best.pt or latest.pt
#   OUT         report directory (default: the checkpoint's run dir)
#   SETS        default empty = cont,region,set4 (see above)
#   POLICY      default argmax,sampled
#   N           episodes per set (default 4096 = the runs' eval_n, so the cached oracles of
#               the run are reused; another N caches its oracles in OUT/oracles_n<N>)
#   ATLAS       scenario atlas, default /ix1/kchen/evv/multireflection/rl_runs/atlas/atlas_v1.pt
#   TABLES      eval tables dir, default /ix1/kchen/evv/multireflection/rl_runs/eval_sets
#   EXTRA_ARGS  appended to eval_policy.py (word-split), e.g. "--tag it1200 --compile" or
#               "--name eval_region_r3_it1600 --compare <other report>.json"
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
echo "sets        : ${SETS:-cont,region,set4} ($N episodes, policies $POLICY)"
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
