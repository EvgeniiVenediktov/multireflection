#!/bin/bash
# encoder_l40s.slurm - CNN frame-encoder pretraining on all four geometry.SETUPS (PLAN
# phase 5 recipe: 8000 steps, batch 256, seed 0), then eval_encoder.py on the result
# (four-setup sets and measured-cell sets), on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/encoder_l40s.slurm
#   sbatch --export=ALL,SCENARIOS=16384,STEPS=16000 rl/cluster/encoder_l40s.slurm   # full grid
#
# Outputs: /ix1/kchen/evv/multireflection/rl_runs/encoder/<jobid>/
#     cnn_w160_lr0.001_b256_8k_s0_set4/{model.pt, config.json, train.csv, val.csv,
#                                       metrics.json, metrics_set4.json}
#     encoder_tables.md, encoder_tables_set4.md
# and a copy of model.pt at the fixed path the PPO job reads by default (ENCODER_INIT in
# ppo_l40s.slurm), overwritten by every run of this job:
#     /ix1/kchen/evv/multireflection/rl_runs/encoder/cnn_w160_lr0.001_b256_8k_s0_set4.pt
# Chain the PPO job after it:
#   jid=$(sbatch --parsable rl/cluster/encoder_l40s.slurm)
#   sbatch --dependency=afterok:${jid%%;*} rl/cluster/ppo_l40s.slurm
#
# Settings from the environment: STEPS (8000), SEED (0), EXTRA_ARGS (appended to
# pretrain_encoder.py, e.g. EXTRA_ARGS="--wandb" for W&B project rl-encoder).
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>
#
# Runs .venv/bin/python directly (built by cluster/check_env.sh from uv.lock). Never
# installs anything.

#SBATCH --job-name=mfl-encoder
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
RUN_DIR="$RUNS_ROOT/encoder/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
STEPS="${STEPS:-8000}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SCENARIOS="${SCENARIOS:-0}"      # > 0: frames from that many atlas-grid scenarios + the named setups
FIXED_CKPT="$RUNS_ROOT/encoder/cnn_w160_lr0.001_b256_8k_s0_set4.pt"

# No modules: the venv carries its own CPython and CUDA runtime. Slurm does not source
# ~/.bashrc, so purge explicitly.
module purge

cd "$PROJECT_DIR/rl"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
mkdir -p "$RUN_DIR"

# Triton / inductor kernel caches on node-local scratch (wiped at job end)
CACHE="${SLURM_SCRATCH:-$RUN_DIR}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
export WANDB_DIR="$RUN_DIR"

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
$VENV_PY -c "import torch, triton; print('torch       :', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, '| available', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "run dir     : $RUN_DIR"
echo "=================================================================="

SRUN=(srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}")
"${SRUN[@]}" "$VENV_PY" pretrain_encoder.py --arch cnn --steps "$STEPS" --seed "$SEED" \
    --setups all --scenarios "$SCENARIOS" --runs-dir "$RUN_DIR" $EXTRA_ARGS

RUN_SUB=$(ls -d "$RUN_DIR"/cnn_* | head -n 1)
echo "trained     : $RUN_SUB"
# Four-setup sets (metrics_set4.json, encoder_tables_set4.md), then the measured-cell
# sets of the phase-5 benchmark (metrics.json, encoder_tables.md); tables in $RUN_DIR
# Full range: 4096 grid scenarios drawn with another seed than the training scenarios
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --scenarios 4096 --scenario-seed 1 --no-speed --no-figures --out "$RUN_DIR" "$RUN_SUB"
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --setups all --no-figures --out "$RUN_DIR" "$RUN_SUB"
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --no-figures --out "$RUN_DIR" "$RUN_SUB"

cp "$RUN_SUB/model.pt" "$FIXED_CKPT"
echo "=================================================================="
echo "finished    : $(date)"
echo "checkpoint  : $RUN_SUB/model.pt"
echo "copied to   : $FIXED_CKPT (ppo_l40s.slurm default ENCODER_INIT)"
ls -lh "$RUN_DIR" "$RUN_SUB" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
