#!/bin/bash
# train_l40s.slurm - ResNet-18 tilt regression on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch cluster/train_l40s.slurm
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>
#
# Runs .venv/bin/python directly (built by cluster/check_env.sh from uv.lock; re-run it
# after changing uv.lock). Never installs anything. Resource choices, storage layout and
# the staging rationale are explained in cluster/README.md.

#SBATCH --job-name=mfl-resnet18
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
RUN_DIR="$DATA_ROOT/runs/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"

# No modules: the venv carries its own CPython and CUDA runtime, and a cuda module would
# shadow it through LD_LIBRARY_PATH. Slurm does not source ~/.bashrc, so purge explicitly.
module purge

cd "$PROJECT_DIR"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
$VENV_PY -c "import torch; print('torch       :', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=================================================================="

# Stage the 228,800 small JPEGs onto node-local NVMe; reading them off /ix1 every epoch
# would be slow and hammer a shared filesystem. $SLURM_SCRATCH is wiped at job end, so
# checkpoints go to $RUN_DIR on /ix1.
STAGE="${SLURM_SCRATCH:-/tmp/$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$STAGE"
echo "staging $DATA_TAR -> $STAGE"
t0=$(date +%s)
cp "$DATA_TAR" "$STAGE/"
tar -xzf "$STAGE/$(basename "$DATA_TAR")" -C "$STAGE"
rm -f "$STAGE/$(basename "$DATA_TAR")"
DATA_DIR="$STAGE/dark512"
echo "staged $(ls "$DATA_DIR" | wc -l) files in $(( $(date +%s) - t0 ))s"
df -h "$STAGE" | tail -1

mkdir -p "$RUN_DIR"
# Keep the W&B run files next to the checkpoints on /ix1 instead of ./wandb on /ihome.
export WANDB_DIR="$RUN_DIR"

# bs 512 / lr 2.8e-3: 8x the local batch of 64, lr scaled by sqrt(8). ~36 GB of the 45 GB.
# W&B auth comes from ~/.netrc (wandb login); add --no-wandb for an unlogged run.
# --cpus-per-task is passed explicitly: srun does not reliably inherit it from sbatch
# (Slurm 22.05+), and the loader workers would otherwise be pinned to one core.
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" "$VENV_PY" train/train_resnet_direct.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$RUN_DIR" \
    --name "resnet18_l40s_${SLURM_JOB_ID:-manual}" \
    --batch-size 512 \
    --lr 0.0028 \
    --epochs 96 \
    --num-workers 10 \
    --val-num-workers 2 \
    --wandb-project multireflection

echo "=================================================================="
echo "finished    : $(date)"
echo "checkpoints : $RUN_DIR"
ls -lh "$RUN_DIR" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
