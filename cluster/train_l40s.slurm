#!/bin/bash
# train_l40s.slurm - ResNet-18 tilt regression on one CRCD L40S, then the offline
# alignment sweep (utils/eval_batched.py) on the best checkpoint.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch cluster/train_l40s.slurm
#
# Fine-tune from an existing checkpoint, fewer epochs (values are read from the environment):
#   sbatch --export=ALL,START_CKPT=/ix1/kchen/evv/multireflection/runs/3854472/resnet18_l40s_3854472_best_model.pth,EPOCHS=49 \
#       cluster/train_l40s.slurm
# Extra flags for train/train_resnet_direct.py go in EXTRA_ARGS (word-split, so no spaces in values):
#   sbatch --export=ALL,EXTRA_ARGS="--noise-min 0.0 --noise 0.2" cluster/train_l40s.slurm
# Another image bank (an archive $DATA_ROOT/data/<DATASET>.tar.gz that extracts to <DATASET>/):
#   sbatch --export=ALL,DATASET=dark256,EXTRA_ARGS="--resolution 256" cluster/train_l40s.slurm
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
DATASET="${DATASET:-dark512}"
DATA_TAR="$DATA_ROOT/data/$DATASET.tar.gz"
RUN_DIR="$DATA_ROOT/runs/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
EPOCHS="${EPOCHS:-98}"
START_CKPT="${START_CKPT:-}"        # absolute path to a state dict, empty = from scratch
EXTRA_ARGS="${EXTRA_ARGS:-}"        # appended to the training command

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
DATA_DIR="$STAGE/$DATASET"
echo "staged $(ls "$DATA_DIR" | wc -l) files in $(( $(date +%s) - t0 ))s"
df -h "$STAGE" | tail -1

mkdir -p "$RUN_DIR"
# Keep the W&B run files next to the checkpoints on /ix1 instead of ./wandb on /ihome.
# A fixed run id lets the eval stage below resume the same run and attach its summary.
export WANDB_DIR="$RUN_DIR"
export WANDB_RUN_ID="l40s-${SLURM_JOB_ID:-manual}"

# bs 512 / lr 2.8e-3: 8x the local batch of 64, lr scaled by sqrt(8). ~36 GB of the 45 GB.
# W&B auth comes from ~/.netrc (wandb login); add --no-wandb for an unlogged run.
# --cpus-per-task is passed explicitly: srun does not reliably inherit it from sbatch
# (Slurm 22.05+), and the loader workers would otherwise be pinned to one core.
RUN_NAME="resnet18_l40s_${SLURM_JOB_ID:-manual}"
if [ "$DATASET" != dark512 ]; then
    RUN_NAME="resnet18_l40s_${DATASET}_${SLURM_JOB_ID:-manual}"
fi
START_ARGS=()
if [ -n "$START_CKPT" ]; then
    [ -f "$START_CKPT" ] || { echo "no checkpoint at $START_CKPT"; exit 1; }
    RUN_NAME="resnet18_l40s_ft_${SLURM_JOB_ID:-manual}"
    START_ARGS=(--starting-checkpoint "$START_CKPT")
    echo "fine-tuning : $START_CKPT"
fi
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" "$VENV_PY" train/train_resnet_direct.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$RUN_DIR" \
    --name "$RUN_NAME" \
    "${START_ARGS[@]}" \
    --batch-size 512 \
    --lr 0.0028 \
    --epochs "$EPOCHS" \
    --num-workers 10 \
    --val-num-workers 2 \
    --wandb-project multireflection \
    $EXTRA_ARGS

# Offline alignment sweep on the best checkpoint, using the staged image bank: once from
# every 0.1 deg grid position, once from every validation image (positions this run never
# trained on). Results go next to the checkpoint; summaries are attached to the W&B run
# above as eval/* and eval_val/*.
echo "evaluating  : $RUN_DIR/${RUN_NAME}_best_model.pth"
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" "$VENV_PY" utils/eval_batched.py \
    --checkpoint "$RUN_DIR/${RUN_NAME}_best_model.pth" \
    --data-dir "$DATA_DIR" \
    --out-dir "$RUN_DIR/eval" \
    --grid-step 0.1 \
    --threshold 0.97 \
    --wandb --wandb-project multireflection
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" "$VENV_PY" utils/eval_batched.py \
    --checkpoint "$RUN_DIR/${RUN_NAME}_best_model.pth" \
    --data-dir "$DATA_DIR" \
    --out-dir "$RUN_DIR/eval_val" \
    --starts-file "$RUN_DIR/val_names.txt" \
    --threshold 0.97 \
    --wandb --wandb-project multireflection --wandb-prefix eval_val

echo "=================================================================="
echo "finished    : $(date)"
echo "checkpoints : $RUN_DIR"
ls -lh "$RUN_DIR" "$RUN_DIR/eval" "$RUN_DIR/eval_val" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
