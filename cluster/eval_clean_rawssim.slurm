#!/bin/bash
# eval_clean_rawssim.slurm - clean closed-loop eval of the five hold models with the adopted
# stop rule: converged when the unrounded SSIM >= 0.97 (--raw-ssim) OR the last predicted
# correction has a norm < STOP_MOTION deg (--stop-motion-steps 1); max 10 corrections.
# One array task per model; output <run dir>/clean_rawssim_k1/.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch -p rtx6k cluster/eval_clean_rawssim.slurm

#SBATCH --job-name=mfl-rawk1
#SBATCH --cluster=gpu
#SBATCH --partition=rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=00:12:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
STOP_MOTION="${STOP_MOTION:-0.04}"
CPUS="${SLURM_CPUS_PER_TASK:-12}"
STEM="occ15-40img-rot90-fill0-255_n00-20_e98_hold"

# checkpoint | model input px | output run dir
MODELS=(
    "$DATA_ROOT/runs/r64_${STEM}_3881600/r64_${STEM}_3881600.pth|64|$DATA_ROOT/runs/r64_${STEM}_3881600"
    "$DATA_ROOT/runs/r128_${STEM}_3881604/r128_${STEM}_3881604.pth|128|$DATA_ROOT/runs/r128_${STEM}_3881604"
    "$DATA_ROOT/runs/r256_${STEM}_3881612/r256_${STEM}_3881612.pth|256|$DATA_ROOT/runs/r256_${STEM}_3881612"
    "$DATA_ROOT/runs/r512_${STEM}_3881616/r512_${STEM}_3881616.pth|512|$DATA_ROOT/runs/r512_${STEM}_3881616"
    "$DATA_ROOT/runs/3943793/resnet18_l40s_dark256_3943793_best_model.pth|256|$DATA_ROOT/runs/3943793"
)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
IFS='|' read -r CKPT MODEL_RES RUN_DIR <<< "${MODELS[$TASK]}"
OUT_DIR="$RUN_DIR/clean_rawssim_k1"

module purge
cd "$PROJECT_DIR"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
echo "job ${SLURM_ARRAY_JOB_ID:-}_${TASK} on $(hostname), partition ${SLURM_JOB_PARTITION:-?}, $(date)"
echo "checkpoint $CKPT ($MODEL_RES px) -> $OUT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

STAGE="${SLURM_SCRATCH:-/tmp/$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$STAGE"
t0=$(date +%s)
if command -v pigz >/dev/null; then tar -I pigz -xf "$DATA_TAR" -C "$STAGE"; else tar -xzf "$DATA_TAR" -C "$STAGE"; fi
echo "staged $(ls "$STAGE/dark512" | wc -l) files in $(( $(date +%s) - t0 ))s"

srun --cpus-per-task="$CPUS" "$VENV_PY" utils/eval_sweep.py \
    --checkpoint "$CKPT" \
    --data-dir "$STAGE/dark512" \
    --model-resolution "$MODEL_RES" \
    --grid-step 0.1 \
    --workers "$CPUS" \
    --conditions clean \
    --raw-ssim \
    --stop-motion "$STOP_MOTION" \
    --stop-motion-steps 1 \
    --out-dir "$OUT_DIR"
echo "done $(date)"
