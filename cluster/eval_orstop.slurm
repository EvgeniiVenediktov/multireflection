#!/bin/bash
# eval_orstop.slurm - the paper's Table 3 robustness sweep with the OR stop rule, as a job
# array of short single-GPU tasks over (model x condition group).
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch cluster/eval_orstop.slurm                 # all 55 tasks
#   sbatch --array=3,17 cluster/eval_orstop.slurm    # resubmit selected tasks
#
# A trajectory is converged when the SSIM stop test passes (rounded SSIM >= 0.97) OR the
# last two predicted corrections both have a norm < STOP_MOTION deg (eval_batched.py
# --stop-motion); max 10 corrections. Perturbed conditions run with --perturb-ssim (the stop
# test sees the perturbed frame), clean once. Every task runs all perturbation SEEDS of its
# conditions via utils/eval_sweep.py into <run dir>/sweep10_orstop/task_<group>/.
# Task index = model * 11 + group (models and groups below).
#
# The image bank is staged to $SLURM_SCRATCH: the extracted data/dark512 on /ix1 is partial
# (93,383 of 228,800 images), and staging the archive takes ~50 s.

#SBATCH --job-name=mfl-orstop
#SBATCH --cluster=gpu
#SBATCH --partition=a100,a100_nvlink,h200,l40s,rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --array=0-54
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
STOP_MOTION="${STOP_MOTION:-0.04}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
CPUS="${SLURM_CPUS_PER_TASK:-12}"
TAG="hold"
STEM="occ15-40img-rot90-fill0-255_n00-20_e98_${TAG}"

# checkpoint | model input px | output run dir
MODELS=(
    "$DATA_ROOT/runs/r64_${STEM}_3881600/r64_${STEM}_3881600.pth|64|$DATA_ROOT/runs/r64_${STEM}_3881600"
    "$DATA_ROOT/runs/r128_${STEM}_3881604/r128_${STEM}_3881604.pth|128|$DATA_ROOT/runs/r128_${STEM}_3881604"
    "$DATA_ROOT/runs/r256_${STEM}_3881612/r256_${STEM}_3881612.pth|256|$DATA_ROOT/runs/r256_${STEM}_3881612"
    "$DATA_ROOT/runs/r512_${STEM}_3881616/r512_${STEM}_3881616.pth|512|$DATA_ROOT/runs/r512_${STEM}_3881616"
    "$DATA_ROOT/runs/3943793/resnet18_l40s_dark256_3943793_best_model.pth|256|$DATA_ROOT/runs/3943793"
)
# condition groups (eval_sweep.py names); clean is deterministic and runs once, with noise_0.10
COND_GROUPS=(
    "clean,noise_0.10" "noise_0.30" "brightness_0.4" "brightness_1.6" "occlusion_1x30" "occlusion_2x30"
    "occlusion_2x50" "occlusion_2x50_bright" "occlusion_2x50_rotbright" "occlusion_2_mixedfill"
    "combined_harsh"
)

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
IFS='|' read -r CKPT MODEL_RES RUN_DIR <<< "${MODELS[$(( TASK / ${#COND_GROUPS[@]} ))]}"
CONDS="${COND_GROUPS[$(( TASK % ${#COND_GROUPS[@]} ))]}"
OUT_DIR="$RUN_DIR/sweep10_orstop/task_${CONDS//,/+}"

module purge
cd "$PROJECT_DIR"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

echo "=================================================================="
echo "job         : ${SLURM_ARRAY_JOB_ID:-}_${TASK} (${SLURM_JOB_ID:-}) on $(hostname), partition ${SLURM_JOB_PARTITION:-?}"
echo "started     : $(date)"
echo "checkpoint  : $CKPT (input $MODEL_RES px)"
echo "conditions  : $CONDS, seeds $SEEDS, stop-motion $STOP_MOTION"
echo "output      : $OUT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=================================================================="

STAGE="${SLURM_SCRATCH:-/tmp/$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$STAGE"
t0=$(date +%s)
if command -v pigz >/dev/null; then tar -I pigz -xf "$DATA_TAR" -C "$STAGE"; else tar -xzf "$DATA_TAR" -C "$STAGE"; fi
DATA_DIR="$STAGE/dark512"
echo "staged $(ls "$DATA_DIR" | wc -l) files in $(( $(date +%s) - t0 ))s"

t0=$(date +%s)
srun --cpus-per-task="$CPUS" "$VENV_PY" utils/eval_sweep.py \
    --checkpoint "$CKPT" \
    --data-dir "$DATA_DIR" \
    --model-resolution "$MODEL_RES" \
    --grid-step 0.1 \
    --workers "$CPUS" \
    --conditions "$CONDS" \
    --perturb-seeds "$SEEDS" \
    --perturb-ssim \
    --stop-motion "$STOP_MOTION" \
    --out-dir "$OUT_DIR"
echo "sweep done in $(( $(date +%s) - t0 ))s at $(date)"
