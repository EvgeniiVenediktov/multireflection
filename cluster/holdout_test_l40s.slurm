#!/bin/bash
# holdout_test_l40s.slurm - evaluation of one checkpoint on the spatial hold-out split on one
# CRCD L40S: open-loop TEST evaluation (holdout/eval_test.py), then an offline closed-loop
# replay (utils/eval_batched.py) from the split's closed-loop origins.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/runs/<name>/<name>.pth,MODEL_RES=256 \
#       cluster/holdout_test_l40s.slurm
#
# Frames come from dark512 for every model: perturbations act on the 512 px frame, which is
# box-averaged to MODEL_RES for the model.
# Knobs: CKPT (required), MODEL_RES (512), SPLIT (split.csv of the hold-out split),
# OUT_DIR (<checkpoint dir>/holdout_test), EXTRA_ARGS (appended to eval_test.py, word-split,
# e.g. EXTRA_ARGS="--conditions clean,noise_0.20 --nn-resolution 128"),
# CLOSED_LOOP (1: run the closed-loop stage, 0: skip), ORIGINS (start file, default
# closed_loop_origins.txt next to the default split), CL_CONDITIONS (comma-separated
# utils/eval_sweep.CONDITIONS names run after clean, perturbation seed 0).
# Closed-loop outputs: $OUT_DIR/closed_loop/<condition>/.

#SBATCH --job-name=mfl-holdout
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
CKPT="${CKPT:?set CKPT to the absolute checkpoint path}"
MODEL_RES="${MODEL_RES:-512}"
SPLIT="${SPLIT:-$DATA_ROOT/data/holdout_split/split.csv}"
OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")/holdout_test}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CLOSED_LOOP="${CLOSED_LOOP:-1}"
ORIGINS="${ORIGINS:-$DATA_ROOT/data/holdout_split/closed_loop_origins.txt}"
CL_CONDITIONS="${CL_CONDITIONS:-occlusion_2x30,occlusion_2x30_rotbright,noise_0.20}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

module purge

cd "$PROJECT_DIR"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
[ -f "$SPLIT" ] || { echo "no split at $SPLIT"; exit 1; }
if [ "$CLOSED_LOOP" = "1" ]; then
    [ -f "$ORIGINS" ] || { echo "no closed-loop origins at $ORIGINS"; exit 1; }
    # Resolve every condition before the long stages, so a typo fails the job at once
    for c in clean ${CL_CONDITIONS//,/ }; do
        "$VENV_PY" holdout/condition_flags.py "$c" >/dev/null
    done
fi

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "checkpoint  : $CKPT (input $MODEL_RES px)"
echo "split       : $SPLIT"
echo "output      : $OUT_DIR"
echo "closed loop : $([ "$CLOSED_LOOP" = "1" ] && echo "clean,$CL_CONDITIONS from $ORIGINS" || echo off)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=================================================================="

# Stage the image bank on node-local NVMe, as in train_l40s.slurm.
STAGE="${SLURM_SCRATCH:-/tmp/$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$STAGE"
t0=$(date +%s)
cp "$DATA_TAR" "$STAGE/"
tar -xzf "$STAGE/$(basename "$DATA_TAR")" -C "$STAGE"
rm -f "$STAGE/$(basename "$DATA_TAR")"
DATA_DIR="$STAGE/dark512"
echo "staged $(ls "$DATA_DIR" | wc -l) files in $(( $(date +%s) - t0 ))s"

srun --cpus-per-task="$CPUS" "$VENV_PY" holdout/eval_test.py \
    --checkpoint "$CKPT" \
    --data-dir "$DATA_DIR" \
    --model-resolution "$MODEL_RES" \
    --split "$SPLIT" \
    --workers "$CPUS" \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS

if [ "$CLOSED_LOOP" = "1" ]; then
    for c in clean ${CL_CONDITIONS//,/ }; do
        FLAGS=$("$VENV_PY" holdout/condition_flags.py "$c")
        echo "===== closed loop: $c ====="
        srun --cpus-per-task="$CPUS" "$VENV_PY" utils/eval_batched.py \
            --checkpoint "$CKPT" \
            --data-dir "$DATA_DIR" \
            --model-resolution "$MODEL_RES" \
            --starts-file "$ORIGINS" \
            --workers "$CPUS" \
            --out-dir "$OUT_DIR/closed_loop/$c" \
            $FLAGS --perturb-seed 0
    done
fi

command -v crc-job-stats >/dev/null && crc-job-stats "${SLURM_JOB_ID:-}" || true
