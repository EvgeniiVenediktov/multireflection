#!/bin/bash
# encoder_cam_l40s.slurm - camera-mode encoder training (pretrain_encoder.py --camera) on one
# CRCD L40S: 3-channel CameraRenderer inputs (render_camera.CameraRanges augmentation
# config), camera-frame spot map + count targets.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/encoder_cam_l40s.slurm
#
# Default: the encoder of the r6 policy (INIT_AGENT, ppo.py best.pt of run 4423352, tag r6),
# stem copied to 3 channels at 1/3; heads and model config from INIT (the D7 continuous
# encoder that r2-r6 descend from); frames of SCENARIOS = 16,384 continuous scenarios
# (separation SEP_RANGE 20-400 mm, laser LASER_RANGE -8..8 deg, offset radius 1.5 mm) plus
# the four named setups; batch 256, peak lr 3e-4, warmup 300; trained to convergence
# (CONVERGE=1, default: pretrain_encoder.py --converge, the lr drops on validation-loss
# plateaus and the run stops after the last one; STEPS 400000 is only a cap, reaching it
# fails the job with exit code 3 so an afterok chain never starts from an unconverged
# encoder). CONVERGE=0: fixed cosine schedule over STEPS (to 1 %). W&B project rl-encoder.
# Run name e.g.
#     cnn_w160_lr0.0003_b256_400k_s0_ft-r6_cont20-400x16384_cam
#
# Outputs: /ix1/kchen/evv/multireflection/rl_runs/encoder/<jobid>/<run name>/
#     {model.pt, config.json, train.csv, val.csv}   (val.csv: camera set + clean set)
# and a copy of model.pt next to the job dirs: .../rl_runs/encoder/<run name>.pt
#
# Settings from the environment: INIT, INIT_AGENT, INIT_AGENT_TAG, SCENARIOS, SEP_RANGE,
# LASER_RANGE, LASER_OFFSET_RADIUS, STEPS, CONVERGE, LR, WARMUP, BATCH, SEED, WANDB (default --wandb;
# WANDB= switches W&B off), EXTRA_ARGS (appended to pretrain_encoder.py)
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>

#SBATCH --job-name=mfl-encoder-cam
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
RUN_DIR="$RUNS_ROOT/encoder/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
INIT="${INIT-$RUNS_ROOT/encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt}"
INIT_AGENT="${INIT_AGENT-$RUNS_ROOT/4423352/best.pt}"
INIT_AGENT_TAG="${INIT_AGENT_TAG:-r6}"
SCENARIOS="${SCENARIOS:-16384}"
SEP_RANGE="${SEP_RANGE-20 400}"
LASER_RANGE="${LASER_RANGE:--8 8}"
LASER_OFFSET_RADIUS="${LASER_OFFSET_RADIUS:-1.5}"
CONVERGE="${CONVERGE:-1}"
STEPS="${STEPS:-400000}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-300}"
BATCH="${BATCH:-256}"
SEED="${SEED:-0}"
WANDB="${WANDB---wandb}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

module purge
cd "$PROJECT_DIR/rl"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
mkdir -p "$RUN_DIR"
CACHE="${SLURM_SCRATCH:-$RUN_DIR}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"
export WANDB_DIR="$RUN_DIR"

TRAIN_ARGS=(--camera --arch cnn --steps "$STEPS" --lr "$LR" --warmup "$WARMUP" --batch "$BATCH"
            --seed "$SEED" --setups all --scenarios "$SCENARIOS" --runs-dir "$RUN_DIR")
[ "$CONVERGE" = "1" ] && TRAIN_ARGS+=(--converge)
if [ -n "$SEP_RANGE" ] && [ "$SCENARIOS" -gt 0 ]; then
    TRAIN_ARGS+=(--sep-range $SEP_RANGE --laser-range $LASER_RANGE
                 --laser-offset-radius "$LASER_OFFSET_RADIUS")
fi
if [ -n "$INIT" ]; then
    [ -f "$INIT" ] || { echo "no checkpoint at INIT=$INIT"; exit 1; }
    TRAIN_ARGS+=(--init "$INIT")
fi
if [ -n "$INIT_AGENT" ]; then
    [ -f "$INIT_AGENT" ] || { echo "no checkpoint at INIT_AGENT=$INIT_AGENT"; exit 1; }
    TRAIN_ARGS+=(--init-agent "$INIT_AGENT" --init-agent-tag "$INIT_AGENT_TAG")
fi

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "run dir     : $RUN_DIR"
echo "init        : heads ${INIT:-none}, encoder ${INIT_AGENT:-from INIT} ($INIT_AGENT_TAG)"
echo "args        : ${TRAIN_ARGS[*]} $WANDB $EXTRA_ARGS"
echo "=================================================================="

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" -u pretrain_encoder.py \
    "${TRAIN_ARGS[@]}" $WANDB $EXTRA_ARGS

RUN_SUB=$(ls -d "$RUN_DIR"/cnn_* | head -n 1)
cp "$RUN_SUB/model.pt" "$RUNS_ROOT/encoder/$(basename "$RUN_SUB").pt"
echo "trained     : $RUN_SUB -> $RUNS_ROOT/encoder/$(basename "$RUN_SUB").pt"
echo "finished    : $(date)"
crc-job-stats 2>/dev/null || true
