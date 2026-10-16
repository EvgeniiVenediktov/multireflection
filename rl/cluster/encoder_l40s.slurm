#!/bin/bash
# encoder_l40s.slurm - CNN frame-encoder pretraining / fine-tuning (pretrain_encoder.py),
# then eval_encoder.py on the result, on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/encoder_l40s.slurm        # default: fine-tune on continuous scenarios (D7)
#
# Default (D7): fine-tune the grid encoder INIT (run 4407576, 16,384 grid scenarios, 16k
# steps; the encoder of PPO run 4407578) on SCENARIOS = 16,384 continuous scenarios
# (separation SEP_RANGE 20-400 mm, laser pitch / yaw LASER_RANGE -8..8 deg, laser offset
# uniform over the M1 hole disk of LASER_OFFSET_RADIUS 1.5 mm, beams that reach M2 only)
# plus the four named setups, STEPS 8000 at the fine-tuning defaults of
# pretrain_encoder.py (peak lr 3e-4, warmup 100, cosine to 1 %). Run name e.g.
#     cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384
# Then the fine-tuned encoder and the starting one (INIT's run dir, copied into RUN_DIR/
# init_<name>/ so nothing is written next to INIT) are evaluated side by side:
#   - held-out continuous scenarios (EVAL_SCENARIOS 4096, --scenario-seed 1) over the
#     same ranges, per separation bin 20-50, ..., 350-400 (gain outside 50-200 mm, loss
#     inside): metrics_<ranges tag>.json, encoder_tables_<ranges tag>.md
#   - the 50-200 mm grid sets of reports/encoder_tables_full.md (4096 grid scenarios,
#     seed 1): metrics_full.json, encoder_tables_full.md
#   - the four setups: metrics_set4.json, encoder_tables_set4.md
#   - the measured cell (phase-5 benchmark): metrics.json, encoder_tables.md
# Estimate: 8k steps at the 16k-step grid run's rate, ~30-60 min including the frame
# scans of 16,388 setups; evaluation ~10 min per set kind for both runs.
#
#   sbatch --export=ALL,INIT=,SEP_RANGE=,SCENARIOS=16384,STEPS=16000 rl/cluster/encoder_l40s.slurm  # grid, from scratch
#   sbatch --export=ALL,INIT=,SEP_RANGE=,SCENARIOS=0 rl/cluster/encoder_l40s.slurm                  # four setups, from scratch
#
# Outputs: /ix1/kchen/evv/multireflection/rl_runs/encoder/<jobid>/
#     <run name>/{model.pt, config.json, train.csv, val.csv, metrics*.json}
#     init_<INIT run name>/{model.pt, config.json, metrics*.json}   (with INIT)
#     encoder_tables*.md
# and a copy of model.pt next to the job dirs, named after the run (overwritten only by a
# run of the same name):
#     /ix1/kchen/evv/multireflection/rl_runs/encoder/<run name>.pt
# e.g. .../encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt, the
# ENCODER_INIT of the D7 PPO run (ppo_l40s.slurm). Chain the PPO job after it:
#   jid=$(sbatch --parsable rl/cluster/encoder_l40s.slurm)
#   sbatch --dependency=afterok:${jid%%;*} --export=ALL,ENCODER_INIT=<that copy>,... rl/cluster/ppo_l40s.slurm
#
# Settings from the environment (empty INIT / SEP_RANGE switch the fine-tuning / the
# continuous scenarios off):
#   INIT               pretrain_encoder.py checkpoint to fine-tune; default
#                      /ix1/kchen/evv/multireflection/rl_runs/encoder/4407576/cnn_w160_lr0.001_b256_16k_s0_grid16384/model.pt
#   SCENARIOS          16384 (0: the four setups only)
#   SEP_RANGE          "20 400" (empty: grid scenarios with SCENARIOS > 0)
#   LASER_RANGE        "-8 8" (bounds of the landing-point sampler)
#   LASER_OFFSET_RADIUS 1.5 (0: the measured laser offsets)
#   STEPS              8000 (pass 16000 for a from-scratch grid run)
#   SEED               0
#   EVAL_SCENARIOS     4096 held-out scenarios for the continuous / grid evaluation
#   EXTRA_ARGS         appended to pretrain_encoder.py, e.g. EXTRA_ARGS="--wandb" (W&B
#                      project rl-encoder) or "--lr 1e-4"
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
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
RUN_DIR="$RUNS_ROOT/encoder/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
INIT="${INIT-$RUNS_ROOT/encoder/4407576/cnn_w160_lr0.001_b256_16k_s0_grid16384/model.pt}"
SCENARIOS="${SCENARIOS:-16384}"
SEP_RANGE="${SEP_RANGE-20 400}"
LASER_RANGE="${LASER_RANGE:--8 8}"
LASER_OFFSET_RADIUS="${LASER_OFFSET_RADIUS:-1.5}"
STEPS="${STEPS:-8000}"
SEED="${SEED:-0}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-4096}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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

TRAIN_ARGS=(--arch cnn --steps "$STEPS" --seed "$SEED" --setups all --scenarios "$SCENARIOS"
            --runs-dir "$RUN_DIR")
CONT_ARGS=()
if [ -n "$SEP_RANGE" ] && [ "$SCENARIOS" -gt 0 ]; then
    CONT_ARGS=(--sep-range $SEP_RANGE --laser-range $LASER_RANGE
               --laser-offset-radius "$LASER_OFFSET_RADIUS")
fi
if [ -n "$INIT" ]; then
    [ -f "$INIT" ] || { echo "no checkpoint at INIT=$INIT"; exit 1; }
    TRAIN_ARGS+=(--init "$INIT")
fi

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
$VENV_PY -c "import torch, triton; print('torch       :', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, '| available', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "run dir     : $RUN_DIR"
echo "init        : ${INIT:-none (from scratch)}"
echo "scenarios   : $SCENARIOS ${CONT_ARGS[*]:-(grid / setups)}"
echo "=================================================================="

SRUN=(srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}")
"${SRUN[@]}" "$VENV_PY" pretrain_encoder.py "${TRAIN_ARGS[@]}" "${CONT_ARGS[@]}" $EXTRA_ARGS

RUN_SUB=$(ls -d "$RUN_DIR"/cnn_* | head -n 1)
RUN_NAME=$(basename "$RUN_SUB")
echo "trained     : $RUN_SUB"
EVAL_RUNS=("$RUN_SUB")
if [ -n "$INIT" ]; then
    # The starting encoder, evaluated next to the new one (a copy: no files next to INIT)
    INIT_DIR=$(dirname "$INIT")
    INIT_SUB="$RUN_DIR/init_$(basename "$INIT_DIR")"
    mkdir -p "$INIT_SUB"
    cp "$INIT" "$INIT_SUB/model.pt"
    cp "$INIT_DIR/config.json" "$INIT_SUB/config.json"
    for f in train.csv val.csv; do [ -f "$INIT_DIR/$f" ] && cp "$INIT_DIR/$f" "$INIT_SUB/$f"; done
    EVAL_RUNS+=("$INIT_SUB")
fi

# Tables in $RUN_DIR. Held-out scenarios: --scenario-seed 1 (the training scenarios use
# --scenario-seed 0)
if [ ${#CONT_ARGS[@]} -gt 0 ]; then
    "${SRUN[@]}" "$VENV_PY" eval_encoder.py --scenarios "$EVAL_SCENARIOS" --scenario-seed 1 "${CONT_ARGS[@]}" \
        --no-speed --no-figures --out "$RUN_DIR" "${EVAL_RUNS[@]}"
fi
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --scenarios "$EVAL_SCENARIOS" --scenario-seed 1 --no-speed --no-figures --out "$RUN_DIR" "${EVAL_RUNS[@]}"
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --setups all --no-speed --no-figures --out "$RUN_DIR" "${EVAL_RUNS[@]}"
"${SRUN[@]}" "$VENV_PY" eval_encoder.py --no-figures --out "$RUN_DIR" "${EVAL_RUNS[@]}"

FIXED_CKPT="$RUNS_ROOT/encoder/$RUN_NAME.pt"
cp "$RUN_SUB/model.pt" "$FIXED_CKPT"
echo "=================================================================="
echo "finished    : $(date)"
echo "checkpoint  : $RUN_SUB/model.pt"
echo "copied to   : $FIXED_CKPT (ENCODER_INIT for ppo_l40s.slurm)"
ls -lh "$RUN_DIR" "$RUN_SUB" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
