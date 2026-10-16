#!/bin/bash
# ppo_l40s.slurm - PPO training of the alignment agent (rl/ppo.py) on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/ppo_l40s.slurm                       # production run, defaults below
#
# Defaults of ppo.py: four setups, B = 1024 envs, rollout 128, K = 16, frozen encoder
# initialized from ENCODER_INIT, 200M env steps, eval every 200 iterations (~15 min at an
# estimated 4-5 s per iteration on the L40S; the RTX A2000 takes 21 s), latest.pt every
# 15 min, W&B project herriott-rl (auth from ~/.netrc; add --no-wandb to EXTRA_ARGS for an
# unlogged run). The W&B display name is built at start, e.g.
# ppo_cnn-pre4_k16_set4_b1024_s0_<jobid>. Per-iteration rollout / update / eval times are
# in train.csv and W&B (time_rollout_s, time_update_s, time_eval_s, fps*).
#
# Environment variables (sbatch --export=ALL,NAME=value,...):
#   ENCODER_INIT  encoder checkpoint (pretrain_encoder.py model.pt); default the copy that
#                 encoder_l40s.slurm writes:
#                 /ix1/kchen/evv/multireflection/rl_runs/encoder/cnn_w160_lr0.001_b256_8k_s0_set4.pt
#   RUN_ID        run directory name under /ix1/kchen/evv/multireflection/rl_runs/
#                 (default: this job id). If <run dir>/latest.pt exists the job resumes it
#                 (--resume); otherwise it starts a new run there.
#   EXTRA_ARGS    appended to ppo.py (word-split, so no spaces inside values)
#
# After the encoder job (chained):
#   jid=$(sbatch --parsable rl/cluster/encoder_l40s.slurm)
#   sbatch --dependency=afterok:${jid%%;*} rl/cluster/ppo_l40s.slurm
#
# Learning sanity checks (short jobs; set --time on the sbatch line):
#   (a) measured cell only, every episode starts at the nominal tilt (-0.38, 0.27); the
#       agent must reach the trapped region and stop. Oracle return from that start
#       0.7275 (3 moves to (0.21, 1.07), 2733 -> 11271 mm, P_ref = P_max); eval/all/
#       oracle_return in the logs. 64 identical eval episodes:
#   sbatch --time=03:00:00 --export=ALL,EXTRA_ARGS="--setups measured --init-center -0.38 0.27 --init-radius 0 --total-env-steps 30000000 --eval-every 10 --eval-n 64 --eval-envs 64 --name-tag fixnom" rl/cluster/ppo_l40s.slurm
#   (a') the same with starts uniform within +-0.3 deg of nominal (curriculum step):
#   sbatch --time=03:00:00 --export=ALL,EXTRA_ARGS="--setups measured --init-center -0.38 0.27 --init-radius 0.3 --total-env-steps 30000000 --eval-every 20 --eval-n 1024 --eval-envs 1024 --name-tag r03" rl/cluster/ppo_l40s.slurm
#   (b) four setups, random starts over the full +-2 deg box, B = 256; references on the
#       same eval set: always-stop 0, scripted 0.079, setup-blind homing 0.276,
#       setup-aware homing 0.828, oracle 0.862:
#   sbatch --time=03:00:00 --export=ALL,EXTRA_ARGS="--num-envs 256 --total-env-steps 30000000 --eval-every 50 --name-tag sanity" rl/cluster/ppo_l40s.slurm
#
# Resume / continue (same run dir; the config comes from latest.pt, only the run-control
# flags total-env-steps, eval-every, ckpt-minutes, wandb, wandb-project, unfreeze-at,
# encoder-lr-scale, compile, target-kl may change):
#   sbatch --export=ALL,RUN_ID=<jobid of the first job> rl/cluster/ppo_l40s.slurm
#   sbatch --export=ALL,RUN_ID=<jobid>,EXTRA_ARGS="--total-env-steps 300000000 --unfreeze-at 1600" rl/cluster/ppo_l40s.slurm
#
# Time limit: Slurm sends SIGUSR1 600 s before the limit (--signal below) and SIGTERM on
# scancel; ppo.py then drops the current rollout, writes latest.pt and exits 0, so the
# run continues with a resume job.
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>      (latest.pt is written on the SIGTERM)
#
# Runs .venv/bin/python directly (built by cluster/check_env.sh from uv.lock; torch 2.14
# cu130, triton 3.8). Never installs anything.

#SBATCH --job-name=mfl-ppo
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
RUNS_ROOT="${RUNS_ROOT:-/ix1/kchen/evv/multireflection/rl_runs}"
RUN_DIR="$RUNS_ROOT/${RUN_ID:-${SLURM_JOB_ID:-manual}}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
ENCODER_INIT="${ENCODER_INIT:-$RUNS_ROOT/encoder/cnn_w160_lr0.001_b256_8k_s0_set4.pt}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# No modules: the venv carries its own CPython and CUDA runtime, and a cuda module would
# shadow it through LD_LIBRARY_PATH. Slurm does not source ~/.bashrc, so purge explicitly.
module purge

cd "$PROJECT_DIR/rl"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
mkdir -p "$RUN_DIR"

# W&B files next to the checkpoints; Triton / inductor kernel caches on node-local scratch
export WANDB_DIR="$RUN_DIR"
CACHE="${SLURM_SCRATCH:-$RUN_DIR}"
export TRITON_CACHE_DIR="$CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor_cache"

if [ -f "$RUN_DIR/latest.pt" ]; then
    START_ARGS=(--resume)
    MODE="resume $RUN_DIR/latest.pt"
else
    [ -f "$ENCODER_INIT" ] || { echo "no encoder checkpoint at $ENCODER_INIT"; exit 1; }
    START_ARGS=(--encoder-init "$ENCODER_INIT")
    MODE="new run, encoder from $ENCODER_INIT"
fi

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "cpus        : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "python      : $VENV_PY ($($VENV_PY --version 2>&1))"
$VENV_PY -c "import torch, triton; print('torch       :', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, '| available', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "run dir     : $RUN_DIR"
echo "mode        : $MODE"
echo "extra args  : $EXTRA_ARGS"
echo "=================================================================="

# --cpus-per-task is passed explicitly: srun does not reliably inherit it from sbatch
# (Slurm 22.05+). srun forwards the --signal USR1 to ppo.py.
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" ppo.py \
    --run-dir "$RUN_DIR" \
    "${START_ARGS[@]}" \
    $EXTRA_ARGS

echo "=================================================================="
echo "finished    : $(date)"
ls -lh "$RUN_DIR" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
