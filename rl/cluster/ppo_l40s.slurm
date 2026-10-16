#!/bin/bash
# ppo_l40s.slurm - PPO training of the alignment agent (rl/ppo.py) on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/ppo_l40s.slurm                       # production run, defaults below
#
# Defaults of ppo.py (stage 2): the full scenario range (--setups full: atlas ATLAS,
# regret-prioritized scenario sampler, 10 % impossible scenarios), B = 1024 envs, rollout
# 128, K = 16, frozen encoder initialized from ENCODER_INIT, 1e9 env steps (the 1-day limit
# and a resume job handle the rest), lr warmup 20 iterations then cosine to 10 %, adaptive
# entropy (target 2.0 -> 0.7 nats over 80 % of the steps), eval every 200 iterations (~15
# min at 4.4 s per iteration with the frozen encoder, run 4398136) on the full set and the
# four-setup set (deterministic and sampled policies), latest.pt every 15 min, W&B project
# herriott-rl (auth from ~/.netrc; add --no-wandb to EXTRA_ARGS for an unlogged run). The
# W&B display name is built at start, e.g. ppo_cnn-pre4_k16_full-plr_b1024_s0_<jobid>, with
# _ft<run> when INIT_FROM is set and _uf<it> with --unfreeze-at. Per-iteration rollout /
# update / eval times are in train.csv and W&B (time_rollout_s, time_update_s, time_eval_s,
# fps*). Four-setup runs as before: add --setups all to EXTRA_ARGS (ATLAS then unused).
#
# Startup of a new full run: the atlas is built once if ATLAS is missing (82,768
# scenarios x 401^2 M2 cells, 13.3 G rays; estimate 1-2 min on the L40S, logged as
# "atlas: built in ... s") and loaded otherwise (seconds); the oracle of the full eval set
# (4096 x 401^2 states) and of the four-setup set are computed once and cached in the run
# dir (oracle_full.pt, oracle_eval.pt; logged as "oracle plan (...): ... s").
#
# Environment variables (sbatch --export=ALL,NAME=value,...):
#   ENCODER_INIT  encoder checkpoint (pretrain_encoder.py model.pt); default the copy that
#                 encoder_l40s.slurm writes:
#                 /ix1/kchen/evv/multireflection/rl_runs/encoder/cnn_w160_lr0.001_b256_8k_s0_set4.pt
#   ATLAS         scenario atlas (scenarios.py), passed as --atlas to new runs; built there
#                 if missing. Default /ix1/kchen/evv/multireflection/rl_runs/atlas/atlas_v1.pt
#   INIT_FROM     previous ppo.py checkpoint for --init-from (new runs only; a resume
#                 ignores it): every tensor whose name and shape match, and its value-norm
#                 statistics (the 4-way setup head is skipped, the scenario head is new).
#                 Default empty (no init). E.g.
#                 /ix1/kchen/evv/multireflection/rl_runs/4398136/best.pt
#   RUN_ID        run directory name under /ix1/kchen/evv/multireflection/rl_runs/
#                 (default: this job id). If <run dir>/latest.pt exists the job resumes it
#                 (--resume); otherwise it starts a new run there.
#   EXTRA_ARGS    appended to ppo.py (word-split, so no spaces inside values)
#
# After the encoder job (chained):
#   jid=$(sbatch --parsable rl/cluster/encoder_l40s.slurm)
#   sbatch --dependency=afterok:${jid%%;*} rl/cluster/ppo_l40s.slurm
#
# Stage 2 run (fine-tuned from run 4398136; the frozen four-setup encoder does not
# transfer to the full range, see reports/encoder_tables_full.md, so it trains from the
# start at 0.1 x lr). Smoke first (builds the atlas once; atlas + oracles + 20 iterations
# + 2 evals, ~20 min); set --eval-every of the run to ~900 s / its time per iteration:
#   sbatch --time=01:00:00 --export=ALL,INIT_FROM=/ix1/kchen/evv/multireflection/rl_runs/4398136/best.pt,EXTRA_ARGS="--unfreeze-at 0 --encoder-lr-scale 0.1 --total-env-steps 2621440 --eval-every 10 --eval-n 1024 --no-wandb --name-tag smoke" rl/cluster/ppo_l40s.slurm
#   sbatch --export=ALL,INIT_FROM=/ix1/kchen/evv/multireflection/rl_runs/4398136/best.pt,EXTRA_ARGS="--unfreeze-at 0 --encoder-lr-scale 0.1 --eval-every 40" rl/cluster/ppo_l40s.slurm
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
#   sbatch --time=03:00:00 --export=ALL,EXTRA_ARGS="--setups all --num-envs 256 --total-env-steps 30000000 --eval-every 50 --name-tag sanity" rl/cluster/ppo_l40s.slurm
#
# Resume / continue (same run dir; the config comes from latest.pt, only the run-control
# flags total-env-steps, eval-every, ckpt-minutes, wandb, wandb-project, unfreeze-at,
# encoder-lr-scale, compile, target-kl, eval-sampled, enc-checkpoint may change; ATLAS and
# INIT_FROM are not passed):
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
ATLAS="${ATLAS:-$RUNS_ROOT/atlas/atlas_v1.pt}"
INIT_FROM="${INIT_FROM:-}"
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
    START_ARGS=(--encoder-init "$ENCODER_INIT" --atlas "$ATLAS")
    MODE="new run, encoder from $ENCODER_INIT, atlas $ATLAS"
    if [ -n "$INIT_FROM" ]; then
        [ -f "$INIT_FROM" ] || { echo "no checkpoint at INIT_FROM=$INIT_FROM"; exit 1; }
        START_ARGS+=(--init-from "$INIT_FROM")
        MODE="$MODE, init from $INIT_FROM"
    fi
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
