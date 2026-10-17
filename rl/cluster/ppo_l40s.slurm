#!/bin/bash
# ppo_l40s.slurm - PPO training of the alignment agent (rl/ppo.py) on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch rl/cluster/ppo_l40s.slurm                       # production run, defaults below
#
# Defaults of ppo.py (stage 2, D7): continuous scenarios (--setups full --scenarios pool:
# separation SEP_RANGE 20-400 mm, laser offset uniform over the M1 hole disk of
# LASER_OFFSET_RADIUS 1.5 mm, laser pitch / yaw from a uniform M2 landing point within
# LASER_RANGE -8..8 deg, beams clipped by the hole or missing M2 discarded; a pool of
# 16,384 scanned scenarios, 1/16 rescanned per iteration, regret-prioritized replay within
# it, 10 % impossible scenarios), B = 1024 envs, rollout 128, K = 16, frozen encoder
# initialized from ENCODER_INIT, 1e9 env steps (the 1-day limit and a resume job handle
# the rest), lr warmup 20 iterations then cosine to 10 %, adaptive entropy (target 2.0 ->
# 0.7 nats over 80 % of the steps), eval every 200 iterations (--eval-sets, default
# cont,cont_samp,region,set4): eval_cont/ and eval_cont_samp/ (4096 seeded continuous
# scenarios of the training ranges, deterministic and sampled; best.pt is selected on
# eval_cont/), eval_region/ (4096 good continuous scenarios, 1024 per target-region bin:
# area of the M2 box cells with path >= 0.9 P_max < 0.02, 0.02-0.1, 0.1-0.5, >= 0.5 deg^2)
# and eval/ (the four setups), deterministic; the grid benchmarks of run 4407578 (eval_full/,
# eval_offgrid/, eval_extrap_short/, eval_extrap_long/; measured laser offsets) only when
# listed, e.g. --eval-sets cont,cont_samp,region,set4,full,offgrid,extrap_short,extrap_long,
# latest.pt every 15 min, W&B project herriott-rl (auth from ~/.netrc; add --no-wandb to
# EXTRA_ARGS for an unlogged run). The W&B display name is built at start, e.g.
# ppo_cnn-ftcont_k16_cont20-400-plr_b1024_s0_ft4407578_<jobid> (encoder origin, scenario
# source and range, _ft<run> when INIT_FROM is set, _uf<it> with --unfreeze-at, _e2e with
# --no-encoder-freeze, _ee<N> / _eu<F> with --encoder-epochs / --encoder-train-until,
# _pr<F> with a --pool-refresh other than 0.0625, _nb<F> / _rmix / _bmregion with
# --near-best-frac / --pool-region-mix / --best-metric region).
# Per-iteration rollout / update / eval times are in train.csv and W&B (time_rollout_s,
# time_update_s, time_eval_s, fps*), the pool refresh in pool/refresh_s (1.25 s per
# iteration on the L40S at the former default --pool-refresh 0.125, so ~0.6 s at the
# default 0.0625; runs resumed from older checkpoints keep their 0.125). Grid-atlas runs
# as before (run 4407578):
# SCENARIOS=atlas; four-setup runs: add --setups all to EXTRA_ARGS (ATLAS then unused).
#
# Startup of a new pool run: the pool is filled (16,384 slots: ~39k candidates, ~32k box
# scans of 401^2 M2 cells; estimate ~10 s on the L40S, logged as "pool: ... filled in
# ... s"); the atlas ATLAS is loaded only for grid benchmark sets (built once if missing:
# 82,768 scenarios, 22 s on the L40S); the continuous and region eval tables are built
# once in EVAL_TABLES if missing (cont20-400_n4096_seed0_v1.pt, ~10k candidates, seconds;
# cont20-400_region0.02-0.1-0.5_n4096_seed0_v1.pt, ~43k candidates, 35k box scans, ~33 s
# on an RTX A2000) and loaded otherwise, the off-grid tables likewise; the oracles of the
# sets (4096 x 401^2 states each) are computed once and cached in the run dir
# (oracle_cont.pt, oracle_region.pt, oracle_eval.pt, ...; logged as "oracle plan (...):
# ... s"). A grid hold-out (atlas runs only):
# EXTRA_ARGS="--holdout-frac 0.05" (seeded by --holdout-seed; adds eval_holdout/ and the
# -ho5 marker to the run name; best.pt stays selected on eval_full/, which then draws from
# the training scenarios only).
#
# Environment variables (sbatch --export=ALL,NAME=value,...):
#   ENCODER_INIT  encoder checkpoint (pretrain_encoder.py model.pt), new runs only; it
#                 replaces the encoder of INIT_FROM. "none" (or set empty): no
#                 --encoder-init, the encoder comes from INIT_FROM (required then), and
#                 the run name's encoder origin from that checkpoint's encoder_info.
#                 Default (unset) the copy of
#                 the D7 fine-tuned encoder that encoder_l40s.slurm writes:
#                 /ix1/kchen/evv/multireflection/rl_runs/encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt
#   SCENARIOS     pool (default, continuous) or atlas (grid), new runs only
#   SEP_RANGE     "20 400" (mm), LASER_RANGE "-8 8" (deg), LASER_OFFSET_RADIUS 1.5 (mm; 0 =
#                 the measured offsets): the pool's ranges, new runs only
#   ATLAS         scenario atlas (scenarios.py), passed as --atlas to new runs (pool runs:
#                 the fixed benchmark sets only); built there if missing. Default
#                 /ix1/kchen/evv/multireflection/rl_runs/atlas/atlas_v1.pt
#   INIT_FROM     previous ppo.py checkpoint for --init-from (new runs only; a resume
#                 ignores it): every tensor whose name and shape match, and its value-norm
#                 statistics (an old 4-way setup head is skipped; the scenario head is
#                 re-initialized when its normalization or width differs, e.g. 4407578's
#                 three-parameter 50-200 mm head in a pool run). Default empty (no init).
#                 E.g. /ix1/kchen/evv/multireflection/rl_runs/4407578/best.pt
#   RUN_ID        run directory name under /ix1/kchen/evv/multireflection/rl_runs/
#                 (default: this job id). If <run dir>/latest.pt exists the job resumes it
#                 (--resume); otherwise it starts a new run there.
#   EVAL_TABLES   off-grid and continuous eval tables dir, passed as --eval-tables to new
#                 and resumed runs. Default /ix1/kchen/evv/multireflection/rl_runs/eval_sets
#   EXTRA_ARGS    appended to ppo.py (word-split, so no spaces inside values)
#
# D7 run (continuous scenarios, separation 20-400 mm; PLAN.md D7), chained after the
# encoder fine-tuning job; transformer and heads from run 4407578 (its scenario head is
# re-initialized: 5 parameters over the new ranges), the new encoder frozen:
#   ENC=/ix1/kchen/evv/multireflection/rl_runs/encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt
#   INIT=/ix1/kchen/evv/multireflection/rl_runs/4407578/best.pt
#   jid=$(sbatch --parsable rl/cluster/encoder_l40s.slurm)
#   sid=$(sbatch --parsable --dependency=afterok:${jid%%;*} --time=01:00:00 --export=ALL,ENCODER_INIT=$ENC,INIT_FROM=$INIT,EXTRA_ARGS="--total-env-steps 2621440 --eval-every 10 --eval-n 1024 --no-wandb --name-tag smoke" rl/cluster/ppo_l40s.slurm)
#   sbatch --dependency=afterok:${sid%%;*} --export=ALL,ENCODER_INIT=$ENC,INIT_FROM=$INIT rl/cluster/ppo_l40s.slurm
# (smoke: pool fill, tables, oracles of 1024-episode sets, 20 iterations with refreshes,
# 2 evals, ~20-30 min; check pool/refresh_s and time_rollout_s in its train.csv and set
# --eval-every of the main run to ~900 s / its time per iteration if 200 is far off)
#
# Stage 2 run (fine-tuned from run 4398136; the frozen four-setup encoder does not
# transfer to the full range, see reports/encoder_tables_full.md, so it trains from the
# start at 0.1 x lr). Smoke first (builds the atlas once; atlas + oracles + 20 iterations
# + 2 evals, ~20 min); set --eval-every of the run to ~900 s / its time per iteration:
# (grid atlas, as run 4407578: SCENARIOS=atlas)
#   sbatch --time=01:00:00 --export=ALL,SCENARIOS=atlas,INIT_FROM=/ix1/kchen/evv/multireflection/rl_runs/4398136/best.pt,EXTRA_ARGS="--unfreeze-at 0 --encoder-lr-scale 0.1 --total-env-steps 2621440 --eval-every 10 --eval-n 1024 --no-wandb --name-tag smoke" rl/cluster/ppo_l40s.slurm
#   sbatch --export=ALL,SCENARIOS=atlas,INIT_FROM=/ix1/kchen/evv/multireflection/rl_runs/4398136/best.pt,EXTRA_ARGS="--unfreeze-at 0 --encoder-lr-scale 0.1 --eval-every 40" rl/cluster/ppo_l40s.slurm
#
# Encoder trained for part of the run, first PPO epoch only (r4b): everything, the
# fine-tuned encoder included, from a previous pool run's best checkpoint (ENCODER_INIT=none
# keeps its encoder; snapshot best.pt to a fixed file first, the run keeps writing it).
# The encoder trains at 0.1 x lr with gradients in the first of the 4 PPO epochs
# (--encoder-epochs 1; then the z of all stored frames are recomputed once without grad
# and the other 3 epochs train the rest) until 30 % of the env steps (--encoder-train-until
# 0.3: frozen from iteration 1145 of 3815, logged as "iteration 1145: encoder frozen for
# the rest of the run"). Expected update ~12-13 s per iteration while the encoder trains
# (36 s with gradients in all 4 epochs), 2.8 s after; --eval-every 60 is ~15 min in the
# first phase at ~15 s per iteration (~5 min after the switch):
#   INIT=/ix1/kchen/evv/multireflection/rl_runs/4410759/best_snapshot_it<N>.pt
#   sbatch --export=ALL,ENCODER_INIT=none,INIT_FROM=$INIT,EXTRA_ARGS="--no-encoder-freeze --encoder-lr-scale 0.1 --lr 1e-4 --total-env-steps 500000000 --encoder-epochs 1 --encoder-train-until 0.3 --eval-every 60" rl/cluster/ppo_l40s.slurm
# (run name ppo_cnn-ftcont_k16_cont20-400-plr_b1024_s0_ft4410759_e2e_ee1_eu0.3_<jobid>)
#
# Hard-target fine-tuning (PLAN.md "Future fine-tuning"): from a pool run's best snapshot,
# encoder frozen (ENCODER_INIT=none keeps its encoder); a quarter of the good-scenario
# episodes start on a ring 0.05-0.5 deg around the slot's best tilt (--near-best-frac,
# --near-best-radius; covers the neighbouring lower-order pattern), the good pool slots
# 20 / 20 / 25 / 35 % per target-region bin (--pool-region-mix, rejection with per-bin
# quotas: the refresh scans ~3.2x the candidates of the natural pool, ~2 s instead of
# ~0.6 s per iteration on the L40S, the start fill ~30 s; pool/refresh_s, pool/region_*,
# pool/rejected in the logs), best.pt on the mean over the four eval_region/ bins
# (--best-metric region, logged as best/value), the entropy target from the base
# policy's ~1.5 nats; ~6 s per iteration, so --eval-every 150 is ~15 min:
#   INIT=/ix1/kchen/evv/multireflection/rl_runs/4413333/best_snapshot_it<N>.pt
#   sbatch --time=06:00:00 --export=ALL,ENCODER_INIT=none,INIT_FROM=$INIT,EXTRA_ARGS="--near-best-frac 0.25 --near-best-radius 0.05 0.5 --pool-region-mix 0.2,0.2,0.25,0.35 --best-metric region --lr 1e-4 --total-env-steps 200000000 --ent-target-start 1.5 --ent-target-end 0.7 --eval-every 150" rl/cluster/ppo_l40s.slurm
# (run name ppo_cnn-ftcont_k16_cont20-400-plr_b1024_s0_ft4413333_nb0.25_rmix_bmregion_<jobid>)
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
# encoder-lr-scale, encoder-epochs, encoder-train-until, compile, target-kl, eval-sampled,
# enc-checkpoint, eval-sets, eval-tables, near-best-frac, near-best-radius, best-metric
# may change (a new best metric restarts the best value); ATLAS, INIT_FROM,
# SCENARIOS, the ranges, pool-refresh and pool-region-mix are not passed (the pool's
# slots, refresh share, region quotas and stream position come from latest.pt; a changed
# total-env-steps moves the encoder-train-until step with it, like the lr schedule); a run
# started before --eval-sets keeps the sets it evaluated (cont, full, offgrid, extrap_*,
# set4, each also sampled) unless EXTRA_ARGS="--eval-sets cont,cont_samp,region,set4" is given; new keys
# (e.g. eval_region/) of a resumed run go to W&B only: eval.csv keeps its header):
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
# Unset: the default encoder; "none" or empty: the encoder of INIT_FROM
ENCODER_INIT="${ENCODER_INIT-$RUNS_ROOT/encoder/cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384.pt}"
SCENARIOS="${SCENARIOS:-pool}"
SEP_RANGE="${SEP_RANGE:-20 400}"
LASER_RANGE="${LASER_RANGE:--8 8}"
LASER_OFFSET_RADIUS="${LASER_OFFSET_RADIUS:-1.5}"
ATLAS="${ATLAS:-$RUNS_ROOT/atlas/atlas_v1.pt}"
INIT_FROM="${INIT_FROM:-}"
EVAL_TABLES="${EVAL_TABLES:-$RUNS_ROOT/eval_sets}"
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
    START_ARGS=(--atlas "$ATLAS" --scenarios "$SCENARIOS")
    if [ -z "$ENCODER_INIT" ] || [ "$ENCODER_INIT" = "none" ]; then
        [ -n "$INIT_FROM" ] || { echo "ENCODER_INIT=${ENCODER_INIT:-(empty)} needs INIT_FROM (the encoder comes from it)"; exit 1; }
        MODE="new run, encoder from INIT_FROM"
    else
        [ -f "$ENCODER_INIT" ] || { echo "no encoder checkpoint at $ENCODER_INIT"; exit 1; }
        START_ARGS+=(--encoder-init "$ENCODER_INIT")
        MODE="new run, encoder from $ENCODER_INIT"
    fi
    MODE="$MODE, scenarios $SCENARIOS, atlas $ATLAS"
    if [ "$SCENARIOS" = "pool" ]; then
        START_ARGS+=(--sep-range $SEP_RANGE --laser-range $LASER_RANGE
                     --laser-offset-radius "$LASER_OFFSET_RADIUS")
        MODE="$MODE, separation $SEP_RANGE mm, laser $LASER_RANGE deg, offsets r $LASER_OFFSET_RADIUS mm"
    fi
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
echo "eval tables : $EVAL_TABLES"
echo "extra args  : $EXTRA_ARGS"
echo "=================================================================="

# --cpus-per-task is passed explicitly: srun does not reliably inherit it from sbatch
# (Slurm 22.05+). srun forwards the --signal USR1 to ppo.py.
srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" "$VENV_PY" ppo.py \
    --run-dir "$RUN_DIR" \
    "${START_ARGS[@]}" \
    --eval-tables "$EVAL_TABLES" \
    $EXTRA_ARGS

echo "=================================================================="
echo "finished    : $(date)"
ls -lh "$RUN_DIR" || true
echo "=================================================================="

# CRCD wrapper: appends the job's resource utilization. Must run after srun.
crc-job-stats 2>/dev/null || true
