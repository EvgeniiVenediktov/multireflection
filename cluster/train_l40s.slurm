#!/bin/bash
# train_l40s.slurm - ResNet-18 tilt regression on one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch cluster/train_l40s.slurm
#
# Monitor:  squeue -M gpu -u $USER      |  crc-job-stats <jobid>
# Cancel:   scancel -M gpu <jobid>
#
# Uses the uv-managed venv built by cluster/check_env.sh, which matches uv.lock and so
# matches local development exactly. A zero-install fallback on the CRCD module
# python/pytorch_251_311_cu124 also works - see cluster/README.md.

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

# Resource rationale (measured, see cluster/README.md):
#   1 GPU   - single-GPU job; the model is 11M params and the dataset 4.8 GB. Nothing to
#             gain from multi-GPU, and it would only lengthen the queue wait.
#   12 CPU  - the l40s nodes are 64 cores / 4 GPUs, so 16 cores is the per-GPU fair share.
#             The loader needs far less (2 workers fed an A2000 as well as 12), but the
#             L40S is several times faster, so this leaves real headroom.
#   96 GB   - matches the partition default (DefMemPerCPU=8000 x 12 CPUs), so asking for
#             less would only give up free headroom: memory is billed at weight 0 on the
#             GPU cluster, where jobs are charged per card. Far more than the 4.8 GB
#             dataset needs; it lands on node-local NVMe anyway.
#   1 day   - the default "normal" QOS allows 3 days; GPU jobs top out at "long" (6 days,
#             add --qos=long). Shorter QOS means higher priority, so do not over-request.
#             See README for the runtime estimate.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
RUN_DIR="$DATA_ROOT/runs/${SLURM_JOB_ID:-manual}"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"

# ---------------------------------------------------------------------------
# Environment.
#
# Slurm does not source ~/.bashrc or ~/.profile, so this has to be explicit.
#
# module purge and nothing else: the venv carries its own CPython and its own CUDA runtime
# (the torch wheels bundle nvidia-* packages). Loading a cuda module would be actively
# harmful - cuda/12.9.0 prepends its lib64 to LD_LIBRARY_PATH and would shadow the CUDA 13
# runtime in the venv. Only the driver matters, and these nodes run 595.91.07.
#
# Note neither this wheel nor the CRCD module ships native sm_89 (Ada) kernels; both run
# correctly on the L40S through CUDA's forward binary compatibility from sm_86. Verified.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stage the dataset onto node-local NVMe.
#
# This is the important part. The archive holds 228,800 files averaging 22 KB. Reading
# those repeatedly over /ix1 (a shared network filesystem) is exactly the many-small-files
# pattern such filesystems are worst at, and it would also hammer a resource other people
# share. $SLURM_SCRATCH is node-local NVMe with several TB free, and it is wiped when the
# job ends, so anything worth keeping must be copied back out before then.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Train.
#
# batch_size 256 with lr 2e-3: 4x the local batch of 64, scaled by sqrt(4)=2 from 1e-3.
# Peak memory is roughly 70 MiB/image, so ~21 GB of the L40S's 45 GB.
# Compute nodes have outbound internet, so W&B logs online. Drop --wandb-project and add
# --no-wandb for an unlogged run.
# ---------------------------------------------------------------------------
srun "$VENV_PY" train/train_resnet_direct.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$RUN_DIR" \
    --name "resnet18_l40s_${SLURM_JOB_ID:-manual}" \
    --batch-size 256 \
    --lr 0.002 \
    --epochs 96 \
    --num-workers 10 \
    --val-num-workers 2 \
    --wandb-project multireflection

echo "=================================================================="
echo "finished    : $(date)"
echo "checkpoints : $RUN_DIR"
ls -lh "$RUN_DIR" || true
echo "=================================================================="

# CRCD wrapper: prints the job's resource utilization. Must run after srun, at the end.
crc-job-stats 2>/dev/null || true
