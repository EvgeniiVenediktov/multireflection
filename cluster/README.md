# Training on Pitt CRCD

Everything here targets the CRCD `gpu` cluster (`ssh crc`, SLURM 23.11, multi-cluster —
GPU jobs need `--cluster=gpu`).

| | path |
|---|---|
| project files | `/ihome/kchen/evv13/multireflection` |
| dataset + checkpoints | `/ix1/kchen/evv/multireflection` |
| staged data during a job | `$SLURM_SCRATCH` (node-local NVMe, wiped at job end) |

## One-time setup

```bash
# from this machine - push the code (exclude .git, it is 259 MB and not needed there)
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    --exclude 'graphs' --exclude 'spie-archive' --exclude 'wandb' \
    ./ crc:/ihome/kchen/evv13/multireflection/

# and the dataset (~4.7 GB, resumable)
rsync -avP dark512.tar.gz crc:/ix1/kchen/evv/multireflection/data/

# then on the cluster
ssh crc
cd /ihome/kchen/evv13/multireflection
bash cluster/check_env.sh              # builds the venv from uv.lock (~6 GB, once)
./.venv/bin/python -m wandb login      # once; writes ~/.netrc
```

## Submit

```bash
sbatch cluster/train_l40s.slurm

squeue -M gpu -u $USER          # queue state
crc-job-stats <jobid>           # utilization after it runs
scancel -M gpu <jobid>
tail -f logs/mfl-resnet18-<jobid>.out
```

Checkpoints land in `/ix1/kchen/evv/multireflection/runs/<jobid>/`. After training the job
runs `utils/eval_batched.py` on the best checkpoint against the staged image bank, from the
0.1 deg grid (`runs/<jobid>/eval/`) and from the run's validation images
(`runs/<jobid>/eval_val/`, positions never trained on); summaries and heatmaps are attached
to the training's W&B run as `eval/*` and `eval_val/*` (all stages share `WANDB_RUN_ID`).

`sbatch --export=ALL,START_CKPT=<path>,EPOCHS=<n>,EXTRA_ARGS="<flags>"` fine-tunes from a
checkpoint, changes the epoch count, or appends training flags (see the script header).

## Resource request, and why

```
--cluster=gpu --partition=l40s --gres=gpu:1 --cpus-per-task=12 --mem=96G --time=1-00:00:00
```

**1 GPU.** ResNet-18 is 11M parameters and the dataset is 4.8 GB. There is nothing for a
second GPU to do, and multi-GPU requests only lengthen the queue wait.

**L40S (48 GB).** Verified available: a test allocation was granted immediately, and the
partition has 19 nodes. Driver is **595.91.07**, which supports CUDA 13, so the
`torch 2.14 + cu130` resolution in `uv.lock` runs unmodified — no CUDA module needed, since
the PyPI wheels bundle their own runtime.

**12 CPUs.** The l40s nodes are 64 cores / 4 GPUs, so 16 cores is the per-GPU fair share.
Profiling on the local A2000 showed the loader is not the bottleneck at all (2 workers fed
the GPU exactly as well as 12), but the L40S is several times faster, so this is headroom
rather than a tuned number.

**96 GB RAM.** This is the partition default (`DefMemPerCPU=8000` x 12 CPUs), so requesting
less would only give up free headroom — on the GPU cluster memory is billed at weight 0 and
jobs are charged per card. Far more than the 4.8 GB dataset needs.

**1 day.** The default `normal` QOS allows 3 days; GPU jobs top out at `long` (6 days, add
`--qos=long`). Shorter QOS carries higher priority, so do not over-request. See the estimate
below.

Validated with `sbatch --test-only`: the scheduler accepts every directive and reported it
would start within minutes on `gpu-n56`.

### Environment

Built once from `uv.lock`, so the cluster runs the same versions as local development:
torch 2.14.0+cu130, numpy 2.4.6, cv2 5.0.0, tqdm 4.70.0, wandb 0.30.0 on Python 3.11.

```bash
bash cluster/check_env.sh      # installs uv to ~/.local/bin, then uv sync
```

CRCD permits this. The documented restriction is only against installing to the *system* —
*"Users do not have privileges to install Python packages to the system"* — while conda
envs, venvs and `pip install --user` are explicitly supported. All three were verified to
work on this account, and uv is a single 48 MB static binary in `~/.local/bin`.

**The job never installs or syncs anything.** `train_l40s.slurm` runs `.venv/bin/python`
directly, so it needs no network and starts instantly. The flip side is that it cannot
notice a stale venv: **re-run `check_env.sh` after changing `uv.lock`.**

**No modules are loaded — `module purge` and nothing else.** The venv carries its own
CPython and its own CUDA runtime (the torch wheels bundle `nvidia-*` packages). Loading a
CUDA module would be actively harmful: `module show cuda/12.9.0` prepends its `lib64` to
`LD_LIBRARY_PATH`, which would shadow the CUDA 13 runtime in the venv. Only the driver
matters at runtime, and the GPU nodes run 595.91.07.

### GPU architecture

The wheel is compiled for `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` — **no native `sm_89`**,
which is what the L40S is. It runs correctly anyway: CUDA guarantees binary compatibility
forward across minor revisions within a major compute capability, so `sm_86` cubins execute
on `sm_89`, and cuDNN/cuBLAS ship their own Ada kernels. Verified on `gpu-n64`: correct
matmul results and 2681 img/s on a conv stack.

Worth knowing this is not specific to the CUDA 13 wheel — the CRCD module
`python/pytorch_251_311_cu124` (torch 2.5.1) has the same gap and works for the same reason.
`cluster/gpu_check.py` reports all of this on demand:

```bash
srun -M gpu -p l40s --gres=gpu:1 -c 8 --mem=32G -t 00:05:00 \
    ./.venv/bin/python cluster/gpu_check.py
```

### Storage tiers — measured

Measured from the login node, 1 GiB sequential (`dd`, `O_DIRECT`) and 600 random 22 KB file
reads, which is the shape of this dataset:

| | `/ix1/kchen/evv` | `/vast/kchen` |
|---|---|---|
| sequential write | 330 MB/s | 1.1 GB/s |
| sequential read | 331 MB/s | 584 MB/s |
| random 22 KB files | 732 files/s (1.37 ms) | 1083 files/s (0.92 ms) |

`/vast` is faster on every axis and CRCD's docs recommend it for ML training data; the group
has 1 TB of it entirely unused. **Caveat on these numbers:** `O_DIRECT` bypasses the client
page cache but not the server's, and the files were freshly written, so ZFS ARC and the NVMe
cache tier are likely serving them. Treat this as relative performance, not as evidence
about the underlying media.

For this workflow the tier does not matter: the job reads one 4.5 GB archive sequentially,
once (~14 s on `/ix1` vs ~8 s on `/vast`), then works entirely from node-local NVMe. `/ix1`
as specified is the right place.

What the numbers *do* show is why staging matters — at 732 files/s, reading the JPEGs
directly off `/ix1` every epoch would cap throughput below what the GPU wants, and `/vast`
at 1083 files/s would not comfortably clear it either.

### Other partitions

`--partition` alternatives on the same cluster, if `l40s` is busy:

| partition | GPU | per node | notes |
|---|---|---|---|
| `l40s` | L40S 48 GB | 4 GPU / 64 CPU / 515 GB | default choice here |
| `a100` | A100 40 GB | 4 GPU / 64 CPU / 515 GB | cluster default partition, often busy |
| `rtx6k` | RTX6000 96 GB | 8 GPU / 128 CPU / 1.5 TB | often has idle GPUs |
| `h200` | H200 141 GB | 8 GPU / 128 CPU / 3 TB | 2 nodes, in demand |
| `preempt` | mixed | — | shortest wait, but jobs can be killed |

Use `crc-idle -g` to see what is actually free before choosing.

## Why the data is staged to `$SLURM_SCRATCH`

The archive contains **228,800 files averaging 22 KB**. Reading those repeatedly straight
from `/ix1` is the many-small-files access pattern shared network filesystems handle worst,
and it degrades a resource other groups depend on. The job copies one 4.5 GB archive across
the network, extracts it to node-local NVMe (6.6 TB free), and reads from there.

`$SLURM_SCRATCH` is deleted when the job ends, which is why `--checkpoint-dir` points at
`/ix1`. Anything not copied out before the job finishes is gone.

## Runtime estimate

This is an **estimate**, not a measurement — the only figures actually measured are from a
local RTX A2000, which reached 157 img/s and was power-limited at 67 W of 70 W.

An L40S is several times that. At a guessed 600–900 img/s, one epoch over the 183,040-image
train split takes 3.5–5 minutes, so 96 epochs plus validation lands somewhere around
**7–12 hours**. The 1-day request covers that with margin.

The training script prints `img/s` every epoch, so the first epoch of the first run tells
you the real number. Adjust `--time` afterwards — a tighter wall time queues sooner.

## Batch size

The job uses `--batch-size 256 --lr 0.002`. Memory is ~70 MiB/image, so roughly 21 GB of
the L40S's 45 GB. The learning rate is the local `1e-3` scaled by sqrt(256/64) = 2.

Do not raise the batch size expecting speed. On the A2000, throughput was flat from bs=16
to bs=128 and then *collapsed* at 160 (87 img/s) once memory pressure set in. The L40S is
fast enough that larger batches may genuinely help there, but that is an assumption to
verify with `--batch-size`, not something the local profiling supports.

## Things to check on the first run

1. `crc-job-stats <jobid>` after it completes — if GPU utilization is low, the loader is
   starving and `--num-workers` should go up.
2. The `img/s` in the epoch lines, to replace the estimate above with a real number.
3. Whether `--batch-size 384` or `512` improves throughput on this GPU. It will not change
   memory safety much (~27 GB / ~36 GB) but may change convergence, so re-tune the LR.
