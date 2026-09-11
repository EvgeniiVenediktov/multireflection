#!/bin/bash
# submit_64.sh - spatial hold-out pipeline on CRCD for one resolution (default 64 px): three
# training seeds on TRAIN (model selection on VAL), each followed by the TEST-hole evaluation and
# the offline closed-loop replay from the 108 origins (cluster/holdout_test_l40s.slurm).
#
#   cd /ihome/kchen/evv13/multireflection && bash holdout/submit_64.sh
#   RES=128 bash holdout/submit_64.sh
#   RES=512 SBATCH_ARGS="--partition=h200 --cpus-per-task=32 --mem=160G" \
#       TRAIN_EXTRA="--compile --num-workers 24" bash holdout/submit_64.sh
#
# SBATCH_ARGS overrides the training job's #SBATCH lines (partition, CPUs, memory); TRAIN_EXTRA is
# appended to the training command (a later --num-workers wins over the script's). Needs the split
# on /ix1: data/holdout_split/{split.csv,train_names.txt,val_names.txt,closed_loop_origins.txt}
# (built by holdout/make_split.py) and data/dark<RES>.tar.gz. The full-data control is submitted
# later with holdout/submit_control.sh, once the selected epochs are known.
set -euo pipefail
cd /ihome/kchen/evv13/multireflection
D=/ix1/kchen/evv/multireflection/data
R=/ix1/kchen/evv/multireflection/runs
S=$D/holdout_split
RES=${RES:-64}
SEEDS=${SEEDS:-"0 1 2"}
SBATCH_ARGS=${SBATCH_ARGS:-}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
# cluster/train_l40s.slurm names the run resnet18_l40s_<DATASET>_<job>, without the tag for dark512
TAG="dark${RES}_"; [ "$RES" = 512 ] && TAG=""
for f in split.csv train_names.txt val_names.txt closed_loop_origins.txt; do
    [ -f "$S/$f" ] || { echo "missing $S/$f"; exit 1; }
done

for seed in $SEEDS; do
    jid=$(sbatch --parsable $SBATCH_ARGS --job-name=hold-r$RES-s$seed \
        --export=ALL,DATASET=dark$RES,EXTRA_ARGS="--resolution $RES --seed $seed --train-keys-file $S/train_names.txt --val-keys-file $S/val_names.txt $TRAIN_EXTRA" \
        cluster/train_l40s.slurm)
    jid=${jid%%;*}
    tid=$(sbatch --parsable --dependency=afterok:$jid --job-name=holdtest-r$RES-s$seed \
        --export=ALL,CKPT=$R/$jid/resnet18_l40s_${TAG}${jid}_best_model.pth,MODEL_RES=$RES \
        cluster/holdout_test_l40s.slurm)
    echo "r$RES seed $seed: train $jid -> test ${tid%%;*}"
done
squeue -M gpu -u "$USER" -h -o "%i %j %T %R"
