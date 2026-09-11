#!/bin/bash
# submit_control.sh - full-data control for the spatial hold-out: the same recipe trained on
# TRAIN + VAL + TEST for a fixed number of epochs (the epoch selected on VAL by the hole-trained
# runs, from their train_summary.json), saving the final-epoch weights. Its closed-loop replay is
# compared with the hole-trained models; its TEST-hole numbers are not a hold-out result, since it
# trained on those positions.
#
#   cd /ihome/kchen/evv13/multireflection && EPOCHS=<selected epoch> bash holdout/submit_control.sh
set -euo pipefail
cd /ihome/kchen/evv13/multireflection
D=/ix1/kchen/evv/multireflection/data
R=/ix1/kchen/evv/multireflection/runs
S=$D/holdout_split
RES=${RES:-64}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:?set EPOCHS to the selected epoch of the hole-trained runs}

# Every position; VAL is passed only so the script has something to report per epoch.
[ -f "$S/all_names.txt" ] || cat "$S/train_names.txt" "$S/val_names.txt" "$S/test_names.txt" | sort > "$S/all_names.txt"
[ "$(wc -l < "$S/all_names.txt")" -eq 228800 ] || { echo "all_names.txt is not 228800 lines"; exit 1; }

jid=$(sbatch --parsable --job-name=holdctl-r$RES-e$EPOCHS \
    --export=ALL,DATASET=dark$RES,EPOCHS=$EPOCHS,EXTRA_ARGS="--resolution $RES --seed $SEED --save-last --train-keys-file $S/all_names.txt --val-keys-file $S/val_names.txt" \
    cluster/train_l40s.slurm)
jid=${jid%%;*}
tid=$(sbatch --parsable --dependency=afterok:$jid --job-name=holdctltest-r$RES \
    --export=ALL,CKPT=$R/$jid/resnet18_l40s_dark${RES}_${jid}_last_model.pth,MODEL_RES=$RES,OUT_DIR=$R/$jid/holdout_test_last \
    cluster/holdout_test_l40s.slurm)
echo "control r$RES epochs $EPOCHS: train $jid -> test/closed loop ${tid%%;*}"
