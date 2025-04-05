#!/usr/bin/env bash
# run_all_eval.sh

LEADTIMES=("24h" "72h" "120h")
CONFIGS=("normal" "normal_mixed" "mixed" "mixed_u")
DATAS=("rf" "f")

root_raw="/home/groups/ai/buelte/precip/Singapur-Trip-25/data"
root_processed="data/EUPPBench"

for LT in "${LEADTIMES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    for DATA_SPLIT in "${DATAS[@]}"; do
      FOLDER="trained_models/${LT}_${CFG}"
      echo "Evaluating data=$DATA_SPLIT for leadtime=$LT, config=$CFG"
      python eval.py \
        --leadtime "$LT" \
        --dir "$FOLDER" \
        --data "$DATA_SPLIT" \
        --root_raw "$root_raw" \
        --root_processed "$root_processed"
    done
  done
done
