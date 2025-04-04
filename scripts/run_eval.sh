#!/usr/bin/env bash
# run_all_eval.sh

LEADTIMES=("24h" "72h" "120h")
CONFIGS=("normal" "normal_mixed" "mixed" "mixed_u")
DATAS=("rf" "f")

for LT in "${LEADTIMES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    for DATA_SPLIT in "${DATAS[@]}"; do
      FOLDER="trained_models/${LT}_${CFG}"
      echo "Evaluating data=$DATA_SPLIT for leadtime=$LT, config=$CFG"
      python eval.py \
        --leadtime "$LT" \
        --folder "$FOLDER" \
        --data "$DATA_SPLIT"
    done
  done
done
