#!/usr/bin/env bash
# run_all_train.sh
# Example: train multiple runs. Adjust as needed.

LEADTIMES=("24h" "72h" "120h")
CONFIGS=("normal" "normal_mixed" "mixed" "mixed_u")
RUN_IDS=("0")

root_raw="/home/groups/ai/buelte/precip/Singapur-Trip-25/data"
root_processed="data/EUPPBench"

for LT in "${LEADTIMES[@]}"; do
  for CFG in "${CONFIGS[@]}"; do
    for RUN_ID in "${RUN_IDS[@]}"; do
      TARGET_DIR="trained_models/${LT}_${CFG}"
      echo "Running training for leadtime=$LT, config=$CFG, run_id=$RUN_ID"
      python train.py \
        --leadtime "$LT" \
        --dir "$TARGET_DIR" \
        --run_id "$RUN_ID" \
        --root_raw "$root_raw" \
        --root_processed "$root_processed"
    done
  done
done
