#!/usr/bin/env bash
# run_eval.sh
# Example usage:
#    ./scripts/run_eval.sh 24h configs/24h_mixed.json runX rf --no_graph

LEADTIME=$1
CONFIG=$2
RUN_ID=$3
DATA_SPLIT=$4   # "rf" or "f"
EXTRA=$5        # e.g. "--no_graph"

python eval.py \
  --leadtime "${LEADTIME}" \
  --config "${CONFIG}" \
  --run_id "${RUN_ID}" \
  --data "${DATA_SPLIT}" \
  ${EXTRA}
