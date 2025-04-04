#!/usr/bin/env bash
# run_train.sh
# Example usage:
#    ./scripts/run_train.sh 24h configs/24h_mixed.json runX --no_graph

LEADTIME=$1
CONFIG=$2
RUN_ID=$3

# optional 4th argument could be e.g. "--no_graph", or empty
NO_GRAPH=$4

# Example command (adjust path as needed):
python train.py \
  --leadtime "${LEADTIME}" \
  --config "${CONFIG}" \
  --run_id "${RUN_ID}" \
  ${NO_GRAPH}
