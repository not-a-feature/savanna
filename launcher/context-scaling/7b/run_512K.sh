#!/bin/bash
set -euo pipefail

CONFIGS=("12" "13" "14")
for config in "${CONFIGS[@]}"; do
  CMD="./7b-context-scaling/run.sh ${config}"
  echo $CMD
  $CMD
  # Wait for user to press enter
  read -s -n 1 -p "Press enter to continue..."
done
