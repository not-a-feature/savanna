#/bin/bash
set -euo pipefail
DEFAULT_LOG_FILE="/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-v2/202410281731/logs/40b_train_v2/40b-train-n256-v2_85368_date_24-10-28_time_17-40-21.log"
DEFAULT_PATTERN="LOAD_CHECKPOINT"
LOG_FILE=$DEFAULT_LOG_FILE
PATTERN=${1:-$DEFAULT_PATTERN}

grep $PATTERN $LOG_FILE