#!/bin/bash
set -euo pipefail

source "utilities.sh"

#  cat ../../7b-context-extension-n32-evo1-32K/202411100226/model_configs/7b-evo1-32K.yml | grep -A2 save

ITERATION=12500
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints"
ROPE_SCALE="5x"

echo "ITERATION: ${ITERATION}, ROPE_SCALE: ${ROPE_SCALE}"

# ------------- #
echo "Step 1: Merge MP2 -> MP1"

MP_SIZE=1
SOURCE_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-n32-5x-32K/7b-5x-32K/202411100320/global_step${ITERATION}"
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/32K/MP${MP_SIZE}/global_step${ITERATION}"

# check_substring "$SOURCE_DIR" "$ROPE_SCALE"
# check_substring "$OUTPUT_DIR" "$ROPE_SCALE"
# check_directory_does_not_exist "$OUTPUT_DIR"

# CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --mp_size $MP_SIZE"


# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR
echo "Step 1: Done"

# # ------------- #

echo "Step 2: Extend 32K -> 64K"

NUM_GROUPS=256
SEQ_LENGTH=32768
TARGET_LENGTH=65536
TARGET_LEN_STR="64K"

SOURCE_DIR=$OUTPUT_DIR
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/${TARGET_LEN_STR}/MP1/global_step${ITERATION}"

# check_substring "$OUTPUT_DIR" "$ROPE_SCALE"
# check_substring "$OUTPUT_DIR" "$TARGET_LEN_STR"
# check_directory_does_not_exist "$OUTPUT_DIR"

# CMD="python conversion/extend_filter.py \
#     --source_dir $SOURCE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

# ls -lth $OUTPUT_DIR
echo "Step 2: Done"

# ------------- #

# echo "Step 3: Check ${TARGET_LEN_STR} filter"

# SOURCE_DIR=$OUTPUT_DIR

# CMD="python checks/check_filter_lens.py \
#     --source_dir $SOURCE_DIR \
#     --num_groups $NUM_GROUPS \
#     --seq_len $SEQ_LENGTH \
#     --target_seq_len $TARGET_LENGTH"

# echo $CMD
# eval $CMD

echo "Step 3: Done"

# # ------------- #

echo "Step 4: Convert MP1 -> MP8"

MP_SIZE=8
SOURCE_DIR=$OUTPUT_DIR
OUTPUT_DIR="${CHECKPOINT_DIR}/7b-context-extension/${ROPE_SCALE}/64K/MP${MP_SIZE}/global_step${ITERATION}"

check_directory_does_not_exist "$OUTPUT_DIR"

CMD="python conversion/convert_checkpoint_model_parallel_evo2.py \
    --source_dir $SOURCE_DIR \
    --output_dir $OUTPUT_DIR \
    --mp_size $MP_SIZE"

echo $CMD
eval $CMD

ls -lth $OUTPUT_DIR
echo "Step 4: Done"