#!/usr/bin/env bash

# For SIFT-SMALL
# example usage: ./compute_gt_for_topk.sh CCANN /home/data/deadpool/ANN/SIFT-SMALL/data/siftsmall_base.bin bigann-small /home/data/deadpool/ANN/SIFT-SMALL/data/siftsmall_query.bin 10

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
ANN_NAME=$1
UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

DATA_BIN=$2
# generated topk ground truth
QUERY_BIN=$3
TOPK=$4

if [[ "${TARGET_WORKLOAD}" == "bigann-small" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-SMALL/data/
    TYPE=uint8
fi

"$UTILS_PATH"/utils/compute_groundtruth "$TYPE" "$DATA_BIN" "$QUERY_BIN" "$TOPK" "$TARGET_DIR/top${TOPK}_groundtruth.bin"