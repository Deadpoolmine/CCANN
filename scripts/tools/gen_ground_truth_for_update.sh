#!/usr/bin/env bash

# For SIFT-SMALL
# example usage: bash gen_ground_truth_for_update.sh PipeANN /home/data/straho/pipe_ann_data/bigann/truth.bin bigann-100m $((1000000*100)) 1000000 10

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
ANN_NAME=$1
UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

ORIG_TRUTH_BIN=$2
WORKLOAD=$3
INSERTS=$4
N_BATCHES=$5
TOPK=$6
INSERT_ONLY=1

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-SMALL/update_gt/
    TOTAL_NPTS=10000
elif [[ "${WORKLOAD}" == "bigann-1m" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-1M/update_gt/
    TOTAL_NPTS=1000000
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-100M/update_gt/
    TOTAL_NPTS=100000000
fi

BATCH_NPTS=$(("$INSERTS" + "$TOTAL_NPTS"))

"$UTILS_PATH"/gt_update "$ORIG_TRUTH_BIN" "$TOTAL_NPTS" "$BATCH_NPTS" "$N_BATCHES" "$TOPK" "$TARGET_DIR" "$INSERT_ONLY"

echo "Use $ORIG_TRUTH_BIN as the original ground truth to generate ground truth for $WORKLOAD with $INSERTS insertions in total, $N_BATCHES points per batch, top-$TOPK neighbors."
echo "Generated ground truth files are stored in $TARGET_DIR."