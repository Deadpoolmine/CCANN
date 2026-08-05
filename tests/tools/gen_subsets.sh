#!/usr/bin/env bash

# For SIFT-SMALL
# example usage: ./gen_subsets.sh CCANN /home/data/deadpool/ANN/SIFT-1M/data/sift_groundtruth.bin bigann-20k /home/data/deadpool/ANN/SIFT-1M/data/sift_query.bin 100

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
ANN_NAME=$1
UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

ORIG_DATA_BIN=$2
TARGET_WORKLOAD=$3
# generated topk ground truth
ORIG_QUERY_BIN=$4
TOPK=$5

if [[ "${TARGET_WORKLOAD}" == "bigann-2m" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-2M/data/
    TOTAL_NPTS=2000000
    TYPE=uint8
    PREFIX=2M
elif [[ "${TARGET_WORKLOAD}" == "bigann-20k" ]]; then
    TARGET_DIR=/home/data/deadpool/ANN/SIFT-20K/data/
    TOTAL_NPTS=20000
    TYPE=uint8
    PREFIX=20K
fi

mkdir -p "$TARGET_DIR"

# get original data bin dir

"$UTILS_PATH"/change_pts "$TYPE" "$ORIG_DATA_BIN" "$TOTAL_NPTS"
mv "${ORIG_DATA_BIN}${TOTAL_NPTS}" "$TARGET_DIR/$PREFIX.bbin"

echo "Use $ORIG_DATA_BIN as the data bin file to generate subsets for $TARGET_WORKLOAD at $TARGET_DIR, with $TOTAL_NPTS points."

"$UTILS_PATH"/utils/compute_groundtruth "$TYPE" "$TARGET_DIR/$PREFIX.bbin" "$ORIG_QUERY_BIN" "$TOPK" "$TARGET_DIR/${PREFIX}_groundtruth.bin"