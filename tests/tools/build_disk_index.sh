#!/usr/bin/env bash

# example usage: ./build_index.sh CCANN SIFT /home/data/deadpool/ANN/SIFT-SMALL/data/siftsmall_base.bin /home/data/deadpool/ANN/SIFT-SMALL/index/siftsmall

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
ANN_NAME=$1
UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

DATASET_FAMILY=$2
DATA_FILE=$3
INDEX_PREFIX=$4

if [[ "${DATASET_FAMILY}" == "SIFT" || "${DATASET_FAMILY}" == "DEEP" || "${DATASET_FAMILY}" == "SPACEV100M" ]]; then
    if [[ "${DATASET_FAMILY}" == "SIFT" ]]; then
        TYPE=uint8
    elif [[ "${DATASET_FAMILY}" == "DEEP" ]]; then
        TYPE=float
    elif [[ "${DATASET_FAMILY}" == "SPACEV100M" ]]; then
        TYPE=int8
    fi
    MAX_OUT_NEIGHBORS=96
    CANDIDATE_POOL_SIZE=128
    MEM_PQ_BUDGET=3.3
    MAX_MEM=256
elif [[ "${DATASET_FAMILY}" == "SIFT1B" ]]; then
    TYPE=uint8
    MAX_OUT_NEIGHBORS=128
    CANDIDATE_POOL_SIZE=200
    MEM_PQ_BUDGET=33
    MAX_MEM=500
elif [[ "${DATASET_FAMILY}" == "SPACEV1B" ]]; then
    TYPE=int8
    MAX_OUT_NEIGHBORS=128
    CANDIDATE_POOL_SIZE=200
    MEM_PQ_BUDGET=43
    MAX_MEM=500
else
    echo "Unsupported dataset family. Supported: SIFT, DEEP, SPACEV100M, SIFT1B, SPACEV1B"
    exit 1
fi

THREADS=56
SIMILARITY=l2
SINGLE_FILE_INDEX=0

# Usage:
# build/tests/build_disk_index <data_type (float/int8/uint8)> <data_file.bin> <index_prefix_path> <R>  <L>  <B>  <M>  <T> <similarity metric (cosine/l2) case sensitive>. <single_file_index (0/1)>
"$UTILS_PATH"/build_disk_index $TYPE "$DATA_FILE" "$INDEX_PREFIX" \
    $MAX_OUT_NEIGHBORS \
    $CANDIDATE_POOL_SIZE \
    $MEM_PQ_BUDGET \
    $MAX_MEM \
    $THREADS \
    $SIMILARITY \
    $SINGLE_FILE_INDEX