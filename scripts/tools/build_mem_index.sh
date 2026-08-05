#!/usr/bin/env bash

# example usage: ./build_index.sh PipeANN SIFT /home/data/deadpool/ANN/SIFT-SMALL/data/siftsmall_base.bin /home/data/deadpool/ANN/SIFT-SMALL/index/siftsmall

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
elif [[ "${DATASET_FAMILY}" == "SIFT1B" ]]; then
    TYPE=uint8
elif [[ "${DATASET_FAMILY}" == "SPACEV1B" ]]; then
    TYPE=int8
else
    echo "Unsupported dataset family. Supported: SIFT, DEEP, SPACEV100M, SIFT1B, SPACEV1B"
    exit 1
fi

# Usage:
# build/tests/build_disk_index <data_type (float/int8/uint8)> <data_file.bin> <index_prefix_path> <R>  <L>  <B>  <M>  <T> <similarity metric (cosine/l2) case sensitive>. <single_file_index (0/1)>

export INDEX_PREFIX="$INDEX_PREFIX"

OUT_PREFIX="$DATA_FILE"_sample_rate_0.01
"$UTILS_PATH"/utils/gen_random_slice "$TYPE" "$DATA_FILE" "$OUT_PREFIX" 0.01

SAMPLE_DATA_FILE="${OUT_PREFIX}_data.bin"
SAMPLE_IDS_FILE="${OUT_PREFIX}_ids.bin"

rm -f "${INDEX_PREFIX}_mem.index"
MEM_INDEX_FILE="${INDEX_PREFIX}_mem.index"
"$UTILS_PATH"/build_memory_index "$TYPE" "$SAMPLE_DATA_FILE" "$SAMPLE_IDS_FILE" \
    "$MEM_INDEX_FILE" \
    1 0 32 64 1.2 24 l2