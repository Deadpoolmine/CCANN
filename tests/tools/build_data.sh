#!/usr/bin/env bash

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
CONVERTERS_PATH=$ABS_PATH/converters


# default usage: ./build_data.sh <raw_data_dir> <data_dir>
# example: ./build_data.sh /home/data/deadpool/ANN/RAW/siftsmall /home/data/deadpool/ANN/SIFT-SMALL/data

TARGET_RAW_DIR=$1
TARGET_DATA_DIR=$2

for file in "$TARGET_RAW_DIR"/*; do
    filename=$(basename "$file")
    echo "Processing $filename ..."
    if [[ "$filename" == *.fvecs ]]; then
        "$CONVERTERS_PATH"/fvecs_to_bin "$file" "$TARGET_DATA_DIR/${filename%.fvecs}.bin"
    elif [[ "$filename" == *.ivecs ]]; then
        "$CONVERTERS_PATH"/ivecs_to_bin "$file" "$TARGET_DATA_DIR/${filename%.ivecs}.bin"
    elif [[ "$filename" == *.bvecs ]]; then
        "$CONVERTERS_PATH"/bvecs_to_bin "$file" "$TARGET_DATA_DIR/${filename%.bvecs}.bin"
    fi
done