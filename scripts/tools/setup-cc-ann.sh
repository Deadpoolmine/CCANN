#!/usr/bin/env bash

set -e

ABS_PATH=$(cd "$(dirname "$0")" && pwd)
ANN_NAME=$1
CONFIG=$2
TIMING=$3

if [[ "$CONFIG" != "cc-ann" ]]; then
    echo "Unsupported configuration: $CONFIG (only cc-ann is supported)" >&2
    exit 2
fi

FLAGS=""
if [[ "$TIMING" == "1" ]]; then
    FLAGS="-DANN_TIMING"
fi

cd "$ABS_PATH/../../$ANN_NAME"
bash build_flags.sh "$FLAGS"
