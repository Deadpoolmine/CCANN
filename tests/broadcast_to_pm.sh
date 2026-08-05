#!/usr/bin/env bash

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/common.sh"

INDEX_PATH=$1
FS_NAME=$2

broadcast_to_pm "$INDEX_PATH" "$FS_NAME"