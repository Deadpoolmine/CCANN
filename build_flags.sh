#!/usr/bin/env bash
set -e

if [[ -n "$1" && "$1" != "-DANN_TIMING" ]]; then
    echo "Unsupported compile flags: $1 (only -DANN_TIMING is supported)" >&2
    exit 2
fi

mkdir -p build
cd build
if [[ "$1" == "-DANN_TIMING" ]]; then
    cmake .. -DCCANN_TIMING=ON
else
    cmake .. -DCCANN_TIMING=OFF
fi
make -j56
