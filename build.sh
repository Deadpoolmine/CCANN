#!/usr/bin/env bash

export ADDITIONAL_DEFINITIONS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY"

mkdir -p build
cd build
cmake ..
make -j56