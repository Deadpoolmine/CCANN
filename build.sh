#!/usr/bin/env bash

mkdir -p build
cd build
cmake .. -DCCANN_TIMING=OFF
make -j56
