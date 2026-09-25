#!/bin/bash

set -e
cmake -S . -B build -DCCANN_BUILD_PYTHON=OFF
cmake --build build --target quick_start -j4
build/tests/quick_start "$@"
