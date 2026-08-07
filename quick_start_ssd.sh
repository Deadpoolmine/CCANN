#!/bin/bash

mkdir -p /mnt/pmem0
rm -rf /mnt/pmem0/*

bash build.sh
build/tests/quick_start