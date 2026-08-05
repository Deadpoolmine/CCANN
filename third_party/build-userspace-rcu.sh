#!/usr/bin/env bash
ABSPATH=$(cd "$(dirname "$0")"; pwd)
cd userspace-rcu || exit
mkdir -p build
./bootstrap # skip if using tarball
./configure --prefix="$ABSPATH"/userspace-rcu/build
make -j32
make install -j32