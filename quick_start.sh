#!/bin/bash

sudo bash scripts/setups/setup-ext4-dax.sh
sudo bash build.sh
sudo build/tests/quick_start