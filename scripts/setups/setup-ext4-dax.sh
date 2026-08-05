#!/usr/bin/env bash

function check_if_dev_exits() {
    local dev=$1
    output=$(ls -l "$dev")
    if [[ -z "$output" ]]; then
        echo "Device $dev does not exist"
        return 1
    fi
    return 0
}

function try_switch_ns0_0_mode() {
    local mode=$1

    if [[ "${mode}" == "devdax" ]]; then
        if check_if_dev_exits "/dev/dax0.0"; then
            echo "File \"/dev/dax0.0\" exists"
        else
            sudo ndctl disable-namespace namespace0.0
            sudo ndctl destroy-namespace namespace0.0 --force
            sudo ndctl create-namespace -m devdax
        fi
    fi

    if [[ "${mode}" == "fsdax" ]]; then
        if check_if_dev_exits "/dev/pmem0"; then
            echo "File \"/dev/pmem0\" exists"
        else
            sudo ndctl disable-namespace namespace0.0
            sudo ndctl destroy-namespace namespace0.0 --force
            sudo ndctl create-namespace -m fsdax
        fi
    fi
}

function try_switch_ns1_0_mode() {
    local mode=$1

    if [[ "${mode}" == "devdax" ]]; then
        if check_if_dev_exits "/dev/dax1.0"; then
            echo "File \"/dev/dax1.0\" exists"
        else
            sudo ndctl disable-namespace namespace1.0
            sudo ndctl destroy-namespace namespace1.0 --force
            sudo ndctl create-namespace -m devdax
        fi
    fi

    if [[ "${mode}" == "fsdax" ]]; then
        if check_if_dev_exits "/dev/pmem1"; then
            echo "File \"/dev/pmem1\" exists"
        else
            sudo ndctl disable-namespace namespace1.0
            sudo ndctl destroy-namespace namespace1.0 --force
            sudo ndctl create-namespace -m fsdax
        fi
    fi
}

try_switch_ns0_0_mode "fsdax"
try_switch_ns1_0_mode "fsdax"

# this script is executed under the target file system directory
sudo umount /mnt/pmem0

sudo mkfs.ext4 -F -b 4096 /dev/pmem0
sudo mount -t ext4 -o dax /dev/pmem0 /mnt/pmem0

sudo umount /mnt/pmem1

sudo mkfs.ext4 -F -b 4096 /dev/pmem1
sudo mount -t ext4 -o dax /dev/pmem1 /mnt/pmem1