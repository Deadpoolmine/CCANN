#!/usr/bin/env bash

# this script is executed under the target file system directory
sudo umount /mnt/pmem0

sudo rmmod nova
sudo insmod nova.ko

sudo mount -t NOVA -o init,data_cow /dev/pmem0 /mnt/pmem0/