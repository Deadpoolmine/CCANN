#!/usr/bin/env bash

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/../common.sh"

ANN_NAME=$1
CONFIG=$2
TIMING=$3

case "${CONFIG}" in
    "cc-ann-pne") # use para search
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DNO_ACC_OPT "
    ;;
    "cc-ann-pne-pm") # use para search
        FLAGS="-DDELTA_PRUNING -DNO_ACC_OPT "
    ;;
    "cc-ann-pne-pm-early-exit") # use para search
        FLAGS="-DDELTA_PRUNING -DEARLY_EXIT -DNO_ACC_OPT "
    ;;
    "cc-ann-pne-soft-insert") # use para search
        FLAGS="-DDELTA_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DNO_ACC_OPT "
    ;;
    "cc-ann-pne-soft-insert-acc") # para search
        FLAGS="-DDELTA_PRUNING -DEARLY_EXIT -DASYNC_INSERTION "
    ;;
    "cc-ann-pne-soft-insert-acc-va") # use beam search
        FLAGS="-DDELTA_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann-pne-soft-insert-acc-threshold") # TODO: use para search, identify the max concurrency
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DNO_ACC_OPT "
    ;;
esac

# "cc-ann-pne-soft-insert-acc-va") # use para search
#     FLAGS="-DDELTA_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
# ;;

if [[ "$TIMING" == "1" ]]; then
    FLAGS+=" -DANN_TIMING "
fi

cd $ABS_PATH/../../$ANN_NAME || exit
bash build_flags.sh "$FLAGS"
cd - || exit