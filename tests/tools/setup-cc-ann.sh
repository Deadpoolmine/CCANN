#!/usr/bin/env bash

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/../common.sh"

ANN_NAME=$1
CONFIG=$2
TIMING=$3

case "${CONFIG}" in
    "cc-ann-naive")
        # CCANN with pipelined PM reads
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS "
    ;;
    "cc-ann-no-async-insert")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann-acc")
        FLAGS="-DBATCH_PRUNING -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann-no-early-exit")
        FLAGS="-DBATCH_PRUNING -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann-journal")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DFINE_GRAINED_CONCURRENCY -DJ_ANN "
    ;;
    "cc-ann")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
    ;;
    "cc-ann-no-iss")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DNO_ISS "
    ;;
    "cc-ann-dist")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DDIRECT_READ_CC "
    ;;
    "cc-ann-dist-large")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DANN_LARGE -DDIRECT_READ_CC "
    ;;
    "cc-ann-no-acc")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DNO_ACC_OPT "
    ;;
    "cc-ann-recover")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DPM_RECOVERY "
    ;;
    "cc-ann-recover-opt")
        FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY -DISS_OPT -DPM_RECOVERY "
    ;;
    "odinann")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DODIN_ANN "
    ;;
    "odinann-dist")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DODIN_ANN -DDIRECT_READ_CC "
    ;;
    "odinann-dist-large")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DODIN_ANN -DANN_LARGE -DDIRECT_READ_CC "
    ;;
    "odinann-immediate-no-cc")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DODIN_ANN -DODIN_ANN_IMMEDIATE_NO_CC "
    ;;
    "diskann")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT "
    ;;
    "ccann")
        FLAGS="-DDELTA_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT "
    ;;
    "cc-ann-soft-insert") # use para search
        FLAGS="-DBATCH_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DNO_ACC_OPT -DASYNC_INSERTION "
    ;;
    "cc-ann-no-soft-insert") # use para search
        FLAGS="-DBATCH_PRUNING -DPIPE_PM_READS -DNO_PM_READ_OPT -DNO_ACC_OPT "
    ;;
esac

if [[ "$TIMING" == "1" ]]; then
    FLAGS+=" -DANN_TIMING "
fi

cd $ABS_PATH/../../$ANN_NAME || exit
bash build_flags.sh "$FLAGS"
cd - || exit