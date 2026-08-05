#!/usr/bin/env bash

# set -x 

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/common.sh"

ANN_NAME=$1
ON_PM=$2
CC_SUPPORT=$3
PERF=$4
if [[ -z "$ANN_NAME" ]]; then
    ANN_NAME=PipeANN
fi
if [[ -z "$ON_PM" ]]; then
    ON_PM=0
fi
if [[ -z "$CC_SUPPORT" ]]; then
    CC_SUPPORT=0
fi
if [[ -z "$PERF" ]]; then
    PERF=0
fi
UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

NUM_THREADS=$5
if [[ -z "$NUM_THREADS" ]]; then
    NUM_THREADS=8
fi

WORKLOAD=$6
if [[ -z "$WORKLOAD" ]]; then
    WORKLOAD="bigann-small"
    # WORKLOAD="bigann-100m"
fi

SEARCH_MODE=$7
if [[ -z "$SEARCH_MODE" ]]; then
    SEARCH_MODE="4"
fi

# compile
cd $ABS_PATH/../$ANN_NAME || exit
# FLAGS="-DDELTA_PRUNING "
# FLAGS="-DBATCH_PRUNING -DASYNC_INSERTION -DANN_TIMING "
# FLAGS="-DBATCH_PRUNING "
FLAGS="-DBATCH_PRUNING -DANN_TIMING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
FLAGS="-DBATCH_PRUNING -DANN_TIMING -DEARLY_EXIT -DFINE_GRAINED_CONCURRENCY "

FLAGS="-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "

# FLAGS="-DBATCH_PRUNING -DANN_TIMING -DFINE_GRAINED_CONCURRENCY "
# FLAGS="-DBATCH_PRUNING -DANN_TIMING -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY "
# FLAGS="-DBATCH_PRUNING -DANN_TIMING -DASYNC_INSERTION "
# FLAGS="-DBATCH_PRUNING -DANN_TIMING -DBATCH_SEARCH "
# FLAGS="-DBATCH_PRUNING -DANN_TIMING "
if (( CC_SUPPORT == 1 )); then
    FLAGS+="-DCC_ANN"
elif (( CC_SUPPORT == 2 )); then
    FLAGS+="-DJ_ANN"
fi
bash build_flags.sh "$FLAGS"
cd - || exit

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/SIFT-SMALL/data/
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-SMALL/index/
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/SIFT-100M/data/
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-100M/index/
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/DEEP-100M/data/
    INDEX_PATH=/home/data/deadpool/ANN/DEEP-100M/index/
fi

if (( ON_PM == 1 )); then
    INDEX_PATH=/mnt/pmem0/index
fi

function print_conf() {
    echo "Configuration (of the last run):"

    if (( CC_SUPPORT == 2 )); then
        echo "Enable J-ANN"
    elif (( CC_SUPPORT == 1 )); then
        echo "Enable CC-ANN"
    elif (( CC_SUPPORT == 0 )); then
        echo "Disable CC-ANN"
    fi

    if (( ON_PM == 0 )); then
        echo "Running on NVMe SSD."
    else
        echo "Running on persistent memory."
    fi
}

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    # DATA_BIN=$DATA_PATH/siftsmall_base.bin
    INDEX_PREFIX=$INDEX_PATH/siftsmall
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    # DATA_BIN=$DATA_PATH/100M.bbin
    INDEX_PREFIX=$INDEX_PATH/100M
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    INDEX_PREFIX=$INDEX_PATH/100M
fi

if [[ "${WORKLOAD}" == *"bigann"* ]]; then
    # Path to 1B bigann dataset
    DATA_BIN=/home/data/straho/data/bigann/bigann.bbin
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    DATA_BIN=/home/data/straho/data/deep/200M.fbin
fi


L_disk=128
VECS_PER_STEP=5000
NUM_STEPS=10
INSERT_THREADS=$NUM_THREADS
BEAM_WIDTH=32

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    rm $INDEX_PATH/siftsmall_shadow_disk.index
    rm $INDEX_PATH/siftsmall_shadow_journal.log
    rm $INDEX_PATH/siftsmall_shadow_pq_compressed.bin
    rm $INDEX_PATH/siftsmall_shadow_pq_pivots.bin
    rm $INDEX_PATH/siftsmall_merge_journal*
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    rm $INDEX_PATH/100M_shadow_disk.index
    rm $INDEX_PATH/100M_shadow_journal.log
    rm $INDEX_PATH/100M_shadow_pq_compressed.bin
    rm $INDEX_PATH/100M_shadow_pq_pivots.bin
    rm $INDEX_PATH/100M_merge_journal*
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    rm $INDEX_PATH/100M_shadow_disk.index
    rm $INDEX_PATH/100M_shadow_journal.log
    rm $INDEX_PATH/100M_shadow_pq_compressed.bin
    rm $INDEX_PATH/100M_shadow_pq_pivots.bin
    rm $INDEX_PATH/100M_merge_journal*
fi

clean_cache

if (( PERF == 1 )); then    
    flamegraph_dir=/home/deadpool/Downloads/Flamegraph
    PERF_OUT_PREFIX=$ABS_PATH/perf_out/insert_"$ANN_NAME"
    mkdir -p $PERF_OUT_PREFIX
    PERF_DATA=$PERF_OUT_PREFIX/perf_data_insert_"$ANN_NAME"_$(date +%Y%m%d_%H%M%S).data
    PERF_UNFOLDED=$PERF_OUT_PREFIX/perf_unfolded_insert_"$ANN_NAME"_$(date +%Y%m%d_%H%M%S).unfolded
    PERF_FOLDED=$PERF_OUT_PREFIX/perf_folded_insert_"$ANN_NAME"_$(date +%Y%m%d_%H%M%S).folded
    PERF_SVG=$PERF_OUT_PREFIX/perf_svg_insert_"$ANN_NAME"_$(date +%Y%m%d_%H%M%S).svg
    sudo perf record -g -F 5000 -o $PERF_DATA -- \
        $UTILS_PATH/insert uint8 $DATA_BIN $L_disk $VECS_PER_STEP $NUM_STEPS $INSERT_THREADS $INDEX_PREFIX $BEAM_WIDTH "$SEARCH_MODE"

    perf script -i $PERF_DATA > $PERF_UNFOLDED
    $flamegraph_dir/stackcollapse-perf.pl $PERF_UNFOLDED > $PERF_FOLDED
    $flamegraph_dir/flamegraph.pl --title "Insertion Flamegraph ($ANN_NAME)" $PERF_FOLDED > $PERF_SVG

    rm $PERF_DATA $PERF_UNFOLDED $PERF_FOLDED
else
    # valgrind --tool=helgrind 
    $UTILS_PATH/insert uint8 $DATA_BIN $L_disk $VECS_PER_STEP $NUM_STEPS $INSERT_THREADS $INDEX_PREFIX $BEAM_WIDTH "$SEARCH_MODE"
fi


print_conf
