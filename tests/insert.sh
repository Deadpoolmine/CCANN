#!/usr/bin/env bash

# set -x 

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/common.sh"

ANN_NAME=$1 # CCANN by default
ON_PM=$2 # 0 by default
PERF=$3 # disabled by default
NUM_THREADS=$4 # 8 by default
WORKLOAD=$5 # bigann-small by default
VECS_PER_STEP=$6 # 5000 by default
NUM_STEPS=$7 # 1 by default
SEARCH_MODE=$8 # PIPE_SEARCH by default

if [[ -z "$ANN_NAME" ]]; then
    ANN_NAME=CCANN
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
if [[ -z "$NUM_THREADS" ]]; then
    NUM_THREADS=8
fi
if [[ -z "$WORKLOAD" ]]; then
    WORKLOAD="bigann-small"
    # WORKLOAD="bigann-100m"
fi
if [[ -z "$VECS_PER_STEP" ]]; then
    VECS_PER_STEP=5000
fi
if [[ -z "$NUM_STEPS" ]]; then
    NUM_STEPS=1
fi
if [[ -z "$SEARCH_MODE" ]]; then
    SEARCH_MODE="4"
fi

UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/SIFT-SMALL/data/
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-SMALL/index/
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/SIFT-100M/data/
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-100M/index/
elif [[ "${WORKLOAD}" == "bigann-1m" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/SIFT-1M/data/
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-1M/index/
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    DATA_PATH=/home/data/deadpool/ANN/DEEP-100M/data/
    INDEX_PATH=/home/data/deadpool/ANN/DEEP-100M/index/
fi

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
    rm $INDEX_PATH/100M_merge*
    rm $INDEX_PATH/100M_shadow*
elif [[ "${WORKLOAD}" == "bigann-1m" ]]; then
    rm $INDEX_PATH/1M_shadow_disk.index
    rm $INDEX_PATH/1M_shadow_journal.log
    rm $INDEX_PATH/1M_shadow_pq_compressed.bin
    rm $INDEX_PATH/1M_shadow_pq_pivots.bin
    rm $INDEX_PATH/1M_merge_journal*
    rm $INDEX_PATH/1M_merge*
    rm $INDEX_PATH/1M_shadow*
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    rm $INDEX_PATH/100M_shadow_disk.index
    rm $INDEX_PATH/100M_shadow_journal.log
    rm $INDEX_PATH/100M_shadow_pq_compressed.bin
    rm $INDEX_PATH/100M_shadow_pq_pivots.bin
    rm $INDEX_PATH/100M_merge_journal*
    rm $INDEX_PATH/100M_merge*
    rm $INDEX_PATH/100M_shadow*
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
    TYPE_PREFIX=uint8
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    # DATA_BIN=$DATA_PATH/100M.bbin
    INDEX_PREFIX=$INDEX_PATH/100M
    TYPE_PREFIX=uint8
elif [[ "${WORKLOAD}" == "bigann-1m" ]]; then
    # DATA_BIN=$DATA_PATH/1M.bbin
    INDEX_PREFIX=$INDEX_PATH/1M
    TYPE_PREFIX=uint8
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    INDEX_PREFIX=$INDEX_PATH/100M
    TYPE_PREFIX=float
fi

if [[ "${WORKLOAD}" == *"bigann"* ]]; then
    DATA_BIN=/home/data/straho/data/bigann/bigann.bbin
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    DATA_BIN=/home/data/straho/data/deep/200M.fbin
fi

L_disk=128
# VECS_PER_STEP=5000
# NUM_STEPS=1
INSERT_THREADS=$NUM_THREADS
BEAM_WIDTH=32

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    rm $INDEX_PATH/siftsmall_shadow_disk.index
    rm $INDEX_PATH/siftsmall_shadow_journal.log
    rm $INDEX_PATH/siftsmall_shadow_pq_compressed.bin
    rm $INDEX_PATH/siftsmall_shadow_pq_pivots.bin
    rm $INDEX_PATH/siftsmall_merge_journal*
elif [[ "${WORKLOAD}" == "bigann-1m" ]]; then
    rm $INDEX_PATH/1M_shadow_disk.index
    rm $INDEX_PATH/1M_shadow_journal.log
    rm $INDEX_PATH/1M_shadow_pq_compressed.bin
    rm $INDEX_PATH/1M_shadow_pq_pivots.bin
    rm $INDEX_PATH/1M_merge_journal*
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
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
        $UTILS_PATH/insert "$TYPE_PREFIX" $DATA_BIN $L_disk $VECS_PER_STEP $NUM_STEPS $INSERT_THREADS $INDEX_PREFIX $BEAM_WIDTH

    perf script -i $PERF_DATA > $PERF_UNFOLDED
    $flamegraph_dir/stackcollapse-perf.pl $PERF_UNFOLDED > $PERF_FOLDED
    $flamegraph_dir/flamegraph.pl --title "Insertion Flamegraph ($ANN_NAME)" $PERF_FOLDED > $PERF_SVG

    rm $PERF_DATA $PERF_UNFOLDED $PERF_FOLDED
else
    # valgrind --tool=helgrind 
    $UTILS_PATH/insert "$TYPE_PREFIX" "$DATA_BIN" $L_disk "$VECS_PER_STEP" "$NUM_STEPS" "$INSERT_THREADS" "$INDEX_PREFIX" $BEAM_WIDTH "$SEARCH_MODE"
fi


print_conf
