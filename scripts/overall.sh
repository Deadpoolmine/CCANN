#!/usr/bin/env bash

set -x 

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

source "$ABS_PATH/common.sh"

ANN_NAME=$1
ON_PM=$2
PERF=$3
NUM_INSERT_THREADS=$4
NUM_SEARCH_THREADS=$5
VECS_PER_STEP=$6
NUM_STEPS=$7
WORKLOAD=$8
SEARCH_MODE=$9
INSERT_ONLY=${10}

if [[ -z "$ANN_NAME" ]]; then
    ANN_NAME=PipeANN
fi
if [[ -z "$ON_PM" ]]; then
    ON_PM=0
fi
if [[ -z "$PERF" ]]; then
    PERF=0
fi
if [[ -z "$NUM_INSERT_THREADS" ]]; then
    NUM_INSERT_THREADS=1
fi
if [[ -z "$NUM_SEARCH_THREADS" ]]; then
    NUM_SEARCH_THREADS=8
fi
if [[ -z "$WORKLOAD" ]]; then
    # WORKLOAD="bigann-small"
    WORKLOAD="bigann-100m"
fi
if [[ -z "$SEARCH_MODE" ]]; then
    SEARCH_MODE="4"
fi
if [[ -z "$VECS_PER_STEP" ]]; then
    VECS_PER_STEP=5000
fi
if [[ -z "$NUM_STEPS" ]]; then
    NUM_STEPS=1
fi
if [[ -z "$INSERT_ONLY" ]]; then
    INSERT_ONLY=0
fi

UTILS_PATH=$ABS_PATH/../$ANN_NAME/build/tests

if [[ "${WORKLOAD}" == "bigann-small" ]]; then
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-SMALL/index/
    TRUTH_SET=/home/data/deadpool/ANN/SIFT-SMALL/update_gt
    QUERY_BIN=/home/data/deadpool/ANN/SIFT-SMALL/data/siftsmall_query.bin
    TRUTH_SET_OFFSET=0
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    INDEX_PATH=/home/data/deadpool/ANN/SIFT-100M/index/
    TRUTH_SET=/home/data/deadpool/ANN/SIFT-100M/update_gt_100_1000000
    TRUTH_SET_OFFSET=0
    QUERY_BIN=/home/data/deadpool/ANN/SIFT-100M/data/bigann_query.bbin
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    INDEX_PATH=/home/data/deadpool/ANN/DEEP-100M/index/
    TRUTH_SET=/home/data/deadpool/ANN/DEEP-100M/update_gt_100_1000000
    TRUTH_SET_OFFSET=0
    QUERY_BIN=/home/data/deadpool/ANN/DEEP-100M/data/100M_queries.fbin
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
    INDEX_PREFIX=$INDEX_PATH/siftsmall
    TYPE_PREFIX=uint8
elif [[ "${WORKLOAD}" == "bigann-100m" ]]; then
    INDEX_PREFIX=$INDEX_PATH/100M
    TYPE_PREFIX=uint8
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    INDEX_PREFIX=$INDEX_PATH/100M
    TYPE_PREFIX=float
fi

L_disk=128
BEAM_WIDTH=4
SEARCH_THREADS=$NUM_SEARCH_THREADS
INSERT_THREADS=$NUM_INSERT_THREADS
RECALL=10
SEARCH_MEM_L=0 # L for in-memory search
# SEARCH_MEM_L=10 # L for in-memory search
SEARCH_BEAM_WIDTH=4 # beam width for search
SEARCH_L=(25 128) # do not search too much during insertion

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
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    rm $INDEX_PATH/100M_shadow_disk.index
    rm $INDEX_PATH/100M_shadow_journal.log
    rm $INDEX_PATH/100M_shadow_pq_compressed.bin
    rm $INDEX_PATH/100M_shadow_pq_pivots.bin
    rm $INDEX_PATH/100M_merge_journal*
    rm $INDEX_PATH/100M_merge*
    rm $INDEX_PATH/100M_shadow*
fi

if [[ "${WORKLOAD}" == *"bigann"* ]]; then
    DATA_BIN=/home/data/straho/data/bigann/bigann.bbin
elif [[ "${WORKLOAD}" == "deep-100m" ]]; then
    DATA_BIN=/home/data/straho/data/deep/200M.fbin
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
        $UTILS_PATH/test_insert_search "$TYPE_PREFIX" $L_disk $SEARCH_THREADS $SEARCH_MODE $INDEX_PREFIX $QUERY_BIN $TRUTH_SET $RECALL $BEAM_WIDTH $SEARCH_BEAM_WIDTH $SEARCH_MEM_L $SEARCH_L

    perf script -i $PERF_DATA > $PERF_UNFOLDED
    $flamegraph_dir/stackcollapse-perf.pl $PERF_UNFOLDED > $PERF_FOLDED
    $flamegraph_dir/flamegraph.pl --title "Insertion Flamegraph ($ANN_NAME)" $PERF_FOLDED > $PERF_SVG

    rm $PERF_DATA $PERF_UNFOLDED $PERF_FOLDED
else
    # valgrind --tool=helgrind

    $UTILS_PATH/test_insert_search "$TYPE_PREFIX" "$DATA_BIN" $L_disk "$VECS_PER_STEP" "$NUM_STEPS" "$INSERT_THREADS" "$SEARCH_THREADS" "$SEARCH_MODE" "$INDEX_PREFIX" "$QUERY_BIN" "$TRUTH_SET" "$TRUTH_SET_OFFSET" $RECALL $BEAM_WIDTH $SEARCH_BEAM_WIDTH $SEARCH_MEM_L $INSERT_ONLY ${SEARCH_L[@]}
fi


print_conf
