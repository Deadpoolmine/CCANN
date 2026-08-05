#!/usr/bin/env bash

set -x

source ../common.sh

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

NUM_INSERT_THREADS=1
NUM_SEARCH_THREADS_ODINANN=8 # 5K QPS
MEDIAS=(1)

ANNS=( "cc-ann" )
WORKLOADS=("bigann-100m" "deep-100m")
WORKLOADS=("deep-100m")

TABLE_NAME="$ABS_PATH/performance-comparison-table-timing"
table_create "$TABLE_NAME" "workload media ANN insert_jobs search_jobs time(s) tput(QPS) lat50(us) lat90(us) lat99(us) lat999(us)"

function mean () {
    values=$1
    count=$(echo "$values" | wc -l)
    sum=$(echo "$values" | awk '{sum+=$1} END {print sum}')
    average=$(echo "$sum / $count" | bc -l)
    average=$(printf "%.2f" "$average")
    echo "$average"
}

mkdir -p "$ABS_PATH/TIMING_DATA"

for WORKLOAD in "${WORKLOADS[@]}"; do
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
        rm $INDEX_PATH/100M_shadow*
        rm $INDEX_PATH/100M_merge*
    fi

    for media in "${MEDIAS[@]}"; do
        # TODO: if the not the same layout, we need other migration
        if (( media == 1 )); then
            echo "ON PM selected"
            bash "$ABS_PATH"/../migrate_to_pm.sh "$INDEX_PATH" "ext4-dax"
        fi
        for ann in "${ANNS[@]}"; do
            if [[ "${ann}" == *"odinann"* ]]; then
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh CCANN "odinann" 1
                insert_t=$NUM_INSERT_THREADS
                search_t=$NUM_SEARCH_THREADS_ODINANN
            elif [[ "${ann}" == *"cc-ann"* ]]; then
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh CCANN "$ann" 1
                insert_t=$NUM_INSERT_THREADS
                search_t=$NUM_SEARCH_THREADS_CC_ANN
            fi

            echo "Running with $insert_t insert threads and $search_t search threads ..."
            n_batches=5
            batch_size=10000

            if [[ "${ann}" == *"odinann"* ]]; then
                if [[ "${ann}" == "odinann-improve" ]]; then
                    # pipe search
                    search_mode=2
                elif [[ "${ann}" == "odinann" ]]; then
                    # beam search
                    search_mode=0
                fi
            elif [[ "${ann}" == *"cc-ann"* ]]; then
                search_mode=4
                # search_mode=0
            fi

            output_path="$ABS_PATH/TIMING_DATA/${WORKLOAD}_media${media}_ann${ann}"
            bash "$ABS_PATH"/../overall.sh CCANN "$media" 0 "$insert_t" "$search_t" $batch_size $n_batches "$WORKLOAD" "$search_mode" 1 |& tee "$output_path"

            OUTPUT=$(cat "$output_path")

            tot_time=$(echo "$OUTPUT" | grep "Inserted" | grep "points" | awk '{print $6}' | sed 's/s//g')
            lat50=$(echo "$OUTPUT" | grep "50p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat90=$(echo "$OUTPUT" | grep "90p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat99=$(echo "$OUTPUT" | grep "99p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat999=$(echo "$OUTPUT" | grep "99.9p insertion time" | awk '{print $6}' | sed 's/us//g')

            tot_time=$(mean "$tot_time")
            lat50=$(mean "$lat50")
            lat90=$(mean "$lat90")
            lat99=$(mean "$lat99")
            lat999=$(mean "$lat999")
            tput=$(echo "$OUTPUT" | grep "Average insertion throughput" | awk '{print $5}' | sed 's/points\/s//g')

            table_add_row "$TABLE_NAME" "$WORKLOAD $media $ann $insert_t $search_t $tot_time $tput $lat50 $lat90 $lat99 $lat999"

        done
    done
done