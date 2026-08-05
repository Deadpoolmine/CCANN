#!/usr/bin/env bash

source ../common.sh

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

NUM_INSERT_THREADS=4
NUM_SEARCH_THREADS_CC_ANN=4 # 5K QPS
NUM_SEARCH_THREADS_ODINANN_IMPROVE=8 # 5K QPS
NUM_SEARCH_THREADS_ODINANN=24 # 5K QPS
MEDIAS=(1)

# "odinann-improve" 

ANNS=( "odinann" "cc-ann" )
WORKLOADS=("bigann-100m" "deep-100m")

TABLE_NAME="$ABS_PATH/performance-comparison-table"
table_create "$TABLE_NAME" "workload media ANN num_job time(s) tput(QPS) lat50(us) lat90(us) lat99(us) lat999(us)"

function mean () {
    values=$1
    count=$(echo "$values" | wc -l)
    sum=$(echo "$values" | awk '{sum+=$1} END {print sum}')
    average=$(echo "$sum / $count" | bc -l)
    average=$(printf "%.2f" "$average")
    echo "$average"
}

mkdir -p "$ABS_PATH/DATA"

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

    for media in "${MEDIAS[@]}"; do
        # TODO: if the not the same layout, we need other migration
        if (( media == 1 )); then
            echo "ON PM selected"
            bash "$ABS_PATH"/../migrate_to_pm.sh "$INDEX_PATH" "ext4-dax"
        fi
        for ann in "${ANNS[@]}"; do
            if [[ "${ann}" == *"odinann"* ]]; then
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh PipeANN "odinann" 0
                insert_t=$NUM_INSERT_THREADS
                if [[ "${ann}" == "odinann-improve" ]]; then
                    search_t=$NUM_SEARCH_THREADS_ODINANN_IMPROVE
                else
                    search_t=$NUM_SEARCH_THREADS_ODINANN
                fi
            elif [[ "${ann}" == *"cc-ann"* ]]; then
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh PipeANN "$ann" 0
                insert_t=$NUM_INSERT_THREADS
                search_t=$NUM_SEARCH_THREADS_CC_ANN
            fi

            echo "Running with $insert_t insert threads and $search_t search threads ..."
            n_batches=100
            batch_size=1000000

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

            # ANN_NAME=$1
            # ON_PM=$2
            # PERF=$3
            # NUM_INSERT_THREADS=$4
            # NUM_SEARCH_THREADS=$5
            # VECS_PER_STEP=$6
            # NUM_STEPS=$7
            # WORKLOAD=$8
            # SEARCH_MODE=$9

            output_path="$ABS_PATH/DATA/${WORKLOAD}_media${media}_ann${ann}"
            bash "$ABS_PATH"/../overall.sh PipeANN "$media" 0 "$insert_t" "$search_t" $batch_size $n_batches "$WORKLOAD" "$search_mode" |& tee "$output_path"

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

            table_add_row "$TABLE_NAME" "$WORKLOAD $media $ann $t $tot_time $tput $lat50 $lat90 $lat99 $lat999"

        done
    done
done