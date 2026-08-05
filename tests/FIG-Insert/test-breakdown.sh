#!/usr/bin/env bash

source ../common.sh

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

THREADS=(1 4 8 32)
MEDIAS=(1)

ANNS=("cc-ann" "cc-ann-pne" "cc-ann-pne-pm" "cc-ann-pne-pm-early-exit" "cc-ann-pne-soft-insert" "cc-ann-pne-soft-insert-acc" "cc-ann-pne-soft-insert-acc-va")
# ANNS=( "cc-ann-pne-soft-insert-acc" "cc-ann-pne-soft-insert-acc-va" "cc-ann")
# ANNS=( "cc-ann" )

# beam search (acc) + async insert + fine-grained concurrency = 10000 QPS with 32 threads

WORKLOADS=("bigann-100m")

TABLE_NAME="$ABS_PATH/performance-comparison-table-breakdown"
table_create "$TABLE_NAME" "workload media ANN num_job time(s) tput(QPS) lat50(us) lat90(us) lat99(us) lat999(us)"

# [insert.cpp:76:INFO] Inserted 5000 points in 2.87791s
# [insert.cpp:76:INFO] Average insertion throughput: 1738.56 points/s
# [insert.cpp:77:INFO] 10p insertion time : 3046 us
# [insert.cpp:78:INFO] 50p insertion time : 3907 us
# [insert.cpp:79:INFO] 90p insertion time : 6587 us
# [insert.cpp:80:INFO] 99p insertion time : 13525 us
# [insert.cpp:81:INFO] 99.9p insertion time : 35327 us

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
        # if (( media == 1 )); then
        #     echo "ON PM selected"
        #     bash "$ABS_PATH"/../migrate_to_pm.sh "$INDEX_PATH" "ext4-dax"
        # fi
        for ann in "${ANNS[@]}"; do
            bash "$ABS_PATH"/../tools/setup-cc-ann-breakdown.sh CCANN "$ann" 0
            for t in "${THREADS[@]}"; do
                echo "Running insert with $t threads ..."
                n_batches=1
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
                fi

                OUTPUT=$(bash "$ABS_PATH"/../insert.sh CCANN "$media" 0 "$t" "$WORKLOAD" $batch_size $n_batches "$search_mode")
                
                output_path="$ABS_PATH/DATA/insert_${WORKLOAD}_media_${media}_ann_${ann}_threads_${t}.txt"
                echo "$OUTPUT" > "$output_path"

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
done