#!/usr/bin/env bash

source ../common.sh

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

THREADS=(1 2 4 8 16 32)
MEDIAS=(1)
CANDIDATE_LS=( 128 )

ANNS=("cc-ann" "pipeann" "diskann")
# ANNS=("cc-ann")
WORKLOADS=("bigann-100m" "deep-100m")
# WORKLOADS=("bigann-100m")

TABLE_NAME="$ABS_PATH/performance-comparison-table-tput"
table_create "$TABLE_NAME" "workload media ANN num_job L(candidate) tput(QPS) mean(ms) lat50(ms) lat90(ms) lat95(ms) lat99(ms) lat999(ms) recall@10"

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
        if (( media == 1 )); then
            echo "ON PM selected"
            bash "$ABS_PATH"/../migrate_to_pm.sh "$INDEX_PATH" "ext4-dax"
        fi
        for ann in "${ANNS[@]}"; do        
            if [[ "${ann}" == "pipeann" ]]; then
                search_mode=2 # pipe search
            elif [[ "${ann}" == *"cc-ann"* ]]; then
                search_mode=4 # cc-ann search
            elif [[ "${ann}" == *"diskann"* ]]; then
                search_mode=0 # beam search
            fi
            
            if [[ "${ann}" == "pipeann" ]]; then
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh PipeANN "cc-ann-naive" 0
            else
                bash "$ABS_PATH"/../tools/setup-cc-ann.sh PipeANN "$ann" 0
            fi

            for t in "${THREADS[@]}"; do
                for L in "${CANDIDATE_LS[@]}"; do
                    echo "Running search with $t threads and L=$L ..."

                    temp_path="$ABS_PATH/DATA/${WORKLOAD}_media${media}_ann${ann}_threads${t}_L${L}.txt"
                    
                    OUTPUT=$(bash "$ABS_PATH"/../search.sh PipeANN "$media" 0 "$t" "$WORKLOAD" "$search_mode" "$L" 2>&1)
                    echo "$OUTPUT" > "$temp_path"
                    # Parse output
                    parsed_output=$(bash "$ABS_PATH"/parse.sh "$temp_path")

                    Ls=$(grep 'Ls:' <<< "$parsed_output" | awk '{print $2}')
                    QPS=$(grep 'QPS:' <<< "$parsed_output" | awk '{print $2}')
                    Mean_Lat=$(grep 'Mean_Lat:' <<< "$parsed_output" | awk '{print $2}')
                    Lat50=$(grep 'Lat50:' <<< "$parsed_output" | awk '{print $2}')
                    Lat90=$(grep 'Lat90:' <<< "$parsed_output" | awk '{print $2}')
                    Lat95=$(grep 'Lat95:' <<< "$parsed_output" | awk '{print $2}')
                    Lat99=$(grep 'Lat99:' <<< "$parsed_output" | awk '{print $2}')
                    Lat999=$(grep 'Lat999:' <<< "$parsed_output" | awk '{print $2}')
                    Recall10=$(grep 'Recall@10:' <<< "$parsed_output" | awk '{print $2}')

                    table_add_row "$TABLE_NAME" "$WORKLOAD $media $ann $t $Ls $QPS $Mean_Lat $Lat50 $Lat90 $Lat95 $Lat99 $Lat999 $Recall10"
                done
            done
        done
    done
done