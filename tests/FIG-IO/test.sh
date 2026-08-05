#!/usr/bin/env bash

source ../common.sh

ABS_PATH=$(cd "$(dirname "$0")" && pwd)

THREADS=(1 2 4 8 16 32)
# 0 is no CC, 1 is CC-ANN, 2 is J-ANN
# CC_MODES=(0 2)
CC_MODES=(1)
# 0 is on SSD, 1 is on PM
# MEDIAS=(0 1)
MEDIAS=(0)

TABLE_NAME="$ABS_PATH/performance-comparison-table"
table_create "$TABLE_NAME" "media cc_mode num_job time(s) lat50(us) lat90(us) lat99(us) lat999(us)"

# [insert.cpp:76:INFO] Inserted 5000 points in 2.87791s
# [insert.cpp:77:INFO] 10p insertion time : 3046 us
# [insert.cpp:78:INFO] 50p insertion time : 3907 us
# [insert.cpp:79:INFO] 90p insertion time : 6587 us
# [insert.cpp:80:INFO] 99p insertion time : 13525 us
# [insert.cpp:81:INFO] 99.9p insertion time : 35327 us

for media in "${MEDIAS[@]}"; do
    for cc in "${CC_MODES[@]}"; do
        for t in "${THREADS[@]}"; do
            echo "Running insert with $t threads ..."
            OUTPUT=$(bash "$ABS_PATH"/../insert.sh CCANN "$media" "$cc" 0 "$t")
            
            tot_time=$(echo "$OUTPUT" | grep "Inserted" | grep "points" | awk '{print $6}' | sed 's/s//g')
            lat50=$(echo "$OUTPUT" | grep "50p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat90=$(echo "$OUTPUT" | grep "90p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat99=$(echo "$OUTPUT" | grep "99p insertion time" | awk '{print $6}' | sed 's/us//g')
            lat999=$(echo "$OUTPUT" | grep "99.9p insertion time" | awk '{print $6}' | sed 's/us//g')

            table_add_row "$TABLE_NAME" "$media $cc $t $tot_time $lat50 $lat90 $lat99 $lat999"
        done
    done
done
