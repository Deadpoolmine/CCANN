#!/usr/bin/env bash

function setup_pm_fs() {
    FS_NAME=$1
    MOUNT_POINT=$2
    
    local abs_path
    local fs_path

    abs_path=$(cd "$(dirname "$0")" && pwd)
    fs_path=$abs_path/../FileSystems/$FS_NAME

    sudo mkdir -p "$MOUNT_POINT"
    echo "Setting up $FS_NAME on $MOUNT_POINT ..."
    if [ -e "$fs_path" ]; then
        cd "$fs_path" || exit
        make -j"$(nproc)"
        bash "$abs_path/setups/setup-$FS_NAME.sh"
        cd - || exit
    else
        # try mount directly
        bash "$abs_path/setups/setup-$FS_NAME.sh"
    fi
}

function start_http_server() {
    local path="$1"
    local port="$2"

    # check if RangeHTTPServer is installed
    if ! python3 -c "import RangeHTTPServer" &> /dev/null; then
        echo "RangeHTTPServer not found, installing it ..."
        pip install rangehttpserver
    fi

    echo "[INFO] Starting HTTP server for $path on port $port ..."
    cd "$path" || exit
    python3 -m RangeHTTPServer "$port" >/dev/null 2>&1 &
    sleep 1 # give it some time to start
}

function stop_http_server() {
    local port="$1"
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
}

function migrate_to_pm() {
    local index_path="$1"
    local fs_name="$2"

    setup_pm_fs "$fs_name" "/mnt/pmem0"

    # check whether aria2c is installed
    if ! command -v aria2c &> /dev/null; then
        echo "aria2c could not be found, using rsync for migration ..."
        rsync -a --info=progress2 "$index_path/" "/mnt/pmem0/index/"
    else
        echo "aria2c found, using aria2c for migration ..."
        
        index_port=8082
        target_index="/mnt/pmem0/index"

        start_http_server "$index_path" "$index_port"

       echo "[INFO] Copying files from $index_path to $target_index ..."
        for file in "$index_path"/*; do
            filename=$(basename "$file")
            aria2c -x16 -s28 -d "$target_index" -o "$filename" "http://127.0.0.1:$index_port/$filename"
        done

        stop_http_server "$index_port"
    fi


    echo "Data and index files have been migrated to persistent memory."
}

function clean_cache() {
    echo 1 > /proc/sys/vm/drop_caches
    echo 2 > /proc/sys/vm/drop_caches
    echo 3 > /proc/sys/vm/drop_caches
}

function where_is_script() {
    local script=$1
    cd "$( dirname "$script" )" && pwd
}

function table_create () {
    local TABLE_NAME
    local COLUMNS
    TABLE_NAME=$1
    COLUMNS=$2
    echo "$COLUMNS" >"$TABLE_NAME"
}

function table_add_row () {
    local TABLE_NAME
    local ROW
    TABLE_NAME=$1
    ROW=$2
    echo "$ROW" >> "$TABLE_NAME"
}