#!/bin/bash
# 文件名: cpu_monitor.sh

LOGFILE="./cpu_usage.log"
echo > "$LOGFILE"  # 清空日志文件
while true; do
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOGFILE"
    top -b -n1 | head -n 10 >> "$LOGFILE"
    echo "" >> "$LOGFILE"
    sleep 60   # 每 60 秒 = 1 分钟
done
