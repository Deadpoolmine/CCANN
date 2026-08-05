#!/bin/bash
# extract_table_auto_kv.sh

logfile="$1"
if [ -z "$logfile" ]; then
  echo "Usage: $0 logfile.txt"
  exit 1
fi

# 找包含 Ls 的表头行
header_line=$(grep -n '^[[:space:]]*Ls' "$logfile" | head -1 | cut -d: -f1)
if [ -z "$header_line" ]; then
  echo "❌ 未找到表头行（包含 'Ls'）"
  exit 1
fi

data_start=$((header_line + 2))
data=$(sed -n "${data_start},\$p" "$logfile" | grep -E '^[[:space:]]*[0-9]')

header="Ls QPS Mean_Lat Lat50 Lat90 Lat95 Lat99 Lat999 Recall@10 Disk_IOs"

awk -v header="$header" '
BEGIN { n=split(header, h, /[[:space:]]+/) }
{
  # 去掉前导空格，防止第一个字段为空
  sub(/^[[:space:]]+/, "", $0)
  m=split($0, v, /[[:space:]]+/)
  for (i=1; i<=n && i<=m; i++) {
    printf "%s: %s\n", h[i], v[i]
  }
}
' <<< "$data"
