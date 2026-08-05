import re
from collections import Counter

with open("DISTANCE-LOG", "r") as f:
    distances = []
    lines = f.readlines()
    for line in lines:
        match = re.search(r"Distance: (\d+)", line)
        if match:
            distances.append(int(match.group(1)))

counter = Counter(distances)
print(f"总共有 {len(distances)} 个距离")
print(f"有 {len(counter)} 个不同的距离")
# print("每个距离出现的次数：")
# for dist, count in counter.items():
#     print(f"{dist}: {count}")
