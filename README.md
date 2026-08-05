# CCANN: Crash-Consistent Approximate Nearest Neighbor Search

CCANN 是首个面向持久内存（PM）的**崩溃一致性**图基向量近似最近邻搜索（ANNS）索引，基于 [OdinANN (FAST'26)](https://www.usenix.org/conference/fast26/) 和 [PipeANN (OSDI'25)](https://www.usenix.org/conference/osdi25/) 构建。CCANN 的核心创新是 **Soft Insert** — 受文件系统 soft update 启发的逐向量崩溃一致性持久化机制，在毫秒级内将每个向量安全持久化到 PM，同时提供高性能的向量插入和搜索。

## 🙏 Acknowledgments

CCANN 深切感谢以下开源项目：

- **[OdinANN](https://github.com/thustorage/PipeANN)** (FAST'26): Direct Insert for consistently stable performance in billion-scale graph-based vector search. 提供了基础的 **direct insert** 机制 (`src/update/direct_insert.cpp`) 和 **dynamic index** 框架 (`src/update/dynamic_index.cpp`)。
- **[PipeANN](https://github.com/thustorage/PipeANN)** (OSDI'25): Low-latency graph-based vector search via aligning best-first search with SSD. 提供了 **pipelined SSD I/O** 搜索引擎 (`src/search/pipe_search.cpp`)、**coroutine-based search** (`src/search/coro_search.cpp`)、以及高效的 **SSD index layout** (`src/ssd_index.cpp`)。

CCANN 在这些巨人的肩膀上进一步提供了 PM 的崩溃一致性保证。

---

## 🧠 Core Innovation: Soft Insert

### 动机：可靠性-性能的不可调和矛盾

现有图基 ANNS 索引（包括 OdinANN）依赖**图粒度的写时复制（COW）合并**来保证崩溃一致性。这一粗粒度操作严重浪费 PM 的细粒度 I/O 能力——即使使用 PM 加速，单次合并仍需 **>400 秒**，且高频合并会扼杀插入吞吐至接近零，造成可靠性-性能的不可调和矛盾。

### Soft Insert 机制

CCANN 提出 **Soft Insert**，受文件系统 soft update 启发：推理并执行**图感知的写入顺序**，利用 PM 的 `clwb`/`sfence` 原语强制执行有序持久化，同时将插入阶段从搜索阶段解耦并延迟，使其与下一批搜索并发执行以隐藏计算开销。

**三条写入顺序规则：**

1. **先持久化向量数据和标签，再更新位置表**：确保向量数据在对外可见前已安全落盘。
2. **邻居更新先于邻居位置表更新**：保证边的完整性在索引结构变更前已持久化。
3. **目标向量可见后方可使邻居指向它**：防止悬空引用，确保图的拓扑一致性。

### 三项增强技术

为进一步提升 Soft Insert 的性能与可靠性，CCANN 引入了三项关键技术：

| 技术 | 描述 | 效果 |
|------|------|------|
| **并行节点扩展 (PNE)** | 并行化搜索阶段的邻居距离计算，引入早期退出策略减少无效计算 | 吞吐提升 28.2%–52.2% |
| **自适应并发控制 (ACC)** | 根据 CPU 利用率动态调整并发度，防止 PM 带宽饱和 | 稳定高吞吐，避免性能崩塌 |
| **索引快照服务 (ISS)** | 记录最新完整插入的向量 ID，将崩溃恢复范围缩小至 ∼1K 向量 | 恢复仅需数十秒（百万级）至数分钟（十亿级） |

### 性能

| 指标 | 结果 |
|------|------|
| **插入吞吐** (SIFT-100M, DEEP-100M) | 超越 OdinANN **99%–404%** |
| **插入延迟** | 降低 **78.9%–82.8%**（vs. OdinANN） |
| **搜索吞吐** | 超越 PipeANN/DiskANN **54.4%–6.61×** |
| **逐向量持久化延迟 (P99)** | **1.7–4.8 ms**（合并方案需 7.0–8.8 分钟） |
| **崩溃恢复** | 数十秒（百万级）至数分钟（十亿级），全图扫描重建需数天 |
| **十亿级插入吞吐** | 保持 **∼3.58×** 优势 |

---

## Code Modifications (vs. OdinANN/PipeANN)

以下文件包含 Soft Insert 核心实现及 CCANN 特有增强：

### 核心修改文件

| File | Modification |
|------|-------------|
| `include/v2/journal.h` | **Soft Insert 持久化日志**：epoch 级日志记录，崩溃恢复回放，节点插入/删除的日志条目格式 |
| `include/v2/page_cache.h` | PM 页缓存：崩溃一致性分配，epoch 感知的淘汰策略 |
| `include/v2/lock_table.h` | 细粒度、epoch 感知的并发软插入锁表 |
| `include/v2/dynamic_index.h` | 动态索引扩展：epoch 管理，崩溃恢复钩子 |
| `src/update/direct_insert.cpp` | **Soft Insert 主逻辑**：epoch 批量插入，选择性日志，崩溃安全节点分配，PMEM 传输优化 (`PMEM_F_MEM_NODRAIN`) |
| `src/update/dynamic_index.cpp` | Epoch 提交/回滚，恢复编排，启动时日志回放 |
| `src/update/delete_merge.cpp` | 软删除：日志化 tombstone 标记 |
| `src/index.cpp` | 恢复初始化，索引加载时日志验证 |
| `src/ssd_index.cpp` | 扩展 PM 感知的崩溃一致性页管理 |

### 配置标志

| Flag | Description |
|------|-------------|
| `SOFT_INSERT` | 启用 Soft Insert 崩溃一致性协议（核心标志） |
| `ASYNC_INSERTION` | 搜索与插入解耦，零延迟影响的一致性保证 |
| `FINE_GRAINED_CONCURRENCY` | 乐观并发索引表，崩溃安全的原子操作 |
| `BATCH_PRUNING` | 减少邻居修剪时的计算开销 |
| `BATCH_SEARCH` | 批量扩展优化搜索收敛 |

---

# PM-ANN (Memory Index)

CCANN inherits OdinANN's PM-ANN design — a PM-specific memory index with full crash consistency after Soft Insert:

- PM-specific. Extremely high search/insertion performance: reach to 5K QPS at near the same accuracy.

  - Accelerated search path: 
    - Fast sync read for mitigated locking; 
    - Speculative expansion.
    - Adjusted in-memory search structure for entry point optimization.
  
  - Accelerated insertion path with PM-specific crash consistency awareness (approximate crash consistency via Soft Insert: do not need to protect many updates as they can be recomputed).
  
  - Reduced contention using optimistic concurrent index table.

### Configurations

- BATCH_PRUNING: Do not compute distances to all neighbors of a node at once during pruning. Instead, compute distance batch by batch until the degree is satisfied to minimize computation overhead.

- ANN_TIMING: Whether to enable timing.

- BATCH_SEARCH: Do not expand all neighbors of a node at once, especially when the search procedure converges. Also, do not do async read on PM, in order to mitigate lock contention and optimize CPU usage. This is because PM read can be byte-grained and very fast.

- ASYNC_INSERTION: Enable asynchronous insertion with decoupled search and insertion procedures, search can perfectly overlap with insertions.

- FINE_GRAINED_CONCURRENCY: Directly use atomicity of concurrent hash table without coarse-grained locks (i.e., the id2loc is visible by either old or new). For read-consistency, we wrap the id2loc lookup and neighbor retrieval in one critical section.

### Memory Index Build

```
cd tests/tools
# build memory index for CCANN with SIFT-family datasets
# ANN_NAME DATASET_FAMILY BASE_FILE INDEX_PREFIX
bash ./build_mem_index.sh CCANN SIFT /path/to/SIFT-SMALL/data/siftsmall_base.bin /path/to/SIFT-SMALL/index/siftsmall
```

# CCANN (SSD-Based Billion-Scale Search)

CCANN inherits PipeANN's billion-scale SSD-based vector search engine — a **low-latency, billion-scale, and updatable** graph-based vector store on SSD. Features:

* **Extremely low search latency**: <1ms in billion-scale vectors (top-10, 90% recall), only 1.14x-2.02x of in-memory graph-based index but **>10x** less memory usage (e.g., **40GB** for billion-scale datasets).

* **High search throughput**: 20K QPS in billion-scale vectors (top-10, 90% recall), higher than [DiskANN](https://github.com/microsoft/DiskANN) with `beam_width = 8` (latency-optimal) and [SPANN](https://github.com/microsoft/SPTAG).

* **Efficient vector updates**: `insert` and `delete` are supported with minimal interference with concurrent search (fluctuates only **1.07X**) and reduced memory usage (only **<90GB** for billion-scale datasets).

## Build CCANN

### Basic Configurations

* CPU: X86 and ARM CPUs are tested. SIMD (e.g., AVX2, AVX512) will boost performance.

* DRAM: ~40GB (search-only) or ~90GB (search-update) per billion vectors, which may increase for larger product quantization (PQ) table size (Here we assume 32B per vector).

* SSD: ~700GB for SIFT with 1B vectors, ~900GB for SPACEV with 1.4B vectors.

* OS: Linux kernel supporting `io_uring` (e.g., >= 5.15) delivers best performance. Otherwise, set `USE_AIO` option to `ON` to use `libaio` instead. We recommend using `Ubuntu 22.04`, but `Ubuntu 18.04` and `20.04` are also tested (`USE_AIO` option should be enabled).

* Compiler: `c++17` should be supported.

* Vector dataset: less than 2B vectors to avoid integer overflow, each record size (`vector_size + 4 + 4 * num_neighbors`) is less than 4KB (>= 4KB is supported by search-only workloads but not well-tested, and not supported by updates).

### Software Dependencies

For `Ubuntu >= 22.04`, the command to install them:

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev libmkl-full-dev libjemalloc-dev
```

The `libmkl` could be replaced by other `BLAS` libraries (e.g., OpenBlas).


### Build the Repository

First, build `liburing`. The compiled `liburing` library is in its `src` folder.

```bash
cd third_party/liburing
./configure
make -j
```


Then, build CCANN.

```bash
bash ./build.sh
```

## Quick Start (Search-Only)

This section introduces how to build disk index and search using CCANN.
To maximize search performance, `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` definitions should be enabled in `CMakeLists.txt`.

### For DiskANN Users

For DiskANN users with on-disk indexes, enabling CCANN only requires building an in-memory index (<10min for billion-scale datasets).
An example for SIFT100M (the hard-coded paths should be modified):

```bash
# Build in-memory index. Modify the INDEX_PREFIX and DATA_PATH beforehand.
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index file name prefix.
export DATA_PATH=/mnt/nvme/data/bigann/100M.bbin
build/tests/utils/gen_random_slice uint8 ${DATA_PATH} ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2 24 l2

# Search the on-disk index. Modify the query and ground_truth path beforehand.
# build/tests/search_disk_index <data_type> <index_prefix> <nthreads> <I/O pipeline width (max for CCANN)> <query file> <truth file> <top-K> <similarity> <search_mode (2 for CCANN)> <L of in-memory index> <Ls for on-disk index> 
build/tests/search_disk_index uint8 ${INDEX_PREFIX} 1 32 /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/data/bigann/100M_gt.bin 10 l2 2 10 10 20 30 40
```

Then, you should see results like this:

```
Search parameters: #threads: 1,  beamwidth: 32
... some outputs during index loading ...
[search_disk_index.cpp:216:INFO] Use two ANNS for warming up...
[search_disk_index.cpp:219:INFO] Warming up finished.
     L   I/O Width         QPS    Mean Lat     P99 Lat   Mean Hops    Mean IOs   Recall@10
=========================================================================================
    10          32     1952.03      490.99     3346.00        0.00       22.28       67.11
    20          32     1717.53      547.84     1093.00        0.00       31.11       84.53
    30          32     1538.67      608.31     1231.00        0.00       41.02       91.04
    40          32     1420.46      655.24     1270.00        0.00       52.50       94.23
```

### For Others Starting from Scratch

This part introduces how to download the datasets, build the on-disk index, and then search on it using CCANN.

#### Download the Datasets

* SIFT100M and SIFT1B from [BIGANN](http://corpus-texmex.irisa.fr/);
* DEEP1B using [deep1b_gt repository](https://github.com/matsui528/deep1b_gt) (Thanks, matsui528!);
* SPACEV100M and SPACEV1B from [SPTAG](https://github.com/microsoft/SPTAG).

If the links above are not available, you could get the datasets from [Big ANN benchmarks](https://big-ann-benchmarks.com/neurips21.html).

If the datasets follow `ivecs` or `fvecs` format, you could transfer them into `bin` format using:
```bash
build/tests/utils/bvecs_to_bin bigann_base.bvecs bigann.bin # for byte vecs (SIFT), bigann_base.bvecs -> bigann.bin
build/tests/utils/fvecs_to_bin base.fvecs deep.bin # for float vecs (DEEP) base.fvecs -> deep.bin
build/tests/utils/ivecs_to_bin idx_1000M.ivecs idx_1000M.ibin # for int vecs (SIFT groundtruth) idx_1000M.ivecs -> idx_1000M.ibin
```

We need the `bin` files for `base`, `query`, and `groundtruth`.

The SPACEV1B dataset is divided into several sub-files in SPTAG. You could follow [SPTAG SPACEV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) for how to read the dataset.
To concatenate them, just save the dataset's numpy `array` to `bin` format (the following Python code might be used).

```py
# bin format:
# | 4 bytes for num_vecs | 4 bytes for vector dimension (e.g., 100 for SPACEV) | flattened vectors |
def bin_write(vectors, filename):
    with open(filename, 'wb') as f:
        num_vecs, vector_dim = vectors.shape
        f.write(struct.pack('<i', num_vecs))
        f.write(struct.pack('<i', vector_dim))
        f.write(vectors.tobytes())

def bin_read(filename):
    with open(filename, 'rb') as f:
        num_vecs = struct.unpack('<i', f.read(4))[0]
        vector_dim = struct.unpack('<i', f.read(4))[0]
        data = f.read(num_vecs * vector_dim * 4)  # 4 bytes per float
        vectors = np.frombuffer(data, dtype=np.float32).reshape((num_vecs, vector_dim))
    return vectors
```

The dataset should contain a ground truth file for its full set.
Some datasets also contain the ground truth of subsets (first $k$ vectors). For example, SIFT100M's (the first 100M vectors of SIFT1B) ground truth could be found in `idx_100M.ivecs` of SIFT1B dataset.

#### Prepare the 100M Subsets

To generate the 100M subsets (SIFT100M, SPACEV100M, and DEEP100M) using the 1B datasets, use `change_pts` (for `bin`) and `pickup_vecs.py` (in the `deep1b_gt` repository, for `fvecs`).
```bash
# for SIFT, assume that the dataset is converted into bigann.bin
build/tests/change_pts uint8 /mnt/nvme/data/bigann/bigann.bin 100000000
mv /mnt/nvme/data/bigann/bigann.bin100000000 /mnt/nvme/data/bigann/100M.bbin

# for DEEP, assume that the dataset is in fvecs format.
python pickup_vecs.py --src ../base.fvecs --dst ../100M.fvecs --topk 100000000
# If DEEP is already in fbin format, change_pts also works.

# for SPACEV, assume that the dataset is concatenated into a huge file vectors.bin
build/tests/change_pts int8 /mnt/nvme/data/SPACEV1B/vectors.bin 100000000
mv /mnt/nvme/data/SPACEV1B/vectors.bin100000000 /mnt/nvme/data/SPACEV1B/100M.bin
```

Then, use `compute_groundtruth` to calculate the ground truths of 100M subsets. (DEEP100M's ground truth could be calculated using deep1b_gt).
If you want to evaluate **search-update workloads**, please **calculate top-1000 vectors** instead of top-100, from which we will pickup top-k vectors for different intervals.

Take SIFT100M as an example (in fact, its ground truth could also be found in SIFT1B):
```bash
# for SIFT100M, assume that the dataset is at /mnt/nvme/data/bigann/100M.bbin, the query is at /mnt/nvme/data/bigann/bigann_query.bbin 
# output: /mnt/nvme/data/bigann/100M_gt.bin
build/tests/utils/compute_groundtruth uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/data/bigann/bigann_query.bbin 1000 /mnt/nvme/data/bigann/100M_gt.bin
```

#### Build On-Disk Index

CCANN uses the same on-disk index as DiskANN.

```bash
# Usage:
# build/tests/build_disk_index <data_type (float/int8/uint8)> <data_file.bin> <index_prefix_path> <R>  <L>  <B>  <M>  <T> <similarity metric (cosine/l2) case sensitive>. <single_file_index (0/1)>
build/tests/build_disk_index uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme2/indices/bigann/100m 96 128 3.3 256 112 l2 0
```

The final index files will share a prefix of `/mnt/nvme2/indices/bigann/100m`.

Parameter explanation:

* R: maximum out-neighbors
* L: candidate pool size during build (build in fact conducts vector searches to optimize graph edges)
* B: in-memory PQ-compressed vector size. Our goal is to use 32 bytes per vector; higher-dimensional vectors might require more bytes.
* M: maximum memory used during build, 256GB is sufficient for the 100M index to be built totally in memory.
* T: number of threads used during build. Our machine has 112 threads.

We use the following parameters when building indexes:

| Dataset              | type             | R   | L   | B   | M   | T   | similarity |
| -------------------- | ---------------- | --- | --- | --- | --- | --- | ---------- |
| SIFT/DEEP/SPACEV100M | uint8/float/int8 | 96  | 128 | 3.3 | 256 | 112 | L2         |
| SIFT1B               | uint8            | 128 | 200 | 33  | 500 | 112 | L2         |
| SPACEV1B             | int8             | 128 | 200 | 43  | 500 | 112 | L2         |

#### Build In-Memory Entry-Point Index (Optional)

An in-memory index is optional but could significantly improve performance by optimizing the entry point. By selecting `mem_L` to 0 in `search_disk_index`, the in-memory index is automatically skipped.

You could build it in the following way (take SIFT100M as an example):

```bash
# dataset: SIFT100M, {output prefix}: {INDEX_PREFIX}_SAMPLE_RATE_0.01
export INDEX_PREFIX=/mnt/nvme2/indices/bigann/100m # on-disk index file name prefix.
# first, generate random slice, sample rate is 0.01.
build/tests/utils/gen_random_slice uint8 /mnt/nvme/data/bigann/100M.bbin ${INDEX_PREFIX}_SAMPLE_RATE_0.01 0.01
# output files: {output prefix}_data.bin and {outputprefix}_ids.bin
# mem index: {INDEX_PREFIX}_mem.index
# All the in-memory indexes are built using R = 32, L = 64.
build/tests/build_memory_index uint8 ${INDEX_PREFIX}_SAMPLE_RATE_0.01_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_0.01_ids.bin ${INDEX_PREFIX}_mem.index 0 0 32 64 1.2 24 l2
```

The output in-memory index should reside in three files: `100m_mem.index`, `100m_mem.index.data`, and `100m_mem.index.tags`.

## Quick Start (Search-Update)

This part shows how to evaluate these two workloads:

* **Search-insert workload**: Insert the second 100M vectors into an index built with the first 100M vectors in a dataset, with concurrent search.

* **Search-insert-delete workload**: Insert the second 100M vectors and delete the first 100M vectors in a dataset, with concurrent search.

The recall is calculated after every 1M vectors are inserted/deleted.

### Prerequisites

* Prepare datasets and run search-only CCANN, by referring to [Quick Start (Search-Only)](#quick-start-search-only).
* Disable `-DREAD_ONLY_TESTS` and `-DNO_MAPPING` flags.
* The in-memory index is optional.

### Generate Ground-Truths

In our evaluation, we insert the second 100M vectors in SIFT1B/DEEP1B into the initial index built using its first 100M vectors.
We require ground truth for every `[0, 100M+iM)` vectors, where `0 <= i <= 100`.
The trivial approach is to calculate all of them, but it is costly.

We take a tricky approach: **select top-10 vectors for each interval** from the **top-1000** in SIFT1B (or the first 200M vectors in SIFT). 
Thus, only one top-1000 of the whole dataset should be calculated.
Amazingly, this approach is very likely to succeed.

```bash
# build/tests/gt_update <file> <index_npts> <tot_npts> <batch_npts> <target_topk> <target_dir> <insert_only>
# /mnt/nvme/data/bigann/truth.bin is the top-1000 for SIFT.
# 100M vectors in the index, 200M vectors in total, each batch contains 1M vectors.
# <file> <index_npts> <total_npts> <batch_npts> <n_batches> <target_topk> <target_dir> <insert_only>
build/tests/gt_update /mnt/nvme/data/bigann/truth.bin 100000000 200000000 1000000 10 /mnt/nvme/indices_upd/bigann_gnd/1B_topk 1
```

The output files will be stored in `/mnt/nvme/indices_upd/bigann_gnd/1B_topk/gt_{i * batch_npts}.bin`, each `bin` contains the ground truth for `[0, index_npts + i*batch_npts)`.

For workload change (insert the second 100M, delete the first 100M), set the last parameter `insert_only` to `false`.
Then, it generates ground truth for every `[i*batch_npts, index_npts + i*batch_npts)` vectors.

### Prepare Tags (Optional)

Each vector corresponds to one tag, which does not change during updates (but its ID may change during `merge`, and location on disk may change during `insert`).

Use `gen_tags` to generate an identity mapping (vector ID -> tag) for the vector dataset.
For CCANN, this is optional, but this is necessary for FreshDiskANN.

```bash
# build/tests/gen_tags <type[int8/uint8/float]> <base_data_file> <index_file_prefix> <false>
build/tests/gen_tags uint8 /mnt/nvme/data/bigann/100M.bbin /mnt/nvme/indices_upd/bigann/100M false
```

Other mappings could also be generated by modifying the `gen_tags.cpp`.

### Run The Benchmark

**Search-insert workload.** Please run `test_insert_search`. Its workflow:

* Copy the original index to `_shadow.index`, to avoid polluting it.
* Execute `num_step` steps, each step inserts `vecs_per_step` vector and calculate recall.
* Concurrent search with the `Ls` are executed.

An example to insert the second 100M vectors into the SIFT100M dataset:

```bash
# build/tests/test_insert_search <type[int8/uint8/float]> <data_bin> <L_disk> <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode> <index_prefix> <query_file> <truthset_prefix> <truthset_l_offset> <recall@> <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>
# 500M_topk stores the above-mentioned ground truth.
build/tests/test_insert_search uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 1000000 100 10 32 0 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd_insert/500M_topk 0 10 4 4 0 20 30 40 50 60 80 100
```

Notes:

* This test takes ~1 day to complete. To reduce the evaluation time, set `num_step` to a smaller value (e.g., 10) or use a smaller dataset (e.g., SIFT2M).
* `data_bin` should contain all the data (200M vectors in this setup), or larger (e.g., SIFT1B).
* `search_mode` is set to `0` for best-first search. **If you are evaluating OdinANN, use `2` (CCANN) instead**.
* `L_disk` is set to `128` for 100M-scale datasets and `160` for billion-scale datasets.
* Set `mem_L` to non-zero when using the [in-memory index](#build-in-memory-entry-point-index-optional).

**Search-insert-delete workload.** Please run `overall_performance`. Its workflow:
* Copy the original index to `_shadow.index`, to avoid polluting it.
* Execute `step` steps, each step inserts and deletes `index_points / step` vector and calculate recall.
* Concurrent search with the `Ls` are executed.

An example to insert the second 100M vectors and delete the first 100M in SIFT100M dataset:

```bash
# tests/overall_performance <type[int8/uint8/float]> <data_bin> <L_disk> <indice_path> <query_file> <truthset_prefix> <recall@> <#beam_width> <step> <Lsearch> <L2>
build/tests/overall_performance uint8 /mnt/nvme/data/bigann/bigann_200M.bbin 128 /mnt/nvme/indices_upd/bigann/100M /mnt/nvme/data/bigann/bigann_query.bbin /mnt/nvme/indices_upd/bigann_gnd/500M_topk 10 4 100 20 30
```

### Notes

* The index is **not crash-consistent** after updates currently, journaling could be adopted for it.

* To save a **consistent index snapshot** after updates, use `final_merge` similar to `test_insert_search`.

* For better performance, please select the `search_mode` to `2` (CCANN) in `test_insert_search`, and set the `search_beam_width` to 32.
The in-memory index could also be used (but it is immutable during updates).


## Reproduce Results in Our Papers

The scripts we use for evaluation are placed in the `scripts/` directory. For details, please refer to:

* [CCANN](./README-CCANN.md) for search-only scripts in `tests-ccann`.
* [OdinANN](./README-OdinANN.md) for search-update scripts in `tests-odinann`.


## Cite Our Paper

If you use this repository in your research, please cite our papers:
```
@inproceedings {fast26odinann,
  author = {Hao Guo and Youyou Lu},
  title = {OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search},
  booktitle = {24th USENIX Conference on File and Storage Technologies (FAST 26)},
  year = {2026},
  address = {Santa Clara, CA},
  publisher = {USENIX Association}
}

@inproceedings {osdi25ccann,
  author = {Hao Guo and Youyou Lu},
  title = {Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year = {2025},
  address = {Boston, MA},
  pages = {171--186},
  publisher = {USENIX Association}
}
```

## Acknowledgments

CCANN is built upon and gratefully acknowledges the following open-source projects:

- **[OdinANN](https://github.com/thustorage/PipeANN)** (FAST'26): The direct insert mechanism and dynamic index framework form the foundation of CCANN's update path. We extend OdinANN's insert logic with crash-consistent Soft Insert.
- **[PipeANN](https://github.com/thustorage/PipeANN)** (OSDI'25): The pipelined SSD I/O search engine, coroutine-based search, and efficient SSD index layout provide CCANN's billion-scale search capability.
- **[DiskANN / FreshDiskANN](https://github.com/microsoft/DiskANN)**: The original graph-based disk index design that inspired both PipeANN and OdinANN.

**Core CCANN Contribution — Soft Insert:** A lightweight crash-consistency protocol for PM-resident ANN indices. See the [Soft Insert](#-core-innovation-soft-insert) section above for details.
