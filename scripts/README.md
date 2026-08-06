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

* CCANN's **Soft Insert** ensures per-vector crash consistency via graph-aware write ordering. After a crash, recovery replays the minimal journal to restore the last committed epoch (see [Soft Insert](#-core-innovation-soft-insert) above).

* To save a **consistent index snapshot** after updates, use `final_merge` similar to `test_insert_search`.

* For better performance, select `search_mode` to `2` (CCANN) in `test_insert_search`, and set `search_beam_width` to 32. The in-memory index can also be used (it is immutable during updates).


## Reproduce Results

The evaluation scripts are placed in the `scripts/` directory:

| Directory | Description |
|-----------|-------------|
| `scripts/FIG-Search/` | Search throughput and latency benchmarks |
| `scripts/FIG-Insert/` | Insert performance and breakdown tests |
| `scripts/FIG-Overall/` | End-to-end overall performance |
| `scripts/FIG-Breakdown/` | Component-level performance breakdown |
| `scripts/tools/` | Build, setup, and utility scripts |