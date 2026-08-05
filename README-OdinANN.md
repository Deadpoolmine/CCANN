# [FAST'26 Artifact] OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search

Welcome to the artifact repository of FAST'26 accepted paper: *OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search*!
This repository contains the implementation of OdinANN, and scripts to reproduce the experiment results in our paper.

Should there be any questions, please **contact the authors in HotCRP** (or GitHub issue). The authors will respond to each question as soon as possible.

## Evaluate the Artifact

We have prepared corresponding scripts to evaluate this artifact. 

### Hello-World Example (Artifact Functional, eta: 1 minute)

First, compile OdinANN (refer to [this](./README.md)).

```bash
bash ./build.sh
```

Generate groundtruths for SIFT1M dataset.

```bash
# 1M vectors in total, 10K vectors per batch, top-10 results.
# directly filter the top-10 results using SIFT2M groundtruths.
# <file> <index_npts> <total_npts> <batch_npts> <target_topk> <target_dir> <insert_only>
build/tests/gt_update /path/to/idx_2M.ibin 1000000 10000 10 /path/to/groundtruths 1
```

Modify the `test_insert_search` line in `scripts/tests-odinann/hello_world.sh`:
```bash
# /path/to/full-data.bin refers to a data bin with more than SIFT2M dataset (e.g., SIFT2M, SIFT100M, and SIFT1B are OK)
# /path/to/SIFT1M-index_disk.index should be the SIFT1M index file.
# /path/to/bigann_query.bin should be the bigann query file
# /path/to/groundtruths should be the groundtruth folder above. For example, /path/to/groundtruths/gt_0.bin and /path/to/groundtruths/gt_10000.bin both exist. 
build/tests/test_insert_search uint8 /path/to/full-data.bin 64 10000 1 10 32 0 /path/to/SIFT1M-index /path/to/bigann_query.bin /path/to/groundtruths 0 10 4 4 0 20

# _shadow is the hard-coded suffix for the copied index file.
rm /path/to/SIFT1M-index_shadow
```

Then, you could run the following command for functionality:

```bash
bash scripts/tests-odinann/hello_world.sh
```

This script runs `test_insert_search`, which inserts 10K vectors into a index built using 1M vectors.
You could get a similar output as below. Note that the QPS, latency, and Recall might differ slightly.
```
This is a hello-world example for OdinANN, it inserts 10K vectors into a 1M index.
[test_insert_search.cpp:369:INFO] num insert threads: 10
... some outputs during index loading ...
[utils.h:291:INFO] Reading truthset file /mnt/nvme/indices_upd/bigann_gnd_insert/2M_topk/gt_0.bin...
[utils.h:300:INFO] Metadata: #pts = 10000, #dims = 10...
  Ls        QPS           Mean Lat      50 Lat      90 Lat      95 Lat      99 Lat    99.9 Lat   Recall@10    Disk IOs
==============================================================================
  20     2902.91            8.8294       6.733      18.721       22.44      30.237       39.81      92.136      34.269
... some other outputs during concurrent insert-search ...
[direct_insert.cpp:285:INFO] Processed 9075 tasks, throughput: 1814.97 tasks/sec.
... some other outputs during concurrent insert-search ...
[utils.h:291:INFO] Reading truthset file /mnt/nvme/indices_upd/bigann_gnd_insert/2M_topk/gt_10000.bin...
[utils.h:300:INFO] Metadata: #pts = 10000, #dims = 10...
  Ls        QPS           Mean Lat      50 Lat      90 Lat      95 Lat      99 Lat    99.9 Lat   Recall@10    Disk IOs
==============================================================================
  20     9213.54           1.34341       1.139       2.199       2.617       3.181       3.708      92.155     34.2034
... some other outputs for storing the index ...
[test_insert_search.cpp:342:INFO] Store finished.
```

If you can see this output, then everything is OK, and you can start running the artifact.

### Run All Experiments (Results Reproduced, eta: 30 days)


#### Baselines

We use FreshDiskANN and SPFresh.

* **FreshDiskANN** is in [this repo](https://github.com/g4197/FreshDiskANN-baseline), you should [prepare tags](./README.md) to make it run. To handle memory leak, we restart FreshDiskANN after every merge.

* **SPFresh** is in [its repo](https://github.com/SPFresh/SPFresh). We modified the `SPFresh/SPFresh.h` in order to skip vector deletes in some experiments.

#### Scripts

We prepare scripts for all the experiments.

```bash
bash scripts/tests-odinann/figX.sh # X = 6, ..., 12
```

This generates the data used by Figure X. Here, we briefly introduce them:

* `fig6.sh` runs OdinANN, DiskANN and SPFresh using an insert-search workload on SIFT100M dataset. (eta: 4d)
* `fig7.sh` is similar to `fig6.sh`, but runs on DEEP100M dataset. (eta: 4d)
* `fig8.sh` runs the same workload using SIFT1B dataset (insert 200M vectors into the index with 800M vectors). (eta: 8d)
* `fig9.sh` inserts vectors using more insert threads. (eta: 1h)
* `fig10.sh` depends on `fig6.sh` and `fig7.sh`,without running additional experiments.
* `fig11.sh` runs the same workload using indexes with different out-neighbors (R). (eta: 5h)
* `fig12.sh` runs the insert-delete-search workload. (eta: 6d)

#### Preparation for Experiments

`fig8.sh` requires a SIFT800M index (the first 800M vectors of SIFT1B). To generate ground truths, use `gt_update`:
```bash
# Here we use 100M as the index_npts, to match the truthset_l_offset (700M) in fig8.sh
# <file> <index_npts> <total_npts> <batch_npts> <n_batches> <target_topk> <target_dir> <insert_only>
build/tests/gt_update /path/to/bigann_groundtruth.bin 100000000 1000000000 1000000 10 /path/to/groundtruths 1
```

Other experiments require 100M indexes (the first 100M vectors of SIFT1B and DEEP1B). To generate ground truths:
```bash
build/tests/gt_update /path/to/dataset_groundtruth.bin 100000000 200000000 1000000 10 /path/to/groundtruths 1 # 0 for fig12.sh
```

To reduce the run time, reduce the `num_step` parameter of `test_insert_search`.
The `overall_performance.cpp` should be modified similar to `test_insert_search.cpp`.

### Plot the Figures

#### (Recommended) For Visual Studio Code users

Please install the Jupyter extension in VSCode. Then, please open `scripts/plotting.ipynb`.

Please use the Python 3.10.12 (`/usr/bin/python`) kernel.

Then, you can run each cell from top to bottom. The first cell contains prelude functions and definitions, so please run it first. Each other cell plots a figure.

In each cell, set the variable `USE_EXAMPLE` to `True` to use our provided example data for plotting, `False` to use the data generated by the scripts above.

### Appendix

#### Code Structure

We mainly introduce the `src` folder, which contains the main code.

```bash
src/
├── CMakeLists.txt
├── index.cpp # in-memory Vamana index
├── ssd_index.cpp # on-disk index (search-only)
├── search # search algorithms, details in README-CCANN.md
│   ├── beam_search.cpp # best-first search
│   ├── coro_search.cpp # best-first search with inter-request scheduling
│   ├── page_search.cpp # search algorithm in Starling (SIGMOD '24)
│   └── pipe_search.cpp # our CCANN search algorithm
├── update
│   ├── delete_merge.cpp # delete and merge implementation
│   ├── direct_insert.cpp # insert implementation
│   └── dynamic_index.cpp # on-disk index wrapper (search-insert)
└── utils # some utils (mainly for index building)
    ├── aux_utils.cpp
    ├── distance.cpp
    ├── linux_aligned_file_reader.cpp # io_uring and AIO support
    ├── math_utils.cpp
    ├── partition_and_pq.cpp
    ├── prune_neighbors.cpp
    └── utils.cpp
```

#### Indexes Used

We pre-built the indexes in our evaluation.
Here are the related files:

```bash
/mnt/nvme/data/
├── bigann
│   ├── 100M.bbin # SIFT100M dataset
│   ├── 100M_gt.bin # SIFT100M ground truth
│   ├── truth.bin # SIFT1B ground truth
│   ├── bigann.bbin # SIFT1B dataset
│   └── bigann_query.bbin # SIFT query
└── deep
    ├── 100M.fbin # DEEP100M dataset
    ├── 100M_gt.bin # DEEP100M ground truth
    └── queries.fbin # DEEP query

/mnt/nvme/indices_upd
├── bigann # SIFT100M index
│   ├── 100M_disk.index
│   ├── 100M_disk.index.tags
│   ├── 100M_pq_compressed.bin
│   └── 100M_pq_pivots.bin
├── bigann_gnd
│   └── 500M_topk # generated ground truth for insert-delete-search
├── bigann_gnd_insert
│   ├── 1B_topk # generated ground truth for insert-search
│   └── 500M_topk # generated ground truth for insert-search
├── bigann_varl # parameter study for different R (Figure 11)
├── deep_gnd_insert
│   └── 200M_topk # generated ground truth for insert-search
└── sift1b # SIFT1B index (800M vectors, insert 200M vectors)
    ├── 800M_disk.index
    ├── 800M_disk.index.tags
    ├── 800M_pq_compressed.bin
    └── 800M_pq_pivots.bin

/mnt/nvme2
├── indices_spann # Indexes for SPFresh
├── SPFresh # SPFresh code
├── DiskANN # FreshDiskANN code
└── CCANN # source code of CCANN+OdinANN

# For each index of OdinANN, important files:
/mnt/nvme/indices_upd/SIFT1B/
├── 1B_disk.index # full graph index
├── 1B_mem.index # in-memory graph
├── 1B_mem.index.tags # in-memory vector ID -> on-disk vector ID
├── 1B_pq_compressed.bin # PQ-compressed vectors
└── 1B_pq_pivots.bin # PQ pivots
```

## Cite Our Paper

If you use this repository in your research, please cite our papers:

Hao Guo and Youyou Lu. OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search. To appear in the 24th USENIX Conference on File and Storage Technologies (FAST '26), Santa Clara CA USA, February 2026.

Hao Guo and Youyou Lu. Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD. To appear in the 19th USENIX Symposium on Operating Systems Design and Implementation (OSDI '25), Boston MA USA, July 2025.

## Acknowledgments

CCANN is based on [DiskANN and FreshDiskANN](https://github.com/microsoft/DiskANN/tree/diskv2), we really appreciate it.
