# CCANN: Crash-Consistent Approximate Nearest Neighbor Search

CCANN is the first **crash-consistent** graph-based approximate nearest neighbor search (ANNS) index for persistent memory (PM), built upon [OdinANN (FAST'26)](https://www.usenix.org/conference/fast26/) and [PipeANN (OSDI'25)](https://www.usenix.org/conference/osdi25/). Its core innovation is **Soft Insert** — a per-vector crash-consistent persistence mechanism inspired by filesystem soft updates, which durably persists each vector to PM in milliseconds while delivering high-performance insertion and search.

## 🧠 Core Innovation: Soft Insert

### Motivation: The Reliability–Performance Tension

Existing graph-based ANNS indices (including OdinANN) rely on **graph-granularity copy-on-write (COW) compaction** to ensure crash consistency. This coarse-grained operation severely wastes PM's fine-grained I/O capability — even with PM acceleration, a single compaction still takes **>400 seconds**, and frequent compaction throttles insertion throughput to near zero, creating an irreconcilable tension between reliability and performance.

### The Soft Insert Mechanism

CCANN introduces **Soft Insert**, inspired by filesystem soft updates: it reasons about and enforces a **graph-aware write ordering**, leveraging PM's `clwb`/`sfence` primitives to guarantee ordered persistence. Simultaneously, the insert phase is decoupled from and deferred behind the search phase, executing concurrently with the next batch of searches to hide computation overhead.

**Three Write-Ordering Rules:**

1. **Persist vector data and tags before updating the location table**: ensures vector data is safely flushed before becoming externally visible.
2. **Neighbor updates precede neighbor location-table updates**: guarantees edge integrity is persisted before index structural changes.
3. **The target vector must be visible before neighbors can point to it**: prevents dangling references and ensures graph topological consistency.

### Three Enhancement Techniques

To further boost Soft Insert's performance and reliability, CCANN introduces three key techniques:

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Parallel Node Expansion (PNE)** | Parallelizes neighbor distance computation during search with early-exit to reduce wasted work | 28.2%–52.2% throughput gain |
| **Adaptive Concurrency Control (ACC)** | Dynamically adjusts concurrency based on CPU utilization to prevent PM bandwidth saturation | Stable high throughput, avoids performance collapse |
| **Index Snapshot Service (ISS)** | Records the ID of the latest fully-inserted vector, shrinking crash recovery scope to ∼1K vectors | Recovery in tens of seconds (100M-scale) to minutes (billion-scale) |

### Performance

| Metric | Result |
|--------|--------|
| **Insert throughput** (SIFT-100M, DEEP-100M) | Outperforms OdinANN by **99%–404%** |
| **Insert latency** | Reduced by **78.9%–82.8%** (vs. OdinANN) |
| **Search throughput** | Outperforms PipeANN/DiskANN by **54.4%–6.61×** |
| **Per-vector persistence latency (P99)** | **1.7–4.8 ms** (compaction-based schemes require 7.0–8.8 minutes) |
| **Crash recovery** | Tens of seconds (100M-scale) to minutes (billion-scale); full graph scan rebuild takes days |
| **Billion-scale insert throughput** | Maintains **∼3.58×** advantage |

---

## Code Modifications (vs. OdinANN/PipeANN)

The following files contain the core Soft Insert implementation and CCANN-specific enhancements:

### Core Modified Files

| File | Modification |
|------|-------------|
| `include/v2/journal.h` | **Soft Insert persistent journal**: epoch-level logging, crash recovery replay, journal entry formats for node insertion/deletion |
| `include/v2/page_cache.h` | PM page cache: crash-consistent allocation, epoch-aware eviction |
| `include/v2/lock_table.h` | Fine-grained, epoch-aware lock table for concurrent soft inserts |
| `include/v2/dynamic_index.h` | Dynamic index extended with epoch management and crash recovery hooks |
| `src/update/direct_insert.cpp` | **Soft Insert main logic**: epoch-batched insertion, selective journaling, crash-safe node allocation, PMEM transfer optimizations (`PMEM_F_MEM_NODRAIN`) |
| `src/update/dynamic_index.cpp` | Epoch commit/rollback, recovery orchestration, journal replay on startup |
| `src/update/delete_merge.cpp` | Soft delete with journaled tombstone markers |
| `src/index.cpp` | Recovery initialization, journal validation on index load |
| `src/ssd_index.cpp` | Extended with PM-aware crash-consistent page management |

### Configuration Flags

| Flag | Description |
|------|-------------|
| `SOFT_INSERT` | Enable Soft Insert crash-consistency protocol (core flag) |
| `ASYNC_INSERTION` | Decouple search from insertion for zero-latency-impact consistency |
| `FINE_GRAINED_CONCURRENCY` | Optimistic concurrent index table with crash-safe atomic operations |
| `BATCH_PRUNING` | Reduce computation overhead during neighbor pruning |
| `BATCH_SEARCH` | Optimize search convergence with batched expansion |

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
cd scripts/tools
# build memory index for CCANN with SIFT-family datasets
# ANN_NAME DATASET_FAMILY BASE_FILE INDEX_PREFIX
bash ./build_mem_index.sh CCANN SIFT /path/to/SIFT-SMALL/data/siftsmall_base.bin /path/to/SIFT-SMALL/index/siftsmall
```

# CCANN

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

## Quick Start

> **Prerequisite:** PM (Persistent Memory) hardware with DAX support is required.  
> Without PM, run `sudo bash scripts/setups/setup-ext4-dax.sh` to configure PM namespaces.

A self-contained quick start test (`tests/quick_start.cpp`) is provided to verify the PM-based insert and search pipeline **without any external dataset**.

### 1. Setup PM Environment (DAX)

```bash
# Create ext4-DAX filesystems on PM namespaces
sudo bash scripts/setups/setup-ext4-dax.sh
```

This script configures PM namespaces (`namespace0.0`, `namespace1.0`) in `fsdax` mode and mounts them as ext4-DAX at `/mnt/pmem0` and `/mnt/pmem1`.

### 2. Build (with cc-ann PM flags)

```bash
cd third_party/liburing && ./configure && make -j && cd ../..

# Uses cc-ann flags: -DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY
# (see scripts/tools/setup-cc-ann.sh for all configurations)
bash ./build.sh
```

### 3. Run Quick Start Test

```bash
cd build

# Default: 500 base + 500 insert, 128-dim float vectors, L2
./tests/quick_start

# Custom parameters
./tests/quick_start \
    --base 10000 --dim 256 --insert 1000 \
    --R 64 --L 128 --Ld 128 --K 20 --Ls 150 \
    --bw 4 --threads 16

# Show all options
./tests/quick_start --help
```

### What the Test Does

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Build disk index | Generates random vectors, builds memory index, creates PQ-compressed disk layout on PM |
| 2 | Init `DynamicSSDIndex` | Opens the PM-based disk index for insert & search |
| 3 | **PM Insert** | Inserts random vectors via Soft Insert (`sync_index.insert()`) |
| 4 | **PM Search** | Searches with brute-force ground truth, reports Recall@K |
| 5 | Cleanup | Removes temporary files |

### Example Output

The following is an example output of the quick start test:

```text
╔══════════════════════════════════════════════════════════╗
║    CCANN PM Quick Start — PM Insert & Search Test        ║
║          Self-contained — no external dataset             ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║         CCANN PM Quick Start — Insert & Search           ║
╚══════════════════════════════════════════════════════════╝

=== Configuration ===
  Data type:      float
  Dimensions:     128
  Base points:    500
  Insert points:  500
  R (max degree): 32
  L_build:        64
  L_disk:         64
  beam_width:     2
  Threads:        16
  K (top-K):      10
  L_search:       80

=== Step 1: Generate base data & build disk index ===
  Disk index built! Time: 0.497s

=== Step 2: Init DynamicSSDIndex ===
  Disk index loaded, 500 points

=== Step 3: Generate insert data & PM insert ===
  Inserted 100/500 pts
  Inserted 200/500 pts
  Inserted 300/500 pts
  Inserted 400/500 pts
  Inserted 500/500 pts
  Insert done! Time: 0.315s, throughput: 1586 ops/s

=== Step 4: Search test ===

╔══════════════════════════════════════════════════════════╗
║                    Search Results                        ║
╚══════════════════════════════════════════════════════════╝

  Queries:        10
  K (top-K):      10
  Recall@10:      91.00%
  Avg latency:    0.37 ms

  --- Top-10 results ---

  Query 0:
    Top-1: tag=883 dist=0.35
    Top-2: tag=672 dist=0.35
    ...

=== Step 5: Cleanup ===
  Temp files cleaned

╔══════════════════════════════════════════════════════════╗
║                PM Quick Start Complete                   ║
╠══════════════════════════════════════════════════════════╣
║  Base idx:            500  pts                           ║
║  Inserted:            500  pts                           ║
║  Build:             0.497  s                             ║
║  Insert:            0.315  s                             ║
║  Recall@10:         91.00  %                             ║
╚══════════════════════════════════════════════════════════╝
```

## Cite Our Paper

TBD

CCANN builds upon and gratefully acknowledges:
- **[OdinANN](https://github.com/thustorage/PipeANN)** (FAST'26) for the direct insert and dynamic index framework.
- **[PipeANN](https://github.com/thustorage/PipeANN)** (OSDI'25) for the pipelined SSD I/O search engine.
- **[DiskANN / FreshDiskANN](https://github.com/microsoft/DiskANN)** for the original graph-based disk index design.
