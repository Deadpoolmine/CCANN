#include "ssd_index.h"
#include "v2/dynamic_index.h"

#include <index.h>
#include <cstddef>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int NUM_SEARCH_THREADS = 32;

int search_mode = BEAM_SEARCH;

int begin_time = 0;
ccann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus(const std::string &filename) {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  LOG(INFO) << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB";
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(const std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T, typename TagT>
void search_with_threads(int64_t query_num, int query_dim, const float *query, int recall_at, int mem_L, int L,
                         int beam_width, int search_mode, int NUM_SEARCH_THREADS, int *query_result_tags,
                         float *query_result_dists, ccann::QueryStats *stats, double *latency_stats,
                         ccann::DynamicSSDIndex<T, TagT> &sync_index)  // 假设 sync_index 是你的 Index 对象
{
  std::atomic<int64_t> next_idx{0};

  auto worker = [&]() {
    while (true) {
      int64_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
      if (i >= query_num)
        break;

      // 调用原搜索函数
      sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                        query_result_dists + i * recall_at, stats + i, true);

      // 记录延迟
      latency_stats[i] = stats[i].total_us / 1000.0;  // ms

      if (search_mode == BEAM_SEARCH) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(NUM_SEARCH_THREADS);
  for (int t = 0; t < NUM_SEARCH_THREADS; ++t) {
    threads.emplace_back(worker);
  }

  for (auto &t : threads) {
    t.join();
  }
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, _u32 mem_L, _u64 L,
                        uint32_t beam_width, ccann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  if (NUM_SEARCH_THREADS == 0) {
    return;
  }
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    LOG(INFO) << "current truthfile: " << truthset_file;
    ccann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float *query_result_dists = new float[recall_at * query_num];
  TagT *query_result_tags = new TagT[recall_at * query_num];

  for (_u32 q = 0; q < query_num; q++) {
    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  ccann::QueryStats *stats = new ccann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cerr << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cerr << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    latency_stats[i] = stats[i].total_us / 1000.0;  // convert to ms
    if (search_mode == BEAM_SEARCH) {
      // Here we follow the original paper 's settings...
      // For PipeSearch, do not sleep is faster.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = ccann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  float mean_ios =
      (float) ccann::get_mean_stats(stats, query_num, [](const ccann::QueryStats &stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cerr << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(_u64) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(_u64) (0.999 * ((double) query_num))] << std::setw(12) << recall << std::setw(12)
            << mean_ios << std::endl;

  LOG(INFO) << "search current time: " << current_time;
  disk_io = mean_ios;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void search(const unsigned L_disk, const std::string &index_prefix, const std::string &query_file,
            std::string &truthset_file, const int recall_at, const std::vector<_u64> &Lsearch,
            const unsigned beam_width, const uint32_t search_beam_width, const uint32_t search_mem_L,
            ccann::Distance<T> *dist_cmp) {
  ccann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", NUM_SEARCH_THREADS);
  std::vector<T> data_load;
  size_t dim{};

  ccann::Timer timer;

  LOG(INFO) << "Loading queries";
  T *query = NULL;
  size_t query_num, query_dim;
  ccann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  ccann::Metric metric = ccann::Metric::L2;
  ccann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, (search_mem_L > 0), true);

  LOG(INFO) << "Searching before inserts: ";

  uint64_t res = 0;

  // GetTruthFileName(truthset_file, res + truthset_l_offset);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  std::vector<double> ref_diskio;
  for (size_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width, sync_index,
                       truthset_file, false, true, diskio);
    ref_diskio.push_back(diskio);
  }
}

int main(int argc, char **argv) {
  if (argc < 13) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <L_disk>"
              << " <search_threads> <search_mode>"
              << " <index_prefix> <query_file> <truthset_prefix> <recall@>"
              << " <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  int arg_no = 2;
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);

  NUM_SEARCH_THREADS = (int) std::atoi(argv[arg_no++]);
  LOG(INFO) << "num search threads: " << NUM_SEARCH_THREADS;

  search_mode = std::atoi(argv[arg_no++]);
  LOG(INFO) << "search mode: " << search_mode;

  std::string index_prefix(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);

  // truthsets lie in the same folder, with gt_X.bin as the truthset of [0, x + 100M) vectors.
  std::string truthset(argv[arg_no++]);

  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_mem_L = (unsigned) std::atoi(argv[arg_no++]);  // 0 if no mem_index is used.
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  if (std::string(argv[1]) == std::string("int8")) {
    ccann::DistanceL2Int8 dist_cmp;
    search<int8_t, unsigned>(L_disk, index_prefix, query_file, truthset, recall_at, Lsearch, beam_width,
                             search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    ccann::DistanceL2UInt8 dist_cmp;
    search<uint8_t, unsigned>(L_disk, index_prefix, query_file, truthset, recall_at, Lsearch, beam_width,
                              search_beam_width, search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    ccann::DistanceL2 dist_cmp;
    search<float, unsigned>(L_disk, index_prefix, query_file, truthset, recall_at, Lsearch, beam_width,
                            search_beam_width, search_mem_L, &dist_cmp);
  } else
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
}
