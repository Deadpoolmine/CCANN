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

int NUM_INSERT_THREADS = 10;
int NUM_SEARCH_THREADS = 32;

int search_mode = BEAM_SEARCH;

int begin_time = 0;
pipeann::Timer globalTimer;

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

template<typename T, typename TagT>
void sync_search_kernel_distribute(T *query, size_t query_num, size_t query_dim, const int recall_at, _u32 mem_L,
                                   _u64 L, uint32_t beam_width,
                                   std::vector<std::unique_ptr<pipeann::DynamicSSDIndex<T, TagT>>> &indices,
                                   std::string &truthset_file, bool merged, bool calRecall, double &disk_io) {
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
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  unsigned num_indices = indices.size();
  float *query_result_dists = new float[num_indices * recall_at * query_num];
  TagT *query_result_tags = new TagT[num_indices * recall_at * query_num];
  TagT *final_query_result_tags = new TagT[recall_at * query_num];

  for (size_t idx = 0; idx < num_indices; ++idx) {
    for (_u32 q = 0; q < query_num; q++) {
      for (_u32 r = 0; r < (_u32) recall_at; r++) {
        query_result_tags[idx * query_num * recall_at + q * recall_at + r] = std::numeric_limits<TagT>::max();
        query_result_dists[idx * query_num * recall_at + q * recall_at + r] = std::numeric_limits<float>::max();
      }
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cerr << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cerr << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();

  omp_set_nested(1);

#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
#pragma omp parallel for num_threads(num_indices) schedule(static)
    for (size_t idx = 0; idx < num_indices; ++idx) {
      pipeann::DynamicSSDIndex<T, TagT> &sync_index = *(indices[idx]);
      auto res_pos = idx * query_num * recall_at + i * recall_at;
      sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + res_pos,
                        query_result_dists + res_pos, stats + i, true);

      if (search_mode == BEAM_SEARCH) {
        // Here we follow the original paper 's settings...
        // For PipeSearch, do not sleep is faster.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }
    latency_stats[i] = stats[i].total_us / 1000.0;  // convert to ms
  }
  auto e = std::chrono::high_resolution_clock::now();

  // merge query results from different indices
  for (_u32 q = 0; q < query_num; q++) {
    // use a min-heap to merge
    auto cmp = [&](const std::pair<TagT, float> &a, const std::pair<TagT, float> &b) { return a.second > b.second; };
    std::priority_queue<std::pair<TagT, float>, std::vector<std::pair<TagT, float>>, decltype(cmp)> min_heap(cmp);

    uint32_t tag_start = 0;
    for (size_t idx = 0; idx < num_indices; ++idx) {
      for (_u32 r = 0; r < (_u32) recall_at; r++) {
        TagT tag = query_result_tags[idx * query_num * recall_at + q * recall_at + r];
        uint32_t _tag = ((uint32_t) tag) + tag_start;
        float dist = query_result_dists[idx * query_num * recall_at + q * recall_at + r];
        if (tag != std::numeric_limits<TagT>::max()) {
          min_heap.push(std::make_pair(_tag, dist));
        }
        
        pipeann::DynamicSSDIndex<T, TagT> &sync_index = *(indices[idx]);
        tag_start += sync_index._disk_index->num_points;
      }
    }

    for (_u32 r = 0; r < (_u32) recall_at; r++) {
      if (!min_heap.empty()) {
        final_query_result_tags[q * recall_at + r] = min_heap.top().first;
        min_heap.pop();
      } else {
        LOG(ERROR) << "Not enough results found when merging from multiple indices.";
        crash();
      }
    }
  }

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

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
void search_distribute(const unsigned L_disk, int vecs_per_step, int num_steps,
                       const std::vector<std::string> &index_prefixs, const std::string &query_file,
                       std::string &truthset_file, size_t truthset_l_offset, const int recall_at,
                       const std::vector<_u64> &Lsearch, const unsigned beam_width, const uint32_t search_beam_width,
                       const uint32_t search_mem_L, pipeann::Distance<T> *dist_cmp) {
  pipeann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", NUM_SEARCH_THREADS + NUM_INSERT_THREADS);
  std::vector<T> data_load;
  size_t dim{};

  pipeann::Timer timer;

  LOG(INFO) << "Loading queries";
  T *query = NULL;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  std::vector<std::unique_ptr<pipeann::DynamicSSDIndex<T, TagT>>> indices;
  indices.reserve(index_prefixs.size());

  for (const auto &index_prefix : index_prefixs) {
    // pipeann::DynamicSSDIndex<T, TagT> index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
    // search_mode,
    //                                         (search_mem_L > 0), false);
    auto index = std::make_unique<pipeann::DynamicSSDIndex<T, TagT>>(
        paras, index_prefix, index_prefix + "_merge", dist_cmp, metric, search_mode, (search_mem_L > 0), false);
    indices.push_back(std::move(index));
  }

  LOG(INFO) << "Searching before inserts: ";

  uint64_t res = 0;

  begin_time = globalTimer.elapsed() / 1.0e6f;

  std::vector<double> ref_diskio;
  for (size_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel_distribute<T, TagT>(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j],
                                           search_beam_width, indices, truthset_file, false, true, diskio);
    ref_diskio.push_back(diskio);
  }
}

int main(int argc, char **argv) {
  if (argc < 18) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <L_disk>"
              << " <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode> <N_index>"
              << " <index_prefix1> ... <index_prefixN> <query_file> <truthset_prefix> <truthset_l_offset> <recall@>"
              << " <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  int arg_no = 2;
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);

  // 1M vectors per step.
  int vecs_per_step = (int) std::atoi(argv[arg_no++]);

  // 100 steps for 100M + 100M test, 200 steps for 800M + 200M test.
  int num_steps = (int) std::atoi(argv[arg_no++]);

  NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);
  NUM_SEARCH_THREADS = (int) std::atoi(argv[arg_no++]);
  LOG(INFO) << "num insert threads: " << NUM_INSERT_THREADS;
  LOG(INFO) << "num search threads: " << NUM_SEARCH_THREADS;

  search_mode = std::atoi(argv[arg_no++]);
  LOG(INFO) << "search mode: " << search_mode;

  int N_index = (int) std::atoi(argv[arg_no++]);
  std::vector<std::string> index_prefixes;
  for (int i = 0; i < N_index; ++i) {
    index_prefixes.push_back(std::string(argv[arg_no++]));
  }

  std::string query_file(argv[arg_no++]);

  // truthsets lie in the same folder, with gt_X.bin as the truthset of [0, x + 100M) vectors.
  std::string truthset(argv[arg_no++]);

  // We generate truth files every 1M vectors, gt_X.bin is the truthset of [0, x + 100M) vectors.
  // This should be 0 for 100M + 100M test, and 700M for 800M + 200M test.
  size_t truthset_l_offset = (size_t) std::atoll(argv[arg_no++]);
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
    pipeann::DistanceL2Int8 dist_cmp;
    search_distribute<int8_t, unsigned>(L_disk, vecs_per_step, num_steps, index_prefixes, query_file, truthset,
                                        truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width,
                                        search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    search_distribute<uint8_t, unsigned>(L_disk, vecs_per_step, num_steps, index_prefixes, query_file, truthset,
                                         truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width,
                                         search_mem_L, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2 dist_cmp;
    search_distribute<float, unsigned>(L_disk, vecs_per_step, num_steps, index_prefixes, query_file, truthset,
                                       truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width,
                                       search_mem_L, &dist_cmp);
  } else
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
}
