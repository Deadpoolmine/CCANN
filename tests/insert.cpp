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

int search_mode = PARA_SEARCH;

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

template<typename T, typename TagT>
float insertion_kernel(T *data_load, ccann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                       size_t dim) {
  ccann::Timer timer;
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  LOG(INFO) << "Begin Insert";
  std::atomic_size_t success(0);

#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (_s64 i = 0; i < (_s64) insert_vec.size(); i++) {
    ccann::Timer insert_timer;
    sync_index.insert(data_load + dim * i, insert_vec[i]);
    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(insert_latencies.begin(), insert_latencies.end());
  LOG(INFO) << "Inserted " << insert_vec.size() << " points in " << time_secs << "s";
  LOG(INFO) << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us";
  LOG(INFO) << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us";
  LOG(INFO) << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us";
  LOG(INFO) << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us";
  LOG(INFO) << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us";
  return time_secs;
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load, size_t &dim) {
  LOG(INFO) << "l_start: " << l_start << " r_start: " << r_start << " n: " << n;

  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }

  for (uint64_t i = r_start; i < r_start + n; ++i) {
    insert_tags.push_back(i);
  }

  // load data, load n vecs from r_start.
  int npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);
  reader.seekg(0, reader.beg);
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));

  dim = dim_i32;

  size_t data_dim = dim_i32;
  data_load.resize(n * data_dim);
  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char *) data_load.data(), sizeof(T) * n * data_dim);
}

template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
            const std::string &index_prefix, const unsigned beam_width, ccann::Distance<T> *dist_cmp) {
  ccann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", NUM_SEARCH_THREADS + NUM_INSERT_THREADS);
  std::vector<T> data_load;
  size_t dim{};

  ccann::Timer timer;

  ccann::Metric metric = ccann::Metric::L2;
  ccann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, true);

  uint64_t res = 0;

  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);
  float total_time_secs = 0;
  int inMemorySize = 0;
  uint64_t index_npts = sync_index._disk_index->num_points;
  for (int i = 0; i < num_steps; i++) {
    LOG(INFO) << "Batch: " << i << " Total Batch : " << num_steps;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    uint64_t ed = st + index_npts;
    LOG(INFO) << "st: " << st << " ed: " << ed;
    get_trace<T, TagT>(data_bin, st, ed, vecs_per_step, delete_vec, insert_vec, data_load, dim);

    // std::future<void> insert_future = std::async(std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
    //                                              std::ref(sync_index), std::ref(insert_vec), dim);

    // int total_queries = 0;
    // std::future_status insert_status;
    // do {
    //   insert_status = insert_future.wait_for(std::chrono::seconds(5));
    //   if (insert_status == std::future_status::deferred) {
    //     LOG(INFO) << "deferred\n";
    //   }
    //   if (insert_status == std::future_status::ready) {
    //     LOG(INFO) << "Insertions complete!\n";
    //   }
    // } while (insert_status != std::future_status::ready);
    total_time_secs += insertion_kernel<T, TagT>(data_load.data(), sync_index, insert_vec, dim);

    inMemorySize += insert_vec.size();
  }
  LOG(INFO) << "Total insertion time for " << (vecs_per_step * num_steps) << " points: " << total_time_secs << "s";
  LOG(INFO) << "Average insertion throughput: " << ((float) (vecs_per_step * num_steps) / total_time_secs)
            << " points/s";
}

int main(int argc, char **argv) {
  if (argc < 9) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk>"
              << " <vecs_per_step> <num_steps> <insert_threads> "
              << " <index_prefix> "
              << " <#beam_width> <search_mode>";
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);

  // 1M vectors per step.
  int vecs_per_step = (int) std::atoi(argv[arg_no++]);

  // 100 steps for 100M + 100M test, 200 steps for 800M + 200M test.
  int num_steps = (int) std::atoi(argv[arg_no++]);

  NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);
  LOG(INFO) << "num insert threads: " << NUM_INSERT_THREADS;

  std::string index_prefix(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);

  if (arg_no < argc) {
    search_mode = std::atoi(argv[arg_no++]);
    LOG(INFO) << "search mode: " << search_mode;
  } else {
    search_mode = PARA_SEARCH;
    LOG(INFO) << "search mode not set, use default PARA_SEARCH: " << search_mode;
  }

  if (std::string(argv[1]) == std::string("int8")) {
    ccann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, beam_width, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    ccann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, beam_width, &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    ccann::DistanceL2 dist_cmp;
    update<float, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, beam_width, &dist_cmp);
  } else
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
}
