/**
 * CCANN Quick Start Test — PM-based Insert & Search
 * Self-contained -- no external dataset required
 *
 * Pipeline (following insert.cpp PM pattern):
 *   1. Generate random data -> build disk index
 *   2. Insert vectors via DynamicSSDIndex
 *   3. Search and evaluate Recall
 *
 * Build: cmake
 * Run: ./build/tests/quick_start
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "aux_utils.h"
#include "ssd_index.h"
#include "v2/dynamic_index.h"
#include "parameters.h"
#include "utils.h"
#include "neighbor.h"
#include "partition_and_pq.h"

// ============================================================
// Helper: generate random vectors as .bin
// ============================================================
template<typename T>
void generate_random_bin(const std::string &filepath, size_t npts, size_t dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<T> data(npts * dim);
  for (size_t i = 0; i < npts * dim; ++i)
    data[i] = static_cast<T>(dist(gen));

  // L2 normalize
  for (size_t i = 0; i < npts; ++i) {
    float norm = 0.0f;
    for (size_t j = 0; j < dim; ++j)
      norm += static_cast<float>(data[i * dim + j]) * static_cast<float>(data[i * dim + j]);
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
      for (size_t j = 0; j < dim; ++j)
        data[i * dim + j] = static_cast<T>(static_cast<float>(data[i * dim + j]) / norm);
    }
  }

  ccann::save_bin<T>(filepath, data.data(), npts, dim);
  std::cout << "[gen] random data -> " << filepath << " (" << npts << " pts, " << dim << " dims)" << std::endl;
}

// ============================================================
// Helper: brute-force ground truth
// ============================================================
template<typename T>
void brute_force(const std::vector<T> &data, const std::vector<T> &query,
                 size_t npts, size_t dim, size_t K,
                 std::vector<unsigned> &gt_ids, std::vector<float> &gt_dists) {
  gt_ids.resize(K);
  gt_dists.resize(K);
  std::vector<std::pair<float, unsigned>> dists(npts);
  for (size_t i = 0; i < npts; ++i) {
    float d = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      float diff = static_cast<float>(query[j]) - static_cast<float>(data[i * dim + j]);
      d += diff * diff;
    }
    dists[i] = {d, static_cast<unsigned>(i)};
  }
  std::partial_sort(dists.begin(), dists.begin() + K, dists.end());
  for (size_t k = 0; k < K; ++k) {
    gt_ids[k] = dists[k].second;
    gt_dists[k] = dists[k].first;
  }
}

// ============================================================
// Core: PM-based quick start (following insert.cpp)
// ============================================================
template<typename T, typename TagT = uint32_t>
int run_pm_quick_start(const std::string &dtype_name, ccann::Distance<T> *dist_cmp,
                       size_t base_npts, size_t dim, size_t insert_npts,
                       unsigned R, unsigned L_build, unsigned L_disk,
                       unsigned beam_width, unsigned num_threads,
                       unsigned K, unsigned L_search) {
  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════╗\n"
            << "║         CCANN PM Quick Start — Insert & Search           ║\n"
            << "╚══════════════════════════════════════════════════════════╝\n"
            << std::endl;

  std::cout << "=== Configuration ===" << std::endl;
  auto label = [](const std::string &s) { std::cout << "  " << std::left << std::setw(16) << s; };
  label("Data type:");     std::cout << dtype_name << std::endl;
  label("Dimensions:");    std::cout << dim << std::endl;
  label("Base points:");   std::cout << base_npts << std::endl;
  label("Insert points:"); std::cout << insert_npts << std::endl;
  label("R (max degree):");std::cout << R << std::endl;
  label("L_build:");       std::cout << L_build << std::endl;
  label("L_disk:");        std::cout << L_disk << std::endl;
  label("beam_width:");    std::cout << beam_width << std::endl;
  label("Threads:");       std::cout << num_threads << std::endl;
  label("K (top-K):");     std::cout << K << std::endl;
  label("L_search:");      std::cout << L_search << std::endl;
  std::cout << std::endl;

  // ============================================================
  // Step 1: Generate data & build disk index (PM-aware)
  //   Requires cc-ann compile flags (-DBATCH_PRUNING -DEARLY_EXIT -DASYNC_INSERTION -DFINE_GRAINED_CONCURRENCY)
  //   Build: memory index -> PQ compress -> create_disk_layout
  // ============================================================
  std::cout << "=== Step 1: Generate base data & build disk index ===" << std::endl;

  std::string data_file = "/tmp/ccann_pm_qs_data.bin";
  std::string index_prefix = "/mnt/pmem0/ccann_pm_qs";
  std::string pq_pivots_path = index_prefix + "_pq_pivots.bin";
  std::string pq_compressed_path = index_prefix + "_pq_compressed.bin";
  std::string mem_index_path = index_prefix + "_mem.index";
  std::string disk_index_path = index_prefix + "_disk.index";

  // Clean up old files
  std::remove(data_file.c_str());
  std::remove(disk_index_path.c_str());
  std::remove(pq_pivots_path.c_str());
  std::remove(pq_compressed_path.c_str());
  std::remove(mem_index_path.c_str());
  std::remove((mem_index_path + ".data").c_str());

  generate_random_bin<T>(data_file, base_npts, dim);

  auto t0 = std::chrono::high_resolution_clock::now();

  // 1a. PQ training & pivot generation
  double p_val = std::min(0.1, 256000.0 / base_npts);
  float *train_data = nullptr;
  size_t train_size, train_dim;
  gen_random_slice<T>(data_file, p_val, train_data, train_size, train_dim);

  size_t num_pq_chunks = ccann::calculate_num_pq_chunks(base_npts * dim * sizeof(T) * 2.5, base_npts, (uint32_t) dim);

  generate_pq_pivots(train_data, train_size, (uint32_t) train_dim, 256,
                     (uint32_t) num_pq_chunks, 15, pq_pivots_path);
  delete[] train_data;

  // 1b. PQ compression
  generate_pq_data_from_pivots<T>(data_file, 256, (uint32_t) num_pq_chunks,
                                  pq_pivots_path, pq_compressed_path);

  // 1c. Build Vamana memory index
  {
    ccann::Parameters mem_params;
    mem_params.Set<unsigned>("L", L_build);
    mem_params.Set<unsigned>("R", R);
    mem_params.Set<unsigned>("C", 750);
    mem_params.Set<float>("alpha", 1.2f);
    mem_params.Set<bool>("saturate_graph", false);
    mem_params.Set<unsigned>("num_threads", num_threads);

    ccann::Index<T, TagT> mem_index(ccann::Metric::L2, dim, base_npts,
                                    false, false, false);
    mem_index.build(data_file.c_str(), base_npts, mem_params);
    mem_index.save(mem_index_path.c_str());
  }

  // 1d. Create disk layout (same as build_disk_index internal)
  ccann::create_disk_layout<T, TagT>(mem_index_path, data_file, "",
                                     pq_pivots_path, pq_compressed_path,
                                     false, disk_index_path);

  std::remove(mem_index_path.c_str());
  std::remove((mem_index_path + ".data").c_str());

  auto t1 = std::chrono::high_resolution_clock::now();
  double build_time = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  Disk index built! Time: " << build_time << "s" << std::endl;

  // ============================================================
  // Step 2: Init DynamicSSDIndex
  // ============================================================
  std::cout << "\n=== Step 2: Init DynamicSSDIndex ===" << std::endl;

  ccann::Parameters paras;
  paras.Set<unsigned>("L_disk", L_disk);
  paras.Set<unsigned>("R_disk", 0);
  paras.Set<float>("alpha_disk", 1.2f);
  paras.Set<unsigned>("C", 384);
  paras.Set<unsigned>("beamwidth", beam_width);
  paras.Set<unsigned>("nodes_to_cache", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  ccann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge",
                                               dist_cmp, ccann::Metric::L2, PARA_SEARCH, false);

  uint64_t disk_npts = sync_index._disk_index->num_points;
  std::cout << "  Disk index loaded, " << disk_npts << " points" << std::endl;

  // ============================================================
  // Step 3: Generate insert data & PM insert
  // ============================================================
  std::cout << "\n=== Step 3: Generate insert data & PM insert ===" << std::endl;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist_gen(0.0f, 1.0f);

  std::vector<T> insert_data(insert_npts * dim);
  for (size_t i = 0; i < insert_npts * dim; ++i)
    insert_data[i] = static_cast<T>(dist_gen(gen));
  for (size_t i = 0; i < insert_npts; ++i) {
    float norm = 0.0f;
    for (size_t j = 0; j < dim; ++j)
      norm += static_cast<float>(insert_data[i * dim + j]) * static_cast<float>(insert_data[i * dim + j]);
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
      for (size_t j = 0; j < dim; ++j)
        insert_data[i * dim + j] = static_cast<T>(static_cast<float>(insert_data[i * dim + j]) / norm);
    }
  }

  auto t_ins0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < insert_npts; ++i) {
    TagT tag = static_cast<TagT>(disk_npts + i);
    sync_index.insert(insert_data.data() + dim * i, tag);
    if ((i + 1) % std::max((size_t)1, insert_npts / 5) == 0) {
      std::cout << "  Inserted " << (i + 1) << "/" << insert_npts << " points" << std::endl;
    }
  }
  auto t_ins1 = std::chrono::high_resolution_clock::now();
  double insert_time = std::chrono::duration<double>(t_ins1 - t_ins0).count();
  std::cout << "  Insert done! Time: " << insert_time << "s, throughput: "
            << (insert_npts / insert_time) << " ops/s" << std::endl;

  // ============================================================
  // Step 4: Search test
  // ============================================================
  std::cout << "\n=== Step 4: Search test ===" << std::endl;

  size_t total_npts = base_npts + insert_npts;
  // Load full dataset for brute-force GT
  std::vector<T> all_data(total_npts * dim);
  {
    std::ifstream reader(data_file, std::ios::binary);
    int n_i32, d_i32;
    reader.read((char *) &n_i32, sizeof(int));
    reader.read((char *) &d_i32, sizeof(int));
    reader.read((char *) all_data.data(), base_npts * dim * sizeof(T));
  }
  std::memcpy(all_data.data() + base_npts * dim, insert_data.data(), insert_npts * dim * sizeof(T));

  size_t num_queries = 10;
  std::vector<std::vector<T>> queries(num_queries, std::vector<T>(dim));
  for (size_t q = 0; q < num_queries; ++q) {
    float norm = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
      queries[q][j] = static_cast<T>(dist_gen(gen));
      norm += static_cast<float>(queries[q][j]) * static_cast<float>(queries[q][j]);
    }
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
      for (size_t j = 0; j < dim; ++j)
        queries[q][j] = static_cast<T>(static_cast<float>(queries[q][j]) / norm);
    }
  }

  double total_search_time = 0.0;
  size_t total_hits = 0;
  ccann::QueryStats query_stats;
  std::vector<TagT> result_tags(K);
  std::vector<float> result_dists(K);

  for (size_t q = 0; q < num_queries; ++q) {
    auto ts0 = std::chrono::high_resolution_clock::now();
    sync_index.search(queries[q].data(), K, 0 /* pure disk */, L_search, beam_width,
                      result_tags.data(), result_dists.data(), &query_stats, true);
    auto ts1 = std::chrono::high_resolution_clock::now();
    total_search_time += std::chrono::duration<double>(ts1 - ts0).count();

    std::vector<unsigned> gt_ids;
    std::vector<float> gt_dists;
    brute_force<T>(all_data, queries[q], total_npts, dim, K, gt_ids, gt_dists);

    size_t hits = 0;
    for (size_t k = 0; k < K; ++k)
      for (size_t g = 0; g < K; ++g)
        if (result_tags[k] == static_cast<TagT>(gt_ids[g])) { hits++; break; }
    total_hits += hits;
  }

  double avg_latency_ms = (total_search_time / num_queries) * 1000.0;
  double recall = (double) total_hits / (num_queries * K) * 100.0;

  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════╗\n"
            << "║                    Search Results                        ║\n"
            << "╚══════════════════════════════════════════════════════════╝\n"
            << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  label("Queries:");       std::cout << num_queries << std::endl;
  label("K (top-K):");     std::cout << K << std::endl;
  label(std::string("Recall@") + std::to_string(K) + ":"); std::cout << recall << "%" << std::endl;
  label("Avg latency:");   std::cout << avg_latency_ms << " ms" << std::endl;

  // Detailed results
  std::cout << "\n  --- Top-" << K << " results ---" << std::endl;
  for (size_t q = 0; q < std::min(num_queries, (size_t) 3); ++q) {
    sync_index.search(queries[q].data(), K, 0, L_search, beam_width,
                      result_tags.data(), result_dists.data(), &query_stats, true);
    std::cout << "\n  Query " << q << ":" << std::endl;
    for (size_t k = 0; k < K; ++k)
      std::cout << "    Top-" << (k + 1) << ": tag=" << result_tags[k]
                << " dist=" << result_dists[k] << std::endl;
  }

  // ============================================================
  // Step 5: Cleanup
  // ============================================================
  std::cout << "\n=== Step 5: Cleanup ===" << std::endl;
  std::remove(data_file.c_str());
  std::remove((index_prefix + "_disk.index").c_str());
  std::remove((index_prefix + "_pq_pivots.bin").c_str());
  std::remove((index_prefix + "_pq_compressed.bin").c_str());
  std::remove((index_prefix + "_disk.index_medoids.bin").c_str());
  std::remove((index_prefix + "_disk.index_centroids.bin").c_str());
  // Shadow files from DynamicSSDIndex
  std::remove((index_prefix + "_shadow_disk.index").c_str());
  std::remove((index_prefix + "_shadow_pq_compressed.bin").c_str());
  std::cout << "  Temp files cleaned" << std::endl;

  // Summary
  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════╗\n"
            << "║                PM Quick Start Complete                   ║\n"
            << "╠══════════════════════════════════════════════════════════╣\n";

  // Use consistent label width + value width + unit so all rows align
  auto print_summary_row = [](const std::string &label, const std::string &value, const std::string &unit) {
    std::cout << "║  " << std::left << std::setw(14) << label
              << std::right << std::setw(10) << value
              << "  " << std::left << std::setw(3) << unit
              << "                           ║\n";
  };

  print_summary_row("Base idx:",   std::to_string(base_npts),                 "pts");
  print_summary_row("Inserted:",   std::to_string(insert_npts),               "pts");
  {
    std::ostringstream t; t << std::fixed << std::setprecision(3) << build_time;
    print_summary_row("Build:",     t.str(),                                  "s");
  }
  {
    std::ostringstream t; t << std::fixed << std::setprecision(3) << insert_time;
    print_summary_row("Insert:",    t.str(),                                  "s");
  }
  {
    std::ostringstream t; t << std::fixed << std::setprecision(2) << recall;
    print_summary_row("Recall@" + std::to_string(K) + ":", t.str(),           "%");
  }
  std::cout << "╚══════════════════════════════════════════════════════════╝\n"
            << std::endl;

  return 0;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char **argv) {
  std::string dtype = "float";
  size_t base_npts = 500;
  size_t dim = 128;
  size_t insert_npts = 500;
  unsigned R = 32;
  unsigned L_build = 64;
  unsigned L_disk = 64;
  unsigned beam_width = 2;
  unsigned num_threads = std::min(std::thread::hardware_concurrency(), 16u);
  if (num_threads == 0) num_threads = 4;
  unsigned K = 10;
  unsigned L_search = 80;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--dtype" && i + 1 < argc) dtype = argv[++i];
    else if (arg == "--base" && i + 1 < argc) base_npts = std::stoul(argv[++i]);
    else if (arg == "--dim" && i + 1 < argc) dim = std::stoul(argv[++i]);
    else if (arg == "--insert" && i + 1 < argc) insert_npts = std::stoul(argv[++i]);
    else if (arg == "--R" && i + 1 < argc) R = std::stoul(argv[++i]);
    else if (arg == "--L" && i + 1 < argc) L_build = std::stoul(argv[++i]);
    else if (arg == "--Ld" && i + 1 < argc) L_disk = std::stoul(argv[++i]);
    else if (arg == "--bw" && i + 1 < argc) beam_width = std::stoul(argv[++i]);
    else if (arg == "--threads" && i + 1 < argc) num_threads = std::stoul(argv[++i]);
    else if (arg == "--K" && i + 1 < argc) K = std::stoul(argv[++i]);
    else if (arg == "--Ls" && i + 1 < argc) L_search = std::stoul(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      std::cout << "CCANN PM Quick Start -- PM-based Insert & Search\n"
                << "Usage: ./quick_start [options]\n\n"
                << "Options:\n"
                << "  --dtype TYPE    float (default), int8, uint8\n"
                << "  --base N        Base points (default: 500)\n"
                << "  --dim D         Vector dimensions (default: 128)\n"
                << "  --insert N      Insert points (default: 500)\n"
                << "  --R R           Max degree (default: 32)\n"
                << "  --L L           Build L (default: 64)\n"
                << "  --Ld Ld         Disk L (default: 64)\n"
                << "  --bw BW         Beam width (default: 2)\n"
                << "  --threads T     Threads (default: min(cores,16))\n"
                << "  --K K           Top-K (default: 10)\n"
                << "  --Ls LS         Search L (default: 80)\n"
                << "  --help, -h      Show help\n"
                << std::endl;
      return 0;
    }
  }

  std::cout << "\n"
            << "╔══════════════════════════════════════════════════════════╗\n"
            << "║    CCANN PM Quick Start — PM Insert & Search Test        ║\n"
            << "║          Self-contained — no external dataset             ║\n"
            << "╚══════════════════════════════════════════════════════════╝\n"
            << std::endl;

  if (dtype == "float") {
    ccann::DistanceL2 dist_cmp;
    return run_pm_quick_start<float>("float", &dist_cmp, base_npts, dim, insert_npts,
                                     R, L_build, L_disk, beam_width, num_threads, K, L_search);
  } else if (dtype == "int8") {
    ccann::DistanceL2Int8 dist_cmp;
    return run_pm_quick_start<int8_t>("int8", &dist_cmp, base_npts, dim, insert_npts,
                                      R, L_build, L_disk, beam_width, num_threads, K, L_search);
  } else if (dtype == "uint8") {
    ccann::DistanceL2UInt8 dist_cmp;
    return run_pm_quick_start<uint8_t>("uint8", &dist_cmp, base_npts, dim, insert_npts,
                                       R, L_build, L_disk, beam_width, num_threads, K, L_search);
  } else {
    std::cerr << "Error: unsupported data type '" << dtype << "'. Supported: float, int8, uint8" << std::endl;
    return 1;
  }
}

