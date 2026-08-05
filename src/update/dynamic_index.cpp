#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <mutex>
#include <vector>

#include <algorithm>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>
#include <sys/mman.h>
#include <libpmem.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "aux_utils.h"
#include "ssd_index.h"
#include "parameters.h"

#include "linux_aligned_file_reader.h"

namespace ccann {
#define BLOCK_SIZE (64 * 1024 * 1024)  // 64 MiB
#define MAX_THREADS 32
  void copy_block(const std::string &src, const std::string &dst, size_t offset, size_t size,
                  std::atomic<size_t> &progress) {
    int fd_src = open(src.c_str(), O_RDONLY);
    int fd_dst = open(dst.c_str(), O_WRONLY | O_CREAT, 0666);
    if (fd_src < 0 || fd_dst < 0)
      return;

    std::vector<char> buffer(16 * 1024 * 1024);  // 1 MiB buffer
    size_t copied = 0;
    while (copied < size) {
      size_t to_read = std::min(buffer.size(), size - copied);
      ssize_t r = pread(fd_src, buffer.data(), to_read, offset + copied);
      if (r <= 0)
        break;
      pwrite(fd_dst, buffer.data(), r, offset + copied);
      copied += r;
      progress += r;
    }

    close(fd_src);
    close(fd_dst);
  }

  void parallel_copy_file(const std::string &src, const std::string &dst) {
    auto time_start = std::chrono::high_resolution_clock::now();
    // 先删除目标文件（如果存在）
    if (std::filesystem::exists(dst)) {
      std::filesystem::remove(dst);
    }

    size_t filesize = std::filesystem::file_size(src);

    if (filesize <= BLOCK_SIZE) {
      std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
      std::cout << src << " copied (small file)\n";
      return;
    }

    std::atomic<size_t> progress(0);
    size_t num_threads = std::min(static_cast<size_t>(MAX_THREADS), (filesize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
      size_t offset = i * filesize / num_threads;
      size_t end = (i + 1) * filesize / num_threads;
      size_t sz = end - offset;
      threads.emplace_back(copy_block, src, dst, offset, sz, std::ref(progress));
    }

    // 打印进度线程
    std::thread progress_thread([&]() {
      while (progress < filesize) {
        {
          double pct = 100.0 * progress / filesize;
          std::cout << "\rCopying " << src << ": " << pct << "% " << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });

    for (auto &t : threads)
      t.join();
    progress_thread.join();

    {
      std::cout << "\rCopying " << src << ": 100% done." << std::endl;
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "Total copy time: " << diff.count() << " seconds.\n";
  }

  void copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    auto copy_if_exists = [&](const std::string &src_file, const std::string &dst_file) {
      if (std::filesystem::exists(src_file)) {
        parallel_copy_file(src_file, dst_file);
      }
    };

    std::cout << "Copying disk index from " << prefix_in << " to " << prefix_out << "\n";

    copy_if_exists(prefix_in + "_disk.index", prefix_out + "_disk.index");

    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy_file(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                                 std::filesystem::copy_options::overwrite_existing);
    } else if (std::filesystem::exists(prefix_out + "_disk.index.tags")) {
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    copy_if_exists(prefix_in + "_pq_pivots.bin", prefix_out + "_pq_pivots.bin");
    copy_if_exists(prefix_in + "_pq_compressed.bin", prefix_out + "_pq_compressed.bin");
    copy_if_exists(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned");
    copy_if_exists(prefix_in + "_disk.index.id2loc", prefix_out + "_disk.index.id2loc");
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::build_id2loc_mapping(std::string &index_file, std::string &id2loc_file) {
    std::ifstream index_metadata(index_file, std::ios::binary);
    _u32 nr, nc;
    _u64 npts;
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);
    READ_U64(index_metadata, npts);
    index_metadata.close();

    std::ofstream id2loc_writer(id2loc_file, std::ios::binary);
    LOG(INFO) << "Build ID to Location mapping with " << npts << " points.";
    auto sector_num = DIV_ROUND_UP(npts * sizeof(uint32_t), SECTOR_LEN);
    std::vector<IORequest> writes;

    uint32_t sector_buf[SECTOR_LEN / sizeof(uint32_t)];

    for (uint64_t s = 0; s < sector_num; ++s) {
      memset((void *) sector_buf, 0, SECTOR_LEN);
      for (uint64_t i = 0; i < SECTOR_LEN / sizeof(uint32_t); ++i) {
        uint64_t global_idx = s * (SECTOR_LEN / sizeof(uint32_t)) + i;
        if (global_idx < npts) {
          sector_buf[i] = global_idx;
        } else {
          break;
        }
      }
      IORequest req;
      req.offset = s * SECTOR_LEN;
      req.len = SECTOR_LEN;
      req.buf = (void *) sector_buf;
      writes.push_back(req);
      for (auto &req : writes) {
        id2loc_writer.write((char *) req.buf, req.len);
      }
      writes.clear();
    }

    id2loc_writer.flush();

    LOG(INFO) << "Built ID to Location mapping.";
    id2loc_writer.close();
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(Parameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            ccann::Metric dist_metric, int search_mode, bool use_mem_index,
                                            bool read_only, int cpu_bound) {
    // check if file exists.
    if (!std::filesystem::exists(disk_prefix_in + "_disk.index")) {
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    if (use_mem_index && !std::filesystem::exists(disk_prefix_in + "_mem.index")) {
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in << "_mem.index";
      exit(-1);
    }

    this->active_del[0] = true;
    this->active_del[1] = false;
    this->_dist_metric = dist_metric;

    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    _paras_disk.Set<unsigned>("beamwidth", parameters.Get<unsigned>("beamwidth"));
    _paras_disk.Set<bool>("saturate_graph", 0);

    _num_threads = parameters.Get<_u32>("num_threads");
    _beamwidth = parameters.Get<uint32_t>("beamwidth");

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    reader.reset(new LinuxAlignedFileReader());
    pq_compressed_writer.reset(new LinuxAlignedFileReader());
    tags_writer.reset(new LinuxAlignedFileReader());
    id2loc_writer.reset(new LinuxAlignedFileReader());

    _disk_index = new ccann::SSDIndex<T, TagT>(this->_dist_metric, reader, pq_compressed_writer, tags_writer,
                                                 id2loc_writer, false, false, &_paras_disk);
#ifdef J_ANN
    _disk_index->journals = new v2::Journal<TagT> *[N_JOURNAL];
    for (int i = 0; i < N_JOURNAL; i++) {
      _disk_index->journals[i] = new v2::Journal<TagT>(disk_prefix_out + "_journal" + std::to_string(i) + ".log");
    }
#endif

    std::string id2loc_file(disk_prefix_in + "_disk.index.id2loc");
    std::string index_file(disk_prefix_in + "_disk.index");
    if (!file_exists(id2loc_file)) {
      build_id2loc_mapping(index_file, id2loc_file);
    }

#ifndef NO_POLLUTE_ORIGINAL
    if (read_only) {
      LOG(WARNING)
          << "Read-only mode is enabled. The original index files will not be modified during dynamic updates.";
    } else {
      std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
      copy_index(_disk_index_prefix_in, disk_index_prefix_shadow);
      LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
      _disk_index_prefix_in = disk_index_prefix_shadow;
    }
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH ||
        search_mode == PARA_SEARCH) {
      this->search_mode = search_mode;
      _disk_index->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, or PIPE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, true, use_page_search, cpu_bound);
    if (res != 0) {
      LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
      exit(-1);
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + "_mem.index";  // use the original one.
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      _disk_index->load_mem_index(this->_dist_metric, _disk_index->data_dim, mem_index_path);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::show_statistics() {
    _disk_index->statistics();
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::reset_statistics() {
    _disk_index->reset_statistics();
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    // put in destructor code
    _disk_index->statistics();

    LOG(INFO) << "Waiting for all async threads to finish...";
    if (_disk_index->insert_pool != nullptr) {
      _disk_index->synchronize_insertions();

#ifdef USE_SMALL_THREAD_POOL
      // _disk_index is not automatically deleted.
      delete _disk_index->insert_pool.get();
#endif
    }

    auto final_task = new typename SSDIndex<T, TagT>::CommitTask{
        .pq_coords = std::vector<uint8_t>(), .target_id = 0, .terminate = true, .point = nullptr};
    _disk_index->commit_tasks.push(final_task);
    _disk_index->commit_tasks.push_notify_all();
    if (_disk_index->commit_thread_->joinable()) {
      LOG(INFO) << "Joining commit thread...";
      _disk_index->commit_thread_->join();
      delete _disk_index->commit_thread_;
    }
    LOG(INFO) << "Done";
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    // journal->append(v2::TxType::kInsert, tag);
    auto *deletion_set = &deletion_sets[active_delete_set];
    int target_id = 0;

#ifdef ASYNC_INSERTION
    target_id = _disk_index->async_insert_in_place(point, tag, deletion_set);
#else
    target_id = _disk_index->insert_in_place(point, tag, deletion_set);
#endif

    return target_id;
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l) {
    std::vector<TagT> result_tags(4096);
    std::vector<float> result_distances(4096);
    auto *deletion_set = &deletion_sets[active_delete_set];
    size_t n = 0;
    if (search_mode == BEAM_SEARCH) {
      n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats, deletion_set, dyn_search_l);
    } else if (search_mode == PAGE_SEARCH) {
      n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PIPE_SEARCH) {
      n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PARA_SEARCH) {
      n = _disk_index->para_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode;
      exit(-1);
    }
    std::vector<NeighborTag<TagT>> best_vec;
    for (size_t i = 0; i < n; i++) {
      best_vec.emplace_back(result_tags[i], result_distances[i]);
    }
    std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
    size_t pos = 0;

    for (auto iter : best_vec) {
      if (deletion_set->find(iter.tag) == deletion_set->end()) {
        tags[pos] = iter.tag;
        distances[pos] = iter.dist;
        pos++;
      }
      if (pos == K) {
        return;
      }
    }
    // LOG(INFO) << "Failed to find enough tags after " << i + 1 << " attempts";
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    // journal->append(v2::TxType::kDelete, tag);

    if (active_del[active_delete_set].load() == false) {
      LOG(ERROR) << "Active deletion set indicated as _deletion_set_" << active_delete_set
                 << " but it cannot accept deletions";
    }

    // if not deleted, then buffer the deletion.
    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      deletion_sets[active_delete_set].insert(tag);
      deleted_tags[active_delete_set].push_back(tag);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::save_del_set() {
    int nxt_idx = 1 - active_delete_set, cur_idx = active_delete_set;
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    deletion_sets[nxt_idx].clear();
    deleted_tags[nxt_idx].clear();
    bool expected_active = false;
#ifndef ODIN_ANN_IMMEDIATE_NO_CC
    if (active_del[nxt_idx].compare_exchange_strong(expected_active, true)) {
      LOG(INFO) << "Cleared deletion set " << nxt_idx << " - ready to accept new points";
    } else {
      LOG(INFO) << "Failed to clear deletion set " << nxt_idx;
    }
#endif
    active_delete_set = nxt_idx;
    active_del[cur_idx].store(false);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    // _disk_index_in -> _disk_index_out
    save_del_set();
    ccann::Timer timer;
    merge(nthreads, n_sampled_nbrs);

    // TODO(gh): do we really need to reload disk index?
    // std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
    // _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
#ifndef ODIN_ANN_IMMEDIATE_NO_CC
    LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
#endif
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
#ifdef ODIN_ANN_IMMEDIATE_NO_CC
    _disk_index->merge(_disk_index_prefix_in, _disk_index_prefix_out);
#else
    _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                               deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
#endif
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}  // namespace ccann
