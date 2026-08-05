#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <filesystem>

#include <omp.h>
#include <cmath>
#include "parameters.h"
#include "query_buf.h"
#include "timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "tsl/robin_set.h"

namespace ccann {
#define PRE_MAP_PMEM_SIZE (1 * 1024L * 1024L * 1024L)

  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(ccann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                              std::shared_ptr<AlignedFileReader> &pqCompressedWriter,
                              std::shared_ptr<AlignedFileReader> &tagsWriter,
                              std::shared_ptr<AlignedFileReader> &id2locWriter, bool single_file_index, bool tags,
                              Parameters *params)
      : reader(fileReader), tags_writer(tagsWriter), pq_compressed_writer(pqCompressedWriter),
        id2loc_writer(id2locWriter), data_is_normalized(false), enable_tags(tags) {
    if (m == ccann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = ccann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(ccann::get_distance_function<T>(m));

    // this->pq_reader = new LinuxAlignedFileReader();
    if (params != nullptr) {
      this->beam_width = params->Get<uint32_t>("beamwidth");
      this->l_index = params->Get<uint32_t>("L");
      this->range = params->Get<uint32_t>("R");
      this->maxc = params->Get<uint32_t>("C");
      this->alpha = params->Get<float>("alpha");
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::statistics() {
    uint64_t insert_phase_time = 0;
    uint64_t update_graph_time = 0;
    uint64_t update_neighbor_time = 0;
    uint64_t prune_neighbor_time = 0;
    uint64_t update_metadata_time = 0;
    uint64_t read_nodes_time = 0;
    uint64_t search_phase_time = 0;
    uint64_t update_PQ_vec_time = 0;
    uint64_t prune_search_neighbors_time = 0;
    uint64_t search_graph_time = 0;
    uint64_t populate_pq_dists_time = 0;
    uint64_t calc_best_node_time = 0;
    uint64_t send_best_node_time = 0;
    uint64_t do_read_best_node_time = 0;
    uint64_t poll_all_time = 0;
    uint64_t calc_exact_dist_time = 0;
    uint64_t expand_neighbors_time = 0;
    uint64_t merge_time = 0;
    uint64_t journal_time = 0;
    // IO Stat
    uint64_t calc_best_node_number = 0;
    uint64_t send_best_node_number = 0;
    uint64_t poll_hit_number = 0;
    uint64_t poll_number = 0;
    uint64_t ncalc_for_expanding_neighbors = 0;
    uint64_t ncalc_for_useless_neighbors = 0;
    uint64_t expand_loop_number = 0;
    // IO stat for para search
    uint64_t para_busy_wait_number = 0;
    uint64_t para_wait_for_flight_number = 0;
    uint64_t para_useless_wait_for_flight_number = 0;
    uint64_t para_direct_hit_number = 0;
    uint64_t para_early_exit_number = 0;
    uint64_t para_likely_early_exit_number = 0;

    for (size_t i = 0; i < io_timing_stats.size(); i++) {
      insert_phase_time += io_timing_stats[i]->insert_phase_time.load();
      update_graph_time += io_timing_stats[i]->update_graph_time.load();
      update_neighbor_time += io_timing_stats[i]->update_neighbor_time.load();
      prune_neighbor_time += io_timing_stats[i]->prune_neighbor_time.load();
      update_metadata_time += io_timing_stats[i]->update_metadata_time.load();
      read_nodes_time += io_timing_stats[i]->read_nodes_time.load();
      search_phase_time += io_timing_stats[i]->search_phase_time.load();
      update_PQ_vec_time += io_timing_stats[i]->update_PQ_vec_time.load();
      prune_search_neighbors_time += io_timing_stats[i]->prune_search_neighbors_time.load();
      search_graph_time += io_timing_stats[i]->search_graph_time.load();
      populate_pq_dists_time += io_timing_stats[i]->populate_pq_dists_time.load();
      calc_best_node_time += io_timing_stats[i]->calc_best_node_time.load();
      send_best_node_time += io_timing_stats[i]->send_best_node_time.load();
      do_read_best_node_time += io_timing_stats[i]->do_read_best_node_time.load();
      poll_all_time += io_timing_stats[i]->poll_all_time.load();
      calc_exact_dist_time += io_timing_stats[i]->calc_exact_dist_time.load();
      expand_neighbors_time += io_timing_stats[i]->expand_neighbors_time.load();
      merge_time += io_timing_stats[i]->merge_time.load();
      journal_time += io_timing_stats[i]->journal_time.load();

      calc_best_node_number += io_timing_stats[i]->calc_best_node_number.load();
      send_best_node_number += io_timing_stats[i]->send_best_node_number.load();
      poll_hit_number += io_timing_stats[i]->poll_hit_number.load();
      poll_number += io_timing_stats[i]->poll_number.load();
      ncalc_for_expanding_neighbors += io_timing_stats[i]->ncalc_for_expanding_neighbors.load();
      ncalc_for_useless_neighbors += io_timing_stats[i]->ncalc_for_useless_neighbors.load();
      expand_loop_number += io_timing_stats[i]->expand_loop_number.load();

      para_busy_wait_number += io_timing_stats[i]->para_busy_wait_number.load();
      para_wait_for_flight_number += io_timing_stats[i]->para_wait_for_flight_number.load();
      para_useless_wait_for_flight_number += io_timing_stats[i]->para_useless_wait_for_flight_number.load();
      para_direct_hit_number += io_timing_stats[i]->para_direct_hit_number.load();
      para_early_exit_number += io_timing_stats[i]->para_early_exit_number.load();
      para_likely_early_exit_number += io_timing_stats[i]->para_likely_early_exit_number.load();
    }

    LOG(INFO) << std::left << std::setw(35) << "=== SSDIndex Statistics ===";
    LOG(INFO) << std::left << std::setw(35) << "Max CPUs used: " << this->peak_cpus;
    LOG(INFO) << std::left << std::setw(35) << "|- Search phase time (ns): " << search_phase_time;
    LOG(INFO) << std::left << std::setw(35) << "  |- Update PQ table time (ns): " << update_PQ_vec_time;
    LOG(INFO) << std::left << std::setw(35) << "  |- Search graph time (ns): " << search_graph_time;
    LOG(INFO) << std::left << std::setw(35) << "    |- Populate PQ time (ns): " << populate_pq_dists_time;
    LOG(INFO) << std::left << std::setw(35) << "    |- Poll all time (ns): " << poll_all_time << " (" << poll_hit_number
              << " hits / " << (poll_number - poll_hit_number) << " misses, " << para_busy_wait_number
              << " busy waits, " << para_wait_for_flight_number << " waits for flight, but with waits "
              << para_useless_wait_for_flight_number << " useless)";
    LOG(INFO) << std::left << std::setw(35) << "    |- Send best node time (ns): " << send_best_node_time << " ("
              << send_best_node_number << " times)";
    LOG(INFO) << std::left << std::setw(35) << "      |- Read best node time (ns): " << do_read_best_node_time;
    LOG(INFO) << std::left << std::setw(35) << "    |- Calc best node time (ns): " << calc_best_node_time << " ("
              << calc_best_node_number << " times, early exit " << para_early_exit_number
              << " times, likely early exit (no sending neighbors) " << para_likely_early_exit_number << " times)";
    LOG(INFO) << std::left << std::setw(35) << "      |- Calc dist time (ns): " << calc_exact_dist_time;
    LOG(INFO) << std::left << std::setw(35) << "      |- Expand nbr time (ns): " << expand_neighbors_time << " ("
              << ncalc_for_expanding_neighbors << " calcs, " << ncalc_for_useless_neighbors << " useless calcs)";
    LOG(INFO) << std::left << std::setw(35) << "  |- Prune search time (ns): " << prune_search_neighbors_time;

    LOG(INFO) << std::left << std::setw(35) << "|- Insert phase time (ns): " << insert_phase_time;
    LOG(INFO) << std::left << std::setw(35) << "  |- Read nodes time (ns): " << read_nodes_time;
    LOG(INFO) << std::left << std::setw(35) << "  |- Update graph time (ns): " << update_graph_time;
    LOG(INFO) << std::left << std::setw(35) << "    |- Update neighbor time (ns): " << update_neighbor_time;
    LOG(INFO) << std::left << std::setw(35) << "      |- Prune neighbor time (ns): " << prune_neighbor_time;
    LOG(INFO) << std::left << std::setw(35) << "  |- Update metadata time (ns): " << update_metadata_time;
    LOG(INFO) << std::left << std::setw(35) << "|- Merge time (ns): " << merge_time;
    LOG(INFO) << std::left << std::setw(35) << "|- Journal time (ns): " << journal_time;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reset_statistics() {
    for (size_t i = 0; i < io_timing_stats.size(); i++) {
      io_timing_stats[i]->search_phase_time = 0;
      io_timing_stats[i]->update_PQ_vec_time = 0;
      io_timing_stats[i]->search_graph_time = 0;
      io_timing_stats[i]->populate_pq_dists_time = 0;
      io_timing_stats[i]->poll_all_time = 0;
      io_timing_stats[i]->poll_number = 0;
      io_timing_stats[i]->poll_hit_number = 0;
      io_timing_stats[i]->para_busy_wait_number = 0;
      io_timing_stats[i]->para_wait_for_flight_number = 0;
      io_timing_stats[i]->para_useless_wait_for_flight_number = 0;
      io_timing_stats[i]->send_best_node_time = 0;
      io_timing_stats[i]->send_best_node_number = 0;
      io_timing_stats[i]->do_read_best_node_time = 0;
      io_timing_stats[i]->calc_best_node_time = 0;
      io_timing_stats[i]->calc_best_node_number = 0;
      io_timing_stats[i]->para_early_exit_number = 0;
      io_timing_stats[i]->para_likely_early_exit_number = 0;
      io_timing_stats[i]->calc_exact_dist_time = 0;
      io_timing_stats[i]->expand_neighbors_time = 0;
      io_timing_stats[i]->ncalc_for_expanding_neighbors = 0;
      io_timing_stats[i]->ncalc_for_useless_neighbors = 0;
      io_timing_stats[i]->prune_search_neighbors_time = 0;
      io_timing_stats[i]->insert_phase_time = 0;
      io_timing_stats[i]->read_nodes_time = 0;
      io_timing_stats[i]->update_graph_time = 0;
      io_timing_stats[i]->update_neighbor_time = 0;
      io_timing_stats[i]->prune_neighbor_time = 0;
      io_timing_stats[i]->update_metadata_time = 0;
      io_timing_stats[i]->journal_time = 0;
    }
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
#ifndef ANN_LARGE
      pq_compressed_writer->close();
#endif
      id2loc_writer->close();
      if (enable_tags) {
        tags_writer->close();
      }
    }

    if (medoids != nullptr) {
      delete[] medoids;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(_u64 n_threads) {
    _u64 n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
    }

    for (uint64_t i = 0; i < n_buffers; ++i) {
      uint8_t *thread_pq_buf;
      ccann::alloc_aligned((void **) &thread_pq_buf, 16ul << 20, 256);
      thread_pq_bufs.push_back(thread_pq_buf);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }

    LOG(INFO) << "Setup commit thread for insert...";
    commit_thread_ = new std::thread(&SSDIndex<T, TagT>::insert_commit_thread, this);
#endif
    load_flag = true;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_thread_data() {
    // TODO(gh): destruct thread_queue and other readers.
    for (auto &buf : this->thread_data_bufs) {
      ccann::aligned_free((void *) buf->coord_scratch);
      ccann::aligned_free((void *) buf->sector_scratch);
      ccann::aligned_free((void *) buf->aligned_pq_coord_scratch);
      ccann::aligned_free((void *) buf->aligned_pqtable_dist_scratch);
      ccann::aligned_free((void *) buf->aligned_dist_scratch);
      ccann::aligned_free((void *) buf->aligned_query_T);
      ccann::aligned_free((void *) buf->update_buf);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(1);
    }
    mem_index_ = std::make_unique<ccann::Index<T, uint32_t>>(metric, query_dim, 0, true, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, _u32 num_threads, bool new_index_format, bool use_page_search, int cpu_bound) {
    std::string pq_table_bin, pq_compressed_vectors, disk_index_file, centroids_file, disk_journal_file;
    ccann::Timer load_timer;
    std::string iprefix = std::string(index_prefix);
    pq_table_bin = iprefix + "_pq_pivots.bin";
    pq_compressed_vectors = iprefix + "_pq_compressed.bin";
    disk_index_file = iprefix + "_disk.index";
    this->_disk_index_file = disk_index_file;
    centroids_file = disk_index_file + "_centroids.bin";

    std::ifstream index_metadata(disk_index_file, std::ios::binary);

    size_t tags_offset = 0;
    size_t pq_pivots_offset = 0;
    size_t pq_vectors_offset = 0;
    _u64 disk_nnodes;
    _u64 disk_ndims;
    size_t medoid_id_on_file;
    _u64 file_frozen_id;
    _u32 ckpt_id = 0;

    if (new_index_format) {
      _u32 nr, nc;

      READ_U32(index_metadata, nr);
      READ_U32(index_metadata, nc);

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, disk_ndims);

      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      data_dim = disk_ndims;
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
      if (max_degree != this->range) {
        LOG(ERROR) << "Range mismatch: " << max_degree << " vs " << this->range << ", setting range to " << max_degree;
        this->range = max_degree;
      }

      ckpt_id = nr;

      LOG(INFO) << "Meta-data: # nodes per sector: " << nnodes_per_sector << ", max node len (bytes): " << max_node_len
                << ", max node degree: " << max_degree << ", npts: " << nr << ", dim: " << nc
                << " disk_nnodes: " << disk_nnodes << " disk_ndims: " << disk_ndims;

      if (nnodes_per_sector > this->kMaxElemInAPage) {
        LOG(ERROR) << "nnodes_per_sector: " << nnodes_per_sector << " is greater than " << this->kMaxElemInAPage
                   << ". Please recompile with a higher value of kMaxElemInAPage.";
        return -1;
      }

      READ_U64(index_metadata, this->num_frozen_points);
      READ_U64(index_metadata, file_frozen_id);
      if (this->num_frozen_points == 1) {
        this->frozen_location = file_frozen_id;
        // if (this->num_frozen_points == 1) {
        LOG(INFO) << " Detected frozen point in index at location " << this->frozen_location
                  << ". Will not output it at search time.";
      }
      READ_U64(index_metadata, tags_offset);
      READ_U64(index_metadata, pq_pivots_offset);
      READ_U64(index_metadata, pq_vectors_offset);

      LOG(INFO) << "Tags offset: " << tags_offset << " PQ Pivots offset: " << pq_pivots_offset
                << " PQ Vectors offset: " << pq_vectors_offset;
    } else {  // old index file format
      size_t actual_index_size = get_file_size(disk_index_file);
      size_t expected_file_size;
      READ_U64(index_metadata, expected_file_size);
      if (actual_index_size != expected_file_size) {
        LOG(INFO) << "File size mismatch for " << disk_index_file << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size;
        return -1;
      }

      READ_U64(index_metadata, disk_nnodes);
      READ_U64(index_metadata, medoid_id_on_file);
      READ_U64(index_metadata, max_node_len);
      READ_U64(index_metadata, nnodes_per_sector);
      max_degree = ((max_node_len - data_dim * sizeof(T)) / sizeof(unsigned)) - 1;

      LOG(INFO) << "Disk-Index File Meta-data: # nodes per sector: " << nnodes_per_sector;
      LOG(INFO) << ", max node len (bytes): " << max_node_len;
      LOG(INFO) << ", max node degree: " << max_degree;
    }

    size_per_io = SECTOR_LEN * (nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    LOG(INFO) << "Size per IO: " << size_per_io;

    index_metadata.close();

    pq_pivots_offset = 0;
    pq_vectors_offset = 0;

    LOG(INFO) << "After single file index check, Tags offset: " << tags_offset
              << " PQ Pivots offset: " << pq_pivots_offset << " PQ Vectors offset: " << pq_vectors_offset;

    size_t npts_u64, nchunks_u64;

    // read from file pq_compressed_vectors
    // from pq_vectors_offset read npts_u64(number of points) * nchunks_u64(dim)
    ccann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64, pq_vectors_offset);
    this->num_points = this->init_num_pts = npts_u64;
    this->n_chunks = nchunks_u64;

    this->cur_id = this->num_points;

    LOG(INFO) << "Load compressed vectors from file: " << pq_compressed_vectors << " offset: " << pq_vectors_offset
              << " num points: " << npts_u64 << " n_chunks: " << nchunks_u64;

    // Load PQ pivots (质心)
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64, pq_pivots_offset);

    if (disk_nnodes != num_points) {
      LOG(INFO) << "Mismatch in #points for compressed data file and disk "
                   "index file: "
                << disk_nnodes << " vs " << num_points;
      return -1;
    }

    this->data_dim = pq_table.get_dim();
    this->aligned_dim = ROUND_UP(this->data_dim, 8);

    LOG(INFO) << "Loaded PQ centroids and in-memory compressed vectors. #points: " << num_points
              << " #dim: " << data_dim << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

    // read index metadata
    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    std::string pq_compressed_file(pq_compressed_vectors);
    std::string id2loc_file(disk_index_file + ".id2loc");
    std::string tags_file(disk_index_file + ".tags");

#ifdef ANN_LARGE
    // delete pq_compressed_vectors;  
    // save storage
    // TODO: Remove once ready
    std::filesystem::remove(pq_compressed_vectors);
#endif

#ifdef ISS_OPT
    // Check 1000 points before the end
    ckpt_id = num_points - 1000;
#else
    ckpt_id = 0;
#endif

    reader->open(index_fname, true, false);

#ifndef ANN_LARGE
    pq_compressed_writer->open(pq_compressed_file, true, false);
#endif

    id2loc_writer->open(id2loc_file, true, false);
    if (this->enable_tags) {
      tags_writer->open(tags_file, true, false);
    }

    if (index_fname.find("pmem") != std::string::npos) {
      this->on_pm = true;

      auto index_pre_map_size = reader->file_size();
      index_pre_map_size = ROUND_UP(index_pre_map_size, PRE_MAP_PMEM_SIZE);
      LOG(INFO) << "Pre-mapping first " << index_pre_map_size << " bytes of pmem index file.";
      reader->init_dax(index_pre_map_size);

#ifndef ANN_LARGE
      auto pq_pre_map_size = pq_compressed_writer->file_size();
      pq_pre_map_size = ROUND_UP(pq_pre_map_size, PRE_MAP_PMEM_SIZE);
      pq_compressed_writer->init_dax(pq_pre_map_size);
#endif

      auto id2loc_pre_map_size = id2loc_writer->file_size();
      id2loc_pre_map_size = ROUND_UP(id2loc_pre_map_size, PRE_MAP_PMEM_SIZE);
      LOG(INFO) << "Pre-mapping first " << id2loc_pre_map_size << " bytes of pmem id2loc file.";
      id2loc_writer->init_dax(id2loc_pre_map_size);

      if (this->enable_tags) {
        auto tags_pre_map_size = tags_writer->file_size();
        tags_pre_map_size = ROUND_UP(tags_pre_map_size, PRE_MAP_PMEM_SIZE);
        tags_writer->init_dax(tags_pre_map_size);
      }
    }

    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // Check Consistency via ID2LOC file with the following steps:
    // 0. build allocation manager from ID2LOC file.
    // 1. Check the maximum ID in the ID2LOC file (may be through file size), max_id
    // 2. Check the ANN num_points
    // 3. Need to check vectors id ranging in [num_points, max_id] (both are not checkpointed inserted points)
    // # Check and Fix Index File
    // 4. For vector in [num_points, max_id]
    //    a. vec = id2loc(id)
    //    b. neighbors_ids = vec.neighbors
    //    # Fix all neighbors
    //    c. for each neighbor_id in neighbors_ids:
    //         neighbor_vec = id2loc(neighbor_id)
    //         neighbor_vec.add_if_not_exist(id)
    //         new_neighbor_vec = neighbor_vec.prune();
    //         new_loc = allocate space for new_neighbor_vec
    //         id2loc_writer.write(new_neighbor_vec, new_loc)
    //         update id2loc(neighbor_id) = new_loc
    // # Check and Fix PQ Compressed File
    // 5. For vector in [num_points, max_id]
    //    a. vec = id2loc(id)
    //    b. pq_coords = compress(vec)
    //    c. write pq_coords to pq_compressed_file at id
    //
    // # Set Current points
    // 6. set num_points = max_id + 1
    //
    // # We exclude maintaining consistency of #npts in Tags File/PQ Compressed file
    // # We use the global ANN num_points as the ground truth for all vectors.
    // # We only load [0, num_points) from Tags File/PQ Compressed file.

    // load page layout and set cur_loc
    this->use_page_search_ = use_page_search;
    if (this->on_pm) {
#ifdef PM_RECOVERY
      if (std::filesystem::exists(id2loc_file)) {
        libcuckoo::cuckoohash_map<uint32_t, uint32_t> loc2id;
        // load from id2loc file
        auto size = get_file_size(id2loc_file);
        _u32 potential_ids = size / sizeof(uint32_t);
        auto addr = id2loc_writer->get_dax(size, false);
        _u64 num_points = 0;
        _u32 max_id = 0;
        _u32 max_loc = 0;
        for (_u32 id = 0; id < potential_ids; id++) {
          _u32 loc = *((_u32 *) addr + id);
          // FIXME: how to identify loc 0?
          if (id == 0) {
            this->id2loc_.insert_or_assign(id, loc);
            loc2id.insert_or_assign(loc, id);
            num_points++;
            max_id = id;
          } else {
            if (loc != 0) {
              this->id2loc_.insert_or_assign(id, loc);
              loc2id.insert_or_assign(loc, id);
              num_points++;
              max_id = id;
            }
          }
          if (loc > max_loc) {
            max_loc = loc;
          }
        }

        LOG(INFO) << "Loaded ID2LOC from file: " << id2loc_file << ", #valid_points: " << num_points
                  << ", max_id: " << max_id << ", max_loc: " << max_loc;

        id2loc_writer->put_dax();

        uint64_t page_offset = loc_sector_no(0);
        uint64_t num_sectors = (num_points + nnodes_per_sector - 1) / nnodes_per_sector;

#pragma omp parallel for
        for (size_t i = 0; i < num_sectors; ++i) {
          PageArr tmp_arr;
          for (uint32_t j = 0; j < nnodes_per_sector; ++j) {
            uint32_t loc = i * nnodes_per_sector + j;
            uint32_t id;
            if (loc2id.contains(loc))
              id = loc2id.find(loc);
            else
              id = kInvalidID;
            tmp_arr[j] = id;
          }
          for (uint32_t j = nnodes_per_sector; j < tmp_arr.size(); ++j) {
            tmp_arr[j] = kInvalidID;
          }
          this->page_layout.insert(i + page_offset, tmp_arr);
        }
        this->cur_loc = max_loc + 1;
        if (this->cur_loc % nnodes_per_sector != 0) {
          this->cur_loc += nnodes_per_sector - (this->cur_loc % nnodes_per_sector);
        }
        this->num_points = num_points;
        this->cur_id = max_id + 1;
        LOG(INFO) << "Page Allocator Restored from ID2LOC file. cur_loc: " << this->cur_loc
                  << ", num_points: " << this->num_points << ", cur_id: " << this->cur_id;
        // check from [ckpt_id, max_id]
        char sector_buf[SECTOR_LEN];
        char nbr_sector_buf[SECTOR_LEN];
        LOG(INFO) << "Start to check and fix index from ID " << ckpt_id << " to " << max_id;
        std::vector<uint64_t> hint_pages;
        for (_u32 id = ckpt_id; id <= max_id; id++) {
          if (id % 10000 == 0) {
            LOG(INFO) << "  Checking ID " << id << ", progress: " << 100.0 * (id - ckpt_id) / (max_id - ckpt_id) << "%";
          }
          auto loc = id2loc(id);
          // FIX all neighbors
          auto sector = loc_sector_no(loc);
          // read sector
          auto graph_size = get_file_size(disk_index_file);
          auto graph_addr = reader->get_dax(graph_size, false);
          memcpy(sector_buf, (char *) graph_addr + sector * SECTOR_LEN, SECTOR_LEN);
          auto node_buf = offset_to_loc(sector_buf, loc);
          DiskNode<T> node(id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
          auto nnbrs = node.nnbrs;
          auto nbrs = node.nbrs;
          // recalc PQ coords and write to pq_compressed_file
          auto pq_coords = this->deflate_vector(node.coords);
          reader->put_dax();

#ifndef ANN_LARGE
          auto pq_file_size = ROUND_UP((id + 1) * this->n_chunks * sizeof(_u8), SECTOR_LEN);
          auto pq_file_addr = pq_compressed_writer->get_dax(pq_file_size, false);
          auto target_pq_addr = pq_file_addr + id * this->n_chunks * sizeof(_u8);
          memcpy((char *) target_pq_addr, (char *) pq_coords.data(), pq_coords.size());
          pq_compressed_writer->flush_dax(target_pq_addr, pq_coords.size());
          pq_compressed_writer->barrier_dax();
          pq_compressed_writer->put_dax();
#endif
          // pq_compressed_writer->write((char *) pq_coords.data(), pq_coords.size(), id * this->n_chunks *
          // sizeof(_u8));

          std::set<uint64_t> pages_need_to_read;  // useless for PM
          auto new_locs = this->alloc_loc_compact(nnbrs, hint_pages, pages_need_to_read);

          uint64_t new_max_loc = 0;
          uint64_t extend_fsize = 0;
          for (auto new_loc : new_locs) {
            if (new_loc > new_max_loc)
              new_max_loc = new_loc;
          }
          extend_fsize = loc_sector_no(new_max_loc) * SECTOR_LEN + SECTOR_LEN;
          extend_fsize = extend_fsize > graph_size ? extend_fsize : graph_size;

          graph_addr = reader->get_dax(extend_fsize, false);

          // LOG(INFO) << "Checking neighbors of ID " << id << ", nnbrs: " << nnbrs;
          // Check and fix all neighbors
          for (uint32_t n = 0; n < nnbrs; n++) {
            uint32_t neighbor_id = nbrs[n];
            auto neighbor_loc = id2loc(neighbor_id);
            // LOG(INFO) << "  Checking neighbor ID " << neighbor_id << " at loc " << neighbor_loc;
            // read neighbor node
            auto neighbor_sector = loc_sector_no(neighbor_loc);
            memcpy(nbr_sector_buf, (char *) graph_addr + neighbor_sector * SECTOR_LEN, SECTOR_LEN);
            auto neighbor_node_buf = offset_to_loc(nbr_sector_buf, neighbor_loc);
            DiskNode<T> neighbor_node(neighbor_id, offset_to_node_coords(neighbor_node_buf),
                                      offset_to_node_nhood(neighbor_node_buf));
            // add id if not exist
            std::vector<uint32_t> nhood;
            bool exist = false;
            for (uint32_t m = 0; m < neighbor_node.nnbrs; m++) {
              if (neighbor_node.nbrs[m] == id) {
                exist = true;
              }
              nhood.push_back(neighbor_node.nbrs[m]);
            }
            if (!exist) {
              nhood.push_back(id);
            }
            // Re-prune neighbor
            if (nhood.size() > this->range) {
              // Batch Prune
              QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);

              auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
              std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
              std::vector<float> tgt_dists_batch(PRUNE_BATCH_SIZE, 0.0f), nbr_dists_batch(PRUNE_BATCH_SIZE, 0.0f);
              auto target_id = id;

              bool pruned = false;
              float tgt_nbr_dis = 0;
              compute_pq_dists(target_id, &neighbor_node.id, &tgt_nbr_dis, 1, thread_pq_buf);

              for (size_t k = 0; k < nhood.size(); k += PRUNE_BATCH_SIZE) {
                size_t bsize = std::min((size_t) PRUNE_BATCH_SIZE, nhood.size() - k);
                ANN_START_TIMING(prune_neighbor_time, prune_neighbor_t);
                compute_pq_dists(target_id, nhood.data() + k, tgt_dists_batch.data(), (_u32) bsize, thread_pq_buf);
                compute_pq_dists(neighbor_node.id, nhood.data() + k, nbr_dists_batch.data(), (_u32) bsize,
                                 thread_pq_buf);
                ANN_END_TIMING(prune_neighbor_time, prune_neighbor_t);

                std::vector<TriangleNeighbor> tri_pool(PRUNE_BATCH_SIZE);

                for (size_t j = 0; j < PRUNE_BATCH_SIZE; j++) {
                  tri_pool[j].id = nhood[k + j];
                  tri_pool[j].tgt_dis = tgt_dists_batch[j];
                  tri_pool[j].distance = nbr_dists_batch[j];
                }
                std::sort(tri_pool.begin(), tri_pool.end());

                int to_evict = -1;
                pruned = this->fast_delta_prune_neighbors_pq(tri_pool, to_evict, tgt_nbr_dis);
                if (to_evict != -1) {
                  if ((uint32_t) to_evict != this->range) {
                    nhood.erase(nhood.begin() + k + to_evict);
                  } else {
                    // remove target node
                    nhood.pop_back();
                  }
                  break;
                }

                // assign to the full buffer
                for (size_t j = 0; j < bsize; ++j) {
                  tgt_dists[k + j] = tgt_dists_batch[j];
                  nbr_dists[k + j] = nbr_dists_batch[j];
                }
              }

              if (!pruned) {
                // full prune
                std::vector<TriangleNeighbor> tri_pool(nhood.size());

                for (uint32_t k = 0; k < nhood.size(); k++) {
                  tri_pool[k].id = nhood[k];
                  tri_pool[k].tgt_dis = tgt_dists[k];
                  tri_pool[k].distance = nbr_dists[k];
                }
                std::sort(tri_pool.begin(), tri_pool.end());

                int tgt_idx = -1;
                for (int k = 0; k < (int) nhood.size(); ++k) {
                  if (tri_pool[k].id == target_id) {
                    tgt_idx = k;
                    break;
                  }
                }
                if (unlikely(tgt_idx == -1)) {
                  LOG(ERROR) << "Target ID " << target_id << " not found in tri_pool";
                  exit(-1);
                }
                this->slow_delta_prune_neighbors_pq(tri_pool, nhood, thread_pq_buf, tgt_idx);
              }
              this->push_query_buf(read_data);
            }

            // rewrite neighbor node
            auto w_sector = loc_sector_no(new_locs[n]);
            char *pm_sec = (static_cast<char *>(graph_addr) + w_sector * SECTOR_LEN);
            char *pm_node = offset_to_loc(pm_sec, new_locs[n]);
            DiskNode<T> w_nbr_node_pm(id, offset_to_node_coords(pm_node), offset_to_node_nhood(pm_node));
            w_nbr_node_pm.nnbrs = (_u32) nhood.size();
            *(w_nbr_node_pm.nbrs - 1) = (_u32) nhood.size();  // write to buf
            memcpy(w_nbr_node_pm.coords, neighbor_node.coords, data_dim * sizeof(T));
            memcpy(w_nbr_node_pm.nbrs, nhood.data(), w_nbr_node_pm.nnbrs * sizeof(uint32_t));
            auto node_len = data_dim * sizeof(T) + nhood.size() * sizeof(uint32_t);
            reader->flush_dax(pm_node, node_len);
            reader->barrier_dax();
          }

          // Fix id2loc
          std::vector<uint64_t> orig_locs;
          std::vector<uint32_t> new_ids;
          for (uint32_t i = 0; i < nnbrs; ++i) {
            auto orig_loc = id2loc(nbrs[i]);
            orig_locs.emplace_back(id2loc(nbrs[i]));
            auto page = loc_sector_no(orig_loc);
            // deduplication
            hint_pages.emplace_back(page);

            id2loc_.insert_or_assign(nbrs[i], new_locs[i]);
            new_ids.emplace_back(nbrs[i]);
          }

          std::set<uint64_t> dedup_hint_pages(hint_pages.begin(), hint_pages.end());
          hint_pages.assign(dedup_hint_pages.begin(), dedup_hint_pages.end());

          erase_and_set_loc(orig_locs, new_locs, new_ids);
          reader->put_dax();
        }
      } else {
        this->load_page_layout(index_prefix, nnodes_per_sector, num_points);
      }
#else
      this->load_page_layout(index_prefix, nnodes_per_sector, num_points);
#endif
    } else {
      this->load_page_layout(index_prefix, nnodes_per_sector, num_points);
    }

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    num_medoids = 1;
    this->medoids = new uint32_t[1];
    this->medoids[0] = (_u32) (medoid_id_on_file);

    // set ckpt_id
    this->ckpt_id = this->num_points;

    // sanity check for CC_ANN
    bool enable_cc_ann = false;
    bool enable_journal_ann = false;

    // setup thread pool
    if (cpu_bound == -1) {
      this->num_cpus = std::thread::hardware_concurrency();
    } else {
      this->num_cpus = cpu_bound;
    }
    this->peak_cpus = this->num_cpus;
    for (_u32 i = 0; i < this->num_cpus; i++) {
      this->io_timing_stats.push_back(new ann_stats());
    }

    LOG(INFO) << "Setting up async insert thread pool with " << num_cpus << " threads.";

#ifdef USE_BS_THREAD_POOL
    this->insert_pool = std::make_unique<BS::thread_pool<>>(num_cpus);
#elif USE_SMALL_THREAD_POOL
    this->insert_pool = std::make_unique<ThreadPool>(num_cpus);
#endif

#ifdef CC_ANN
    enable_cc_ann = true;
#endif

#ifdef J_ANN
    enable_journal_ann = true;
#endif

    if (enable_cc_ann && enable_journal_ann) {
      LOG(ERROR) << "Cannot enable both CC-ANN and J-ANN";
      return -1;
    }

    LOG(INFO) << "Index loaded in " << load_timer.elapsed() / (double) 1000000 << "s";
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  _u64 SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::compute_pq_dists(const _u32 src, const _u32 *ids, float *fp_dists, const _u32 count,
                                           uint8_t *aligned_scratch) {
    const _u8 *src_ptr = this->data.data() + (this->n_chunks * src);
    if (unlikely(aligned_scratch == nullptr || count >= 32768)) {
      LOG(ERROR) << "Aligned scratch buffer is null or count is too large: " << count
                 << ". This will lead to memory issues.";
      crash();
    }
    // aggregate PQ coords into scratch
    ::aggregate_coords(ids, count, this->data.data(), this->n_chunks, aligned_scratch);
    // compute distances
    this->pq_table.compute_distances_alltoall(src_ptr, aligned_scratch, fp_dists, count);
  }

  template<typename T, typename TagT>
  std::vector<_u8> SSDIndex<T, TagT>::deflate_vector(const T *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    std::vector<float> fp_vec(this->data_dim);
    for (uint32_t i = 0; i < this->data_dim; i++) {
      fp_vec[i] = (float) vec[i];
    }
    this->pq_table.deflate_vec(fp_vec.data(), pq_coords.data());
    return pq_coords;
  }

  template<>
  std::vector<_u8> SSDIndex<float>::deflate_vector(const float *vec) {
    std::vector<_u8> pq_coords(this->n_chunks);
    this->pq_table.deflate_vec(vec, pq_coords.data());
    return pq_coords;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      ccann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());
      id2loc_.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
    }
    LOG(INFO) << "Loaded " << tags.size() << " tags";
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
