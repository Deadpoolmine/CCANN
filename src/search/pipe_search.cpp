#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <map>
#include <tbb/parallel_for.h>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

#ifndef USE_AIO
#include "liburing.h"
#endif

namespace ccann {
  struct io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_pipe_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    uint32_t original_l_search = l_search;
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    ANN_INIT_TIMING(populate_t);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    this->search_thread_count_++;

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    std::vector<unsigned int> inserts;
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(l_search * 10);

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;

#ifndef OVERLAP_INIT
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
#endif

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_dists](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                             _u8 *pq_coord_scratch) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      ANN_INIT_TIMING(compute_t);

      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);

      // NOTE: add coord_map
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      return cur_expanded_dist;
    };
    // LOG(INFO) << "Start pipe search with mem_L: " << mem_L << ", l_search: " << l_search
    //           << ", beam_width: " << beam_width;
    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      ANN_INIT_TIMING(expand_t);

      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      ANN_START_TIMING(expand_neighbors_time, expand_t);
      if (nbors_cand_size) {
        auto adjust_size = nbors_cand_size;
        // auto cpu1_st = std::chrono::high_resolution_clock::now();
        // 3400ns
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch, pq_coord_scratch);
        auto not_insert = 0;
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            ++not_insert;
            ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
            continue;
          }
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
        // auto cpu1_ed = std::chrono::high_resolution_clock::now();
        // stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();
        // LOG(INFO) << "Computed and pushed " << (nbors_cand_size - not_insert) << " / " << nbors_cand_size
      }
      ANN_END_TIMING(expand_neighbors_time, expand_t);
    };

    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    stats->io_us = 0;
    stats->io_us1 = 0;
    stats->cpu_us = 0;
    stats->cpu_us1 = 0;
    stats->cpu_us2 = 0;
    // search in in-memory index.

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;  // before converge.
#else
    int64_t cur_beam_width = beam_width;  // before converge.
#endif
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    ANN_START_TIMING(populate_pq_dists_time, populate_t);
#ifdef OVERLAP_INIT
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((unsigned) mem_L, (unsigned) l_search), mem_dists.data());
    } else {
      // cannot overlap.
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), dist_scratch);
    } else {
      compute_pq_dists(&medoids[0], 1, dist_scratch, pq_coord_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif
    ANN_END_TIMING(populate_pq_dists_time, populate_t);

    std::queue<io_t> on_flight_ios;

    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      // NOTE: here needs special attention:
      //
      // The correct version is:
      //
      // Insert Thread:         Search Thread:
      //
      // wrLock(id-a)           rdLock(id-a)
      // wrLock(id-b)           rdLock(id-b)
      //
      // unLock(id-a)
      // unLock(id-b)
      //
      // However, if use Pipe style:
      //
      // Search Thread (without interrupt):
      //
      // rdLock(id-a)
      // unLock(id-a)
      //
      // rdLock(id-b)
      // unLock(id-b)
      //
      // However, the case is that
      //
      // Search Thread (with interrupt):
      //
      // rdLock(id-b)
      // still proc
      // rdLock(id-a) <-- deadlock
      //
      // unLock(id-a)
      // unLock(id-b)
      //
      // So, we should keep the lock id order.

      // LOG(INFO) << "Locking node " << item.id << " in thread " << syscall(SYS_gettid);
      ANN_INIT_TIMING(send_best_t);
      // lock the corresponding page.
      uint32_t pid;
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      unsigned loc = 0;
#ifdef FINE_GRAINED_CONCURRENCY
      // This is CCANN-improved
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          pid = loc_sector_no(loc);
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);

          ANN_START_TIMING(send_best_node_time, send_best_t);
          reader->send_io(req, ctx, false);
          ANN_ADD_STAT(send_best_node_number, 1);
          ANN_END_TIMING(send_best_node_time, send_best_t);
        });
      } else {
        // Because of long I/O latency of SSD, we cannot support high concurrency.
        LOG(ERROR) << "Fine grained concurrency is only supported for PM index.";
        crash();
      }
#else
      loc = id2loc(item.id);
      pid = loc_sector_no(loc);
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);

      ANN_START_TIMING(send_best_node_time, send_best_t);
      reader->send_read_no_alloc(req, ctx);
      ANN_ADD_STAT(send_best_node_number, 1);
      ANN_END_TIMING(send_best_node_time, send_best_t);

#ifndef PIPE_PM_READS
      if (this->on_pm) {
        // for PM index, unlock immediately.
        this->unlock_idx(idx_lock_table, item.id);
      }
#endif

#endif
      if (passthrough_page_ref != nullptr)
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    // id_buf_map: id -> buf
    std::unordered_map<unsigned, char *> id_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;

      // there exists on flight ios.
      if (!on_flight_ios.empty()) {
        ANN_ADD_STAT(poll_number, 1);
      }

      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc((char *) io.read_req->buf, io.loc)));
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;

#ifndef PIPE_PM_READS
        // unlock the corresponding page.
        if (!this->on_pm) {
          // enable async only for SSD index.
          this->unlock_idx(idx_lock_table, io.nbr.id);
        }
#else
        this->unlock_idx(idx_lock_table, io.nbr.id);
#endif

        on_flight_ios.pop();
        // LOG(INFO) << "Unlocked node " << io.nbr.id << " in thread " << syscall(SYS_gettid);
      }

      if (n_in + n_out > 0) {
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      // auto io_st = std::chrono::high_resolution_clock::now();
      // retset is ordered (from small to large).
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size /* pool size */ &&
               (retset[marker].flag == false /* on flight */ ||
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          // find the first not sent and not already read neighbor.
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }

        auto id = retset[marker].id;

        // ensure that on_flight ios are also ordered
        // on_flight_ios reflects the order of sending IOs.
        // NOTE: Ensure Lock Order.
        if (!on_flight_ios.empty() &&
            on_flight_ios.back().nbr.id > id) {  // if the last sent IO has id > current id, then wait.
          break;
        }

        n_sent += send_read_req(retset[marker]);
      }
      // auto io_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_ed - io_st).count();
      return n_sent != 0;  // nothing to send.
    };

    auto calc_best_node = [&](int &expand_retries) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      for (marker = 0; marker < cur_list_size; ++marker) {
        // NOTE: after compute_exact_dists_and_push, the retset is changed, so marker should be changed.
        if (!retset[marker].visited) {
          auto it = id_buf_map.find(retset[marker].id);
          bool res = it != id_buf_map.end();
          if (res) {
            auto [id, buf] = *it;
            retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
            retset[marker].visited = true;
            // Do not expand too far nodes.
            compute_exact_dists_and_push(buf, id);
            compute_and_push_nbrs(buf, nk);
            break;
          }
        }
      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto print_state = [&]() {
      LOG(INFO) << "cur_list_size: " << cur_list_size;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        LOG(INFO) << "retset[" << i << "]: " << retset[i].id << ", " << retset[i].distance << ", " << retset[i].flag
                  << ", " << retset[i].visited << ", " << (id_buf_map.find(retset[i].id) != id_buf_map.end());
      }
      LOG(INFO) << "On flight IOs: " << on_flight_ios.size();
      if (on_flight_ios.size() != 0) {
        auto &io = on_flight_ios.front();
        LOG(INFO) << "on_flight_io: " << io.nbr.id << ", " << io.nbr.distance << ", " << io.nbr.flag << ", "
                  << io.page_id << ", " << io.loc << ", " << io.finished();
      }
      usleep(500);
    };

    std::ignore = print_state;

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;
#ifdef OVERLAP_INIT
    if (likely(mem_L != 0)) {
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif
    ANN_INIT_TIMING(poll_t);
    ANN_INIT_TIMING(calc_best_t);

    int expand_retries = 0;
    // LOG(INFO) << "Start One Query Pipe Search: beam_width=" << beam_width;
    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      // auto io1_st = std::chrono::high_resolution_clock::now();
      ANN_START_TIMING(poll_all_time, poll_t);
      auto [n_in, n_out] = poll_all();
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      // n_in: number of nodes that can improve the retset.
      // n_out: number of nodes that can not improve the retset.

#ifdef DYN_PIPE_WIDTH
#ifdef STATIC_POLICY
      constexpr int kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
#else
      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // LOG(INFO) << "Current beam width: " << cur_beam_width << " n_in: " << cur_n_in << " n_out: " << cur_tot;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }
#endif
#endif

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
#ifdef NAIVE_PIPE
        send_best_read_req(cur_beam_width - on_flight_ios.size());
#else
        send_best_read_req(1);
#endif
      }

      // auto io1_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
      ANN_START_TIMING(calc_best_node_time, calc_best_t);
      marker = calc_best_node(expand_retries);
      ANN_END_TIMING(calc_best_node_time, calc_best_t);
      ANN_ADD_STAT(calc_best_node_number, 1);
      max_marker = std::max(max_marker, marker);
    }

    // LOG(INFO) << "Pipe search expanded distribution:";
    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }

    assert(on_flight_ios.size() == 0);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    stats->cpu_us = n_computes;

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    this->search_thread_count_--;
    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    ANN_INIT_TIMING(search_t);
    ANN_START_TIMING(search_graph_time, search_t);
    this->do_pipe_search(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                         dyn_search_l);
    ANN_END_TIMING(search_graph_time, search_t);
    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < expanded_nodes_info.size() && t < k_search && i < l_search; i++) {
      if (i > 0 && expanded_nodes_info[i].id == expanded_nodes_info[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(expanded_nodes_info[i].id);
      if (distances != nullptr) {
        distances[t] = expanded_nodes_info[i].distance;
      }
      t++;
    }

    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
