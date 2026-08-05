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

#include "async_comp.h"
#include "circular_buffer.h"

namespace compctx {
  static thread_local std::unique_ptr<AsyncRing<>> comp_engine = nullptr;
}

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

  struct comp_t {
    Neighbor nbr;
    unsigned nnbrs;
    unsigned *node_nbrs;
    float *nbr_dists;
    IORequest *comp_req;
    bool finished() {
      return comp_req->finished;
    }
  };

#ifdef EARLY_EXIT
  inline float mean(const std::deque<float> &vals) {
    if (vals.empty())
      return 0.0f;
    float sum = std::accumulate(vals.begin(), vals.end(), 0.0f);
    return sum / vals.size();
  }

  inline float variance(const std::deque<float> &vals, float mean_val) {
    if (vals.size() <= 1)
      return 0.0f;
    float accum = 0.0f;
    for (float v : vals) {
      float diff = v - mean_val;
      accum += diff * diff;
    }
    return accum / (vals.size() - 1);  // 无偏估计
  }

#endif

#define NO_EARLY_STOP_FLAG (0)
#define EARLY_STOP_FLAG (-1)
#define LIKELY_EARLY_STOP_FLAG (-2)

  template<typename T, typename TagT>
  AsyncRing<> *SSDIndex<T, TagT>::get_comp_engine() {
    if (unlikely(compctx::comp_engine == nullptr)) {
      compctx::comp_engine = std::make_unique<AsyncRing<>>(
          4, [this](size_t active_workers) { this->calc_thread_count_ -= active_workers; });
      this->calc_thread_count_ += 4;
    }
    return compctx::comp_engine.get();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_para_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info,
                                         tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                         tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                         std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    uint32_t original_l_search = l_search;
    ANN_INIT_TIMING(populate_t);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif
    auto comp_ring = get_comp_engine();

    this->search_thread_count_++;

    auto search_threads = this->search_thread_count_.load();
    auto insert_threads = this->insert_thread_count_.load();
    auto calc_threads = this->calc_thread_count_.load();

#ifdef NO_ACC_OPT
    // disable dynamic adjustment
#else
    auto threshold = this->num_cpus;

    if (this->is_index_inserttable) {
      threshold = this->num_cpus * 2;
    }

    // LOG(INFO) << "threshold for para search dynamic calc thread adjustment: " << threshold;
    if (search_threads + insert_threads + calc_threads >= threshold) {
      if (comp_ring->activated_worker_count() > 1) {
        // do not wait for completion, reduce compute threads.
        comp_ring->remove_worker(true);
        this->calc_thread_count_--;
      }
    } else {
      if (comp_ring->activated_worker_count() < 4) {
        // do not wait for completion, reduce compute threads.
        comp_ring->add_worker();
        this->calc_thread_count_++;
      }
    }
#endif

    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (beam_width > MAX_N_COMPUTES) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_COMPUTES";
      crash();
    }

    QueryBuffer<T> *query_buf = pop_query_buf(query1);

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
    SlidingWindow recentQ(10);
    std::priority_queue<float, std::vector<float>, std::greater<float>> smallestQ;

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
    auto compute_pq_dists = [this, pq_dists, query_buf](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                                        _u8 *pq_coord_scratch) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

#ifdef EARLY_EXIT
    float prev_median = std::numeric_limits<float>::infinity();
    float alpha = 0;
    float alpha_min = 0;
    float alpha_max = 0.3;
    float tau_stable = 0.05;           // 中位数变化小于2%视为平滑
    float tau_volatile = 0.1;          // 大于10%视为波动
    std::deque<float> median_history;  // 存储最近N个窗口中位数
    unsigned median_window = 5;        // 可调，用于检测趋势稳定性
#endif

    auto push_nbrs = [&](unsigned *nbrs, unsigned nnbrs, float *dist_scratch, unsigned &n_in, unsigned &n_out) {
      ANN_INIT_TIMING(compute_t);

      ANN_START_TIMING(expand_neighbors_time, compute_t);
      for (unsigned m = 0; m < nnbrs; ++m) {
        const int nbor_id = nbrs[m];
        const float nbor_dist = dist_scratch[m];
        if (stats != nullptr) {
          stats->n_cmps++;
        }
        ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
        if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
          ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
          n_out++;
          continue;
        }
        n_in++;
        Neighbor nn(nbor_id, nbor_dist, true);
        // Return position in sorted list where nn inserted
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
        if (cur_list_size < l_search) {
          ++cur_list_size;
          if (unlikely(cur_list_size >= retset.size())) {
            retset.resize(2 * cur_list_size);
          }
        }
      }
      ANN_END_TIMING(expand_neighbors_time, compute_t);
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
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      std::sort(retset.begin(), retset.begin() + cur_list_size);
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

    std::queue<comp_t> on_flight_comps;
    // id_buf_map: id -> buf
    std::unordered_map<unsigned, char *> id_buf_map;

    auto send_read_req = [&](Neighbor &item) -> char * {
      ANN_INIT_TIMING(send_best_t);
      ANN_INIT_TIMING(read_best_t);
      ANN_START_TIMING(do_read_best_node_time, read_best_t);
      // lock the corresponding page.
      uint32_t pid;
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      auto loc = 0;
#ifdef FINE_GRAINED_CONCURRENCY
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          pid = loc_sector_no(loc);
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);
          reader->send_io(req, ctx, false);
          if (passthrough_page_ref != nullptr)
            passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
        });
        assert(req.finished == true);
        ANN_ADD_STAT(send_best_node_number, 1);
        // immediately read
        id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));
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
      reader->send_io(req, ctx, false);
      ANN_ADD_STAT(send_best_node_number, 1);
      ANN_END_TIMING(send_best_node_time, send_best_t);
      if (passthrough_page_ref != nullptr)
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);

      // immediately read
      id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));

      // for PM index, unlock immediately.
      this->unlock_idx(idx_lock_table, item.id);
#endif
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) {
        stats->n_ios++;
      }

      ANN_END_TIMING(do_read_best_node_time, read_best_t);

      return offset_to_loc((char *) req.buf, loc);
    };

    auto send_compute_req = [&](Neighbor &item, char *node_buf, bool sync = false) -> bool {
      // create a closure that does the distance computation
      uint64_t &cur_comp_idx = query_buf->comp_idx;
      auto dist_buf = (float *) (((_u8 *) dist_scratch) + cur_comp_idx * 512 * sizeof(float));
      auto pq_buf = pq_coord_scratch + cur_comp_idx * 32768 * 32 * sizeof(_u8);
      auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
      comp_req = IORequest();  // dummy init
      comp_req.finished = false;

      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;

      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }

      auto compute_fn = [this, &compute_pq_dists, node_nbrs, nbors_cand_size, dist_buf, pq_buf,
                         cur_comp_idx]() -> uint64_t {
        ANN_INIT_TIMING(compute_t);
        ANN_START_TIMING(expand_neighbors_time, compute_t);
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_buf, pq_buf);
        ANN_END_TIMING(expand_neighbors_time, compute_t);
        return cur_comp_idx;
      };

      cur_comp_idx = (cur_comp_idx + 1) % MAX_N_COMPUTES;

      if (sync) {
        compute_fn();
        comp_req.finished = true;
      } else {
        comp_ring->send_for_comp(compute_fn);
      }

      on_flight_comps.push(comp_t{item, nbors_cand_size, node_nbrs, dist_buf, &comp_req});

      return true;
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

    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      unsigned n_in = 0, n_out = 0;

      // there exists on flight ios.
      if (!on_flight_comps.empty()) {
        ANN_ADD_STAT(poll_number, 1);
      }

      auto r = comp_ring->poll_all();
      for (auto &cres : r) {
        // find the corresponding comp_t
        auto cur_comp_idx = (uint64_t) cres.result;
        auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
        comp_req.finished = true;
      }

      while (!on_flight_comps.empty() && on_flight_comps.front().finished()) {
        comp_t &comp = on_flight_comps.front();
        unsigned nnbrs = comp.nnbrs;
        unsigned *node_nbrs = comp.node_nbrs;
        float *nbr_dists = comp.nbr_dists;

        push_nbrs(node_nbrs, nnbrs, nbr_dists, n_in, n_out);
        on_flight_comps.pop();
      }

      if (n_in + n_out > 0) {
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      return std::make_pair(n_in, n_out);
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    int marker = 0, max_marker = 0;

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif
    ANN_INIT_TIMING(poll_t);
    ANN_INIT_TIMING(calc_best_t);

    // LOG(INFO) << "Start One Query Para Search: beam_width=" << beam_width;
    int expand_retries = 0;
    bool sent = false;
    int first_unvisited = 0;
    bool wait_for_flight = false;
    bool early_stop = false;

    auto compute_exact_dists_and_push = [&](Neighbor &item, const char *node_buf,
                                            const unsigned id) -> std::pair<float, int> {
      ANN_INIT_TIMING(compute_t);

      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));

      auto cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      int stop_flag = NO_EARLY_STOP_FLAG;
      // NOTE: add coord_map
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

#ifdef EARLY_EXIT
      int start_thresh = l_search / 2;
      if (max_marker > start_thresh) {
        // uint32_t E = std::min((uint32_t) (l_search - start_thresh), 2 * k_search);

        // if (smallestQ.size() < E) {
        //   smallestQ.push(cur_expanded_dist);
        // } else {
        //   if (cur_expanded_dist < smallestQ.top()) {
        //     smallestQ.pop();
        //     smallestQ.push(cur_expanded_dist);
        //   }
        // }
        // // +inf
        // auto R_q = std::numeric_limits<float>::infinity();
        // if (smallestQ.size() == E) {
        //   R_q = smallestQ.top();
        // }

        recentQ.push(cur_expanded_dist);
        if (recentQ.filled()) {
          float med = recentQ.median();
          float delta_med = fabs(med - prev_median) / (prev_median + 1e-6);

          median_history.push_back(med);
          if (median_history.size() > median_window)
            median_history.pop_front();

          float meanM = mean(median_history);
          float varM = variance(median_history, meanM);
          float stdM = std::sqrt(varM);

          bool is_stable = (delta_med < tau_stable) && (stdM < tau_stable * meanM);
          bool is_volatile = (delta_med > tau_volatile) || (stdM > tau_volatile * meanM);

          prev_median = med;
          if (is_stable) {
            // LOG(INFO) << "Search stable detected. Median: " << med << ", Delta: " << delta_med << ", Std: " << stdM
            //           << ", R_q: " << R_q << ", R_q * (1 - alpha): " << R_q * (1 - alpha);
            // alpha = std::min(alpha_max, (float) (alpha + 0.1));  // 更激进
            // LOG(INFO) << "Search stable detected. Increasing alpha to " << alpha;
            // current worst distance in retset
            // only do this when we have enough candidates, and the median is stable
            // if (med >= R_q * (1 - alpha)) {
            if (full_retset.size() > k_search) {
              // can terminate early
              stop_flag = EARLY_STOP_FLAG;
            } else {
              // likely early stop, do not send new computations
              stop_flag = LIKELY_EARLY_STOP_FLAG;
            }
            // }
          }
          // else {
          //   alpha = std::max(alpha_min, (float) (alpha - 0.1));  // 更保守
          //   // LOG(INFO) << "Search volatile detected. Decreasing alpha to " << alpha;
          // }
        }
      }
#endif

      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      return std::make_pair(cur_expanded_dist, stop_flag);
    };

    auto calc_best_node = [&](int &expand_retries, bool &sent) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      for (marker = 0; marker < cur_list_size; ++marker) {
        // NOTE: after compute_exact_dists_and_push, the retset is changed, so marker should be changed.
        if (!retset[marker].visited) {
          auto id = retset[marker].id;

          if ((int64_t) on_flight_comps.size() < cur_beam_width) {
            auto buf = send_read_req(retset[marker]);
            // LOG(INFO) << "Exploring marker @ " << marker;
            retset[marker].visited = true;
            auto [exact_dist, stop_flag] = compute_exact_dists_and_push(retset[marker], buf, id);
            if (stop_flag != NO_EARLY_STOP_FLAG) {
              // LOG(INFO) << "Early stop at expand retries " << expand_retries;
              return stop_flag;
            }
            send_compute_req(retset[marker], buf, false);
            sent = true;
          }
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = marker; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

    while (((first_unvisited = get_first_unvisited()) != -1) || !on_flight_comps.empty()) {
      sent = false;
      ANN_START_TIMING(poll_all_time, poll_t);
      auto [n_in, n_out] = poll_all();
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      expand_retries++;

      if (wait_for_flight && n_in + n_out > 0) {
        if (n_in == 0) {
          ANN_ADD_STAT(para_useless_wait_for_flight_number, 1);
        }
        wait_for_flight = false;
      }

      if (n_in + n_out == 0 && first_unvisited == -1) {
        // LOG(INFO) << "Waiting for on flight computations to finish. On flight comps: " << on_flight_comps.size();
        wait_for_flight = true;
        ANN_ADD_STAT(para_wait_for_flight_number, 1);
        sched_yield();
        continue;
      }

      // if (n_in + n_out > 0) {
      //   if (n_in > 0) {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got n_in " << n_in << " n_out " << n_out
      //               << " max_marker " << max_marker;
      //   } else {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got 0 n_ins ";
      //   }
      // }

      if (early_stop) {
        // tag all remaining as visited.
        for (unsigned i = 0; i < cur_list_size; ++i) {
          retset[i].visited = true;
        }
      } else {
        ANN_START_TIMING(calc_best_node_time, calc_best_t);
        marker = calc_best_node(expand_retries, sent);
        ANN_END_TIMING(calc_best_node_time, calc_best_t);
        ANN_ADD_STAT(calc_best_node_number, 1);
        max_marker = std::max(max_marker, marker);

        if (n_in + n_out == 0 && sent == false && first_unvisited != -1) {
          ANN_ADD_STAT(para_busy_wait_number, 1);
          // LOG(INFO) << "Busy wait due to small beam width.";
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        if (marker == EARLY_STOP_FLAG) {
          // LOG(INFO) << "early stop at expand retries " << expand_retries;
          ANN_ADD_STAT(para_early_exit_number, 1);
          early_stop = true;
        } else if (marker == LIKELY_EARLY_STOP_FLAG) {
          // LOG(INFO) << "likely early stop at expand retries " << expand_retries;
          ANN_ADD_STAT(para_likely_early_exit_number, 1);
          // do nothing, wait for retset to be full.
        }
      }
    }

    // LOG(INFO) << "Pipe search expanded distribution: " << expand_retries;

    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }
    // LOG(INFO) << "Remaining on flight comps: " << on_flight_comps.size();

    assert(on_flight_comps.size() == 0);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    // stats->cpu_us = n_computes;

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    this->search_thread_count_--;
    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_para_search_sync(const T *query1, uint32_t mem_L, uint32_t l_search,
                                              const uint32_t beam_width, std::vector<Neighbor> &expanded_nodes_info,
                                              tsl::robin_map<uint32_t, T *> *coord_map, QueryStats *stats,
                                              tsl::robin_set<uint32_t> *exclude_nodes /* tags */, bool dyn_search_l,
                                              std::vector<uint64_t> *passthrough_page_ref, uint32_t k_search) {
    uint32_t original_l_search = l_search;
    ANN_INIT_TIMING(populate_t);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    this->search_thread_count_++;
    auto search_threads = this->search_thread_count_.load();
    auto insert_threads = this->insert_thread_count_.load();
    auto calc_threads = this->calc_thread_count_.load();
    
    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (beam_width > MAX_N_COMPUTES) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_COMPUTES";
      crash();
    }

    QueryBuffer<T> *query_buf = pop_query_buf(query1);

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
    SlidingWindow recentQ(10);
    std::priority_queue<float, std::vector<float>, std::greater<float>> smallestQ;

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
    auto compute_pq_dists = [this, pq_dists, query_buf](const unsigned *ids, const _u64 n_ids, float *dists_out,
                                                        _u8 *pq_coord_scratch) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

#ifdef EARLY_EXIT
    float prev_median = std::numeric_limits<float>::infinity();
    float alpha = 0;
    float alpha_min = 0;
    float alpha_max = 0.3;
    float tau_stable = 0.05;           // 中位数变化小于2%视为平滑
    float tau_volatile = 0.1;          // 大于10%视为波动
    std::deque<float> median_history;  // 存储最近N个窗口中位数
    unsigned median_window = 5;        // 可调，用于检测趋势稳定性
#endif

    auto push_nbrs = [&](unsigned *nbrs, unsigned nnbrs, float *dist_scratch, unsigned &n_in, unsigned &n_out) {
      ANN_INIT_TIMING(compute_t);

      ANN_START_TIMING(expand_neighbors_time, compute_t);
      for (unsigned m = 0; m < nnbrs; ++m) {
        const int nbor_id = nbrs[m];
        const float nbor_dist = dist_scratch[m];
        if (stats != nullptr) {
          stats->n_cmps++;
        }
        ANN_ADD_STAT(ncalc_for_expanding_neighbors, 1);
        if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
          ANN_ADD_STAT(ncalc_for_useless_neighbors, 1);
          n_out++;
          continue;
        }
        n_in++;
        Neighbor nn(nbor_id, nbor_dist, true);
        // Return position in sorted list where nn inserted
        auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
        if (cur_list_size < l_search) {
          ++cur_list_size;
          if (unlikely(cur_list_size >= retset.size())) {
            retset.resize(2 * cur_list_size);
          }
        }
      }
      ANN_END_TIMING(expand_neighbors_time, compute_t);
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
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch, pq_coord_scratch);
      std::sort(retset.begin(), retset.begin() + cur_list_size);
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

    std::queue<comp_t> on_flight_comps;
    // id_buf_map: id -> buf
    std::unordered_map<unsigned, char *> id_buf_map;

    auto send_read_req = [&](Neighbor &item) -> char * {
      ANN_INIT_TIMING(send_best_t);
      ANN_INIT_TIMING(read_best_t);
      ANN_START_TIMING(do_read_best_node_time, read_best_t);
      // lock the corresponding page.
      uint32_t pid;
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      auto loc = 0;
#ifdef FINE_GRAINED_CONCURRENCY
      if (this->on_pm) {
        loc = id2loc_func(item.id, [&](uint32_t &loc) {
          pid = loc_sector_no(loc);
          req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len);
          reader->send_io(req, ctx, false);
          if (passthrough_page_ref != nullptr)
            passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);
        });
        assert(req.finished == true);
        ANN_ADD_STAT(send_best_node_number, 1);
        // immediately read
        id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));
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
      reader->send_io(req, ctx, false);
      ANN_ADD_STAT(send_best_node_number, 1);
      ANN_END_TIMING(send_best_node_time, send_best_t);
      if (passthrough_page_ref != nullptr)
        passthrough_page_ref->push_back((static_cast<_u64>(pid) * SECTOR_LEN) / SECTOR_LEN);

      // immediately read
      id_buf_map.insert(std::make_pair(item.id, offset_to_loc((char *) req.buf, loc)));

      // for PM index, unlock immediately.
      this->unlock_idx(idx_lock_table, item.id);
#endif
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) {
        stats->n_ios++;
      }

      ANN_END_TIMING(do_read_best_node_time, read_best_t);

      return offset_to_loc((char *) req.buf, loc);
    };

    auto send_compute_req = [&](Neighbor &item, char *node_buf, bool sync = false) -> bool {
      // create a closure that does the distance computation
      uint64_t &cur_comp_idx = query_buf->comp_idx;
      auto dist_buf = (float *) (((_u8 *) dist_scratch) + cur_comp_idx * 512 * sizeof(float));
      auto pq_buf = pq_coord_scratch + cur_comp_idx * 32768 * 32 * sizeof(_u8);
      auto &comp_req = query_buf->comp_reqs[cur_comp_idx];
      comp_req = IORequest();  // dummy init
      comp_req.finished = false;

      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;

      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }

      auto compute_fn = [this, &compute_pq_dists, node_nbrs, nbors_cand_size, dist_buf, pq_buf,
                         cur_comp_idx]() -> uint64_t {
        ANN_INIT_TIMING(compute_t);
        ANN_START_TIMING(expand_neighbors_time, compute_t);
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_buf, pq_buf);
        ANN_END_TIMING(expand_neighbors_time, compute_t);
        return cur_comp_idx;
      };

      cur_comp_idx = (cur_comp_idx + 1) % MAX_N_COMPUTES;

      compute_fn();
      comp_req.finished = true;

      on_flight_comps.push(comp_t{item, nbors_cand_size, node_nbrs, dist_buf, &comp_req});

      return true;
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

    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      unsigned n_in = 0, n_out = 0;

      // there exists on flight ios.
      if (!on_flight_comps.empty()) {
        ANN_ADD_STAT(poll_number, 1);
      }

      while (!on_flight_comps.empty() && on_flight_comps.front().finished()) {
        comp_t &comp = on_flight_comps.front();
        unsigned nnbrs = comp.nnbrs;
        unsigned *node_nbrs = comp.node_nbrs;
        float *nbr_dists = comp.nbr_dists;

        push_nbrs(node_nbrs, nnbrs, nbr_dists, n_in, n_out);
        on_flight_comps.pop();
      }

      if (n_in + n_out > 0) {
        ANN_ADD_STAT(poll_hit_number, 1);
      }
      return std::make_pair(n_in, n_out);
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    int marker = 0, max_marker = 0;

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif
    ANN_INIT_TIMING(poll_t);
    ANN_INIT_TIMING(calc_best_t);

    // LOG(INFO) << "Start One Query Para Search: beam_width=" << beam_width;
    int expand_retries = 0;
    bool sent = false;
    int first_unvisited = 0;
    bool wait_for_flight = false;
    bool early_stop = false;

    auto compute_exact_dists_and_push = [&](Neighbor &item, const char *node_buf,
                                            const unsigned id) -> std::pair<float, int> {
      ANN_INIT_TIMING(compute_t);

      ANN_START_TIMING(calc_exact_dist_time, compute_t);
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));

      auto cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      int stop_flag = NO_EARLY_STOP_FLAG;
      // NOTE: add coord_map
      if (coord_map != nullptr) {
        coord_map->insert(std::make_pair(id, node_fp_coords_copy));
      }

      // LOG(INFO) << "Expanding node " << id << " distance " << cur_expanded_dist;
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

#ifdef EARLY_EXIT
      int start_thresh = l_search / 2;
      if (max_marker > start_thresh) {
        // uint32_t E = std::min((uint32_t) (l_search - start_thresh), 2 * k_search);

        // if (smallestQ.size() < E) {
        //   smallestQ.push(cur_expanded_dist);
        // } else {
        //   if (cur_expanded_dist < smallestQ.top()) {
        //     smallestQ.pop();
        //     smallestQ.push(cur_expanded_dist);
        //   }
        // }
        // // +inf
        // auto R_q = std::numeric_limits<float>::infinity();
        // if (smallestQ.size() == E) {
        //   R_q = smallestQ.top();
        // }

        recentQ.push(cur_expanded_dist);
        if (recentQ.filled()) {
          float med = recentQ.median();
          float delta_med = fabs(med - prev_median) / (prev_median + 1e-6);

          median_history.push_back(med);
          if (median_history.size() > median_window)
            median_history.pop_front();

          float meanM = mean(median_history);
          float varM = variance(median_history, meanM);
          float stdM = std::sqrt(varM);

          bool is_stable = (delta_med < tau_stable) && (stdM < tau_stable * meanM);
          bool is_volatile = (delta_med > tau_volatile) || (stdM > tau_volatile * meanM);

          prev_median = med;
          if (is_stable) {
            // LOG(INFO) << "Search stable detected. Median: " << med << ", Delta: " << delta_med << ", Std: " << stdM
            //           << ", R_q: " << R_q << ", R_q * (1 - alpha): " << R_q * (1 - alpha);
            // alpha = std::min(alpha_max, (float) (alpha + 0.1));  // 更激进
            // LOG(INFO) << "Search stable detected. Increasing alpha to " << alpha;
            // current worst distance in retset
            // only do this when we have enough candidates, and the median is stable
            // if (med >= R_q * (1 - alpha)) {
            if (full_retset.size() > k_search) {
              // can terminate early
              stop_flag = EARLY_STOP_FLAG;
            } else {
              // likely early stop, do not send new computations
              stop_flag = LIKELY_EARLY_STOP_FLAG;
            }
            // }
          }
          // else {
          //   alpha = std::max(alpha_min, (float) (alpha - 0.1));  // 更保守
          //   // LOG(INFO) << "Search volatile detected. Decreasing alpha to " << alpha;
          // }
        }
      }
#endif

      ANN_END_TIMING(calc_exact_dist_time, compute_t);

      return std::make_pair(cur_expanded_dist, stop_flag);
    };

    auto calc_best_node = [&](int &expand_retries, bool &sent) -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      for (marker = 0; marker < cur_list_size; ++marker) {
        // NOTE: after compute_exact_dists_and_push, the retset is changed, so marker should be changed.
        if (!retset[marker].visited) {
          auto id = retset[marker].id;

          if ((int64_t) on_flight_comps.size() < cur_beam_width) {
            auto buf = send_read_req(retset[marker]);
            // LOG(INFO) << "Exploring marker @ " << marker;
            retset[marker].visited = true;
            auto [exact_dist, stop_flag] = compute_exact_dists_and_push(retset[marker], buf, id);
            if (stop_flag != NO_EARLY_STOP_FLAG) {
              // LOG(INFO) << "Early stop at expand retries " << expand_retries;
              return stop_flag;
            }
            send_compute_req(retset[marker], buf, true);
            sent = true;
          }
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = marker; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

    while (((first_unvisited = get_first_unvisited()) != -1) || !on_flight_comps.empty()) {
      sent = false;
      ANN_START_TIMING(poll_all_time, poll_t);
      auto [n_in, n_out] = poll_all();
      ANN_END_TIMING(poll_all_time, poll_t);
      std::ignore = n_in;
      std::ignore = n_out;

      expand_retries++;

      if (wait_for_flight && n_in + n_out > 0) {
        if (n_in == 0) {
          ANN_ADD_STAT(para_useless_wait_for_flight_number, 1);
        }
        wait_for_flight = false;
      }

      if (n_in + n_out == 0 && first_unvisited == -1) {
        // LOG(INFO) << "Waiting for on flight computations to finish. On flight comps: " << on_flight_comps.size();
        wait_for_flight = true;
        ANN_ADD_STAT(para_wait_for_flight_number, 1);
        sched_yield();
        continue;
      }

      // if (n_in + n_out > 0) {
      //   if (n_in > 0) {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got n_in " << n_in << " n_out " << n_out
      //               << " max_marker " << max_marker;
      //   } else {
      //     LOG(INFO) << "Expand retries " << expand_retries << " got 0 n_ins ";
      //   }
      // }

      if (early_stop) {
        // tag all remaining as visited.
        for (unsigned i = 0; i < cur_list_size; ++i) {
          retset[i].visited = true;
        }
      } else {
        ANN_START_TIMING(calc_best_node_time, calc_best_t);
        marker = calc_best_node(expand_retries, sent);
        ANN_END_TIMING(calc_best_node_time, calc_best_t);
        ANN_ADD_STAT(calc_best_node_number, 1);
        max_marker = std::max(max_marker, marker);

        if (n_in + n_out == 0 && sent == false && first_unvisited != -1) {
          ANN_ADD_STAT(para_busy_wait_number, 1);
          // LOG(INFO) << "Busy wait due to small beam width.";
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        if (marker == EARLY_STOP_FLAG) {
          // LOG(INFO) << "early stop at expand retries " << expand_retries;
          ANN_ADD_STAT(para_early_exit_number, 1);
          early_stop = true;
        } else if (marker == LIKELY_EARLY_STOP_FLAG) {
          // LOG(INFO) << "likely early stop at expand retries " << expand_retries;
          ANN_ADD_STAT(para_likely_early_exit_number, 1);
          // do nothing, wait for retset to be full.
        }
      }
    }

    // LOG(INFO) << "Pipe search expanded distribution: " << expand_retries;

    // for (auto &insert : inserts) {
    //   LOG(INFO) << "Insert expanded neighbors: " << insert;
    // }
    // LOG(INFO) << "Remaining on flight comps: " << on_flight_comps.size();

    assert(on_flight_comps.size() == 0);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    // stats->cpu_us = n_computes;

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    this->search_thread_count_--;
    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::para_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats,
                                        tsl::robin_set<uint32_t> *deleted_nodes, bool dyn_search_l) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
#ifdef ANN_LARGE
    this->do_para_search_sync(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                              dyn_search_l, nullptr, k_search);
#else
    this->do_para_search(query1, mem_L, l_search, beam_width, expanded_nodes_info, nullptr, stats, deleted_nodes,
                         dyn_search_l, nullptr, k_search);
#endif
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