#include "aligned_file_io.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <algorithm>
#include <filesystem>
#include <malloc.h>
#include <future>

#include "timer.h"
#include "tsl/robin_map.h"
#include "utils.h"
#include "v2/journal.h"
#include "v2/page_cache.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <tuple>

#include "linux_aligned_file_io.h"
#include <sys/syscall.h>
#include <unistd.h>

namespace ccann {


  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::search_phase(const T *point, tsl::robin_set<uint32_t> *deletion_set,
                                           std::vector<Neighbor> &exp_node_info,
                                           tsl::robin_map<uint32_t, T *> &coord_map, std::vector<uint32_t> &new_nhood,
                                           std::vector<uint64_t> &page_ref, std::vector<uint8_t> &out_pq_coords) {
    assert(reader != nullptr);
    void *ctx = reader->get_ctx();
    ANN_INIT_TIMING(read_t);
    ANN_INIT_TIMING(update_t);
    uint32_t target_id = cur_id++;
    // write PQ.

    // save target PQ vector into memory data.
    std::vector<uint8_t> pq_coords = deflate_vector(point);
    // chunk is dim
    ANN_START_TIMING(update_PQ_vec_time, update_t);
    uint64_t pq_offset = target_id * n_chunks;
    {
      static std::mutex pq_mu;
      std::lock_guard<std::mutex> lock(pq_mu);
      if (this->data.size() < pq_offset + n_chunks) {
        LOG(INFO) << "Resizing PQ data storage from " << this->data.size() << " to "
                  << (1.5 * this->data.size() + n_chunks);
        while (this->data.size() < pq_offset + n_chunks) {
          this->data.resize(1.5 * this->data.size());
        }
      }
      memcpy(this->data.data() + pq_offset, pq_coords.data(), n_chunks);
    }
    ANN_END_TIMING(update_PQ_vec_time, update_t);

    // l_index is candidate size.
    coord_map.reserve(2 * this->l_index);

    ANN_START_TIMING(search_graph_time, read_t);
    // this->do_beam_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, nullptr, deletion_set, false,
    //                      &page_ref);
    QueryStats stats;
    // LOG(INFO) << "Starting pipe search for insert.";

    // NOTE: 10 is hardcoded mem_L for insert search, refer to CCANN configuration.

    void (SSDIndex<T, TagT>::*search_func)(
        const T *, uint32_t, uint32_t, const uint32_t, std::vector<Neighbor> &, tsl::robin_map<uint32_t, T *> *,
        QueryStats *, tsl::robin_set<uint32_t> * /* tags */, bool, std::vector<uint64_t> *, uint32_t) = nullptr;

    if (this->search_mode == BEAM_SEARCH) {
      search_func = &SSDIndex<T, TagT>::do_beam_search;
#ifndef _WIN32
    } else if (this->search_mode == PIPE_SEARCH) {
      search_func = &SSDIndex<T, TagT>::do_pipe_search;
    } else if (this->search_mode == PARA_SEARCH) {
      search_func = &SSDIndex<T, TagT>::do_para_search;
#endif
    } else {
      LOG(ERROR) << "Invalid search mode: " << this->search_mode;
      crash();
    }

    if (this->mem_index_ != nullptr) {
      // this->do_pipe_search(point, 10, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false);
      (this->*search_func)(point, 10, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false,
                           &page_ref, l_index);
    } else {
      // this->do_pipe_search(point, 0, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false);
      (this->*search_func)(point, 0, l_index, beam_width, exp_node_info, &coord_map, &stats, deletion_set, false,
                           &page_ref, l_index);
    }
    ANN_END_TIMING(search_graph_time, read_t);

    ANN_INIT_TIMING(prune_search_neighbors_t);
    ANN_START_TIMING(prune_search_neighbors_time, prune_search_neighbors_t);
    prune_neighbors(coord_map, exp_node_info, new_nhood);
    ANN_END_TIMING(prune_search_neighbors_time, prune_search_neighbors_t);

    out_pq_coords = std::move(pq_coords);
    return target_id;
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::insert_phase_pm(const T *point, const TagT &tag, uint32_t target_id,
                                              std::vector<Neighbor> &exp_node_info,
                                              tsl::robin_map<uint32_t, T *> &coord_map,
                                              std::vector<uint32_t> &new_nhood, std::vector<uint64_t> &page_ref,
                                              std::vector<uint8_t> &in_pq_coords) {
    void *ctx = reader->get_ctx();
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);
    this->is_index_inserttable = true;

    ANN_INIT_TIMING(update_t);
    ANN_INIT_TIMING(update_neighbor_t);
    ANN_INIT_TIMING(read_nodes_t);
    ANN_INIT_TIMING(prune_neighbor_t);
    ANN_INIT_TIMING(journal_t);

    std::set<uint64_t> pages_need_to_read;

    this->insert_thread_count_++;

    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);

    uint64_t max_loc = 0;
    uint64_t extend_fsize = 0;
    void *faddr = nullptr;
    uint64_t fsize = reader->file_size();
    for (auto loc : locs) {
      if (loc > max_loc)
        max_loc = loc;
    }
    extend_fsize = loc_sector_no(max_loc) * SECTOR_LEN + SECTOR_LEN;
    extend_fsize = extend_fsize > fsize ? extend_fsize : fsize;

    faddr = reader->get_dax(extend_fsize, false);

    // LOG(INFO) << "mmaped addr: " << faddr << " len: " << extend_fsize << " for loc up to " << max_loc;

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    std::vector<IORequest> reads, writes, writes_4k;
    std::vector<v2::journal_entry<T>> journal_entries;
    std::vector<FlushRequest> flush_requests;
    assert(new_nhood.size() < MAX_N_EDGES);

    // read old pages for out-of-place update
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      IORequest req(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0);
      reads.push_back(req);
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;

      // (static_cast<char *>(faddr) + node_sector_no(new_nhood[i]) * SECTOR_LEN);
      // NOTE: random PM I/O is super slow, use block read instead.
    }
    // read new pages for RMW (might be in-place update).
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      // writes_4k.push_back(req);
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      // if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
      //   // need to read this page first.
      //   reads.push_back(req);
      // }

      // check if page_buf_map exsits
      auto res = page_buf_map.find(page);
      if (res == page_buf_map.end()) {
        page_buf_map[page] = (static_cast<char *>(faddr) + off);
      } else {
        // already read
        char *addr = res->second;
        IORequest req(off, size_per_io, addr, 0, 0);
        writes_4k.push_back(req);
      }

      // update_buf + i * size_per_io;
      // page_buf_map_pm[off / SECTOR_LEN] = (static_cast<char *>(faddr) + off);
      // static_cast<char *>(reader->mmap(req, true));
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    if (!writes_4k.empty()) {
      writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
      uint64_t start_idx = 0;
      uint64_t cur_off = writes_4k[0].offset;
      uint32_t i;

      for (i = 1; i < writes_4k.size() - 1; ++i) {
        if (writes_4k[i].offset != cur_off + size_per_io) {
          writes.push_back(
              IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
          start_idx = i;
        }
        cur_off = writes_4k[i].offset;
      }

// the last one
// TODO: for PM, only needs one single barrier to be written (figure it out).
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
      }
      writes_4k.pop_back();
    }

    ANN_START_TIMING(read_nodes_time, read_nodes_t);
    // #ifdef DIRECT_READ_CC
    reader->read(reads, ctx);
    // #else
    //     reader->read_alloc(reads, ctx, &page_ref);
    // #endif
    ANN_END_TIMING(read_nodes_time, read_nodes_t);

    // update the target node.
    ANN_START_TIMING(update_graph_time, update_t);

    auto target_sector = loc_sector_no(locs[new_nhood.size()]);
    auto node_buf = offset_to_loc(page_buf_map[target_sector], locs[new_nhood.size()]);
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;

    // LOG(INFO) << "Inserting node " << target_id << " at loc " << locs[new_nhood.size()] << " in Sector "
    //           << target_sector << " (" << target_sector * SECTOR_LEN << ") with " << target_node.nnbrs << "
    //           neighbors.";

    // only store ids? that's good.
    // but where to find the real coordinates?
    // where is the PQ compressed vector data saved?
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));

    // assert(reader->check_addr_in_pm(node_buf) == true);
    if (!reader->check_addr_in_pm(node_buf)) {
      // the buffer is in memory
      // shadow copy to PMem
      char *pm_sec = (static_cast<char *>(faddr) + target_sector * SECTOR_LEN);
      char *pm_node = offset_to_loc(pm_sec, locs[new_nhood.size()]);
      DiskNode<T> target_node_pm(target_id, offset_to_node_coords(pm_node), offset_to_node_nhood(pm_node));
      target_node_pm.nnbrs = new_nhood.size();
      *(target_node_pm.nbrs - 1) = target_node.nnbrs;  // write to buf
      memcpy(target_node_pm.coords, point, data_dim * sizeof(T));
      memcpy(target_node_pm.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
      node_buf = pm_node;
    }

    auto node_len = data_dim * sizeof(T) + target_node.nnbrs * sizeof(uint32_t);
    reader->flush_dax(node_buf, node_len);

    // Step 1. Update Tags in PM
    if (this->enable_tags) {
      auto tag_size = ROUND_UP(2 * sizeof(uint32_t) + (target_id + 1) * sizeof(TagT), SECTOR_LEN);
      // file system allows all zero
      auto tag_dax = tags_writer->get_dax(tag_size, false);
      auto target_tag_offset = 2 * sizeof(uint32_t) + target_id * sizeof(TagT);
      memcpy((char *) tag_dax + target_tag_offset, &tag, sizeof(TagT));
      tags_writer->sync();
      tags_writer->put_dax();
    }

    // Target Vector|Tags -> ID2LOC
    reader->barrier_dax();

    // ANN_END_TIMING(update_graph_time, update_t);
    tags.insert_or_assign(target_id, tag);

    // update the neighbors
    ANN_START_TIMING(update_neighbor_time, update_neighbor_t);
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " "
                   << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      // LOG(INFO) << r_sector << " " << node_sector_no(new_nhood[i]) << " "
      //           << (page_buf_map.find(r_sector) == page_buf_map.end());
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);

      assert(reader->check_addr_in_pm(r_node_buf) == false);
      assert(reader->check_addr_in_pm(r_nbr_node.nbrs) == false);
      assert(reader->check_addr_in_pm(r_nbr_node.nbrs + r_nbr_node.nnbrs) == false);

      // assign the original neighbors
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      // add the new target neighbor
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      assert(reader->check_addr_in_pm(nhood.data()) == false);

      if (nhood.size() > this->range) {  // prune neighbors
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        std::vector<float> tgt_dists_batch(PRUNE_BATCH_SIZE, 0.0f), nbr_dists_batch(PRUNE_BATCH_SIZE, 0.0f);

        bool pruned = false;
        float tgt_nbr_dis = 0;
        compute_pq_dists(target_id, &r_nbr_node.id, &tgt_nbr_dis, 1, thread_pq_buf);

        for (size_t k = 0; k < nhood.size(); k += PRUNE_BATCH_SIZE) {
          size_t bsize = std::min((size_t) PRUNE_BATCH_SIZE, nhood.size() - k);
          ANN_START_TIMING(prune_neighbor_time, prune_neighbor_t);
          compute_pq_dists(target_id, nhood.data() + k, tgt_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          compute_pq_dists(r_nbr_node.id, nhood.data() + k, nbr_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          ANN_END_TIMING(prune_neighbor_time, prune_neighbor_t);

          std::vector<TriangleNeighbor> tri_pool(bsize);

          for (size_t j = 0; j < bsize; j++) {
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

      }

      // Keep an incoming edge when pruning would isolate an outlier.
      if (i == 0 && std::find(nhood.begin(), nhood.end(), target_id) == nhood.end()) {
        if (nhood.size() >= this->range) {
          auto &scratch = read_data->aligned_pq_coord_scratch;
          std::vector<float> distances(nhood.size());
          compute_pq_dists(r_nbr_node.id, nhood.data(), distances.data(), (_u32) nhood.size(), scratch);
          auto farthest = std::max_element(distances.begin(), distances.end()) - distances.begin();
          nhood.erase(nhood.begin() + farthest);
        }
        nhood.push_back(target_id);
      }

      auto w_sector = loc_sector_no(locs[i]);
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));

      if (!reader->check_addr_in_pm(w_node_buf)) {
        // the buffer is in memory
        // shadow copy to PMem
        char *pm_sec = (static_cast<char *>(faddr) + w_sector * SECTOR_LEN);
        char *pm_node = offset_to_loc(pm_sec, locs[i]);
        DiskNode<T> w_nbr_node_pm(new_nhood[i], offset_to_node_coords(pm_node), offset_to_node_nhood(pm_node));
        w_nbr_node_pm.nnbrs = (_u32) nhood.size();
        *(w_nbr_node_pm.nbrs - 1) = (_u32) nhood.size();  // write to buf
        memcpy(w_nbr_node_pm.coords, r_nbr_node.coords, data_dim * sizeof(T));
        memcpy(w_nbr_node_pm.nbrs, nhood.data(), w_nbr_node_pm.nnbrs * sizeof(uint32_t));
        w_node_buf = pm_node;
      }
      // assert(reader->check_addr_in_pm(w_node_buf) == true);

      auto node_len = data_dim * sizeof(T) + w_nbr_node.nnbrs * sizeof(uint32_t);
      flush_requests.push_back(FlushRequest(w_node_buf, node_len));
    }

    reader->put_dax();

    ANN_END_TIMING(update_neighbor_time, update_neighbor_t);

    ANN_INIT_TIMING(update_meta_t);
    std::vector<uint64_t> write_page_ref;
    // reader->wbc_write(writes, ctx, &write_page_ref);

    // NOTE: File System provides atomic writes, ensuring that fallocate with zero populates.
    ANN_START_TIMING(update_metadata_time, update_meta_t);
    // Step 2. Update ID to Location Mapping in PM and DRAM
    // Update id2loc PMem mapping to make Target Vector|Tags Persistent.
    auto id2loc_size = ROUND_UP((target_id + 1) * sizeof(uint32_t), SECTOR_LEN);
    auto id2loc_dax = id2loc_writer->get_dax(id2loc_size, false);
    auto target_id_offset = target_id * sizeof(uint32_t);
    memcpy((char *) id2loc_dax + target_id_offset, &locs[new_nhood.size()], sizeof(uint32_t));

    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);

    // Neighbors -> ID2LOC
    // batch flush caches
    for (auto &flush_req : flush_requests) {
      reader->flush_dax(flush_req.buf, flush_req.len);
    }
    reader->barrier_dax();
    id2loc_writer->sync();

    // We do not need to lock idx_lock_table here, as id2loc_ is concurrent.
    // id2loc_ is already a concurrent hash map.
    // NOTE:
    // We need to ensure the reader-side consistency.
    // just use find_fn to make reader side being atomic.
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      // Atomically update DRAM id2loc and send (id, loc) to background
      // insert_commit_thread for asynchronous PM id2loc table update.
      // Sequence number (id) ensures ordering via the commit priority queue.
      id2loc_insert_or_assign(new_nhood[i], (_u32) locs[i]);
    }

    // NOTE:
    // Delay update allocator
    // i.e., loc2id is not updated immediately after id2loc update.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    id2loc_writer->put_dax();
    id2loc_writer->sync();

    ANN_END_TIMING(update_metadata_time, update_meta_t);

    T *commit_point = nullptr;

    if (this->mem_index_ != nullptr) {
      std::random_device rd;  // Will be used to obtain a seed for the random number engine
      auto x = rd();
      std::mt19937 generator((unsigned) x);
      std::uniform_real_distribution<float> distribution(0, 1);

      if (distribution(generator) < 0.01) {
        // update DRAM index
        commit_point = new T[data_dim];
        memcpy(commit_point, point, data_dim * sizeof(T));
      }
    }

    // Step 3. Update PQ Compressed Vector, this can be done in background
    // Step 4. Update in memory graph if possible
    auto commit_task = new CommitTask{
        .pq_coords = std::move(in_pq_coords),
        .target_id = target_id,
        .point = commit_point,
    };

    commit_tasks.push(commit_task);

    unlock_vec(vec_lock_table, target_id, new_nhood);

    // commit writes (in the background thread.)
    if (!writes.empty()) {
      // std::cout << "Flushing " << writes.size() + 1 << " writes to PMem." << std::endl;
      //   reader->write(writes, ctx);
      // writes.push_back(IORequest(target_sector * SECTOR_LEN, size_per_io, nullptr, 0, 0));
      // for (auto &req : writes) {
      //   char *pm_addr = static_cast<char *>(faddr) + req.offset;
      //   boost::crc_32_type result;
      //   result.process_bytes(pm_addr, req.len);
      //   auto cksum = result.checksum();
      //   // std::cout << "Write to sector " << req.offset / SECTOR_LEN << " len " << req.len << " cksum " << cksum
      //   //           << std::endl;
      // }
    }
    ANN_END_TIMING(update_graph_time, update_t);

    v2::unlockReqs(this->page_lock_table, pages_locked);

    if (search_mode == BEAM_SEARCH)
      reader->deref(&page_ref, ctx);

    this->insert_thread_count_--;

    this->push_query_buf(read_data);
    return target_id;
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::insert_phase(const T *point, const TagT &tag, uint32_t target_id,
                                           std::vector<Neighbor> &exp_node_info,
                                           tsl::robin_map<uint32_t, T *> &coord_map, std::vector<uint32_t> &new_nhood,
                                           std::vector<uint64_t> &page_ref, std::vector<uint8_t> &in_pq_coords) {
    void *ctx = reader->get_ctx();
    QueryBuffer<T> *read_data = this->pop_query_buf(nullptr);

    this->is_index_inserttable = true;

    ANN_INIT_TIMING(update_t);
    ANN_INIT_TIMING(update_neighbor_t);
    ANN_INIT_TIMING(read_nodes_t);
    ANN_INIT_TIMING(prune_neighbor_t);
    ANN_INIT_TIMING(journal_t);
    std::set<uint64_t> pages_need_to_read;

    this->insert_thread_count_++;

    auto locs = this->alloc_loc(new_nhood.size() + 1, page_ref, pages_need_to_read);

    std::set<uint64_t> pages_to_rmw_set;
    for (auto &loc : locs) {
      pages_to_rmw_set.insert(loc_sector_no(loc));
    }
    std::vector<IORequest> pages_to_rmw;
    // ordered because of std::set
    for (auto &page_no : pages_to_rmw_set) {
      pages_to_rmw.push_back(IORequest(page_no * SECTOR_LEN, size_per_io, nullptr, 0, 0));
    }
    // lock the target and the neighbor ids (ensure that sector_no does not change).
    auto pages_locked = v2::lockReqs(this->page_lock_table, pages_to_rmw);
    lock_vec(vec_lock_table, target_id, new_nhood);

    // re-read the candidate pages (mostly in the cache).
    std::unordered_map<uint32_t, char *> page_buf_map;

    auto &update_buf = read_data->update_buf;
    std::vector<IORequest> reads, writes_4k, writes;
    std::vector<v2::journal_entry<T>> journal_entries;

    assert(new_nhood.size() < MAX_N_EDGES);
    // read old pages for out-of-place update
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      reads.push_back(
          IORequest(node_sector_no(new_nhood[i]) * SECTOR_LEN, size_per_io, update_buf + i * size_per_io, 0, 0));
      page_buf_map[node_sector_no(new_nhood[i])] = update_buf + i * size_per_io;
    }

    // read new pages for RMW (might be in-place update).
    for (uint32_t i = new_nhood.size(); i < new_nhood.size() + pages_to_rmw.size(); ++i) {
      auto off = pages_to_rmw[i - new_nhood.size()].offset;
      writes_4k.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      // LOG(INFO) << off / SECTOR_LEN;
      uint64_t page = off / SECTOR_LEN;
      if (pages_need_to_read.find(page) != pages_need_to_read.end()) {
        // need to read this page first.
        reads.push_back(IORequest(off, size_per_io, update_buf + i * size_per_io, 0, 0));
      }
      page_buf_map[off / SECTOR_LEN] = update_buf + i * size_per_io;
    }

    // generate continuous writes from 4k writes.
    // dummy one.
    writes_4k.push_back(IORequest(std::numeric_limits<uint64_t>::max(), 0, nullptr, 0, 0));
    uint64_t start_idx = 0;
    uint64_t cur_off = writes_4k[0].offset;
    uint32_t i;

    for (i = 1; i < writes_4k.size() - 1; ++i) {
      if (writes_4k[i].offset != cur_off + size_per_io) {
        writes.push_back(
            IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
        start_idx = i;
      }
      cur_off = writes_4k[i].offset;
    }

// the last one
// TODO: for PM, only needs one single barrier to be written (figure it out).
    if (writes_4k[i].offset != cur_off + size_per_io) {
      writes.push_back(
          IORequest(writes_4k[start_idx].offset, size_per_io * (i - start_idx), writes_4k[start_idx].buf, 0, 0));
    }
    writes_4k.pop_back();

    std::vector<uint64_t> read_page_ref;
    ANN_START_TIMING(read_nodes_time, read_nodes_t);
    reader->read_alloc(reads, ctx, &read_page_ref);
    ANN_END_TIMING(read_nodes_time, read_nodes_t);

    // update the target node.
    ANN_START_TIMING(update_graph_time, update_t);
    auto sector = loc_sector_no(locs[new_nhood.size()]);
    auto node_buf = offset_to_loc(page_buf_map[sector], locs[new_nhood.size()]);
    DiskNode<T> target_node(target_id, offset_to_node_coords(node_buf), offset_to_node_nhood(node_buf));
    memcpy(target_node.coords, point, data_dim * sizeof(T));
    target_node.nnbrs = new_nhood.size();
    *(target_node.nbrs - 1) = target_node.nnbrs;
    // only store ids? that's good.
    // but where to find the real coordinates?
    // where is the PQ compressed vector data saved?
    memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
    tags.insert_or_assign(target_id, tag);
    auto node_len = data_dim * sizeof(T) + target_node.nnbrs * sizeof(uint32_t);

    // LOG(INFO) << "Target Node at " << locs[new_nhood.size()] << " in Sector " << sector << " (" << sector *
    // SECTOR_LEN
    //           << ") with " << target_node.nnbrs << " neighbors.";

    // update the neighbors
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      auto r_sector = node_sector_no(new_nhood[i]);
      if (page_buf_map.find(r_sector) == page_buf_map.end()) {
        LOG(ERROR) << new_nhood[i] << " "
                   << "Sector " << r_sector << " not found in page_buf_map";
        exit(-1);
      }
      auto r_node_buf = offset_to_node(page_buf_map[r_sector], new_nhood[i]);
      DiskNode<T> r_nbr_node(new_nhood[i], offset_to_node_coords(r_node_buf), offset_to_node_nhood(r_node_buf));
      std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
      // assign the original neighbors
      nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
      // add the new target neighbor
      nhood.emplace_back(target_id);  // attention: we do not reuse IDs.

      // LOG(INFO) << "Original Neighbor Node (" << (new_nhood[i]) << ") at " << id2loc(new_nhood[i]) << " in Sector "
      //           << r_sector << " (" << r_sector * SECTOR_LEN << ")";
      if (nhood.size() > this->range) {  // prune neighbors
        auto &thread_pq_buf = read_data->aligned_pq_coord_scratch;
        std::vector<float> tgt_dists(nhood.size(), 0.0f), nbr_dists(nhood.size(), 0.0f);
        std::vector<float> tgt_dists_batch(PRUNE_BATCH_SIZE, 0.0f), nbr_dists_batch(PRUNE_BATCH_SIZE, 0.0f);

        bool pruned = false;
        float tgt_nbr_dis = 0;
        compute_pq_dists(target_id, &r_nbr_node.id, &tgt_nbr_dis, 1, thread_pq_buf);

        for (size_t k = 0; k < nhood.size(); k += PRUNE_BATCH_SIZE) {
          size_t bsize = std::min((size_t) PRUNE_BATCH_SIZE, nhood.size() - k);
          compute_pq_dists(target_id, nhood.data() + k, tgt_dists_batch.data(), (_u32) bsize, thread_pq_buf);
          compute_pq_dists(r_nbr_node.id, nhood.data() + k, nbr_dists_batch.data(), (_u32) bsize, thread_pq_buf);

          std::vector<TriangleNeighbor> tri_pool(bsize);

          for (size_t j = 0; j < bsize; j++) {
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
      }

      auto w_sector = loc_sector_no(locs[i]);
      auto w_node_buf = offset_to_loc(page_buf_map[w_sector], locs[i]);
      DiskNode<T> w_nbr_node(new_nhood[i], offset_to_node_coords(w_node_buf), offset_to_node_nhood(w_node_buf));
      w_nbr_node.nnbrs = (_u32) nhood.size();
      *(w_nbr_node.nbrs - 1) = (_u32) nhood.size();  // write to buf
      memcpy(w_nbr_node.coords, r_nbr_node.coords, data_dim * sizeof(T));
      memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
      // LOG(INFO) << "New Neighbor Node (" << new_nhood[i] << ") at " << locs[i] << " in Sector "
      //           << w_sector << " (" << w_sector * SECTOR_LEN << ")";
    }

    std::vector<uint64_t> write_page_ref;

    reader->wbc_write(writes, ctx, &write_page_ref);

    // update locs
    // no concurrency issue for target_id (as it can be only inserted).
    id2loc_.insert_or_assign(target_id, locs[new_nhood.size()]);
    auto locked = lock_idx(idx_lock_table, target_id, new_nhood);
    auto page_locked = lock_page_idx(page_idx_lock_table, target_id, new_nhood);
    std::vector<uint64_t> orig_locs;
    for (uint32_t i = 0; i < new_nhood.size(); ++i) {
      orig_locs.emplace_back(id2loc(new_nhood[i]));
      id2loc_.insert_or_assign(new_nhood[i], locs[i]);
    }

    // with lock, for simple concurrency with alloc_loc.
    // Only for convenience, note that locs[new_nhood.size()] -> target.
    new_nhood.push_back(target_id);
    erase_and_set_loc(orig_locs, locs, new_nhood);
    unlock_page_idx(page_idx_lock_table, page_locked);
    unlock_idx(idx_lock_table, locked);
    // LOG(INFO) << "ID " << target_id << " Target loc " << id2loc(target_id);

    unlock_vec(vec_lock_table, target_id, new_nhood);

    // commit writes (in the background thread.)
    ANN_END_TIMING(update_graph_time, update_t);

    // generate journal writes
    // copy all entries to a continuous buffer

    // std::cout << "Flushing " << writes.size() << " writes to PMem." << std::endl;

    ANN_START_TIMING(update_graph_time, update_t);
    reader->write(writes, ctx);
    ANN_END_TIMING(update_graph_time, update_t);
    // for (auto &req : writes) {
    //   void *faddr = reader->get_dax(1 * 1024L * 1024L * 1024L, false);
    //   char *pm_addr = static_cast<char *>(faddr) + req.offset;
    //   boost::crc_32_type result;
    //   result.process_bytes(pm_addr, req.len);
    //   auto cksum = result.checksum();
    //   // std::cout << "Write to sector " << req.offset / SECTOR_LEN << " len " << req.len << " cksum " << cksum
    //   //           << std::endl;
    // }

    v2::unlockReqs(this->page_lock_table, pages_locked);
    reader->deref(&write_page_ref, ctx);

    if (search_mode == BEAM_SEARCH)
      reader->deref(&page_ref, ctx);

    reader->deref(&read_page_ref, ctx);

    this->insert_thread_count_--;
    this->push_query_buf(read_data);
    num_points++;
    return target_id;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::async_insert_in_place(const T *point, const TagT &tag,
                                               tsl::robin_set<uint32_t> *deletion_set) {
    if (this->on_pm)
      return insert_in_place(point, tag, deletion_set);
    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    std::vector<uint32_t> new_nhood;
    std::vector<uint64_t> page_ref;
    std::vector<uint8_t> out_pq_coords;
    uint32_t target_id;

    ANN_INIT_TIMING(search_t);

    ANN_START_TIMING(search_phase_time, search_t);
    target_id = search_phase(point, deletion_set, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    ANN_END_TIMING(search_phase_time, search_t);

    uint32_t (SSDIndex<T, TagT>::*func)(const T *, const TagT &, uint32_t, std::vector<Neighbor> &,
                                        tsl::robin_map<uint32_t, T *> &, std::vector<uint32_t> &,
                                        std::vector<uint64_t> &, std::vector<uint8_t> &) = nullptr;
    if (this->on_pm) {
      func = &SSDIndex<T, TagT>::insert_phase_pm;
    } else {
      func = &SSDIndex<T, TagT>::insert_phase;
    }


    auto search_threads = this->search_thread_count_.load();
    auto insert_threads = this->insert_thread_count_.load();
    auto calc_threads = this->calc_thread_count_.load();
    bool should_async = true;

    if (search_threads + insert_threads + calc_threads >= this->num_cpus * 2) {
      if (calc_threads <= search_threads) {  // calc thread is decreased significantly
        // do not submission, fall back to synchronous insert
        should_async = false;
      }
    }


    if (search_threads + insert_threads + calc_threads > this->peak_cpus) {
      this->peak_cpus = search_threads + insert_threads + calc_threads;
    }

    if (should_async) {
      insert_pool->detach_task([this, point, tag, target_id, exp_node_info = std::move(exp_node_info),
                                coord_map = std::move(coord_map), new_nhood = std::move(new_nhood),
                                page_ref = std::move(page_ref), out_pq_coords = std::move(out_pq_coords),
                                func]() mutable {
        ANN_INIT_TIMING(insert_t);
        ANN_START_TIMING(insert_phase_time, insert_t);
        (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
        ANN_END_TIMING(insert_phase_time, insert_t);
      });
    } else {
      ANN_INIT_TIMING(insert_t);
      ANN_START_TIMING(insert_phase_time, insert_t);
      (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
      ANN_END_TIMING(insert_phase_time, insert_t);
    }

    return target_id;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::synchronize_insertions() {
    insert_pool->wait();
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set) {
    std::lock_guard<std::mutex> insert_lock(insert_mutex_);
    std::vector<Neighbor> exp_node_info;
    tsl::robin_map<uint32_t, T *> coord_map;
    std::vector<uint32_t> new_nhood;
    std::vector<uint64_t> page_ref;
    std::vector<uint8_t> out_pq_coords;
    uint32_t target_id;

    ANN_INIT_TIMING(search_t);

    ANN_START_TIMING(search_phase_time, search_t);
    target_id = search_phase(point, deletion_set, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    ANN_END_TIMING(search_phase_time, search_t);

    uint32_t (SSDIndex<T, TagT>::*func)(const T *, const TagT &, uint32_t, std::vector<Neighbor> &,
                                        tsl::robin_map<uint32_t, T *> &, std::vector<uint32_t> &,
                                        std::vector<uint64_t> &, std::vector<uint8_t> &) = nullptr;
    if (this->on_pm) {
      func = &SSDIndex<T, TagT>::insert_phase_pm;
    } else {
      func = &SSDIndex<T, TagT>::insert_phase;
    }



    ANN_INIT_TIMING(insert_t);
    ANN_START_TIMING(insert_phase_time, insert_t);
    target_id = (this->*func)(point, tag, target_id, exp_node_info, coord_map, new_nhood, page_ref, out_pq_coords);
    ANN_END_TIMING(insert_phase_time, insert_t);
    if (this->on_pm)
      flush_commits();
    return target_id;
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::bg_io_thread() {
    auto ctx = reader->get_ctx();
    auto timer = ccann::Timer();
    uint64_t n_tasks = 0;

    while (true) {
      auto task = bg_io_tasks.pop();
      while (task == nullptr) {
        this->bg_io_tasks.wait_for_push_notify();
        task = bg_io_tasks.pop();
      }

      if (task->terminate) {
        delete task;
        break;
      }

      reader->write(task->writes, ctx);
      v2::unlockReqs(this->page_lock_table, task->pages_to_unlock);
      reader->deref(&task->pages_to_deref, ctx);
      this->push_query_buf(task->thread_data);
      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }
  }

#define COMMIT_INTERVAL 1000

  template<class T, class TagT>
  void SSDIndex<T, TagT>::flush_commits() {
    std::promise<void> completion;
    auto ready = completion.get_future();
    auto task = new CommitTask{{}, 0, false, nullptr, &completion};
    commit_tasks.push(task);
    commit_tasks.push_notify_all();
    ready.get();
  }

  template<class T, class TagT>
  void SSDIndex<T, TagT>::insert_commit_thread() {
    auto timer = ccann::Timer();
    uint64_t n_tasks = 0;
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> commit_queue;

    auto process_commit_queue = [this, &commit_queue]() {
      auto cur_ckpt_id = this->ckpt_id.load();
      uint32_t smallest_commit_id = 0;
      bool ckpt = false;
      while (!commit_queue.empty()) {
        smallest_commit_id = commit_queue.top();
        if (cur_ckpt_id == smallest_commit_id) {
          commit_queue.pop();
          cur_ckpt_id = smallest_commit_id + 1;
          ckpt = true;
        } else {
          // not continuous
          break;
        }
      }
      if (ckpt) {
        // ensure all previous writes are persistent
        reader->barrier_dax();
        id2loc_writer->sync();
        pq_compressed_writer->sync();
        auto pq_header = pq_compressed_writer->get_dax(SECTOR_LEN, false);
        memcpy(pq_header, &cur_ckpt_id, sizeof(uint32_t));
        pq_compressed_writer->sync();
        pq_compressed_writer->put_dax();
        if (enable_tags) {
          tags_writer->sync();
          auto tags_header = tags_writer->get_dax(SECTOR_LEN, false);
          memcpy(tags_header, &cur_ckpt_id, sizeof(uint32_t));
          tags_writer->sync();
          tags_writer->put_dax();
        }
        // update current checkpoint id
        // so we do not check these data during next recovery
        auto index_addr = reader->get_dax(SECTOR_LEN, false);
        auto npts_ofs = 0;
        memcpy((char *) index_addr + npts_ofs, &cur_ckpt_id, sizeof(uint32_t));
        reader->sync();
        reader->put_dax();
        this->ckpt_id.store(cur_ckpt_id);
      }
    };

    while (true) {
      auto task = commit_tasks.pop();
      while (task == nullptr) {
        this->commit_tasks.wait_for_push_notify();
        task = commit_tasks.pop();
      }

      if (task->terminate) {
        LOG(INFO) << "Commit thread received terminate signal.";
        delete task;
        break;
      }

      if (task->completion != nullptr) {
        process_commit_queue();
        task->completion->set_value();
        delete task;
        continue;
      }

      if (task->point) {
        ccann::Parameters paras;
        paras.Set<unsigned>("R", 32);
        paras.Set<unsigned>("L", 64);
        paras.Set<unsigned>("C", 750);
        paras.Set<float>("alpha", 1.2);

        // TODO: fix distribution
        // mem_index_->insert_point(task->point, paras, task->target_id);
        // LOG(INFO) << "Inserted point " << task->target_id << " into in-memory index.";
        delete[] task->point;
      }

      auto pq_coords = task->pq_coords.data();
      auto target_id = task->target_id;

      auto pq_bytes_per_vector = task->pq_coords.size() * sizeof(uint8_t);
      auto pq_size = ROUND_UP(2 * sizeof(uint32_t) + (target_id + 1) * pq_bytes_per_vector, SECTOR_LEN);
      auto pq_addr = this->pq_compressed_writer->get_dax(pq_size, false);
      auto pq_offset = 2 * sizeof(uint32_t) + target_id * pq_bytes_per_vector;
      memcpy((char *) pq_addr + pq_offset, pq_coords, pq_bytes_per_vector);
      pq_compressed_writer->sync();
      this->pq_compressed_writer->put_dax();

      // Drain pending background PM id2loc updates (from id2loc_insert_or_assign).
      // Sequence number (target_id / id) ensures ordering via the commit priority queue.
      {
        auto null_pair = std::make_pair(kInvalidID, kInvalidID);
        auto entry = id2loc_pm_queue.pop();
        if (entry != null_pair) {
          std::vector<std::pair<uint32_t, uint32_t>> id2loc_batch;
          uint32_t max_id = 0;
          do {
            id2loc_batch.push_back(entry);
            if (entry.first > max_id)
              max_id = entry.first;
            entry = id2loc_pm_queue.pop();
          } while (entry != null_pair);

          auto id2loc_size = ROUND_UP((max_id + 1) * sizeof(uint32_t), SECTOR_LEN);
          auto id2loc_dax = id2loc_writer->get_dax(id2loc_size, false);
          for (auto &[id, loc] : id2loc_batch) {
            auto id_offset = id * sizeof(uint32_t);
            memcpy((char *) id2loc_dax + id_offset, &loc, sizeof(uint32_t));
          }
          id2loc_writer->put_dax();
          id2loc_writer->sync();
        }
      }

      commit_queue.push(target_id);
      // commit in batch
      if (n_tasks != 0 && n_tasks % COMMIT_INTERVAL == 0) {
        process_commit_queue();
      }

      delete task;
      ++n_tasks;

      if (timer.elapsed() >= 5000000) {
        LOG(INFO) << "Processed " << n_tasks << " tasks, throughput: " << (double) n_tasks * 1e6 / timer.elapsed()
                  << " tasks/sec.";
        timer.reset();
        n_tasks = 0;
      }
    }

    LOG(INFO) << "Processing remaining commit " << commit_queue.size() << " tasks...";
    process_commit_queue();
    if (!commit_queue.empty()) {
      LOG(INFO) << "Commit queue not empty after termination!";
      crash();
    }
    LOG(INFO) << "Commit thread terminated.";
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;
}  // namespace ccann
