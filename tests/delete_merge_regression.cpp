#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "aux_utils.h"
#include "distance.h"
#include "parameters.h"
#include "utils.h"
#include "v2/dynamic_index.h"

namespace {
constexpr uint32_t kBaseCount = 320;
constexpr uint32_t kDimension = 32;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void set_parameters(ccann::Parameters &params) {
  params.Set<unsigned>("L_disk", 64);
  params.Set<unsigned>("R_disk", 0);
  params.Set<float>("alpha_disk", 1.2f);
  params.Set<unsigned>("C", 384);
  params.Set<unsigned>("beamwidth", 4);
  params.Set<unsigned>("nodes_to_cache", 0);
  params.Set<unsigned>("num_threads", 2);
}

void check_merged_index(const std::string &prefix, const std::set<uint32_t> &expected) {
  std::vector<uint32_t> tags;
  size_t count, dimensions;
  ccann::load_bin<uint32_t>(prefix + "_disk.index.tags", tags, count, dimensions);
  require(dimensions == 1 && count == expected.size(), "merged tag count mismatch");
  require(std::set<uint32_t>(tags.begin(), tags.end()) == expected, "merged tags mismatch");

  std::ifstream graph(prefix + "_disk.index", std::ios::binary);
  uint32_t metadata_rows, metadata_columns;
  std::array<uint64_t, 8> metadata{};
  graph.read(reinterpret_cast<char *>(&metadata_rows), sizeof(metadata_rows));
  graph.read(reinterpret_cast<char *>(&metadata_columns), sizeof(metadata_columns));
  graph.read(reinterpret_cast<char *>(metadata.data()), sizeof(metadata));
  require(graph.good() && metadata_rows == 8 && metadata_columns == 1, "merged graph metadata unreadable");
  require(metadata[0] == expected.size() && metadata[1] == kDimension, "merged graph size mismatch");
  require(metadata[2] < expected.size(), "merged medoid out of range");
  const auto node_size = metadata[3];
  const auto nodes_per_sector = metadata[4];
  require(nodes_per_sector > 0 && node_size >= kDimension * sizeof(float) + sizeof(uint32_t),
          "merged node layout invalid");
  size_t nonempty_nodes = 0;
  for (uint32_t id = 0; id < count; ++id) {
    const auto offset = (1 + id / nodes_per_sector) * SECTOR_LEN + (id % nodes_per_sector) * node_size;
    graph.seekg(offset + kDimension * sizeof(float));
    uint32_t degree;
    graph.read(reinterpret_cast<char *>(&degree), sizeof(degree));
    require(graph.good() && degree <= (node_size - kDimension * sizeof(float) - 4) / 4,
            "merged node degree invalid");
    if (degree != 0) ++nonempty_nodes;
    for (uint32_t edge = 0; edge < degree; ++edge) {
      uint32_t neighbor;
      graph.read(reinterpret_cast<char *>(&neighbor), sizeof(neighbor));
      require(graph.good() && neighbor < count && neighbor != id, "merged neighbor ID invalid");
    }
  }
  require(nonempty_nodes >= count / 2, "merged graph contains too many empty nodes");
}

std::vector<uint32_t> read_neighbors(const std::string &prefix, uint32_t id) {
  std::ifstream graph(prefix + "_disk.index", std::ios::binary);
  std::array<uint64_t, 8> metadata{};
  graph.seekg(8);
  graph.read(reinterpret_cast<char *>(metadata.data()), sizeof(metadata));
  const auto offset = (1 + id / metadata[4]) * SECTOR_LEN + (id % metadata[4]) * metadata[3];
  graph.seekg(offset + kDimension * sizeof(float));
  uint32_t degree;
  graph.read(reinterpret_cast<char *>(&degree), sizeof(degree));
  require(graph.good() && degree <= 32, "cannot read medoid neighbors");
  std::vector<uint32_t> neighbors(degree);
  graph.read(reinterpret_cast<char *>(neighbors.data()), degree * sizeof(uint32_t));
  require(graph.good(), "cannot read medoid neighbors");
  return neighbors;
}

size_t check_search(ccann::DynamicSSDIndex<float, uint32_t> &index, const std::vector<float> &vectors,
                    const std::set<uint32_t> &expected) {
  size_t hits = 0;
  for (uint32_t sample = 0; sample < 12; ++sample) {
    const uint32_t id = sample * 7;
    const uint32_t tag = 1000 + id;
    if (!expected.count(tag)) continue;
    std::array<uint32_t, 5> results{};
    std::array<float, 5> distances{};
    index.search(vectors.data() + id * kDimension, results.size(), 0, 64, 4,
                 results.data(), distances.data(), nullptr);
    for (auto result : results)
      require(expected.count(result) != 0,
              "search returned unexpected tag " + std::to_string(result) + " for query " + std::to_string(tag));
    if (std::find(results.begin(), results.end(), tag) != results.end()) ++hits;
  }
  return hits;
}
}  // namespace

int main() {
  char directory[] = "/tmp/ccann-delete-merge-XXXXXX";
  require(mkdtemp(directory) != nullptr, "cannot create test directory");
  const std::string prefix = std::string(directory) + "/index";
  const std::string merged = prefix + "_merge";
  try {
    std::mt19937 random(23);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> vectors(kBaseCount * kDimension);
    for (auto &value : vectors) value = distribution(random);
    std::vector<uint32_t> tags(kBaseCount);
    std::set<uint32_t> expected;
    for (uint32_t id = 0; id < kBaseCount; ++id) {
      tags[id] = 1000 + id;
      expected.insert(tags[id]);
    }
    const auto data_path = prefix + "_base.bin";
    const auto tag_path = prefix + "_base.tags";
    ccann::save_bin<float>(data_path, vectors.data(), kBaseCount, kDimension);
    ccann::save_bin<uint32_t>(tag_path, tags.data(), kBaseCount, 1);
    require(ccann::build_disk_index_py<float, uint32_t>(data_path.c_str(), prefix.c_str(), 32, 64, 1, 2, 16,
                                                       ccann::Metric::L2, false, tag_path.c_str()),
            "base index build failed");

    ccann::DistanceL2 distance;
    ccann::Parameters params;
    set_parameters(params);
    size_t baseline_hits = 0;
    {
      ccann::DynamicSSDIndex<float, uint32_t> index(params, prefix, merged, &distance,
                                                     ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
      baseline_hits = check_search(index, vectors, expected);
      index.final_merge(2, 20);
      check_merged_index(merged, expected);
      {
        ccann::Parameters verify_params;
        set_parameters(verify_params);
        ccann::DynamicSSDIndex<float, uint32_t> no_delete(verify_params, merged, merged + "_check", &distance,
                                                          ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
        auto hits = check_search(no_delete, vectors, expected);
        require(hits + 2 >= baseline_hits,
                "no-delete merge recall fell from " + std::to_string(baseline_hits) + " to " +
                std::to_string(hits));
      }
    }

    for (int mode : {PIPE_SEARCH, PARA_SEARCH}) {
      ccann::Parameters search_params;
      set_parameters(search_params);
      ccann::DynamicSSDIndex<float, uint32_t> retained(search_params, prefix, merged + "_search", &distance,
                                                        ccann::Metric::L2, mode, false, true, 2);
      require(check_search(retained, vectors, expected) >= 8, "retained search mode lost recall");
    }
    for (int removed_mode : {1, 3}) {
      ccann::Parameters invalid_params;
      set_parameters(invalid_params);
      bool rejected = false;
      try {
        ccann::DynamicSSDIndex<float, uint32_t> removed(invalid_params, prefix, merged + "_removed", &distance,
                                                         ccann::Metric::L2, removed_mode, false, true, 2);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "removed search mode must be rejected");
    }

    {
      ccann::Parameters delete_params;
      set_parameters(delete_params);
      ccann::DynamicSSDIndex<float, uint32_t> index(delete_params, prefix, merged, &distance,
                                                     ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
      const uint32_t medoid_tag = tags[index._disk_index->get_init_ids().at(0)];
      std::set<uint32_t> deleted = {medoid_tag, 1010, 1020, 5000};
      std::array<float, kDimension> added{};
      added.fill(2.0f);
      index.insert(added.data(), 5000);
      added.fill(3.0f);
      index.insert(added.data(), 5001);
      expected.insert(5001);
      for (auto tag : deleted) {
        index.lazy_delete(tag);
        index.lazy_delete(tag);
        expected.erase(tag);
      }
      index.final_merge(2, 20);
      check_merged_index(merged, expected);
      added.fill(4.0f);
      index.insert(added.data(), 5002);
      expected.insert(5002);
      index.lazy_delete(1007);
      expected.erase(1007);
      index.final_merge(2, 20);
      check_merged_index(merged, expected);
      std::thread concurrent_delete([&index] { index.lazy_delete(1009); });
      index.final_merge(2, 20);
      concurrent_delete.join();
      expected.erase(1009);
      index.final_merge(2, 20);
      check_merged_index(merged, expected);
    }

    {
      ccann::Parameters next_params;
      set_parameters(next_params);
      ccann::DynamicSSDIndex<float, uint32_t> restored(next_params, merged, merged + "_next", &distance,
                                                        ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
      auto merged_hits = check_search(restored, vectors, expected);
      require(merged_hits + 2 >= baseline_hits,
              "merged recall fell from " + std::to_string(baseline_hits) + " to " +
              std::to_string(merged_hits));
      restored.lazy_delete(1008);
      expected.erase(1008);
      restored.final_merge(2, 20);
      check_merged_index(merged + "_next", expected);
    }
    {
      ccann::Parameters isolated_params;
      set_parameters(isolated_params);
      const auto compacted = merged + "_next";
      ccann::DynamicSSDIndex<float, uint32_t> isolated(isolated_params, compacted, compacted + "_isolated",
                                                        &distance, ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
      std::vector<uint32_t> current_tags;
      size_t count, dimensions;
      ccann::load_bin<uint32_t>(compacted + "_disk.index.tags", current_tags, count, dimensions);
      const auto medoid = isolated._disk_index->get_init_ids().at(0);
      std::set<uint32_t> to_delete = {current_tags.at(medoid)};
      for (auto neighbor : read_neighbors(compacted, medoid))
        to_delete.insert(current_tags.at(neighbor));
      require(to_delete.size() < expected.size(), "test must leave live points");
      for (auto tag : to_delete) {
        isolated.lazy_delete(tag);
        expected.erase(tag);
      }
      isolated.final_merge(2, 20);
      check_merged_index(compacted + "_isolated", expected);
    }
    {
      ccann::Parameters empty_params;
      set_parameters(empty_params);
      const auto compacted = merged + "_next_isolated";
      ccann::DynamicSSDIndex<float, uint32_t> empty(empty_params, compacted, compacted + "_all_deleted",
                                                     &distance, ccann::Metric::L2, BEAM_SEARCH, false, true, 2);
      for (auto tag : expected) empty.lazy_delete(tag);
      bool rejected = false;
      try {
        empty.final_merge(2, 20);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "merging all deleted nodes must fail explicitly");
      check_merged_index(compacted, expected);
    }
    std::filesystem::remove_all(directory);
  } catch (...) {
    std::fprintf(stderr, "Delete/merge test artifacts: %s\n", directory);
    throw;
  }
}
