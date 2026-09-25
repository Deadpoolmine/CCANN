#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

#include "aux_utils.h"
#include "distance.h"
#include "linux_aligned_file_reader.h"
#include "utils.h"
#include "v2/dynamic_index.h"

namespace py = pybind11;

class PySSDIndex {
 public:
  using Vectors = py::array_t<float, py::array::c_style>;
  using Tags = py::array_t<uint32_t, py::array::c_style>;

  static std::unique_ptr<PySSDIndex> create(const std::string &prefix, Vectors vectors, Tags tags,
                                            ccann::Metric metric, uint32_t threads) {
    if (vectors.ndim() != 2 || tags.ndim() != 1 || vectors.shape(0) != tags.shape(0) ||
        vectors.shape(0) < 256 || vectors.shape(1) == 0 || threads == 0)
      throw py::value_error("Expected at least 256 float32 vectors, matching uint32 tags, and positive threads");
    if (metric != ccann::Metric::L2)
      throw py::value_error("Only L2 is supported by the SSD Python index");
    std::set<uint32_t> unique_tags(tags.data(), tags.data() + tags.shape(0));
    if (unique_tags.size() != static_cast<size_t>(tags.shape(0)))
      throw py::value_error("Tags must be unique");
    if (std::filesystem::exists(prefix + "_disk.index"))
      throw py::value_error("Index already exists at prefix");
    std::string data_path = prefix + "_ccann_data.bin";
    std::string tags_path = prefix + "_ccann_tags.bin";
    ccann::save_bin<float>(data_path, vectors.mutable_data(), vectors.shape(0), vectors.shape(1));
    ccann::save_bin<uint32_t>(tags_path, tags.mutable_data(), tags.shape(0), 1);
    auto chunks = std::min<int>(32, vectors.shape(1));
    if (!ccann::build_disk_index_py<float, uint32_t>(data_path.c_str(), prefix.c_str(), 32, 64, 1,
                                                     threads, chunks, metric, false, tags_path.c_str()))
      throw std::runtime_error("SSD index build failed");
    {
      std::ofstream meta(prefix + "_ccann.meta", std::ios::trunc);
      meta << vectors.shape(1) << ' ' << static_cast<int>(metric) << '\n';
      if (!meta)
        throw std::runtime_error("Failed to write index metadata");
    }
    return load(prefix, metric, threads);
  }

  static std::unique_ptr<PySSDIndex> load(const std::string &prefix, ccann::Metric metric,
                                          uint32_t threads) {
    if (threads == 0)
      throw py::value_error("threads must be positive");
    if (metric != ccann::Metric::L2)
      throw py::value_error("Only L2 is supported by the SSD Python index");
    uint32_t dim;
    int stored_metric;
    std::ifstream meta(prefix + "_ccann.meta");
    if (!(meta >> dim >> stored_metric) || dim == 0 || stored_metric != static_cast<int>(metric))
      throw py::value_error("Missing or incompatible SSD index metadata");
    if (!std::filesystem::exists(prefix + "_disk.index") ||
        !std::filesystem::exists(prefix + "_pq_compressed.bin") ||
        !std::filesystem::exists(prefix + "_pq_pivots.bin") ||
        !std::filesystem::exists(prefix + "_disk.index.tags"))
      throw py::value_error("Missing SSD index files");
    uint32_t graph_count, graph_columns, pq_count, pq_dim, tag_count, tag_dim;
    uint64_t disk_count, disk_dim;
    std::ifstream graph(prefix + "_disk.index", std::ios::binary);
    std::ifstream pq(prefix + "_pq_compressed.bin", std::ios::binary);
    std::ifstream tag_file(prefix + "_disk.index.tags", std::ios::binary);
    graph.read(reinterpret_cast<char *>(&graph_count), sizeof(graph_count));
    graph.read(reinterpret_cast<char *>(&graph_columns), sizeof(graph_columns));
    graph.read(reinterpret_cast<char *>(&disk_count), sizeof(disk_count));
    graph.read(reinterpret_cast<char *>(&disk_dim), sizeof(disk_dim));
    pq.read(reinterpret_cast<char *>(&pq_count), sizeof(pq_count));
    pq.read(reinterpret_cast<char *>(&pq_dim), sizeof(pq_dim));
    tag_file.read(reinterpret_cast<char *>(&tag_count), sizeof(tag_count));
    tag_file.read(reinterpret_cast<char *>(&tag_dim), sizeof(tag_dim));
    if (!graph || !pq || !tag_file || disk_dim != dim || disk_count > pq_count ||
        pq_count != tag_count || pq_dim == 0 || tag_dim != 1 ||
        std::filesystem::file_size(prefix + "_disk.index") < SECTOR_LEN ||
        std::filesystem::file_size(prefix + "_pq_compressed.bin") < 8ull + uint64_t(pq_count) * pq_dim ||
        std::filesystem::file_size(prefix + "_disk.index.tags") < 8ull + uint64_t(tag_count) * sizeof(uint32_t))
      throw py::value_error("Corrupt or incompatible SSD index files");
    if (std::filesystem::exists(prefix + "_disk.index.id2loc") &&
        std::filesystem::file_size(prefix + "_disk.index.id2loc") < uint64_t(pq_count) * sizeof(uint32_t))
      throw py::value_error("Corrupt SSD location mapping");
    auto result = std::unique_ptr<PySSDIndex>(new PySSDIndex(prefix, dim, metric, threads));
    result->open();
    size_t count, loaded_tag_dim;
    std::vector<uint32_t> persisted_tags;
    ccann::load_bin<uint32_t>(prefix + "_disk.index.tags", persisted_tags, count, loaded_tag_dim);
    result->known_tags_.insert(persisted_tags.begin(), persisted_tags.end());
    std::ifstream removed(prefix + "_ccann.removed");
    uint32_t tag;
    while (removed >> tag) {
      result->removed_.insert(tag);
      result->index_->lazy_delete(tag);
    }
    return result;
  }

  void add(Vectors vector, uint32_t tag) {
    validate_vector(vector);
    std::lock_guard<std::mutex> lock(mu_);
    if (!known_tags_.insert(tag).second)
      throw py::value_error("Tag already exists");
    index_->insert(vector.data(), tag);
  }

  py::tuple search(Vectors query, uint32_t k, uint32_t search_l) {
    validate_vector(query);
    if (k == 0 || search_l < k || search_l > 4096)
      throw py::value_error("Require 0 < k <= search_l <= 4096");
    py::array_t<uint32_t> ids(k);
    py::array_t<float> distances(k);
    const auto *query_data = query.data();
    auto *ids_data = ids.mutable_data();
    auto *distances_data = distances.mutable_data();
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mu_);
      index_->search(query_data, k, 0, search_l, 4, ids_data, distances_data, nullptr);
    }
    return py::make_tuple(ids, distances);
  }

  void remove(uint32_t tag) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!known_tags_.count(tag))
      throw py::value_error("Tag does not exist");
    removed_.insert(tag);
    index_->lazy_delete(tag);
    persist_removed();
  }

  void save() {
    std::lock_guard<std::mutex> lock(mu_);
    index_->_disk_index->flush_commits();
    persist_removed();
  }

  uint64_t npoints() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_->_disk_index->cur_id.load() - removed_.size();
  }

  uint32_t dimension() const { return dim_; }

 private:
  PySSDIndex(std::string prefix, uint32_t dim, ccann::Metric metric, uint32_t threads)
      : prefix_(std::move(prefix)), dim_(dim), metric_(metric), threads_(threads) {}

  void persist_removed() {
    std::string pending = prefix_ + "_ccann.removed.tmp";
    {
      std::ofstream file(pending, std::ios::trunc);
      for (uint32_t tag : removed_)
        file << tag << '\n';
      if (!file)
        throw std::runtime_error("Failed to write removed tags");
    }
    int fd = ::open(pending.c_str(), O_RDONLY);
    if (fd < 0 || ::fsync(fd) != 0) {
      if (fd >= 0) ::close(fd);
      throw std::runtime_error("Failed to sync removed tags");
    }
    ::close(fd);
    std::filesystem::rename(pending, prefix_ + "_ccann.removed");
    auto parent = std::filesystem::path(prefix_).parent_path();
    if (parent.empty()) parent = ".";
    fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0 || ::fsync(fd) != 0) {
      if (fd >= 0) ::close(fd);
      throw std::runtime_error("Failed to sync index directory");
    }
    ::close(fd);
  }

  void validate_vector(const Vectors &vector) const {
    if (vector.ndim() != 1 || vector.shape(0) != dim_)
      throw py::value_error("Vector dimension does not match the index");
  }

  void open() {
    ccann::Parameters params;
    params.Set<unsigned>("L_disk", 64);
    params.Set<unsigned>("R_disk", 0);
    params.Set<float>("alpha_disk", 1.2f);
    params.Set<unsigned>("C", 384);
    params.Set<unsigned>("beamwidth", 4);
    params.Set<unsigned>("nodes_to_cache", 0);
    params.Set<unsigned>("num_threads", threads_);
    if (metric_ == ccann::Metric::L2)
      distance_ = std::make_unique<ccann::DistanceL2>();
    else
      distance_ = std::make_unique<ccann::DistanceCosineFloat>();
    index_ = std::make_unique<ccann::DynamicSSDIndex<float, uint32_t>>(
        params, prefix_, prefix_ + "_merge", distance_.get(), metric_, BEAM_SEARCH, false, true, threads_);
    if (index_->_disk_index->data_dim != dim_)
      throw py::value_error("Stored index dimension does not match metadata");
  }

  std::string prefix_;
  uint32_t dim_;
  ccann::Metric metric_;
  uint32_t threads_;
  mutable std::mutex mu_;
  std::set<uint32_t> removed_;
  std::set<uint32_t> known_tags_;
  std::unique_ptr<ccann::Distance<float>> distance_;
  std::unique_ptr<ccann::DynamicSSDIndex<float, uint32_t>> index_;
};

PYBIND11_MODULE(_native, m) {
  m.doc() = "CCANN SSD vector index";
  py::enum_<ccann::Metric>(m, "Metric")
      .value("L2", ccann::Metric::L2)
      .value("COSINE", ccann::Metric::COSINE);
  py::class_<PySSDIndex>(m, "Index")
      .def_static("create", &PySSDIndex::create, py::arg("prefix"), py::arg("vectors").noconvert(),
                  py::arg("tags").noconvert(), py::arg("metric") = ccann::Metric::L2, py::arg("threads") = 4)
      .def_static("load", &PySSDIndex::load, py::arg("prefix"),
                  py::arg("metric") = ccann::Metric::L2, py::arg("threads") = 4)
      .def("add", &PySSDIndex::add, py::arg("vector").noconvert(), py::arg("tag"),
           py::call_guard<py::gil_scoped_release>())
      .def("search", &PySSDIndex::search, py::arg("query").noconvert(), py::arg("k"),
           py::arg("search_l") = 64)
      .def("remove", &PySSDIndex::remove)
      .def("save", &PySSDIndex::save)
      .def_property_readonly("npoints", &PySSDIndex::npoints)
      .def_property_readonly("dimension", &PySSDIndex::dimension);
}
