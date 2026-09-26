#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <exception>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <climits>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "distance.h"
#include "linux_aligned_file_io.h"
#include "utils.h"
#include "v2/dynamic_index.h"

namespace py = pybind11;

namespace {
#ifdef _WIN32
int open_file(const std::string &path, int flags, int mode = 0) {
  return ::_open(path.c_str(), flags | _O_BINARY, mode);
}
int close_file(int fd) { return ::_close(fd); }
int sync_fd(int fd) { return ::_commit(fd); }
int64_t seek_file(int fd, int64_t offset, int origin) { return ::_lseeki64(fd, offset, origin); }
int truncate_file(int fd, uint64_t size) {
  errno_t result = ::_chsize_s(fd, size);
  if (result != 0) errno = result;
  return result == 0 ? 0 : -1;
}
int write_file(int fd, const char *data, size_t size) {
  return ::_write(fd, data, static_cast<unsigned>(std::min<size_t>(size, INT_MAX)));
}
void replace_file(const std::string &source, const std::string &destination) {
  auto from = std::filesystem::u8path(source).wstring();
  auto to = std::filesystem::u8path(destination).wstring();
  if (!::MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                            "publish active generation");
}
#else
int open_file(const std::string &path, int flags, int mode = 0) {
  return ::open(path.c_str(), flags, mode);
}
int close_file(int fd) { return ::close(fd); }
int sync_fd(int fd) { return ::fsync(fd); }
int64_t seek_file(int fd, int64_t offset, int origin) { return ::lseek(fd, offset, origin); }
int truncate_file(int fd, uint64_t size) { return ::ftruncate(fd, size); }
ssize_t write_file(int fd, const char *data, size_t size) { return ::write(fd, data, size); }
void replace_file(const std::string &source, const std::string &destination) {
  std::filesystem::rename(source, destination);
}
#endif
}  // namespace

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
    if (std::filesystem::exists(prefix + "_disk.index") ||
        std::filesystem::exists(prefix + "_ccann.active") ||
        std::filesystem::exists(prefix + "_ccann.flat"))
      throw py::value_error("Index already exists at prefix");
    std::string data_path = prefix + "_ccann_data.bin";
    std::string tags_path = prefix + "_ccann_tags.bin";
    ccann::save_bin<float>(data_path, vectors.mutable_data(), vectors.shape(0), vectors.shape(1));
    ccann::save_bin<uint32_t>(tags_path, tags.mutable_data(), tags.shape(0), 1);
    auto chunks = std::min<int>(32, vectors.shape(1));
    bool built;
    {
      py::gil_scoped_release release;
      built = ccann::build_disk_index_py<float, uint32_t>(data_path.c_str(), prefix.c_str(), 32, 64, 1,
                                                          threads, chunks, metric, false, tags_path.c_str());
    }
    if (!built)
      throw std::runtime_error("SSD index build failed");
    for (const char *suffix : {"_disk.index", "_disk.index.tags", "_pq_compressed.bin", "_pq_pivots.bin"})
      sync_file(prefix + suffix);
    {
      std::ofstream meta(prefix + "_ccann.meta", std::ios::trunc);
      meta << vectors.shape(1) << ' ' << static_cast<int>(metric) << '\n';
      if (!meta)
        throw std::runtime_error("Failed to write index metadata");
    }
    sync_file(prefix + "_ccann.meta");
    sync_directory(prefix + "_ccann.meta");
    auto result = load(prefix, metric, threads);
    if (std::filesystem::exists(prefix + "_disk.index.id2loc"))
      sync_file(prefix + "_disk.index.id2loc");
    return result;
  }

  static std::unique_ptr<PySSDIndex> load(const std::string &prefix, ccann::Metric metric,
                                          uint32_t threads) {
    return load_at(prefix, read_generation(prefix), metric, threads);
  }

  static std::unique_ptr<PySSDIndex> load_at(const std::string &root, uint64_t generation,
                                             ccann::Metric metric, uint32_t threads) {
    std::string prefix = generation_prefix(root, generation);
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
    auto result = std::unique_ptr<PySSDIndex>(new PySSDIndex(root, prefix, generation, dim, metric, threads));
    result->open();
    size_t count, loaded_tag_dim;
    std::vector<uint32_t> persisted_tags;
    ccann::load_bin<uint32_t>(prefix + "_disk.index.tags", persisted_tags, count, loaded_tag_dim);
    result->known_tags_.insert(persisted_tags.begin(), persisted_tags.end());
    auto removed_tags = read_removed(prefix + "_ccann.removed");
#ifdef _WIN32
    std::fprintf(stderr, "Replaying %zu deleted tags\n", removed_tags.size());
    size_t replayed = 0;
#endif
    for (uint32_t tag : removed_tags) {
      if (!result->known_tags_.count(tag))
        throw py::value_error("Deletion log contains an unknown tag");
      result->removed_.insert(tag);
      result->index_->lazy_delete(tag);
#ifdef _WIN32
      if (++replayed % 500 == 0) std::fprintf(stderr, "Replayed %zu deleted tags\n", replayed);
#endif
    }
#ifdef _WIN32
    std::fprintf(stderr, "Deletion replay complete\n");
#endif
    result->compact_if_needed();
    return result;
  }

  void add(Vectors vector, uint32_t tag) {
    validate_vector(vector);
    std::lock_guard<std::mutex> lock(mu_);
    if (!known_tags_.insert(tag).second)
      throw py::value_error("Tag already exists");
    index_->insert(vector.data(), tag);
    compact_if_needed();
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
    if (removed_.count(tag)) return;
    append_removed(tag);
    removed_.insert(tag);
    index_->lazy_delete(tag);
    compact_if_needed();
  }

  void merge() {
    std::lock_guard<std::mutex> lock(mu_);
    compact_if_needed(true);
  }

  void merge_to(const std::string &output_prefix) {
    std::lock_guard<std::mutex> lock(mu_);
    if (index_->_disk_index->cur_id.load() == removed_.size())
      throw py::value_error("Cannot merge an index with no live points");
    write_merged(output_prefix);
  }

  void compact_if_needed(bool force = false) {
    constexpr size_t kMergeThreshold = 10000;
    if (!force && removed_.size() < kMergeThreshold) return;
    if (index_->_disk_index->cur_id.load() == removed_.size()) {
      if (force) throw py::value_error("Cannot merge an index with no live points");
      return;
    }
    uint64_t next_generation = generation_ + 1;
    std::string output_prefix = generation_prefix(root_prefix_, next_generation);
    while (std::filesystem::exists(output_prefix + "_disk.index") ||
           std::filesystem::exists(output_prefix + "_ccann.meta")) {
      output_prefix = generation_prefix(root_prefix_, ++next_generation);
    }
    write_merged(output_prefix);
    auto next = load_at(root_prefix_, next_generation, metric_, threads_);
    sync_file(output_prefix + "_disk.index.id2loc");
    bool published = false;
    std::exception_ptr publication_error;
    try {
      publish_generation(next_generation, published);
    } catch (...) {
      if (!published) throw;
      publication_error = std::current_exception();
    }
    std::string old_prefix = prefix_;
    index_.swap(next->index_);
    distance_.swap(next->distance_);
    prefix_.swap(next->prefix_);
    known_tags_.swap(next->known_tags_);
    removed_.clear();
    generation_ = next_generation;
    next.reset();
    if (!publication_error) cleanup_generation(old_prefix);
    if (publication_error) std::rethrow_exception(publication_error);
  }

  void write_merged(const std::string &output_prefix) {
    if (output_prefix == root_prefix_ || output_prefix == prefix_ ||
        std::filesystem::exists(output_prefix + "_disk.index") ||
        std::filesystem::exists(output_prefix + "_ccann.meta") ||
        std::filesystem::exists(output_prefix + "_ccann.active") ||
        std::filesystem::exists(output_prefix + "_ccann.flat"))
      throw py::value_error("Merge output prefix already exists");
    index_->_disk_index->flush_commits();
    auto deleted = read_removed(prefix_ + "_ccann.removed");
    std::vector<uint32_t> tags(deleted.begin(), deleted.end());
    std::unordered_set<uint32_t> tag_set(deleted.begin(), deleted.end());
    index_->_disk_index->merge_deletes(prefix_, output_prefix, tags, tag_set, threads_, 20);
    for (const char *suffix : {"_disk.index", "_disk.index.tags", "_pq_compressed.bin", "_pq_pivots.bin"})
      sync_file(output_prefix + suffix);
    std::string meta_path = output_prefix + "_ccann.meta";
    int fd = open_file(meta_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open merged metadata");
    std::string meta = std::to_string(dim_) + " " + std::to_string(static_cast<int>(metric_)) + "\n";
    try {
      write_all(fd, meta);
      if (sync_fd(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "sync merged metadata");
    } catch (...) {
      close_file(fd);
      throw;
    }
    close_file(fd);
    sync_directory(meta_path);
  }

  uint64_t npoints() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_->_disk_index->cur_id.load() - removed_.size();
  }

  uint32_t dimension() const { return dim_; }

 private:
  PySSDIndex(std::string root, std::string prefix, uint64_t generation, uint32_t dim,
             ccann::Metric metric, uint32_t threads)
      : root_prefix_(std::move(root)), prefix_(std::move(prefix)), generation_(generation),
        dim_(dim), metric_(metric), threads_(threads) {}

  static std::string generation_prefix(const std::string &root, uint64_t generation) {
    return generation == 0 ? root : root + "_ccann_gen_" + std::to_string(generation);
  }

  static uint64_t read_generation(const std::string &root) {
    std::ifstream file(root + "_ccann.active");
    if (!file) {
      if (std::filesystem::exists(root + "_ccann.active"))
        throw std::runtime_error("Failed to read active index generation");
      return 0;
    }
    std::string line, extra;
    if (!std::getline(file, line) || std::getline(file, extra))
      throw py::value_error("Corrupt active index generation");
    size_t parsed = 0;
    uint64_t generation;
    try {
      generation = std::stoull(line, &parsed);
    } catch (const std::exception &) {
      throw py::value_error("Corrupt active index generation");
    }
    if (generation == 0 || parsed != line.size())
      throw py::value_error("Corrupt active index generation");
    return generation;
  }

  void publish_generation(uint64_t generation, bool &published) {
    std::string path = root_prefix_ + "_ccann.active";
    std::string temporary = path + ".tmp";
    int fd = open_file(temporary, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open active generation");
    try {
      write_all(fd, std::to_string(generation) + "\n");
      if (sync_fd(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "sync active generation");
    } catch (...) {
      close_file(fd);
      throw;
    }
    close_file(fd);
    replace_file(temporary, path);
    published = true;
    sync_directory(path);
  }

  static void cleanup_generation(const std::string &prefix) {
    for (const char *suffix : {"_disk.index", "_disk.index.tags", "_disk.index.id2loc",
                               "_pq_compressed.bin", "_pq_pivots.bin", "_partition.bin.aligned",
                               "_ccann.meta", "_ccann.removed"}) {
      std::error_code error;
      std::filesystem::remove(prefix + suffix, error);
    }
    sync_directory(prefix);
  }

  static void write_all(int fd, const std::string &data) {
    size_t offset = 0;
    while (offset < data.size()) {
      auto written = write_file(fd, data.data() + offset, data.size() - offset);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0)
        throw std::system_error(written < 0 ? errno : EIO, std::generic_category(), "write index file");
      offset += written;
    }
  }

  static void sync_file(const std::string &path) {
#ifdef _WIN32
    int fd = open_file(path, O_RDWR);
#else
    int fd = open_file(path, O_RDONLY);
#endif
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open index file for sync");
    int result = sync_fd(fd);
    int saved_errno = errno;
    close_file(fd);
    if (result != 0)
      throw std::system_error(saved_errno, std::generic_category(), "sync index file");
  }

  static void sync_directory(const std::string &path) {
#ifdef _WIN32
    // Windows does not support POSIX directory fsync; generation publication uses
    // a flushed temporary file and a write-through replacement instead.
    (void)path;
#else
    auto parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) parent = ".";
    int fd = open_file(parent.string(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open index directory");
    int result = sync_fd(fd);
    int saved_errno = errno;
    close_file(fd);
    if (result != 0)
      throw std::system_error(saved_errno, std::generic_category(), "sync index directory");
#endif
  }

  static std::set<uint32_t> read_removed(const std::string &path) {
    std::set<uint32_t> tags;
    if (!std::filesystem::exists(path)) return tags;
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open deletion log");
    uint64_t valid_size = 0;
    std::string line;
    while (std::getline(file, line)) {
      if (file.eof()) break;
      size_t parsed = 0;
      uint64_t value;
      try {
        value = std::stoull(line, &parsed);
      } catch (const std::exception &) {
        throw py::value_error("Corrupt deletion log");
      }
      if (parsed != line.size() || value > std::numeric_limits<uint32_t>::max())
        throw py::value_error("Corrupt deletion log");
      tags.insert(static_cast<uint32_t>(value));
      valid_size += line.size() + 1;
    }
    if (file.bad()) throw std::runtime_error("Failed to read deletion log");
    file.close();
    if (valid_size != std::filesystem::file_size(path)) {
      int fd = open_file(path, O_RDWR);
      if (fd < 0) throw std::system_error(errno, std::generic_category(), "open deletion log for repair");
      int result = truncate_file(fd, valid_size);
      int saved_errno = errno;
      close_file(fd);
      if (result != 0) {
        errno = saved_errno;
        throw std::system_error(errno, std::generic_category(), "repair deletion log");
      }
      sync_file(path);
    }
    return tags;
  }

  void append_removed(uint32_t tag) {
    std::string path = prefix_ + "_ccann.removed";
    bool created = !std::filesystem::exists(path);
    int fd = open_file(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open deletion log");
    if (created) {
      try {
        sync_directory(path);
      } catch (...) {
        close_file(fd);
        throw;
      }
    }
    int64_t start = seek_file(fd, 0, SEEK_END);
    try {
      if (start < 0) throw std::system_error(errno, std::generic_category(), "seek deletion log");
      write_all(fd, std::to_string(tag) + "\n");
      if (sync_fd(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "sync deletion log");
    } catch (...) {
      if (start >= 0) {
        if (truncate_file(fd, start) == 0) sync_fd(fd);
      }
      close_file(fd);
      throw;
    }
    close_file(fd);
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

  std::string root_prefix_;
  std::string prefix_;
  uint64_t generation_;
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
      .def("remove", &PySSDIndex::remove, py::call_guard<py::gil_scoped_release>())
      .def("merge", &PySSDIndex::merge, py::call_guard<py::gil_scoped_release>())
      .def("merge_to", &PySSDIndex::merge_to, py::arg("output_prefix"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("npoints", &PySSDIndex::npoints)
      .def_property_readonly("dimension", &PySSDIndex::dimension);
}
