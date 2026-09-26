#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>
#include "aligned_file_io.h"
#include "v2/lock_table.h"

class WindowsAlignedFileIO : public AlignedFileIO {
 public:
  WindowsAlignedFileIO() = default;
  ~WindowsAlignedFileIO() override;

  void *get_ctx(int flag = 0) override;
  void open(const std::string &fname, bool enable_writes, bool enable_create) override;
  void close() override;
  void read(std::vector<IORequest> &reqs, void *ctx, bool async = false) override;
  void write(std::vector<IORequest> &reqs, void *ctx, bool async = false) override;
  void read_fd(int fd, std::vector<IORequest> &reqs, void *ctx) override;
  void write_fd(int fd, std::vector<IORequest> &reqs, void *ctx) override;
  void atomic_truncate(uint64_t size) override;
  void sync() override;
  uint64_t file_size() override;
  void init_dax(uint64_t hint_size) override;
  void exit_dax() override;
  void *get_dax(uint64_t hint_size, bool init) override;
  void put_dax() override;
  void flush_dax(void *p, uint64_t size) override;
  void barrier_dax() override;
  bool check_addr_in_pm(const void *addr) override;
  void read_alloc(std::vector<IORequest> &reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr) override;
  void send_io(IORequest &req, void *ctx, bool write) override;
  void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) override;
  int send_read_no_alloc(IORequest &req, void *ctx) override;
  int send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx) override;
  int poll(void *ctx) override;
  void poll_all(void *ctx) override;
  void poll_wait(void *ctx) override;

 protected:
  void register_thread(int flag = 0) override;
  void deregister_thread() override;
  void deregister_all_threads() override;

 private:
  void transfer(HANDLE handle, IORequest &req, bool write);
  void unmap_locked();
  void flush_locked();
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
  void *mapped_ = nullptr;
  uint64_t mapped_size_ = 0;
  std::string path_;
  std::shared_mutex mapping_mutex_;
  std::mutex file_mutex_;
};

namespace v2 {
  inline std::vector<uint64_t> lockReqs(SparseLockTable<uint64_t> &table, std::vector<IORequest> &reqs) {
    std::vector<uint64_t> keys;
    for (const auto &req : reqs)
      for (uint64_t i = 0; i < req.len; i += SECTOR_LEN)
        keys.push_back((req.offset + i) / SECTOR_LEN);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (auto key : keys) table.wrlock(key);
    return keys;
  }
  inline void unlockReqs(SparseLockTable<uint64_t> &table, std::vector<uint64_t> &keys) {
    for (auto key : keys) table.unlock(key);
  }
}  // namespace v2
#endif
