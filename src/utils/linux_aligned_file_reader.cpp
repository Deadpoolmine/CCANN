#include "libpmem.h"
#ifndef USE_AIO
#include "linux_aligned_file_reader.h"

#include <filesystem>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sys/mman.h>
#include "aligned_file_reader.h"
#include "liburing.h"
#include <urcu.h>

#define MAX_EVENTS 256

namespace {
  constexpr uint64_t kNoUserData = 0;
  uint64_t write_time = 0;  // in ns
  uint64_t read_time = 0;   // in ns

  bool parallel_io(void *context, int fd, std::vector<IORequest>::iterator start, std::vector<IORequest>::iterator end,
                   uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    bool fail = false;
    uint64_t retries = 0;
    while (true) {
      for (auto it = start; it != end; ++it) {
        auto &req = *it;
        auto sqe = io_uring_get_sqe(ring);
        // if (write)
        //   LOG(INFO) << "[parallel] submit offset: " << req.offset << " len: " << req.len;
        sqe->user_data = (uint64_t) (&req);
        if (write) {
          io_uring_prep_write(sqe, fd, req.buf, req.len, req.offset);
        } else {
          io_uring_prep_read(sqe, fd, req.buf, req.len, req.offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      fail = false;
      for (auto it = start; it != end; ++it) {
        auto &req = *it;
        int ret = 0;
        do {
          ret = io_uring_wait_cqe(ring, &cqe);
        } while (ret == -EINTR);

        if (ret < 0 || cqe->res < 0) {
          fail = true;
          LOG(ERROR) << "Failed " << strerror(-ret) << " " << strerror(-cqe->res) << " " << ring << " " << req.buf
                     << " " << req.len << " " << req.offset;
          break;  // CQE broken.
        }
        io_uring_cqe_seen(ring, cqe);
        // auto req = (IORequest *) cqe->user_data;
        // if (write)
        //   LOG(INFO) << "[write] complete offset: " << req->offset << " len: " << req->len;
      }

      if (fail) {
        retries++;
        if (retries > n_retries) {
          break;
        }
      }

      if (!fail) {  // repeat until no fails.
        break;
      }
    }
    return fail;
  }

  bool serial_io(void *context, int fd, std::vector<IORequest>::iterator start, std::vector<IORequest>::iterator end,
                 uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    bool fail = false;
    uint64_t retries = 0;
    while (true) {
      for (auto it = start; it != end; ++it) {
        if (write) {
          int ret = pwrite(fd, it->buf, it->len, it->offset);
          if (ret < 0) {
            fail = true;
            LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << it->buf << " " << it->len << " "
                       << it->offset;
            break;
          }
        } else {
          int ret = pread(fd, it->buf, it->len, it->offset);
          if (ret < 0) {
            fail = true;
            LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << it->buf << " " << it->len << " "
                       << it->offset;
            break;
          }
        }
      }

      // add fsync here for ordered persistence
      if (write) {
        fsync(fd);
      }

      if (fail) {
        retries++;
        if (retries > n_retries) {
          crash();
          break;
        }
      }

      if (!fail) {  // repeat until no fails.
        break;
      }
    }
    return fail;
  }

  void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    // LOG(INFO) << "Executing " << (write ? "write" : "read") << " of " << reqs.size() << " requests using "
    //           << (ring != nullptr ? "io_uring" : "syscalls");
    if (write) {
      timespec ts_start, ts_end;
      clock_gettime(CLOCK_REALTIME, &ts_start);
#ifdef CC_ANN
      // the last req of reqs
      serial_io(context, fd, reqs.end() - 1, reqs.end(), n_retries, write);
      parallel_io(context, fd, reqs.begin(), reqs.end() - 1, n_retries, write);
#else
      parallel_io(context, fd, reqs.begin(), reqs.end(), n_retries, write);
#endif
      clock_gettime(CLOCK_REALTIME, &ts_end);
      write_time += (ts_end.tv_sec - ts_start.tv_sec) * 1e9 + (ts_end.tv_nsec - ts_start.tv_nsec);
    } else {
      timespec ts_start, ts_end;
      clock_gettime(CLOCK_REALTIME, &ts_start);
      parallel_io(context, fd, reqs.begin(), reqs.end(), n_retries, write);
      clock_gettime(CLOCK_REALTIME, &ts_end);
      read_time += (ts_end.tv_sec - ts_start.tv_sec) * 1e9 + (ts_end.tv_nsec - ts_start.tv_nsec);
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
  this->pm_region.store(nullptr);
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;

  synchronize_rcu();
  rcu_barrier();

  if (this->pm_region != nullptr) {
    this->exit_dax();
  }

  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }

  // io stats
  LOG(INFO) << "Total read time (ns): " << read_time;
  LOG(INFO) << "Total write time (ns): " << write_time;
}

struct io_uring_ctx {
  io_uring *ring;
  ~io_uring_ctx() {
    LOG(INFO) << "Deregistering thread from io_uring.";
    // destroy io_uring
    io_uring_queue_exit(ring);
    delete ring;
    ring = nullptr;
  }
};

namespace ioctx {
  // static thread_local io_uring *ring = nullptr;
  static thread_local std::unique_ptr<io_uring_ctx> ring = nullptr;
};  // namespace ioctx

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring.get()->ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = std::make_unique<io_uring_ctx>();
    ioctx::ring->ring = new io_uring();
    int ret = io_uring_queue_init(MAX_EVENTS, ioctx::ring.get()->ring, flag);
    if (ret < 0) {
      LOG(WARN) << "SQPOLL unsupported or no permission: " << strerror(-ret);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  // io_uring_queue_exit(ioctx::ring);
  // delete ioctx::ring;
  // ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  if (this->file_desc == -1) {
    LOG(ERROR) << "Failed to open file: " << fname << " : " << strerror(errno);
    crash();
  }
  this->file_path = fname;
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  auto current_region = this->pm_region.load();
  auto on_pm = (current_region != nullptr);
  bool fail = false;
  if (on_pm) {
    auto pm_addr = current_region->addr;
    auto pm_size = current_region->size;
    for (auto &req : read_reqs) {
      // assert(req.offset + req.len <= pm_size);
      if (req.offset + req.len > pm_size) {
        fail = true;
        // LOG(WARN) << "PM read exceeds mapped region. req offset: " << req.offset << " len: " << req.len
        //            << " pm_size: " << pm_size;
        // LOG(WARN) << "Falling back to syscall read.";
        break;
      }
      // TODO: using fine-grained unalighed I/O here
      pmem_memcpy(req.buf, (uint8_t *) pm_addr + req.offset, req.len, 0);
    }
  } else {
    execute_io(ctx, this->file_desc, read_reqs);
  }

  if (fail) {
    execute_io(ctx, this->file_desc, read_reqs);
  }

  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 100, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::atomic_truncate(uint64_t size) {
  assert(this->file_desc != -1);
  int ret = ftruncate(this->file_desc, size);
  if (ret != 0) {
    LOG(ERROR) << "Failed to truncate file to size " << size << " : " << strerror(errno);
  }
  fsync(this->file_desc);
}

void LinuxAlignedFileReader::sync() {
  assert(this->file_desc != -1);
  int ret = fsync(this->file_desc);
}

#define EXTEND_FACTOR (1.5)

void *LinuxAlignedFileReader::mmap_hint(IORequest &req, bool write, uint64_t &out_size) {
  assert(this->file_desc != -1);

  int prot = PROT_READ;

  if (write) {
    prot |= PROT_WRITE;
  }

  uint64_t length = req.len;
  uint64_t offset = req.offset;
  uint64_t extend_size = EXTEND_FACTOR * length;

  // check file size first
  struct stat st;
  if (fstat(this->file_desc, &st) != 0) {
    LOG(ERROR) << "Failed to fstat file : " << strerror(errno);
    return nullptr;
  }

  auto exact_allocate = offset + length - st.st_size;
  auto extend_allocate = offset + extend_size - st.st_size;
  // check file system remaining size
  uintmax_t free = std::filesystem::space(this->file_path).free;
  if (write && offset + length > st.st_size && extend_allocate > free) {
    if (free < exact_allocate) {
      LOG(ERROR) << "Not enough space to extend file for mmap: required extra " << exact_allocate << " free " << free << " offset: " << offset << " length: " << length << " file size: " << st.st_size;
      crash();
      return nullptr;
    }
    // try 10G
    extend_size = length + (10ULL << 30);
    LOG(WARN) << "Not enough space to extend file for mmap: try extend to " << extend_size;
    extend_allocate = offset + extend_size - st.st_size;
    if (free < extend_allocate) {
      extend_size = length; // no extend
      LOG(WARN) << "Not enough space to extend file for mmap: disable extension";
    }
  }

  if (offset + length > (uint64_t) st.st_size) {
    if (write) {
      // use fallocate to extend the file
      size_t aligned_size = ROUND_UP(offset + extend_size, SECTOR_LEN);
      int ret = posix_fallocate(this->file_desc, st.st_size, aligned_size - st.st_size);
      // Ensure the new space is zeroed.
      // int ret = fallocate(this->file_desc, FALLOC_FL_ZERO_RANGE, st.st_size, aligned_size - st.st_size);
      LOG(INFO) << "Preallocate file to size " << aligned_size;
      if (ret != 0) {
        LOG(WARN) << "Failed to fallocate file to size " << aligned_size << " : " << strerror(ret);
      }
      length = aligned_size - offset;
    } else {
      LOG(ERROR) << "Failed to mmap file: offset + length exceeds file size.";
      crash();
      return nullptr;
    }
  }

  out_size = length;

  void *addr = ::mmap(nullptr, length, prot, MAP_SHARED | MAP_POPULATE, this->file_desc, offset);
  if (addr == MAP_FAILED) {
    LOG(ERROR) << "Failed to mmap file : " << strerror(errno);
    return nullptr;
  }

  LOG(INFO) << "mapped addr: " << addr << " len: " << length << " offset: " << offset;
  return addr;
}

uint64_t LinuxAlignedFileReader::file_size() {
  assert(this->file_desc != -1);
  struct stat st;
  uint64_t size;

  if (fstat(this->file_desc, &st) != 0) {
    LOG(ERROR) << "Failed to fstat file : " << strerror(errno);
    size = 0;
    return size;
  }
  size = st.st_size;

  return size;
}

void LinuxAlignedFileReader::unmap_dax_region(struct rcu_head *rcu) {
  dax_region *pm_region = caa_container_of(rcu, dax_region, rcu);
  IORequest um_req(0, pm_region->size, nullptr, 0, 0);
  munmap(pm_region->addr, um_req);
  delete pm_region;
}

void unmap_dax_region_callback(struct rcu_head *rcu) {
  LinuxAlignedFileReader::dax_region *pm_region = caa_container_of(rcu, LinuxAlignedFileReader::dax_region, rcu);
  pm_region->owner->unmap_dax_region(rcu);
}

void LinuxAlignedFileReader::munmap(void *addr, IORequest &req) {
  uint64_t length = req.len;
  uint64_t offset = req.offset;

  if (addr == 0 && length == 0) {
    // this is the dummy req.
    return;
  }

  assert(this->file_desc != -1);

  int ret = ::munmap(addr, length);
  if (ret != 0) {
    LOG(ERROR) << "Failed to munmap file : " << strerror(errno);
  }
  LOG(INFO) << "munmaped addr: " << addr << " len: " << length << " offset: " << offset;
}

void *LinuxAlignedFileReader::get_dax(uint64_t hint_size, bool init = false) {
  bool need_alloc = false;

  rcu_register_thread();
  rcu_read_lock();

  auto current_region = this->pm_region.load();
  if (current_region == nullptr) {
    // first time allocation
    need_alloc = true;
  } else {
    if (hint_size > current_region->size) {
      need_alloc = true;
    }
  }

  if (init) {
    need_alloc = true;
  }

  if (need_alloc) {
    uint64_t out_size = 0;
    IORequest mmap_req(0, hint_size, nullptr, 0, 0);
    auto addr = this->mmap_hint(mmap_req, true, out_size);
    auto new_region = new dax_region{addr, out_size, this};
    if (this->pm_region.compare_exchange_weak(current_region, new_region)) {
      // success
      if (current_region) {
        if (current_region->addr != nullptr) {
          call_rcu(&current_region->rcu, unmap_dax_region_callback);
        }
      }
      current_region = this->pm_region.load();
    } else {
      // failed, another thread has expanded
      this->munmap(addr, mmap_req);
      delete new_region;
      // current_region is safe as call_rcu ensures the old region
      // is not freed until all readers are done.
    }
  }

  return current_region->addr;
}

void LinuxAlignedFileReader::put_dax() {
  rcu_read_unlock();
  rcu_unregister_thread();
}

void LinuxAlignedFileReader::init_dax(uint64_t hint_size) {
  get_dax(hint_size, true);
  put_dax();
}

void LinuxAlignedFileReader::exit_dax() {
  if (this->pm_region != nullptr) {
    auto current_region = this->pm_region.load();
    this->unmap_dax_region(&current_region->rcu);
  }
}

void LinuxAlignedFileReader::flush_dax(void *p, uint64_t size) {
  uint64_t flush_num = ROUND_UP(size, 64) / 64;
  for (uint64_t i = 0; i < flush_num; i++) {
    _mm_clwb((char *) p + i * 64);
  }
}

void LinuxAlignedFileReader::barrier_dax() {
  _mm_sfence();
}

bool LinuxAlignedFileReader::check_addr_in_pm(const void *addr) {
  if (this->pm_region.load() == nullptr) {
    return false;
  }

  auto pm_region = this->pm_region.load();
  auto pm_addr = pm_region->addr;
  auto pm_size = pm_region->size;
  if (pm_addr == nullptr) {
    return false;
  }
  if (addr >= pm_addr && addr < (void *) ((char *) pm_addr + pm_size)) {
    return true;
  }
  return false;
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  bool on_pm = this->pm_region.load() != nullptr;
  if (!on_pm) {
    auto sqe = io_uring_get_sqe(ring);
    req.finished = false;
    sqe->user_data = (uint64_t) &req;
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
    }
    io_uring_submit(ring);
  } else {
    auto u_ofs = req.u_offset;
    auto u_len = req.u_len;
    auto buf_ofs = u_ofs - req.offset;
    // io_uring_prep_read(sqe, this->file_desc, (char *)req.buf + buf_ofs, u_len, u_ofs);
    // sqe->flags |= IOSQE_ASYNC;
#ifdef NO_PM_READ_OPT
    pread(this->file_desc, req.buf, req.len, req.offset);
#else
    // pread(this->file_desc, (char *)req.buf + buf_ofs, u_len, u_ofs);
    auto pm_addr = this->pm_region.load()->addr;
    for (uint64_t i = 0; i < u_len; i += 64) {
      _mm_prefetch((char *) pm_addr + u_ofs + i, _MM_HINT_T2);
    }
    // LOG(INFO) << "PM read offset: " << u_ofs << " len: " << u_len;
    pmem_memcpy((char *) req.buf + buf_ofs, (char *) pm_addr + u_ofs, u_len, 0);
#endif
    req.finished = true;
  }
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    reqs[j].finished = false;
    sqe->user_data = (uint64_t) &reqs[j];
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
  }
  io_uring_submit(ring);
}

int LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = io_uring_peek_cqe(ring, &cqe);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = 0;
  do {
    ret = io_uring_wait_cqe(ring, &cqe);
  } while (ret == -EINTR);
  if (ret < 0 || cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
}

#else
#include "linux_aligned_file_reader.h"

#include <libaio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 256

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb iocb_t;

  void execute_io(void *ctx, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS), (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *> cbs(n_ops, nullptr);
      std::vector<io_event_t> evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                        reqs[j + iter * MAX_EVENTS].offset);
        }
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno << "="
                     << ::strerror((int) -ret) << ", try #" << n_tries + 1 << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents((io_context_t) ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            LOG(ERROR) << "io_getevents() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                       << "=" << ::strerror((int) -ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_context_t ctx;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ctx == nullptr)) {
    register_thread(flag);
  }
  return (void *) ioctx::ctx;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ctx == nullptr) {
    int ret = io_setup(MAX_EVENTS, &ioctx::ctx);
    if (ret != 0) {
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_destroy((io_context_t) this->get_ctx());
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  uint64_t n_ops = std::min(reqs.size(), (uint64_t) MAX_EVENTS);
  std::vector<iocb_t *> cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (uint64_t j = 0; j < n_ops; j++) {
    if (write) {
      io_prep_pwrite(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_prep_pread(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    reqs[j].finished = false;  // reset finished flag
  }

  for (uint64_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  // issue reads
  int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
  // if requests didn't get accepted
  if (ret != (int64_t) n_ops) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", " << strerror(errno);
  }
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  iocb_t cb;
  req.finished = false;  // reset finished flag
  if (write) {
    io_prep_pwrite(&cb, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_prep_pread(&cb, this->file_desc, req.buf, req.len, req.offset);
  }
  cb.data = (void *) &req;  // set user data to point to the request

  iocb_t *cbs[1] = {&cb};  // create an array of iocb_t pointers
  int ret = io_submit((io_context_t) ctx, 1, cbs);
  if (ret != 1) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  // Poll a single completed IO request in the io_uring context.
  io_event event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, 1, &event, nullptr);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (ret) {
    IORequest *req = (IORequest *) event.data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  // Poll all completed IO requests in the io_uring context.
  static __thread io_event_t evts[MAX_EVENTS];
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_event_t event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 1, 1, &event, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  IORequest *req = (IORequest *) event.data;
  if (req != nullptr) {
    req->finished = true;
  }
}

#endif

int LinuxAlignedFileReader::send_read_no_alloc(IORequest &req, void *ring) {
#ifdef PIPE_PM_READS
  auto send_io_async = [this](IORequest &r, void *ctx) {
    io_uring *ring = (io_uring *) ctx;
    auto sqe = io_uring_get_sqe(ring);
    r.finished = false;
    sqe->user_data = (uint64_t) &r;
    io_uring_prep_read(sqe, this->file_desc, r.buf, r.len, r.offset);
    io_uring_submit(ring);
  };
#endif

#ifndef READ_ONLY_TESTS
  if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
#ifdef PIPE_PM_READS
    send_io_async(req, ring);
#else
    send_io(req, ring, false);
#endif
  } else {
    req.finished = true;  // mark as finished for cache miss
  }
#else
#ifdef PIPE_PM_READS
  send_io_async(req, ring);
#else
  send_io(req, ring, false);
#endif
#endif
  return 1;
}

int LinuxAlignedFileReader::send_read_no_alloc(std::vector<IORequest> &reqs, void *ring) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;
  // fetch from cache.
  for (auto &req : reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len != SECTOR_LEN) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
      disk_read_reqs.push_back(req);
    }
  }
  send_io(disk_read_reqs, ring, false);
  return disk_read_reqs.size();
#else
  send_io(reqs, ring, false);
  return reqs.size();
#endif
}

void LinuxAlignedFileReader::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;

  // TODO(gh): introduce size_per_io to cache.
  for (auto &req : read_reqs) {
    if (req.offset % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
      crash();
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true)) {
      disk_read_reqs.push_back(req);
    }
  }

  if (disk_read_reqs.size() > 0) {
    read(disk_read_reqs, ctx);
    for (auto &req : disk_read_reqs) {
      v2::cache.put(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true);
    }
  }

  // ref.
  if (page_ref != nullptr) {
    for (auto &req : read_reqs) {
      page_ref->push_back(req.offset / SECTOR_LEN);
    }
  }
#else
  read(read_reqs, ctx);
#endif
}