#include "windows_aligned_file_io.h"

#ifdef _WIN32
#include <io.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include "log.h"

namespace {
  struct MappingPin {
    WindowsAlignedFileIO *owner;
    std::shared_lock<std::shared_mutex> lock;
    unsigned depth;
  };
  thread_local std::vector<MappingPin> mapping_pins;

  MappingPin *find_pin(WindowsAlignedFileIO *owner) {
    for (auto &pin : mapping_pins)
      if (pin.owner == owner) return &pin;
    return nullptr;
  }

  [[noreturn]] void fail(const char *operation) {
    DWORD code = GetLastError();
    LOG(ERROR) << operation << " failed with Win32 error " << code;
    throw std::runtime_error(std::string(operation) + " failed with Win32 error " + std::to_string(code));
  }

  uint64_t checked_end(uint64_t offset, uint64_t length) {
    if (length > std::numeric_limits<uint64_t>::max() - offset)
      throw std::overflow_error("file offset overflow");
    return offset + length;
  }
}

WindowsAlignedFileIO::~WindowsAlignedFileIO() {
  try { close(); } catch (...) {}
}

void *WindowsAlignedFileIO::get_ctx(int) { return this; }
void WindowsAlignedFileIO::register_thread(int) {}
void WindowsAlignedFileIO::deregister_thread() {}
void WindowsAlignedFileIO::deregister_all_threads() {}

void WindowsAlignedFileIO::open(const std::string &fname, bool enable_writes, bool enable_create) {
  std::lock_guard<std::mutex> file_lock(file_mutex_);
  if (file_ != INVALID_HANDLE_VALUE) throw std::logic_error("file already open");
  (void) enable_writes;
  DWORD access = GENERIC_READ | GENERIC_WRITE;
  DWORD disposition = enable_create ? OPEN_ALWAYS : OPEN_EXISTING;
  auto wide_name = std::filesystem::u8path(fname).wstring();
  file_ = CreateFileW(wide_name.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                      disposition, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
  if (file_ == INVALID_HANDLE_VALUE) fail("CreateFileW");
  path_ = fname;
}

void WindowsAlignedFileIO::flush_locked() {
  if (mapped_ && !FlushViewOfFile(mapped_, 0)) fail("FlushViewOfFile");
  if (file_ != INVALID_HANDLE_VALUE && !FlushFileBuffers(file_)) fail("FlushFileBuffers");
}

void WindowsAlignedFileIO::unmap_locked() {
  if (mapped_) {
    flush_locked();
    if (!UnmapViewOfFile(mapped_)) fail("UnmapViewOfFile");
    mapped_ = nullptr;
    mapped_size_ = 0;
  }
  if (mapping_) {
    CloseHandle(mapping_);
    mapping_ = nullptr;
  }
}

void WindowsAlignedFileIO::close() {
  if (find_pin(this)) throw std::logic_error("put_dax required before close");
  std::unique_lock<std::shared_mutex> map_lock(mapping_mutex_);
  std::lock_guard<std::mutex> file_lock(file_mutex_);
  if (file_ == INVALID_HANDLE_VALUE) return;
  unmap_locked();
  if (!CloseHandle(file_)) fail("CloseHandle");
  file_ = INVALID_HANDLE_VALUE;
}

void WindowsAlignedFileIO::transfer(HANDLE handle, IORequest &req, bool write) {
  if (handle == INVALID_HANDLE_VALUE) throw std::logic_error("file is closed");
  if (!req.buf && req.len) throw std::invalid_argument("null I/O buffer");
  uint64_t end = checked_end(req.offset, req.len);
  (void) end;
  auto *buffer = static_cast<uint8_t *>(req.buf);
  uint64_t remaining = req.len;
  uint64_t offset = req.offset;
  req.finished = false;
  while (remaining) {
    DWORD length = static_cast<DWORD>(std::min<uint64_t>(remaining, 1ULL << 30));
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
    overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!overlapped.hEvent) fail("CreateEventW");
    DWORD completed = 0;
    BOOL started = write ? WriteFile(handle, buffer, length, nullptr, &overlapped)
                         : ReadFile(handle, buffer, length, nullptr, &overlapped);
    DWORD error = started ? ERROR_SUCCESS : GetLastError();
    BOOL success = (started || error == ERROR_IO_PENDING)
                   && GetOverlappedResult(handle, &overlapped, &completed, TRUE);
    DWORD result_error = success ? ERROR_SUCCESS : GetLastError();
    CloseHandle(overlapped.hEvent);
    if (!success) {
      if (!write && result_error == ERROR_HANDLE_EOF) {
        std::memset(buffer, 0, static_cast<size_t>(remaining));
        break;
      }
      SetLastError(result_error);
      fail(write ? "WriteFile" : "ReadFile");
    }
    if (completed == 0) {
      if (write) throw std::runtime_error("WriteFile made no progress");
      std::memset(buffer, 0, static_cast<size_t>(remaining));
      break;
    }
    buffer += completed;
    offset += completed;
    remaining -= completed;
  }
  req.finished = true;
}

void WindowsAlignedFileIO::read(std::vector<IORequest> &reqs, void *, bool) {
  std::lock_guard<std::mutex> lock(file_mutex_);
  for (auto &req : reqs) transfer(file_, req, false);
}

void WindowsAlignedFileIO::write(std::vector<IORequest> &reqs, void *, bool) {
  std::lock_guard<std::mutex> lock(file_mutex_);
  for (auto &req : reqs) transfer(file_, req, true);
}

void WindowsAlignedFileIO::read_fd(int fd, std::vector<IORequest> &reqs, void *) {
  auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) throw std::invalid_argument("invalid file descriptor");
  for (auto &req : reqs) transfer(handle, req, false);
}

void WindowsAlignedFileIO::write_fd(int fd, std::vector<IORequest> &reqs, void *) {
  auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) throw std::invalid_argument("invalid file descriptor");
  for (auto &req : reqs) transfer(handle, req, true);
}

uint64_t WindowsAlignedFileIO::file_size() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  LARGE_INTEGER size;
  if (file_ == INVALID_HANDLE_VALUE || !GetFileSizeEx(file_, &size)) fail("GetFileSizeEx");
  return static_cast<uint64_t>(size.QuadPart);
}

void WindowsAlignedFileIO::atomic_truncate(uint64_t size) {
  if (find_pin(this)) throw std::logic_error("put_dax required before truncate");
  std::unique_lock<std::shared_mutex> map_lock(mapping_mutex_);
  std::lock_guard<std::mutex> file_lock(file_mutex_);
  unmap_locked();
  LARGE_INTEGER offset;
  offset.QuadPart = static_cast<LONGLONG>(size);
  if (!SetFilePointerEx(file_, offset, nullptr, FILE_BEGIN) || !SetEndOfFile(file_))
    fail("SetEndOfFile");
  if (!FlushFileBuffers(file_)) fail("FlushFileBuffers");
}

void WindowsAlignedFileIO::sync() {
  if (find_pin(this)) {
    std::lock_guard<std::mutex> file_lock(file_mutex_);
    flush_locked();
    return;
  }
  std::shared_lock<std::shared_mutex> map_lock(mapping_mutex_);
  std::lock_guard<std::mutex> file_lock(file_mutex_);
  flush_locked();
}

void *WindowsAlignedFileIO::get_dax(uint64_t hint_size, bool) {
  hint_size = std::max<uint64_t>(hint_size, SECTOR_LEN);
  if (auto *pin = find_pin(this)) {
    if (hint_size > mapped_size_) throw std::logic_error("cannot grow a pinned mapping");
    ++pin->depth;
    return mapped_;
  }

  std::shared_lock<std::shared_mutex> read_lock(mapping_mutex_);
  if (mapped_ && hint_size <= mapped_size_) {
    mapping_pins.push_back({this, std::move(read_lock), 1});
    return mapped_;
  }
  read_lock.unlock();

  {
    std::unique_lock<std::shared_mutex> write_lock(mapping_mutex_);
    if (!mapped_ || hint_size > mapped_size_) {
      std::lock_guard<std::mutex> file_lock(file_mutex_);
      unmap_locked();
      LARGE_INTEGER size;
      if (!GetFileSizeEx(file_, &size)) fail("GetFileSizeEx");
      uint64_t target = std::max<uint64_t>(hint_size, static_cast<uint64_t>(size.QuadPart));
      if (target > static_cast<uint64_t>(std::numeric_limits<LONGLONG>::max()))
        throw std::overflow_error("mapping exceeds Windows file size limit");
      if (target > static_cast<uint64_t>(size.QuadPart)) {
        LARGE_INTEGER offset;
        offset.QuadPart = static_cast<LONGLONG>(target);
        if (!SetFilePointerEx(file_, offset, nullptr, FILE_BEGIN) || !SetEndOfFile(file_))
          fail("SetEndOfFile");
      }
      mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READWRITE, static_cast<DWORD>(target >> 32),
                                    static_cast<DWORD>(target), nullptr);
      if (!mapping_) fail("CreateFileMappingW");
      mapped_ = MapViewOfFile(mapping_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
      if (!mapped_) fail("MapViewOfFile");
      mapped_size_ = target;
    }
  }
  read_lock.lock();
  if (!mapped_ || hint_size > mapped_size_) {
    read_lock.unlock();
    return get_dax(hint_size, false);
  }
  mapping_pins.push_back({this, std::move(read_lock), 1});
  return mapped_;
}

void WindowsAlignedFileIO::put_dax() {
  for (auto it = mapping_pins.begin(); it != mapping_pins.end(); ++it) {
    if (it->owner == this) {
      if (--it->depth == 0) mapping_pins.erase(it);
      return;
    }
  }
  throw std::logic_error("put_dax without get_dax");
}

void WindowsAlignedFileIO::init_dax(uint64_t hint_size) {
  get_dax(hint_size, true);
  put_dax();
}
void WindowsAlignedFileIO::exit_dax() {
  if (find_pin(this)) throw std::logic_error("put_dax required before exit_dax");
  std::unique_lock<std::shared_mutex> lock(mapping_mutex_);
  std::lock_guard<std::mutex> file_lock(file_mutex_);
  unmap_locked();
}
void WindowsAlignedFileIO::flush_dax(void *p, uint64_t size) {
  if (size == 0) return;
  auto flush = [&] {
    uintptr_t begin = reinterpret_cast<uintptr_t>(mapped_);
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    if (!mapped_ || addr < begin || addr - begin > mapped_size_ ||
        size > mapped_size_ - (addr - begin))
      throw std::out_of_range("flush_dax range is outside mapped file");
    if (!FlushViewOfFile(mapped_, 0)) fail("FlushViewOfFile");
  };
  if (find_pin(this)) flush();
  else {
    std::shared_lock<std::shared_mutex> lock(mapping_mutex_);
    flush();
  }
}
void WindowsAlignedFileIO::barrier_dax() { sync(); }
bool WindowsAlignedFileIO::check_addr_in_pm(const void *addr) {
  auto contains = [&] {
    uintptr_t base = reinterpret_cast<uintptr_t>(mapped_);
    uintptr_t value = reinterpret_cast<uintptr_t>(addr);
    return mapped_ && value >= base && value - base < mapped_size_;
  };
  if (find_pin(this)) return contains();
  std::shared_lock<std::shared_mutex> lock(mapping_mutex_);
  return contains();
}

void WindowsAlignedFileIO::send_io(IORequest &req, void *, bool write) {
  std::lock_guard<std::mutex> lock(file_mutex_);
  transfer(file_, req, write);
}
void WindowsAlignedFileIO::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  if (write) this->write(reqs, ctx);
  else read(reqs, ctx);
}
int WindowsAlignedFileIO::poll(void *) { return 0; }
void WindowsAlignedFileIO::poll_all(void *) {}
void WindowsAlignedFileIO::poll_wait(void *) {}

int WindowsAlignedFileIO::send_read_no_alloc(IORequest &req, void *ctx) {
  if (v2::cache.get(req.offset / SECTOR_LEN, static_cast<uint8_t *>(req.buf))) {
    req.finished = true;
    return 0;
  }
  send_io(req, ctx, false);
  return 1;
}
int WindowsAlignedFileIO::send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx) {
  int reads = 0;
  for (auto &req : reqs) reads += send_read_no_alloc(req, ctx);
  return reads;
}
void WindowsAlignedFileIO::read_alloc(std::vector<IORequest> &reqs, void *ctx,
                                      std::vector<uint64_t> *page_ref) {
  for (auto &req : reqs) {
    auto page = req.offset / SECTOR_LEN;
    if (!v2::cache.get(page, static_cast<uint8_t *>(req.buf), true)) {
      send_io(req, ctx, false);
      v2::cache.put(page, static_cast<uint8_t *>(req.buf), true);
    } else req.finished = true;
    if (page_ref) page_ref->push_back(page);
  }
}
#endif
