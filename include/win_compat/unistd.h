#pragma once

#ifdef _WIN32
#include <BaseTsd.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <fcntl.h>
#include <io.h>
#include <mutex>
#include <sys/stat.h>

using ssize_t = SSIZE_T;

#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif

inline int open(const char *path, int flags, int mode = 0666) {
  return _open(path, flags | _O_BINARY, mode);
}
inline int close(int fd) { return _close(fd); }
inline int fsync(int fd) { return _commit(fd); }
inline int ftruncate(int fd, int64_t length) {
  const errno_t result = _chsize_s(fd, static_cast<__int64>(length));
  if (result != 0) errno = result;
  return result == 0 ? 0 : -1;
}
inline int64_t lseek(int fd, int64_t offset, int origin) { return _lseeki64(fd, offset, origin); }

inline std::mutex &positioned_io_mutex() {
  static std::mutex mutex;
  return mutex;
}
inline ssize_t pread(int fd, void *buffer, size_t length, int64_t offset) {
  std::lock_guard<std::mutex> lock(positioned_io_mutex());
  if (_lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  return _read(fd, buffer, static_cast<unsigned>(std::min<size_t>(length, INT_MAX)));
}
inline ssize_t pwrite(int fd, const void *buffer, size_t length, int64_t offset) {
  std::lock_guard<std::mutex> lock(positioned_io_mutex());
  if (_lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  return _write(fd, buffer, static_cast<unsigned>(std::min<size_t>(length, INT_MAX)));
}
inline unsigned sleep(unsigned seconds) {
  Sleep(seconds * 1000);
  return 0;
}
#endif
