#ifndef LOCK_TABLE_H_
#define LOCK_TABLE_H_

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <shared_mutex>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include "libcuckoo/cuckoohash_map.hh"
#include "log.h"

#define SECTOR_LEN 4096

inline void thread_pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#endif
}

namespace v2 {
  template<class K, class HashFunction = std::hash<K>>
  class SparseLockTable {
   public:
    int tryrdlock(const K &key) { return trylock(key, false); }
    int trywrlock(const K &key) { return trylock(key, true); }
    void rdlock(const K &key) { lock(key, false); }
    void wrlock(const K &key) { lock(key, true); }

    void unlock(const K &key) {
      locks_.erase_fn(key, [&](Entry &entry) {
        if (entry.count == 0) {
          LOG(ERROR) << "SparseLockTable: unlock a non-locked key: " << key;
          std::terminate();
        }
        if (entry.exclusive) entry.mutex->unlock();
        else entry.mutex->unlock_shared();
        return --entry.count == 0;
      });
    }
    size_t size() { return locks_.size(); }

   private:
    struct Entry {
      std::shared_ptr<std::shared_mutex> mutex;
      int count = 0;
      bool exclusive = false;
    };
    libcuckoo::cuckoohash_map<K, Entry, HashFunction> locks_;

    int trylock(const K &key, bool exclusive) {
      int result = 1;
      locks_.upsert(key, [&](Entry &entry, libcuckoo::UpsertContext context) {
        if (context == libcuckoo::UpsertContext::NEWLY_INSERTED)
          entry.mutex = std::make_shared<std::shared_mutex>();
        bool acquired = exclusive ? entry.mutex->try_lock() : entry.mutex->try_lock_shared();
        if (acquired) {
          ++entry.count;
          entry.exclusive = exclusive;
          result = 0;
        }
      });
      return result;
    }
    void lock(const K &key, bool exclusive) {
      auto count = 0;
      while (trylock(key, exclusive) != 0) {
        thread_pause();
        if (++count > 50000000) {
          LOG(ERROR) << "SparseLockTable: lock timeout for key: " << key;
          count = 0;
        }
      }
    }
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseReadLockGuard {
   public:
    SparseReadLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->rdlock(key_);
    }
    ~SparseReadLockGuard() { table_->unlock(key_); }
   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseWriteLockGuard {
   public:
    SparseWriteLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->wrlock(key_);
    }
    ~SparseWriteLockGuard() { table_->unlock(key_); }
   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  struct LockHandle {
    std::shared_mutex *mutex;
    bool exclusive;
    void unlock() const {
      if (exclusive) mutex->unlock();
      else mutex->unlock_shared();
    }
  };

  class LockTable {
   public:
    explicit LockTable(size_t size) : size_(size), locks_(new std::shared_mutex[size]) {}
    LockHandle rdlock(uint32_t key) {
      auto *mutex = &locks_[Hash(key) % size_];
      mutex->lock_shared();
      return {mutex, false};
    }
    LockHandle wrlock(uint32_t key) {
      auto *mutex = &locks_[Hash(key) % size_];
      mutex->lock();
      return {mutex, true};
    }
    uint64_t pos(uint64_t key) { return Hash(key) % size_; }
    bool tryrdlock(uint32_t key) { return locks_[Hash(key) % size_].try_lock_shared(); }
    bool trywrlock(uint32_t key) { return locks_[Hash(key) % size_].try_lock(); }
    void unlock(LockHandle handle) { handle.unlock(); }

   private:
    size_t size_;
    std::unique_ptr<std::shared_mutex[]> locks_;
    static const uint32_t c1 = 0xcc9e2d51;
    static const uint32_t c2 = 0x1b873593;
    static uint32_t fmix(uint32_t h) {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    }
    static uint32_t Rotate32(uint32_t val, int shift) {
      return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
    }
    static uint32_t Mur(uint32_t a, uint32_t h) {
      a *= c1;
      a = Rotate32(a, 17);
      a *= c2;
      h ^= a;
      h = Rotate32(h, 19);
      return h * 5 + 0xe6546b64;
    }
    static uint32_t Hash32Len0to4(const char *s, size_t len) {
      uint32_t b = 0;
      uint32_t c = 9;
      for (size_t i = 0; i < len; i++) {
        signed char v = static_cast<signed char>(s[i]);
        b = b * c1 + static_cast<uint32_t>(v);
        c ^= b;
      }
      return fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
    }
    static uint32_t Hash(uint32_t x) {
      return Hash32Len0to4((const char *) &x, sizeof(uint32_t));
    }
  };

  class LockGuard {
   public:
    explicit LockGuard(LockHandle lock) : lock_(lock) {}
    LockGuard(const LockGuard &) = delete;
    LockGuard &operator=(const LockGuard &) = delete;
    LockGuard(LockGuard &&rhs) noexcept : lock_(rhs.lock_) { rhs.lock_.mutex = nullptr; }
    LockGuard &operator=(LockGuard &&rhs) noexcept {
      if (this != &rhs) {
        if (lock_.mutex) lock_.unlock();
        lock_ = rhs.lock_;
        rhs.lock_.mutex = nullptr;
      }
      return *this;
    }
    ~LockGuard() { if (lock_.mutex) lock_.unlock(); }
   private:
    LockHandle lock_;
  };
}  // namespace v2

#endif  // LOCK_TABLE_H_
