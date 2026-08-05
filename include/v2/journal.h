#ifndef DUMMY_KVS_H_
#define DUMMY_KVS_H_

#include <pthread.h>
// #include <rocksdb/utilities/optimistic_transaction_db.h>
// #include <rocksdb/utilities/transaction.h>
#include <string>
#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "query_buf.h"
#include "utils.h"
#include "libcuckoo/cuckoohash_map.hh"

namespace v2 {
  struct journal_entry_head {
    uint64_t loc;
    uint32_t id = 0;
    uint32_t node_size;
    uint32_t data_dim;
    uint32_t nnbrs;
  };

  // using namespace rocksdb;
  template<typename T>
  struct journal_entry {
    struct journal_entry_head head;
    T *coords = nullptr;
    uint32_t *nbrs = nullptr;
  };

  enum TxType { kInsert, kDelete };
  template<class TagT>
  class Journal {
   public:
    std::string db_name;
    std::shared_ptr<AlignedFileReader> journal_writer;
    std::mutex j_mutex;
    // rocksdb::OptimisticTransactionDB *db;

    Journal(std::string path) : db_name(path) {
      // rocksdb::Options options;
      // options.create_if_missing = true;
      // rocksdb::Status status = rocksdb::OptimisticTransactionDB::Open(options, db_name, &db);
      // if (!status.ok()) {
      //   LOG(INFO) << "Failed to open db: " << db_name;
      //   exit(1);
      // }
      journal_writer = std::make_shared<LinuxAlignedFileReader>();
      journal_writer->open(db_name, true, true);
    }
    ~Journal() {
      // delete db;
      journal_writer->close();
    }

    std::atomic<uint64_t> cur_txid;
    libcuckoo::cuckoohash_map<uint64_t, TagT> running_txs;

    std::string serialize(TxType type, TagT tag) {
      return std::to_string(type) + "_" + std::to_string(tag);
    }

    void deserialize(const std::string &s, TxType &type, TagT &tag) {
      // auto pos = s.find('_');
      // type = (TxType) std::stoi(s.substr(0, pos));
      // tag = (TagT) std::stoi(s.substr(pos + 1));
    }

    void append(TxType type, TagT tag) {
      // uint64_t txid = cur_txid.fetch_add(1);
      // running_txs.insert(txid, tag);
      // db->Put(rocksdb::WriteOptions(), std::to_string(txid), serialize(type, tag));
      // running_txs.erase(txid);
    }

    void checkpoint() {
      // auto locked_table = running_txs.lock_table();
      // uint64_t running_min_tx = std::numeric_limits<uint64_t>::max();
      // for (auto &item : locked_table) {
      //   running_min_tx = std::min(running_min_tx, item.first);
      // }
      // db->Put(rocksdb::WriteOptions(), "checkpoint", std::to_string(running_min_tx));
      // db->SyncWAL();
      // locked_table.unlock();
    }

    template<typename T>
    void append_and_commit_journal(std::vector<journal_entry<T>> &journal_entries) {
      // generate journal writes
      // copy all entries to a continuous buffer
      std::lock_guard<std::mutex> lock(j_mutex);

      std::vector<IORequest> journal_write;
      IORequest journal_req;
      void *journal_buf = nullptr;
      void *commit_buf = nullptr;
      auto journal_ctx = journal_writer->get_ctx();
      if (!journal_entries.empty()) {
        uint64_t journal_size = 0;
        for (auto &entry : journal_entries) {
          auto len = sizeof(journal_entry_head) + entry.head.data_dim * sizeof(T) + entry.head.nnbrs * sizeof(uint32_t);
          journal_size += len;
        }
        journal_size = ROUND_UP(journal_size, SECTOR_LEN);
        ccann::alloc_aligned(&journal_buf, journal_size, SECTOR_LEN);
        memset(journal_buf, 0, journal_size);

        char *ptr = (char *) journal_buf;
        for (auto &entry : journal_entries) {
          memcpy(ptr, &entry.head, sizeof(journal_entry_head));
          ptr += sizeof(journal_entry_head);
          memcpy(ptr, entry.coords, entry.head.data_dim * sizeof(T));
          ptr += entry.head.data_dim * sizeof(T);
          memcpy(ptr, entry.nbrs, entry.head.nnbrs * sizeof(uint32_t));
          ptr += entry.head.nnbrs * sizeof(uint32_t);
        }

        uint64_t j_start = journal_writer->file_size();

        journal_req = IORequest(0, journal_size, journal_buf, 0, 0);
        journal_write.push_back(journal_req);
        journal_writer->write(journal_write, journal_ctx, false);
        journal_writer->sync();

        // commit

        ccann::alloc_aligned(&commit_buf, SECTOR_LEN, SECTOR_LEN);
        auto commit_entry_head = journal_entry_head{(uint64_t) -1, (uint32_t) -1, 0, 0, 0};
        auto commit_entry = journal_entry<T>{commit_entry_head, nullptr, nullptr};
        memset(commit_buf, 0, SECTOR_LEN);

        journal_write.clear();
        auto journal_commit = IORequest(journal_size, SECTOR_LEN, commit_buf, 0, 0);
        journal_write.push_back(journal_commit);
        journal_writer->write(journal_write, journal_ctx, false);
        journal_writer->sync();
      }

      if (journal_buf != nullptr) {
        ccann::aligned_free(journal_buf);
        ccann::aligned_free(commit_buf);
        journal_buf = nullptr;
      }
    }

    inline void clear_journal() {
      std::lock_guard<std::mutex> lock(j_mutex);
      // commit journal. How?
      journal_writer->atomic_truncate(0);
    }
  };
}  // namespace v2
#endif  // DUMMY_KVS_H_