// async_ring.cpp
// Compile: g++ -std=c++17 async_ring.cpp -pthread -O2 -o async_ring
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>
#include <unordered_map>

#include "mpmc.h"

////////////////////////////////////////////////////////////////////////////////
// 设计说明（简短）
// - SQ: submit queue (生产者：主线程/任意线程) -> worker threads 从 SQ 取任务执行。
// - CQ: completion queue (多 producer: workers -> 消费者:主线程通过 poll 轮询获取结果)
// - 默认使用 std::deque + mutex（安全、易用）。若要高性能，把 QueueType 替换为 lock-free 实现（例如
// moodycamel::ConcurrentQueue）
// - 支持携带 task id，返回值用 std::any（也可模板化为具体类型）
////////////////////////////////////////////////////////////////////////////////

using TaskId = uint64_t;

struct CompRequest {
  TaskId id;
  std::function<void()> fn_void;  // wrapper that sets result
};

struct CompResult {
  TaskId id;
  void *result;
  std::exception_ptr eptr;
};

////////////////////////////////////////////////////////////////////////////////
// Queue abstraction points
// DefaultQueue 是基于 mutex 的 deque；高性能场景可替换为 lock-free queue
////////////////////////////////////////////////////////////////////////////////
template<typename T>
class DefaultQueue {
 public:
  DefaultQueue() = default;

  void push(T &&v) {
    std::lock_guard lk(mu_);
    dq_.push_back(std::move(v));
  }

  // try pop (non-blocking). returns true if popped into out.
  bool try_pop(T &out) {
    std::lock_guard lk(mu_);
    if (dq_.empty())
      return false;
    out = std::move(dq_.front());
    dq_.pop_front();
    return true;
  }

  // peek size (approx)
  size_t size() const {
    std::lock_guard lk(mu_);
    return dq_.size();
  }

 private:
  mutable std::mutex mu_;
  std::deque<T> dq_;
};

template<typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() = default;

  // 非阻塞 push
  void push(T &&v) {
    queue_.enqueue(std::move(v));
  }

  // try pop (non-blocking). returns true if popped into out.
  bool try_pop(T &out) {
    return queue_.try_dequeue(out);
  }

  // peek size (approximate)
  size_t size() const {
    return queue_.size_approx();
  }

 private:
  moodycamel::ConcurrentQueue<T> queue_;
};

////////////////////////////////////////////////////////////////////////////////
// AsyncRing: 核心类
// 模板化队列类型，默认为 DefaultQueue
////////////////////////////////////////////////////////////////////////////////
template<typename SQ = ConcurrentQueue<CompRequest>, typename CQ = ConcurrentQueue<CompResult>>
class AsyncRing {
 public:
  AsyncRing(size_t n_workers = std::thread::hardware_concurrency(), std::function<void(size_t)> destructor_fn = nullptr)
      : running_(true), next_id_(1), workers_count_(n_workers), active_worker_count_(n_workers),
        destructor_fn_(std::move(destructor_fn)) {
    if (workers_count_ == 0) {
      workers_count_ = 1;
      active_worker_count_ = 1;
    }
    start_workers(workers_count_);
  }

  ~AsyncRing() {
    LOG(INFO) << "AsyncRing destructor called, closing...";
    if (destructor_fn_) {
      LOG(INFO) << "AsyncRing: custom destructor function called for " << active_worker_count_ << " active workers.";
      destructor_fn_(active_worker_count_);
    }
    close();
  }

  size_t activated_worker_count() {
    return active_worker_count_;
  }

  void remove_worker(bool wait_for_completion = false) {
    if (active_worker_count_ == 1)
      return;

    active_worker_count_--;
    for (size_t i = 0; i < workers_.size(); ++i) {
      if (worker_active_[i]) {
        worker_active_[i] = false;
        break;
      }
    }
    LOG(INFO) << "AsyncRing: removed a worker, current active workers: " << active_worker_count_;
  }

  void add_worker() {
    active_worker_count_++;
    for (size_t i = 0; i < workers_.size(); ++i) {
      if (!worker_active_[i]) {
        // notify worker
        auto cv = &worker_cvs_[i];
        auto mu_ = &worker_mus_[i];

        {
          std::lock_guard<std::mutex> lk(*mu_);
          worker_active_[i] = true;
        }

        cv->notify_one();
        break;
      }
    }
    LOG(INFO) << "AsyncRing: added a worker, current active workers: " << active_worker_count_;
  }

  // 非阻塞提交任务，返回 task id
  // Fn 可以返回任意可拷贝/可移动类型 R
  template<typename Fn>
  TaskId send_for_comp(Fn &&fn) {
    // Wrap fn to capture return value into std::any and catch exceptions.
    TaskId id = next_id_.fetch_add(1, std::memory_order_relaxed);

    // create a packaged wrapper that does the job in worker
    CompRequest req;
    req.id = id;

    // move-construct callable into wrapper
    using R = std::invoke_result_t<Fn>;
    auto wrapper = [fn = std::forward<Fn>(fn)]() mutable -> void * {
      if constexpr (std::is_void<R>::value) {
        fn();
        return nullptr;
      } else {
        R r = fn();
        return (void *) std::move(r);
      }
    };

    // worker will call wrapper() and set result into CQ; but to minimize closure size we
    // capture wrapper into fn_void which calls the wrapper and places result via a lambda
    req.fn_void = [this, wrapper = std::move(wrapper), id]() mutable {
      CompResult cres;
      cres.id = id;
      try {
        cres.result = wrapper();
        cres.eptr = nullptr;
      } catch (...) {
        cres.eptr = std::current_exception();
      }
      cq_.push(std::move(cres));  // push result to completion queue
    };

    sq_.push(std::move(req));
    return id;
  }

  // 非阻塞 poll：一次只返回一个已完成结果（更接近 io_uring 的 cq ring 每次 pop 一条）
  // 返回 std::optional<CompResult>
  std::optional<CompResult> poll() {
    CompResult r;
    if (cq_.try_pop(r)) {
      return r;
    } else {
      return std::nullopt;
    }
  }

  // 非阻塞 poll 多个结果（最多 max_count）
  std::vector<CompResult> poll_multi(size_t max_count = 32) {
    std::vector<CompResult> out;
    out.reserve(std::min<size_t>(max_count, 16));
    for (size_t i = 0; i < max_count; ++i) {
      CompResult r;
      if (!cq_.try_pop(r))
        break;
      out.push_back(std::move(r));
    }
    return out;
  }

  std::vector<CompResult> poll_all() {
    std::vector<CompResult> out;
    while (true) {
      CompResult r;
      if (!cq_.try_pop(r))
        break;
      out.push_back(std::move(r));
    }
    return out;
  }

  // 获取队列近似长度（调试用）
  size_t sq_size() const {
    return sq_.size();
  }
  size_t cq_size() const {
    return cq_.size();
  }

  // 优雅关闭：停止接收任务，等待 worker 退出，清理资源
  void close() {
    // add_all workers
    while (active_worker_count_ < workers_count_) {
      add_worker();
    }

    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false))
      return;  // already closed

    // join workers
    for (auto &t : workers_) {
      if (t.joinable())
        t.join();
    }

    workers_.clear();
  }

 private:
  // blocking pop wrapper for CQ (depends on queue impl). For DefaultQueue we provided wait_pop.
  bool cq_blocking_pop(CompResult &out) {
    // If CQ is DefaultQueue, we can use wait_pop; but we don't have polymorphism.
    // We'll emulate blocking wait by spinning with short sleeps to be generic.
    while (running_.load()) {
      if (cq_.try_pop(out))
        return true;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // try final time
    return cq_.try_pop(out);
  }

  void start_workers(size_t n) {
    workers_.reserve(n);

    worker_mus_ = std::vector<std::mutex>(n);
    worker_cvs_ = std::vector<std::condition_variable>(n);
    worker_active_ = std::vector<bool>(n, true);

    for (size_t i = 0; i < n; ++i) {
      workers_.emplace_back([this, i] { worker_loop(i); });
    }
  }

  void worker_loop(size_t worker_idx) {
    // LOG(INFO) << "Set worker " << worker_idx << " to highest priority nice(-20)";

    while (running_.load(std::memory_order_acquire)) {
      if (!worker_active_[worker_idx]) {
        // use condition variable to avoid busy wait
        auto cv = &worker_cvs_[worker_idx];
        auto mu_ = &worker_mus_[worker_idx];
        std::unique_lock<std::mutex> lk(*mu_);
        cv->wait(lk, [this, worker_idx] { return worker_active_[worker_idx]; });
        continue;
      }

      CompRequest req;
      // blocking pop from SQ. If using DefaultQueue, use wait_pop
      bool got = sq_.try_pop(req);
      if (!got) {
        // either shutdown or spurious wake
        sched_yield();
        continue;
      }
      req.fn_void();
    }
  }

  SQ sq_;
  CQ cq_;
  std::atomic<bool> running_;
  std::atomic<TaskId> next_id_;
  std::vector<std::thread> workers_;
  size_t workers_count_;
  std::vector<bool> worker_active_;
  size_t active_worker_count_;
  // condition var for worker wakeup (if needed)
  std::vector<std::mutex> worker_mus_;
  std::vector<std::condition_variable> worker_cvs_;
  // add a destructor function to custom clean up resources if needed
  std::function<void(size_t)> destructor_fn_;
};

////////////////////////////////////////////////////////////////////////////////
// 示例：如何使用 AsyncRing
////////////////////////////////////////////////////////////////////////////////

// int main() {
//   AsyncRing<> ring(4);  // 4 worker threads

//   // Submit a few tasks
//   auto id1 = ring.send_for_comp([]() -> int {
//     std::this_thread::sleep_for(std::chrono::milliseconds(200));
//     return 100;
//   });

//   auto id2 = ring.send_for_comp([]() -> std::string {
//     std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     return std::string("hello from task");
//   });

//   auto id3 = ring.send_for_comp([]() -> void {
//     std::this_thread::sleep_for(std::chrono::milliseconds(150));
//     // void result
//   });

//   // 主 loop: poll 风格
//   size_t completed = 0;
//   while (completed < 3) {
//     if (auto r = ring.poll()) {
//       ++completed;
//       std::cout << "[poll] got id=" << r->id;
//       if (r->eptr) {
//         try {
//           std::rethrow_exception(r->eptr);
//         } catch (const std::exception &e) {
//           std::cout << " exception: " << e.what();
//         } catch (...) {
//           std::cout << " unknown exception";
//         }
//       } else if (r->result.has_value()) {
//         // try to cast common types
//         if (r->result.type() == typeid(int)) {
//           std::cout << " result(int)=" << std::any_cast<int>(r->result);
//         } else if (r->result.type() == typeid(std::string)) {
//           std::cout << " result(string)=" << std::any_cast<std::string>(r->result);
//         } else {
//           std::cout << " result(type=" << r->result.type().name() << ")";
//         }
//       } else {
//         std::cout << " result(void)";
//       }
//       std::cout << "\n";
//     } else {
//       // no completion yet: 轮询非阻塞模式下，可做其他事情
//       std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }
//   }

//   ring.close();
//   std::cout << "all done\n";
//   return 0;
// }
