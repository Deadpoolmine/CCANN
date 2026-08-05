#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

class ThreadPool {
 public:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable cond_;
  std::atomic<bool> stop_;

  explicit ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ThreadPool() {
    LOG(INFO) << "Destroying ThreadPool...";
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cond_.notify_all();
    for (auto &t : workers_) {
      if (t.joinable())
        t.join();
    }
  }

  void wait_all() {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (tasks_.empty())
          break;
      }
      std::this_thread::yield();
    }
  }

  // 提交任务（带返回值）
  template<typename F, typename... Args>
  auto submit(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task =
        std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_)
        throw std::runtime_error("ThreadPool stopped");
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_.notify_one();
    return res;
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cond_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty())
          return;
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }
};
