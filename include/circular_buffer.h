#include <queue>
#include <unordered_map>
#include <deque>

class SlidingWindow {
  int w;
  std::priority_queue<double> lo;                                             // max-heap
  std::priority_queue<double, std::vector<double>, std::greater<double>> hi;  // min-heap
  std::deque<double> window;
  std::unordered_map<double, int> delayed;  // 记录待删除元素
 public:
  SlidingWindow(int window_size) : w(window_size) {
  }

  void push(double x) {
    window.push_back(x);
    if (lo.empty() || x <= lo.top())
      lo.push(x);
    else
      hi.push(x);
    balance();

    if ((int) window.size() > w) {
      double old = window.front();
      window.pop_front();
      // 延迟删除机制
      if (old <= lo.top()) {
        delayed[old]++;
        prune(lo);
      } else {
        delayed[old]++;
        prune(hi);
      }
      balance();
    }
  }

  bool filled() const {
    return (int) window.size() >= w;
  }

  double median() const {
    if (!filled())
      return 0.0;
    if (w % 2)
      return lo.top();
    else
      return (lo.top() + hi.top()) / 2.0;
  }

  double max() const {
    if (!filled())
      return 0.0;
    return std::max(lo.top(), hi.top());
  }

  double min() const {
    if (!filled())
      return 0.0;
    return std::min(lo.top(), hi.top());
  }

 private:
  void balance() {
    // 保证 |lo| = |hi| (+1)
    while (lo.size() > hi.size() + 1) {
      hi.push(lo.top());
      lo.pop();
    }
    while (hi.size() > lo.size()) {
      lo.push(hi.top());
      hi.pop();
    }
    prune(lo);
    prune(hi);
  }

  void prune(std::priority_queue<double> &heap) {
    while (!heap.empty() && delayed.count(heap.top())) {
      double x = heap.top();
      heap.pop();
      if (--delayed[x] == 0)
        delayed.erase(x);
    }
  }

  void prune(std::priority_queue<double, std::vector<double>, std::greater<double>> &heap) {
    while (!heap.empty() && delayed.count(heap.top())) {
      double x = heap.top();
      heap.pop();
      if (--delayed[x] == 0)
        delayed.erase(x);
    }
  }
};