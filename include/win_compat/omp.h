#pragma once

inline void omp_set_num_threads(int) {}
inline int omp_get_thread_num() { return 0; }
