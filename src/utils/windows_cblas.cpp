#include <cblas.h>

#include <cmath>
#include <stdexcept>

extern "C" float cblas_snrm2(int n, const float *x, int incx) {
  double sum = 0;
  for (int i = 0; i < n; ++i) {
    const double value = x[i * incx];
    sum += value * value;
  }
  return static_cast<float>(std::sqrt(sum));
}

extern "C" void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                             int m, int n, int k, float alpha, const float *a, int lda,
                             const float *b, int ldb, float beta, float *c, int ldc) {
  if (order != CblasRowMajor) throw std::invalid_argument("Only row-major CBLAS is supported");
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double dot = 0;
      for (int inner = 0; inner < k; ++inner) {
        const float left = trans_a == CblasNoTrans ? a[row * lda + inner] : a[inner * lda + row];
        const float right = trans_b == CblasNoTrans ? b[inner * ldb + col] : b[col * ldb + inner];
        dot += static_cast<double>(left) * right;
      }
      const float previous = beta == 0 ? 0 : c[row * ldc + col];
      c[row * ldc + col] = static_cast<float>(alpha * dot + beta * previous);
    }
  }
}
