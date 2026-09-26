#pragma once

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

extern "C" float cblas_snrm2(int n, const float *x, int incx);
extern "C" void cblas_sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                             int m, int n, int k, float alpha, const float *a, int lda,
                             const float *b, int ldb, float beta, float *c, int ldc);
