#ifndef DNNKERNEL_LINEAR_H
#define DNNKERNEL_LINEAR_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {

static void linear(const float *x, const float* weight, const float* bias, int64_t in_features, int64_t out_features, float *y) {
  for (int64_t i = 0; i < out_features; ++i) {
    float sum = 0.f;
    for (int64_t j = 0; j < in_features; ++j) {
      sum += x[j] * weight[i * in_features + j];
    }
    y[i] = sum + bias[i];
  }
}

template <int UNROLL_OCH>
static void linear_opt(const float *x, const float* weight, const float* bias, int64_t in_features, int64_t out_features, float *y) {

  for (int64_t block_i = 0; block_i < out_features; block_i += UNROLL_OCH) {
    float sum[UNROLL_OCH];
#pragma HLS array_partition variable=sum complete

    for (int64_t j = 0; j < in_features; ++j) {
#pragma HLS pipeline II=1
      for (int64_t local_i = 0; local_i < UNROLL_OCH; local_i++) {
#pragma HLS unroll
        int64_t i = block_i + local_i;
        if (i < out_features) {
          float last = (j == 0) ? 0 : sum[local_i];
          sum[local_i] = last + x[j] * weight[i * in_features + j];
        }
      }
    }

    for (int64_t local_i = 0; local_i < UNROLL_OCH; local_i++) {
#pragma HLS unroll
      int64_t i = block_i + local_i;
      if (i < out_features) {
        y[i] = sum[local_i] + bias[i];
      }
    }
  }
}

}  // namespace dnnk

#endif  // DNNKERNEL_LINEAR_H
