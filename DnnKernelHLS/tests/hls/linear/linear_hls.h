#ifndef DNNKERNEL_TEST_LINEAR_HLS_H
#define DNNKERNEL_TEST_LINEAR_HLS_H

#include <stdint.h>

void linear_hls(const float *x, const float* weight, const float* bias, int32_t in_features, int32_t out_features, float *y);
void linear_opt_2_hls(const float *x, const float* weight, const float* bias, int32_t in_features, int32_t out_features, float *y);
void linear_opt_3_hls(const float *x, const float* weight, const float* bias, int32_t in_features, int32_t out_features, float *y);

#endif  // DNNKERNEL_TEST_LINEAR_HLS_H
