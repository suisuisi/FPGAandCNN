#ifndef DNNKERNEL_TEST_RELU_HLS_H
#define DNNKERNEL_TEST_RELU_HLS_H

#include <stdint.h>

void relu_hls(const float* x, int64_t size, float* y);

#endif  // DNNKERNEL_TEST_RELU_HLS_H
