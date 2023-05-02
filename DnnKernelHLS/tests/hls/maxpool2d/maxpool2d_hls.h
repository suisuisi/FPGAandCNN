#ifndef DNNKERNEL_TEST_CONV2D_HLS_H
#define DNNKERNEL_TEST_CONV2D_HLS_H

#include <stdint.h>

void maxpool2d_hls(const float* x, int32_t width, int32_t height, int32_t channels, int32_t stride, float* y);

#endif  // DNNKERNEL_TEST_CONV2D_HLS_H
