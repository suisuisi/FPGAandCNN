#ifndef DNNKERNEL_TEST_CONV2D_HLS_H
#define DNNKERNEL_TEST_CONV2D_HLS_H

#include <stdint.h>

void conv2d_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_pipelined_v1_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_pipelined_v2_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v1_2_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v1_3_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v2_2_2_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v2_2_3_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v2_3_2_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);
void conv2d_unrolled_v2_3_3_hls(const float *x, const float* weight, const float* bias, int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float *y);

#endif  // DNNKERNEL_TEST_CONV2D_HLS_H
