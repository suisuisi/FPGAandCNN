#include "dnn-kernel/conv2d.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void conv2d_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_pipelined_v1_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                             int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_pipelined_v1(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_pipelined_v2_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                             int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_pipelined_v2(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v1_2_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                              int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v1<2>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v1_3_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                              int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v1<3>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v2_2_2_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v2<2, 2>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v2_3_2_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v2<3, 2>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v2_2_3_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v2<2, 3>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}

void conv2d_unrolled_v2_3_3_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize],
                                int32_t width, int32_t height, int32_t in_channels, int32_t out_channels, int32_t ksize, float y[kMaxSize]) {

  dnnk::conv2d_unrolled_v2<3, 3>(x, weight, bias, width, height, in_channels, out_channels, ksize, y);
}
