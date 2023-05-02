#include "dnn-kernel/linear.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void linear_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize], int32_t in_features, int32_t out_features, float y[kMaxSize]) {

  dnnk::linear(x, weight, bias, in_features, out_features, y);
}

void linear_opt_2_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize], int32_t in_features, int32_t out_features, float y[kMaxSize]) {

  dnnk::linear_opt<2>(x, weight, bias, in_features, out_features, y);
}

void linear_opt_3_hls(const float x[kMaxSize], const float weight[kMaxSize], const float bias[kMaxSize], int32_t in_features, int32_t out_features, float y[kMaxSize]) {

  dnnk::linear_opt<3>(x, weight, bias, in_features, out_features, y);
}
