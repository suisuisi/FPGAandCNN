#include "conv2d_hls.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <torch/torch.h>

#include <tests/util.h>

#ifndef TOP_FUNC
#error "TOP_FUNC is not defined"
#endif

static const std::size_t kMaxSize = 65536;

using namespace dnnk;
namespace F = torch::nn::functional;

int main() {
  // Seeds must be fixed because the testbench is executed twice in
  // the cosimulation.
  torch::manual_seed(0);

  int h = 14, w = 14, in_channels = 4, out_channels = 8, ksize = 3;

  auto x_ref = torch::randn({1, in_channels, h, w});
  auto weight_ref = torch::randn({out_channels, in_channels, ksize, ksize});
  auto bias_ref = torch::randn({out_channels});

  float x[kMaxSize], weight[kMaxSize], bias[kMaxSize], y[kMaxSize];
  tensor2array(x_ref, x);
  tensor2array(weight_ref, weight);
  tensor2array(bias_ref, bias);

  auto y_ref = F::detail::conv2d(x_ref, weight_ref, bias_ref, 1, ksize/2, 1, 1);
  TOP_FUNC (x, weight, bias, w, h, in_channels, out_channels, ksize, y);

  if (!verify(y, y_ref)) {
    printf("%sFailed%s\n", Color::red, Color::reset);
    return 1;
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
