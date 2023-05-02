#include "maxpool2d_hls.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <torch/torch.h>

#include <tests/util.h>

static const std::size_t kMaxSize = 65536;

using namespace dnnk;
namespace F = torch::nn::functional;

int main() {
  // Seeds must be fixed because the testbench is executed twice in
  // the cosimulation.
  torch::manual_seed(0);

  int h = 32, w = 32, channels = 4, stride = 2;

  auto x_ref = torch::randn({1, channels, h, w});

  float x[kMaxSize], y[kMaxSize];
  tensor2array(x_ref, x);

  auto y_ref = F::detail::max_pool2d(x_ref, stride, stride, 0, 1, false);
  maxpool2d_hls(x, w, h, channels, stride, y);

  if (!verify(y, y_ref)) {
    printf("%sFailed%s\n", Color::red, Color::reset);
    return 1;
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
