#include "linear_hls.h"

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

  int in_features = 32, out_features = 16;

  auto x_ref = torch::randn({1, in_features});
  auto weight_ref = torch::randn({out_features, in_features});
  auto bias_ref = torch::randn({out_features});

  float x[kMaxSize], weight[kMaxSize], bias[kMaxSize], y[kMaxSize];
  tensor2array(x_ref, x);
  tensor2array(weight_ref, weight);
  tensor2array(bias_ref, bias);

  auto y_ref = F::linear(x_ref, weight_ref, bias_ref);
  TOP_FUNC (x, weight, bias, in_features, out_features, y);

  if (!verify(y, y_ref)) {
    printf("%sFailed%s\n", Color::red, Color::reset);
    return 1;
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
