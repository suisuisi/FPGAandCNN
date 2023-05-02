#include "relu_hls.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <torch/torch.h>

#include <tests/util.h>

using namespace dnnk;
namespace F = torch::nn::functional;

int main() {
  // Seeds must be fixed because the testbench is executed twice in
  // the cosimulation.
  torch::manual_seed(0);

  const std::size_t size_max = 1000;
  auto x_ref = torch::randn({28, 28, 1});
  float x[size_max], y[size_max];
  tensor2array(x_ref, x);

  relu_hls(x, x_ref.numel(), y);
  auto y_ref = F::detail::relu(x_ref, false);

  if (!verify(y, y_ref)) {
    printf("%sFailed%s\n", Color::red, Color::reset);
    return 1;
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
