#include "dnn-kernel/relu.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include <torch/torch.h>

#include <tests/util.h>

using namespace dnnk;
namespace F = torch::nn::functional;

TEST(CPUVerify, ReLU) {
  auto x_ref = torch::randn({28, 28, 1});
  const float* x = tensor2array(x_ref);
  float* y = new float[x_ref.numel()];

  dnnk::relu(x, x_ref.numel(), y);
  auto y_ref = F::detail::relu(x_ref, false);

  EXPECT_TRUE(verify(y, y_ref));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
