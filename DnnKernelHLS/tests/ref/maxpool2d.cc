#include "dnn-kernel/maxpool2d.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include <torch/torch.h>

#include <tests/util.h>

using namespace dnnk;
namespace F = torch::nn::functional;

TEST(CPUVerify, Maxpool2d) {
  torch::manual_seed(0);

  int h = 32, w = 32, channels = 4, stride = 2;

  auto x_ref = torch::randn({1, channels, h, w});

  std::vector<float> x(x_ref.numel());
  tensor2array(x_ref, x.data());

  auto y_ref = F::detail::max_pool2d(x_ref, stride, stride, 0, 1, false);
  std::vector<float> y(y_ref.numel());
  maxpool2d(x.data(), w, h, channels, stride, y.data());

  EXPECT_TRUE(verify(y.data(), y_ref));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
