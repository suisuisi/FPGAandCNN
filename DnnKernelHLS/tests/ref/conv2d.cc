#include "dnn-kernel/conv2d.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include <torch/torch.h>

#include <tests/util.h>

using namespace dnnk;
namespace F = torch::nn::functional;

TEST(CPUVerify, Conv2d) {
  torch::manual_seed(0);

  int h = 14, w = 14, in_channels = 4, out_channels = 8, ksize = 3;

  auto x_ref = torch::randn({1, in_channels, h, w});
  auto weight_ref = torch::randn({out_channels, in_channels, ksize, ksize});
  auto bias_ref = torch::randn({out_channels});

  std::vector<float> x(x_ref.numel());
  std::vector<float> weight(weight_ref.numel());
  std::vector<float> bias(out_channels);
  tensor2array(x_ref, x.data());
  tensor2array(weight_ref, weight.data());
  tensor2array(bias_ref, bias.data());

  auto y_ref = F::detail::conv2d(x_ref, weight_ref, bias_ref, 1, ksize/2, 1, 1);
  std::vector<float> y(y_ref.numel());
  conv2d(x.data(), weight.data(), bias.data(), w, h, in_channels, out_channels, ksize, y.data());

  EXPECT_TRUE(verify(y.data(), y_ref));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
