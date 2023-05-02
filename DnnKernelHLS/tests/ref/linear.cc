#include "dnn-kernel/linear.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include <torch/torch.h>

#include <tests/util.h>

using namespace dnnk;
namespace F = torch::nn::functional;

TEST(CPUVerify, Linear) {
  torch::manual_seed(0);

  int in_channels = 32, out_channels = 16;

  auto x_ref = torch::randn({1, in_channels});
  auto weight_ref = torch::randn({out_channels, in_channels});
  auto bias_ref = torch::randn({out_channels});

  std::vector<float> x(x_ref.numel());
  std::vector<float> weight(weight_ref.numel());
  std::vector<float> bias(out_channels);
  tensor2array(x_ref, x.data());
  tensor2array(weight_ref, weight.data());
  tensor2array(bias_ref, bias.data());

  auto y_ref = F::linear(x_ref, weight_ref, bias_ref);
  std::vector<float> y(y_ref.numel());
  linear(x.data(), weight.data(), bias.data(), in_channels, out_channels, y.data());

  EXPECT_TRUE(verify(y.data(), y_ref));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
