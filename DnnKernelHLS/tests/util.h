#ifndef DNNKERNEL_TEST_UTIL_H
#define DNNKERNEL_TEST_UTIL_H

#include "torch/torch.h"

namespace dnnk {
namespace {

struct Color {
  static constexpr const char* red = "\u001b[31m";
  static constexpr const char* green = "\u001b[32m";
  static constexpr const char* reset = "\u001b[0m";
};

float* tensor2array(const torch::Tensor& tensor) {
  float* ret = new float[tensor.numel()];
  std::memcpy(ret, tensor.data_ptr(), tensor.nbytes());
  return ret;
}

void tensor2array(const torch::Tensor& tensor, float* array) {
  std::memcpy(array, tensor.data_ptr(), tensor.nbytes());
}

bool verify(const float* actual, const torch::Tensor& expect) {
  const float tolerance = 10e-5f;
  auto expect_ptr = expect.data_ptr<float>();

  for (auto i = decltype(expect.numel())(0); i < expect.numel(); ++i) {
    if (std::abs(actual[i] - expect_ptr[i]) >= tolerance) {
      std::cout << i << " : " << actual[i] << " vs " << expect_ptr[i]
                << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace
}  // namespace dnnk

#endif  // DNNKERNEL_TEST_UTIL_H
