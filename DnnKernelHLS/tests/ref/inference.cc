#include "dnn-kernel/inference.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <tests/util.h>

#ifndef PROJECT_ROOT
#error "PROJECT_ROOT is not defined"
#endif

using namespace dnnk;
namespace F = torch::nn::functional;

TEST(CPUVerify, Inference) {
  std::vector<float> x, y;
  std::map<std::string, std::vector<float> > params;

  // load model file
  auto model = torch::jit::load(PROJECT_ROOT "/learning/traced_model.pt");

  // load parameter values from model
  for (const auto& param_ref : model.named_parameters()) {

    // use param_ref.name as key (ex: "conv1.weight")
    params[param_ref.name].resize(param_ref.value.numel());

    // copy image data
    tensor2array(param_ref.value, params[param_ref.name].data());
  }

  // read MNIST dataset
  auto dataset = torch::data::datasets::MNIST(PROJECT_ROOT "/learning/data/MNIST/raw")
    .map(torch::data::transforms::Stack<>());

  // define loader and set batch_size to 1
  auto data_loader =
    torch::data::make_data_loader(std::move(dataset),
                                  torch::data::DataLoaderOptions().batch_size(1));

  // iterate data_loader
  std::size_t num_data = 0;
  std::size_t num_corrects_ref = 0;
  std::size_t num_corrects = 0;
  for (auto& batch : *data_loader) {

    auto x_ref = batch.data;   // shape = (1, 1, 28, 28)
    auto y_label = batch.target; // shape = (1)

    // run inference in pytorch
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_ref);
    auto y_ref = model.forward(inputs).toTensor();

    x.resize(x_ref.numel());
    y.resize(y_ref.numel());

    // run inference
    tensor2array(x_ref, x.data());
    inference(x.data(),
              params.at("conv1.weight").data(), params.at("conv1.bias").data(),
              params.at("conv2.weight").data(), params.at("conv2.bias").data(),
              params.at("fc1.weight").data(), params.at("fc1.bias").data(),
              params.at("fc2.weight").data(), params.at("fc2.bias").data(),
              y.data());

    EXPECT_TRUE(verify(y.data(), y_ref));

    // summarize
    num_data++;

    if (y_label.data_ptr<int64_t>()[0] == y_ref.argmax().data_ptr<int64_t>()[0]) {
      num_corrects_ref++;
    }

    auto argmax = [](const std::vector<float>& vec) {
      return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
    };

    if (y_label.data_ptr<int64_t>()[0] == argmax(y)) {
      num_corrects++;
    }
  }

  std::cout << "accuracy (ref): " << double(num_corrects_ref) / num_data << std::endl;
  std::cout << "accuracy: " << double(num_corrects) / num_data << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
