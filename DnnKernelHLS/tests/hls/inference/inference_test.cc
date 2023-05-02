#include "inference_hls.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <map>

#include <torch/torch.h>
#include <torch/script.h>

#include <tests/util.h>

#ifndef PROJECT_ROOT
#error "PROJECT_ROOT is not defined"
#endif

#ifndef TOP_FUNC
#error "TOP_FUNC is not defined"
#endif

static const std::size_t kMaxSize = 16384;

using namespace dnnk;
namespace F = torch::nn::functional;

int main() {
  // Seeds must be fixed because the testbench is executed twice in
  // the cosimulation.
  torch::manual_seed(0);


  float x[kMaxSize], y[kMaxSize];
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
  std::size_t niters = 0;
  for (auto& batch : *data_loader) {

    auto x_ref = batch.data;   // shape = (1, 1, 28, 28)
    auto y_label = batch.target; // shape = (1)

    // run inference
    tensor2array(x_ref, x);
    TOP_FUNC (x,
              params.at("conv1.weight").data(), params.at("conv1.bias").data(),
              params.at("conv2.weight").data(), params.at("conv2.bias").data(),
              params.at("fc1.weight").data(), params.at("fc1.bias").data(),
              params.at("fc2.weight").data(), params.at("fc2.bias").data(),
              y);


    // run inference in pytorch
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x_ref);
    auto y_ref = model.forward(inputs).toTensor();

    if (!verify(y, y_ref)) {
      printf("%sFailed%s\n", Color::red, Color::reset);
      return 1;
    }

    if (++niters == 4) {
      break;
    }
  }

  printf("%sSucceed!%s\n", Color::green, Color::reset);
  return 0;
}
