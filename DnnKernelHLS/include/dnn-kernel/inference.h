#ifndef DNNKERNEL_INFERENCE_H
#define DNNKERNEL_INFERENCE_H

#include "conv2d.h"
#include "maxpool2d.h"
#include "relu.h"
#include "linear.h"

#include <stdint.h>
#include <algorithm>

namespace dnnk {

template <typename CONV_FUNC, typename MAXPOOL_FUNC, typename RELU_FUNC, typename LINEAR_FUNC>
static void inference_custom(const float* x,
                             const float* weight0, const float* bias0,
                             const float* weight1, const float* bias1,
                             const float* weight2, const float* bias2,
                             const float* weight3, const float* bias3,
                             float* y,
                             CONV_FUNC* conv1_f,
                             RELU_FUNC* relu1_f,
                             MAXPOOL_FUNC* maxpool1_f,
                             CONV_FUNC* conv2_f,
                             RELU_FUNC* relu2_f,
                             MAXPOOL_FUNC* maxpool2_f,
                             LINEAR_FUNC* linear1_f,
                             RELU_FUNC* relu3_f,
                             LINEAR_FUNC* linear2_f) {
#pragma HLS inline

  static const int kWidths[] = {28, 14, 7};
  static const int kHeights[] = {28, 14, 7};
  static const int kChannels[] = {1, 4, 8, 32, 10};

  float x1[kWidths[0] * kHeights[0] * kChannels[1]];
  float x2[kWidths[0] * kHeights[0] * kChannels[1]];
  float x3[kWidths[1] * kHeights[1] * kChannels[1]];
  float x4[kWidths[1] * kHeights[1] * kChannels[2]];
  float x5[kWidths[1] * kHeights[1] * kChannels[2]];
  float x6[kWidths[2] * kHeights[2] * kChannels[2]];
  float x7[kChannels[3]];
  float x8[kChannels[3]];

  // 1st layer
  conv1_f(x, weight0, bias0, kWidths[0], kHeights[0], kChannels[0], kChannels[1], 3, x1);
  relu1_f(x1, kWidths[0] * kHeights[0] * kChannels[1], x2);
  maxpool1_f(x2, kWidths[0], kHeights[0], kChannels[1], 2, x3);

  // 2nd layer
  conv2_f(x3, weight1, bias1, kWidths[1], kHeights[1], kChannels[1], kChannels[2], 3, x4);
  relu2_f(x4, kWidths[1] * kHeights[1] * kChannels[2], x5);
  maxpool2_f(x5, kWidths[1], kHeights[1], kChannels[2], 2, x6);

  // 3rd layer
  linear1_f(x6, weight2, bias2, kWidths[2] * kHeights[2] * kChannels[2], kChannels[3], x7);
  relu3_f(x7, kChannels[3], x8);

  // 4th layer
  linear2_f(x8, weight3, bias3, kChannels[3], kChannels[4], y);
}

template <typename CONV_FUNC, typename MAXPOOL_FUNC, typename RELU_FUNC, typename LINEAR_FUNC>
static void inference_custom(const float* x,
                             const float* weight0, const float* bias0,
                             const float* weight1, const float* bias1,
                             const float* weight2, const float* bias2,
                             const float* weight3, const float* bias3,
                             float* y,
                             CONV_FUNC* conv2d_f,
                             MAXPOOL_FUNC* maxpool2d_f,
                             RELU_FUNC* relu_f,
                             LINEAR_FUNC* linear_f) {
#pragma HLS inline
  inference_custom(x,
                   weight0, bias0,
                   weight1, bias1,
                   weight2, bias2,
                   weight3, bias3,
                   y,
                   conv2d_f, relu_f, maxpool2d_f,
                   conv2d_f, relu_f, maxpool2d_f,
                   linear_f, relu_f,
                   linear_f);
}

static void inference(const float* x,
                      const float* weight0, const float* bias0,
                      const float* weight1, const float* bias1,
                      const float* weight2, const float* bias2,
                      const float* weight3, const float* bias3,
                      float* y) {
#pragma HLS inline

  inference_custom(x,
                   weight0, bias0,
                   weight1, bias1,
                   weight2, bias2,
                   weight3, bias3,
                   y,
                   conv2d, maxpool2d, relu, linear);
}

}  // namespace dnnk

#endif  // DNNKERNEL_INFERENCE_H
