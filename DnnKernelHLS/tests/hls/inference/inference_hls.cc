#include "dnn-kernel/inference.h"

#include <stdint.h>
#include <cstring>
#include <algorithm>

static const std::size_t kMaxSize = 16384;

void inference_hls(const float x[kMaxSize],
                   const float weight0[kMaxSize], const float bias0[kMaxSize],
                   const float weight1[kMaxSize], const float bias1[kMaxSize],
                   const float weight2[kMaxSize], const float bias2[kMaxSize],
                   const float weight3[kMaxSize], const float bias3[kMaxSize],
                   float y[kMaxSize]) {
  dnnk::inference(x,
                  weight0, bias0,
                  weight1, bias1,
                  weight2, bias2,
                  weight3, bias3,
                  y);
}

extern "C" {

void inference_top(const float x[kMaxSize],
                   const float weight0[kMaxSize], const float bias0[kMaxSize],
                   const float weight1[kMaxSize], const float bias1[kMaxSize],
                   const float weight2[kMaxSize], const float bias2[kMaxSize],
                   const float weight3[kMaxSize], const float bias3[kMaxSize],
                   float y[kMaxSize]) {
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9
#pragma HLS interface s_axilite port=x bundle=control
#pragma HLS interface s_axilite port=weight0 bundle=control
#pragma HLS interface s_axilite port=weight1 bundle=control
#pragma HLS interface s_axilite port=weight2 bundle=control
#pragma HLS interface s_axilite port=weight3 bundle=control
#pragma HLS interface s_axilite port=bias0 bundle=control
#pragma HLS interface s_axilite port=bias1 bundle=control
#pragma HLS interface s_axilite port=bias2 bundle=control
#pragma HLS interface s_axilite port=bias3 bundle=control
#pragma HLS interface s_axilite port=y bundle=control
#pragma HLS interface s_axilite port=return bundle=control

  dnnk::inference(x,
                  weight0, bias0,
                  weight1, bias1,
                  weight2, bias2,
                  weight3, bias3,
                  y);
}

void inference_dataflow(const float x[kMaxSize],
                        const float weight0[kMaxSize], const float bias0[kMaxSize],
                        const float weight1[kMaxSize], const float bias1[kMaxSize],
                        const float weight2[kMaxSize], const float bias2[kMaxSize],
                        const float weight3[kMaxSize], const float bias3[kMaxSize],
                        float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9
#pragma HLS interface s_axilite port=x bundle=control
#pragma HLS interface s_axilite port=weight0 bundle=control
#pragma HLS interface s_axilite port=weight1 bundle=control
#pragma HLS interface s_axilite port=weight2 bundle=control
#pragma HLS interface s_axilite port=weight3 bundle=control
#pragma HLS interface s_axilite port=bias0 bundle=control
#pragma HLS interface s_axilite port=bias1 bundle=control
#pragma HLS interface s_axilite port=bias2 bundle=control
#pragma HLS interface s_axilite port=bias3 bundle=control
#pragma HLS interface s_axilite port=y bundle=control
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface ap_ctrl_chain port=return bundle=control

#pragma HLS stable variable=x
#pragma HLS stable variable=weight0
#pragma HLS stable variable=bias0
#pragma HLS stable variable=weight1
#pragma HLS stable variable=bias1
#pragma HLS stable variable=weight2
#pragma HLS stable variable=bias2
#pragma HLS stable variable=weight3
#pragma HLS stable variable=bias3
#pragma HLS stable variable=y

  dnnk::inference(x,
                  weight0, bias0,
                  weight1, bias1,
                  weight2, bias2,
                  weight3, bias3,
                  y);
}


void inference_with_local_buffer(const float x[kMaxSize],
                                 const float weight0[kMaxSize], const float bias0[kMaxSize],
                                 const float weight1[kMaxSize], const float bias1[kMaxSize],
                                 const float weight2[kMaxSize], const float bias2[kMaxSize],
                                 const float weight3[kMaxSize], const float bias3[kMaxSize],
                                 float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9
#pragma HLS interface s_axilite port=x bundle=control
#pragma HLS interface s_axilite port=weight0 bundle=control
#pragma HLS interface s_axilite port=weight1 bundle=control
#pragma HLS interface s_axilite port=weight2 bundle=control
#pragma HLS interface s_axilite port=weight3 bundle=control
#pragma HLS interface s_axilite port=bias0 bundle=control
#pragma HLS interface s_axilite port=bias1 bundle=control
#pragma HLS interface s_axilite port=bias2 bundle=control
#pragma HLS interface s_axilite port=bias3 bundle=control
#pragma HLS interface s_axilite port=y bundle=control
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface ap_ctrl_chain port=return bundle=control

#pragma HLS stable variable=x
#pragma HLS stable variable=weight0
#pragma HLS stable variable=bias0
#pragma HLS stable variable=weight1
#pragma HLS stable variable=bias1
#pragma HLS stable variable=weight2
#pragma HLS stable variable=bias2
#pragma HLS stable variable=weight3
#pragma HLS stable variable=bias3
#pragma HLS stable variable=y

  const std::size_t x_size = 1 * 28 * 28;
  const std::size_t w0_size = 4 * 1 * 3 * 3, b0_size = 4;
  const std::size_t w1_size = 8 * 4 * 3 * 3, b1_size = 8;
  const std::size_t w2_size = 32 * 392, b2_size = 32;
  const std::size_t w3_size = 10 * 32, b3_size = 10;
  const std::size_t y_size = 10;

  float x_local[x_size];
  float w0_local[w0_size], b0_local[b0_size];
  float w1_local[w1_size], b1_local[b1_size];
  float w2_local[w2_size], b2_local[b2_size];
  float w3_local[w3_size], b3_local[b3_size];
  float y_local[y_size];

  // fetch to local buffer
  std::memcpy(x_local, x, x_size * sizeof(float));
  std::memcpy(w0_local, weight0, w0_size * sizeof(float));
  std::memcpy(b0_local, bias0, b0_size * sizeof(float));
  std::memcpy(w1_local, weight1, w1_size * sizeof(float));
  std::memcpy(b1_local, bias1, b1_size * sizeof(float));
  std::memcpy(w2_local, weight2, w2_size * sizeof(float));
  std::memcpy(b2_local, bias2, b2_size * sizeof(float));
  std::memcpy(w3_local, weight3, w3_size * sizeof(float));
  std::memcpy(b3_local, bias3, b3_size * sizeof(float));

  // run inference with local buffer
  dnnk::inference(x_local,
                  w0_local, b0_local,
                  w1_local, b1_local,
                  w2_local, b2_local,
                  w3_local, b3_local,
                  y_local);

  // store to global buffer
  std::memcpy(y, y_local, y_size * sizeof(float));
}


#define DECLARE_INFERENCE_WITH_LOCAL_BUFFER(NAME, CONV_FUNC, MAXPOOL_FUNC, RELU_FUNC, LINEAR_FUNC) \
  void NAME(const float x[kMaxSize],                                   \
            const float weight0[kMaxSize], const float bias0[kMaxSize], \
            const float weight1[kMaxSize], const float bias1[kMaxSize], \
            const float weight2[kMaxSize], const float bias2[kMaxSize], \
            const float weight3[kMaxSize], const float bias3[kMaxSize], \
            float y[kMaxSize]) {                                        \
  _Pragma("HLS dataflow")                                               \
    _Pragma("HLS interface m_axi port=x offset=slave bundle=gmem0")     \
    _Pragma("HLS interface m_axi port=weight0 offset=slave bundle=gmem1") \
    _Pragma("HLS interface m_axi port=weight1 offset=slave bundle=gmem2") \
    _Pragma("HLS interface m_axi port=weight2 offset=slave bundle=gmem3") \
    _Pragma("HLS interface m_axi port=weight3 offset=slave bundle=gmem4") \
    _Pragma("HLS interface m_axi port=bias0 offset=slave bundle=gmem5") \
    _Pragma("HLS interface m_axi port=bias1 offset=slave bundle=gmem6") \
    _Pragma("HLS interface m_axi port=bias2 offset=slave bundle=gmem7") \
    _Pragma("HLS interface m_axi port=bias3 offset=slave bundle=gmem8") \
    _Pragma("HLS interface m_axi port=y offset=slave bundle=gmem9")     \
    _Pragma("HLS interface s_axilite port=x bundle=control")            \
    _Pragma("HLS interface s_axilite port=weight0 bundle=control")      \
    _Pragma("HLS interface s_axilite port=weight1 bundle=control")      \
    _Pragma("HLS interface s_axilite port=weight2 bundle=control")      \
    _Pragma("HLS interface s_axilite port=weight3 bundle=control")      \
    _Pragma("HLS interface s_axilite port=bias0 bundle=control")        \
    _Pragma("HLS interface s_axilite port=bias1 bundle=control")        \
    _Pragma("HLS interface s_axilite port=bias2 bundle=control")        \
    _Pragma("HLS interface s_axilite port=bias3 bundle=control")        \
    _Pragma("HLS interface s_axilite port=y bundle=control")            \
    _Pragma("HLS interface s_axilite port=return bundle=control")       \
    _Pragma("HLS interface ap_ctrl_chain port=return bundle=control")   \
    _Pragma("HLS stable variable=x")                                    \
    _Pragma("HLS stable variable=weight0")                              \
    _Pragma("HLS stable variable=bias0")                                \
    _Pragma("HLS stable variable=weight1")                              \
    _Pragma("HLS stable variable=bias1")                                \
    _Pragma("HLS stable variable=weight2")                              \
    _Pragma("HLS stable variable=bias2")                                \
    _Pragma("HLS stable variable=weight3")                              \
    _Pragma("HLS stable variable=bias3")                                \
    _Pragma("HLS stable variable=y")                                    \
    const std::size_t x_size = 1 * 28 * 28;                             \
  const std::size_t w0_size = 4 * 1 * 3 * 3, b0_size = 4;               \
  const std::size_t w1_size = 8 * 4 * 3 * 3, b1_size = 8;               \
  const std::size_t w2_size = 32 * 392, b2_size = 32;                   \
  const std::size_t w3_size = 10 * 32, b3_size = 10;                    \
  const std::size_t y_size = 10;                                        \
  float x_local[x_size];                                                \
  float w0_local[w0_size], b0_local[b0_size];                           \
  float w1_local[w1_size], b1_local[b1_size];                           \
  float w2_local[w2_size], b2_local[b2_size];                           \
  float w3_local[w3_size], b3_local[b3_size];                           \
  float y_local[y_size];                                                \
  std::memcpy(x_local, x, x_size * sizeof(float));                      \
  std::memcpy(w0_local, weight0, w0_size * sizeof(float));              \
  std::memcpy(b0_local, bias0, b0_size * sizeof(float));                \
  std::memcpy(w1_local, weight1, w1_size * sizeof(float));              \
  std::memcpy(b1_local, bias1, b1_size * sizeof(float));                \
  std::memcpy(w2_local, weight2, w2_size * sizeof(float));              \
  std::memcpy(b2_local, bias2, b2_size * sizeof(float));                \
  std::memcpy(w3_local, weight3, w3_size * sizeof(float));              \
  std::memcpy(b3_local, bias3, b3_size * sizeof(float));                \
                                                                        \
  dnnk::inference_custom(x_local,                                       \
                         w0_local, b0_local,                            \
                         w1_local, b1_local,                            \
                         w2_local, b2_local,                            \
                         w3_local, b3_local,                            \
                         y_local,                                       \
                         CONV_FUNC, MAXPOOL_FUNC, RELU_FUNC, LINEAR_FUNC); \
                                                                        \
  std::memcpy(y, y_local, y_size * sizeof(float));                      \
}

DECLARE_INFERENCE_WITH_LOCAL_BUFFER(inference_pipelined_conv_v1, dnnk::conv2d_pipelined_v1, dnnk::maxpool2d, dnnk::relu, dnnk::linear);
DECLARE_INFERENCE_WITH_LOCAL_BUFFER(inference_pipelined_conv_v2, dnnk::conv2d_pipelined_v2, dnnk::maxpool2d, dnnk::relu, dnnk::linear);
DECLARE_INFERENCE_WITH_LOCAL_BUFFER(inference_unrolledx4_conv_v1, dnnk::conv2d_unrolled_v1<4>, dnnk::maxpool2d, dnnk::relu, dnnk::linear);
DECLARE_INFERENCE_WITH_LOCAL_BUFFER(inference_unrolledx4_conv_v2, (dnnk::conv2d_unrolled_v2<4, 4>), dnnk::maxpool2d, dnnk::relu, dnnk::linear);



void inference_final(const float x[kMaxSize],
                     const float weight0[kMaxSize], const float bias0[kMaxSize],
                     const float weight1[kMaxSize], const float bias1[kMaxSize],
                     const float weight2[kMaxSize], const float bias2[kMaxSize],
                     const float weight3[kMaxSize], const float bias3[kMaxSize],
                     float y[kMaxSize]) {
#pragma HLS dataflow
#pragma HLS interface m_axi port=x offset=slave bundle=gmem0
#pragma HLS interface m_axi port=weight0 offset=slave bundle=gmem1
#pragma HLS interface m_axi port=weight1 offset=slave bundle=gmem2
#pragma HLS interface m_axi port=weight2 offset=slave bundle=gmem3
#pragma HLS interface m_axi port=weight3 offset=slave bundle=gmem4
#pragma HLS interface m_axi port=bias0 offset=slave bundle=gmem5
#pragma HLS interface m_axi port=bias1 offset=slave bundle=gmem6
#pragma HLS interface m_axi port=bias2 offset=slave bundle=gmem7
#pragma HLS interface m_axi port=bias3 offset=slave bundle=gmem8
#pragma HLS interface m_axi port=y offset=slave bundle=gmem9
#pragma HLS interface s_axilite port=x bundle=control
#pragma HLS interface s_axilite port=weight0 bundle=control
#pragma HLS interface s_axilite port=weight1 bundle=control
#pragma HLS interface s_axilite port=weight2 bundle=control
#pragma HLS interface s_axilite port=weight3 bundle=control
#pragma HLS interface s_axilite port=bias0 bundle=control
#pragma HLS interface s_axilite port=bias1 bundle=control
#pragma HLS interface s_axilite port=bias2 bundle=control
#pragma HLS interface s_axilite port=bias3 bundle=control
#pragma HLS interface s_axilite port=y bundle=control
#pragma HLS interface s_axilite port=return bundle=control
#pragma HLS interface ap_ctrl_chain port=return bundle=control

#pragma HLS stable variable=x
#pragma HLS stable variable=weight0
#pragma HLS stable variable=bias0
#pragma HLS stable variable=weight1
#pragma HLS stable variable=bias1
#pragma HLS stable variable=weight2
#pragma HLS stable variable=bias2
#pragma HLS stable variable=weight3
#pragma HLS stable variable=bias3
#pragma HLS stable variable=y

  const std::size_t x_size = 1 * 28 * 28;
  const std::size_t w0_size = 4 * 1 * 3 * 3, b0_size = 4;
  const std::size_t w1_size = 8 * 4 * 3 * 3, b1_size = 8;
  const std::size_t w2_size = 32 * 392, b2_size = 32;
  const std::size_t w3_size = 10 * 32, b3_size = 10;
  const std::size_t y_size = 10;

  float x_local[x_size];
  float w0_local[w0_size], b0_local[b0_size];
  float w1_local[w1_size], b1_local[b1_size];
  float w2_local[w2_size], b2_local[b2_size];
  float w3_local[w3_size], b3_local[b3_size];
  float y_local[y_size];

  // fetch to local buffer
  std::memcpy(x_local, x, x_size * sizeof(float));
  std::memcpy(w0_local, weight0, w0_size * sizeof(float));
  std::memcpy(b0_local, bias0, b0_size * sizeof(float));
  std::memcpy(w1_local, weight1, w1_size * sizeof(float));
  std::memcpy(b1_local, bias1, b1_size * sizeof(float));
  std::memcpy(w2_local, weight2, w2_size * sizeof(float));
  std::memcpy(b2_local, bias2, b2_size * sizeof(float));
  std::memcpy(w3_local, weight3, w3_size * sizeof(float));
  std::memcpy(b3_local, bias3, b3_size * sizeof(float));

  // run inference with local buffer
  dnnk::inference_custom(x_local,
                         w0_local, b0_local,
                         w1_local, b1_local,
                         w2_local, b2_local,
                         w3_local, b3_local,
                         y_local,
                         dnnk::conv2d_unrolled_v2<4, 4>,
                         dnnk::relu,
                         dnnk::maxpool2d,
                         dnnk::conv2d_unrolled_v2<8, 4>,
                         dnnk::relu,
                         dnnk::maxpool2d,
                         dnnk::linear_opt<4>,
                         dnnk::relu,
                         dnnk::linear);

  // store to global buffer
  std::memcpy(y, y_local, y_size * sizeof(float));
}

}
