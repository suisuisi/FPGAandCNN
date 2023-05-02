#ifndef DNNKERNEL_CONV2D_H
#define DNNKERNEL_CONV2D_H

#include <stdint.h>
#include <algorithm>

namespace dnnk {

static void conv2d(const float* x, const float* weight, const float* bias, int32_t width, int32_t height,
                   int32_t in_channels, int32_t out_channels, int32_t ksize, float* y) {
  for (int32_t och = 0; och < out_channels; ++och) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        float sum = 0.f;

        for (int32_t ich = 0; ich < in_channels; ++ich) {
          for (int32_t kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {
              int32_t ph = h + kh - ksize/2;
              int32_t pw = w + kw - ksize/2;

              // zero padding
              if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                continue;
              }

              int64_t pix_idx = (ich * height + ph) * width + pw;
              int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

              sum += x[pix_idx] * weight[weight_idx];
            }
          }
        }

        // add bias
        sum += bias[och];

        y[(och * height + h) * width + w] = sum;
      }
    }
  }
}


static void conv2d_pipelined_v1(const float* x, const float* weight, const float* bias, int32_t width, int32_t height,
                                int32_t in_channels, int32_t out_channels, int32_t ksize, float* y) {
  for (int32_t och = 0; och < out_channels; ++och) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        float sum = 0.f;

        for (int32_t ich = 0; ich < in_channels; ++ich) {
          for (int32_t kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {
#pragma HLS pipeline II=1

              int32_t ph = h + kh - ksize/2;
              int32_t pw = w + kw - ksize/2;

              // zero padding
              if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                continue;
              }

              int64_t pix_idx = (ich * height + ph) * width + pw;
              int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

              sum += x[pix_idx] * weight[weight_idx];
            }
          }
        }

        // add bias
        sum += bias[och];

        y[(och * height + h) * width + w] = sum;
      }
    }
  }
}

static void conv2d_pipelined_v2(const float* x, const float* weight, const float* bias, int32_t width, int32_t height,
                                int32_t in_channels, int32_t out_channels, int32_t ksize, float* y) {
  static const int kShiftRegLength = 4;

  for (int32_t och = 0; och < out_channels; ++och) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        float shift_reg[kShiftRegLength + 1];
#pragma HLS array_partition variable=shift_reg complete

        int32_t glob_idx = 0;
        for (int32_t ich = 0; ich < in_channels; ++ich) {
          for (int32_t kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {
#pragma HLS pipeline II=1

              int32_t ph = h + kh - ksize/2;
              int32_t pw = w + kw - ksize/2;

              // zero padding
              if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                continue;
              }

              int64_t pix_idx = (ich * height + ph) * width + pw;
              int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

              float mul = x[pix_idx] * weight[weight_idx];

              // local sum
              for (int i = 0; i < kShiftRegLength; ++i) {
                if (i == 0) {
                  if (glob_idx < kShiftRegLength) {
                    shift_reg[kShiftRegLength] = mul;
                  } else {
                    shift_reg[kShiftRegLength] = shift_reg[0] + mul;
                  }
                }

                shift_reg[i] = shift_reg[i + 1];
              }

              ++glob_idx;
            }
          }
        }

        // global sum
        float sum = 0.f;
        for (int i = 0; i < kShiftRegLength; ++i) {
#pragma HLS pipeline II=1
          sum += shift_reg[i];
        }

        // add bias
        sum += bias[och];

        y[(och * height + h) * width + w] = sum;
      }
    }
  }
}


template <int UNROLL_X>
static void conv2d_unrolled_v1(const float* x, const float* weight, const float* bias, int32_t width, int32_t height,
                               int32_t in_channels, int32_t out_channels, int32_t ksize, float* y) {

  for (int32_t och = 0; och < out_channels; ++och) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t block_w = 0; block_w < width; block_w += UNROLL_X) {
        float sum[UNROLL_X];
#pragma HLS array_partition variable=sum complete

        for (int32_t ich = 0; ich < in_channels; ++ich) {
          for (int32_t kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {
#pragma HLS pipeline II=4
              for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
                if (block_w + local_w < width) {

                  int32_t w = block_w + local_w;

                  int32_t ph = h + kh - ksize/2;
                  int32_t pw = w + kw - ksize/2;

                  float last = (ich == 0 && kh == 0 && kw == 0) ? 0 : sum[local_w];

                  // zero padding
                  if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                    sum[local_w] = last;
                    continue;
                  }

                  int64_t pix_idx = (ich * height + ph) * width + pw;
                  int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

                  sum[local_w] = last + x[pix_idx] * weight[weight_idx];
                }
              }
            }
          }
        }

        for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
          if (block_w + local_w < width) {

            int32_t w = block_w + local_w;

            // add bias
            y[(och * height + h) * width + w] = sum[local_w] + bias[och];
          }
        }
      }
    }
  }
}


template <int UNROLL_X, int UNROLL_OCH>
static void conv2d_unrolled_v2(const float* x, const float* weight, const float* bias, int32_t width, int32_t height,
                               int32_t in_channels, int32_t out_channels, int32_t ksize, float* y) {

  for (int32_t block_och = 0; block_och < out_channels; block_och += UNROLL_OCH) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t block_w = 0; block_w < width; block_w += UNROLL_X) {
        float sum[UNROLL_OCH][UNROLL_X];
#pragma HLS array_partition variable=sum complete dim=0

        for (int32_t ich = 0; ich < in_channels; ++ich) {
          for (int32_t kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {
#pragma HLS pipeline II=4
              for (int local_och = 0; local_och < UNROLL_OCH; local_och++) {
#pragma HLS unroll
                for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
                  if (block_w + local_w < width && block_och + local_och < out_channels) {

                    int32_t och = block_och + local_och;
                    int32_t w = block_w + local_w;

                    int32_t ph = h + kh - ksize/2;
                    int32_t pw = w + kw - ksize/2;

                    float last = (ich == 0 && kh == 0 && kw == 0) ? 0 : sum[local_och][local_w];

                    // zero padding
                    if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                      sum[local_och][local_w] = last;
                      continue;
                    }

                    int64_t pix_idx = (ich * height + ph) * width + pw;
                    int64_t weight_idx = ((och * in_channels + ich) * ksize + kh) * ksize + kw;

                    sum[local_och][local_w] = last + x[pix_idx] * weight[weight_idx];
                  }
                }
              }
            }
          }
        }

        for (int local_och = 0; local_och < UNROLL_OCH; local_och++) {
#pragma HLS unroll
          for (int local_w = 0; local_w < UNROLL_X; local_w++) {
#pragma HLS unroll
            if (block_w + local_w < width && block_och + local_och < out_channels) {
              int32_t och = block_och + local_och;
              int32_t w = block_w + local_w;

              // add bias
              y[(och * height + h) * width + w] = sum[local_och][local_w] + bias[och];
            }
          }
        }
      }
    }
  }
}

}  // namespace dnnk

#endif  // DNNKERNEL_CONV2D_H
