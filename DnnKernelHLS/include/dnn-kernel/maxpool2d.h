#ifndef DNNKERNEL_MAXPOOL2D_H
#define DNNKERNEL_MAXPOOL2D_H

#include <stdint.h>
#include <cfloat>
#include <algorithm>
#include <limits>

namespace dnnk {

static void maxpool2d(const float *x, int32_t width, int32_t height, int32_t channels, int32_t stride, float *y) {
  for (int ch = 0; ch < channels; ++ch) {
    for (int32_t h = 0; h < height; h += stride) {
      for (int32_t w = 0; w < width; w += stride) {
        float maxval = -FLT_MAX;

        for (int bh = 0; bh < stride; ++bh) {
          for (int bw = 0; bw < stride; ++bw) {
            maxval = std::max(maxval, x[(ch * height + h + bh) * width + w + bw]);
          }
        }

        y[(ch * (height / stride) + (h / stride)) * (width / stride) + w / stride] = maxval;
      }
    }
  }
}

}  // namespace dnnk

#endif  // DNNKERNEL_MAXPOOL2D_H
