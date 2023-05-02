#include "dnn-kernel/maxpool2d.h"

#include <stdint.h>
#include <algorithm>

static const std::size_t kMaxSize = 65536;

void maxpool2d_hls(const float x[kMaxSize], int32_t width, int32_t height, int32_t channels, int32_t stride, float y[kMaxSize]) {

  dnnk::maxpool2d(x, width, height, channels, stride, y);
}
