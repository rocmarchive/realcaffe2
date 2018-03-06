#include "hip/hip_runtime.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/operators/replace_nan_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void
replace_nan_kernel(const T value, const TIndex size, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, size) {
    if (isnan(X[i])) {
      Y[i] = value;
    } else {
      Y[i] = X[i];
    }
  }
}
} // namespace

template <>
template <typename T>
void ReplaceNaNOp<HIPContext>::ReplaceNaN(
    const T& value,
    const TIndex size,
    const T* X,
    T* Y) {
  hipLaunchKernelGGL((replace_nan_kernel), dim3(CAFFE_GET_BLOCKS(size)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), value, size, X, Y);
}
REGISTER_HIP_OPERATOR(ReplaceNaN, ReplaceNaNOp<HIPContext>);
} // namespace caffe2
