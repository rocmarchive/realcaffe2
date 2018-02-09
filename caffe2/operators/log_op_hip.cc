#include "caffe2/core/context_hip.h"
#include "caffe2/operators/math_ops.h"

namespace caffe2 {

struct LogHIPFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, HIPContext* device_context) {
    math::Log<T, HIPContext>(n, x, y, device_context);
  }
};

REGISTER_HIP_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, HIPContext, LogHIPFunctor>);
}
