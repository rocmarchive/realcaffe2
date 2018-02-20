#include "caffe2/operators/batch_matmul_op.h"
#include "caffe2/core/context_hip.h"

namespace caffe2 {

template <>
bool BatchMatMulOp<HIPContext, DefaultEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(BatchMatMul, BatchMatMulOp<HIPContext>);

} // namespace caffe2
