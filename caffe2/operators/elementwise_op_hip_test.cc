#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/context_hip.h"
#include "caffe2/core/flags.h"

CAFFE2_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::HIPContext>(const int N, const bool* x, bool* y) {
  hipMemcpy(y, x, N * sizeof(bool), hipMemcpyHostToDevice);
}

template <>
caffe2::OperatorDef CreateOperatorDef<caffe2::HIPContext>() {
  caffe2::OperatorDef def;
  def.mutable_device_option()->set_device_type(caffe2::HIP);
  return def;
}

TEST(ElementwiseGPUTest, And) {
  if (!caffe2::HasHipGPU())
    return;
  elementwiseAnd<caffe2::HIPContext>();
}

TEST(ElementwiseGPUTest, Or) {
  if (!caffe2::HasHipGPU())
    return;
  elementwiseOr<caffe2::HIPContext>();
}

TEST(ElementwiseGPUTest, Xor) {
  if (!caffe2::HasHipGPU())
    return;
  elementwiseXor<caffe2::HIPContext>();
}

TEST(ElementwiseGPUTest, Not) {
  if (!caffe2::HasHipGPU())
    return;
  elementwiseNot<caffe2::HIPContext>();
}
