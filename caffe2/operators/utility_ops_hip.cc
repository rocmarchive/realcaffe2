/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/context_hip.h"
#include "caffe2/operators/flatten_op.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool WeightedSumOp<HIPContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float>();
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
bool SumOp<HIPContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (Input(0).IsType<float16>()) {
    return DoRunWithType<float16, float16>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
class CopyOnDeviceLikeOp<HIPContext, HIPContext, HIPContext>
    : public Operator<HIPContext> {
 public:
  CopyOnDeviceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<HIPContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(HIPContext);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor<HIPContext>>(0);
    HIPContext context(GetGPUIDForPointer(Input(1).raw_data()));
    output->ResizeLike(input);
    context.template CopyItems<HIPContext, HIPContext>(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

REGISTER_HIP_OPERATOR(Print, PrintOp<HIPContext>);
REGISTER_HIP_OPERATOR(Flatten, FlattenOp<HIPContext>);
REGISTER_HIP_OPERATOR(FlattenToVec, FlattenToVecOp<HIPContext>);
REGISTER_HIP_OPERATOR(Alias, AliasOp<HIPContext>);
REGISTER_HIP_OPERATOR(ResizeLike, ResizeLikeOp<HIPContext>);
REGISTER_HIP_OPERATOR(Sum, SumOp<HIPContext>);
REGISTER_HIP_OPERATOR(WeightedSum, WeightedSumOp<HIPContext>);
// From whatever the current context, ensure the output is TensorCPU
REGISTER_HIP_OPERATOR(
    EnsureCPUOutput,
    CopyOp<HIPContext, CPUContext, HIPContext>);
// From CPU, copy it to whatever the current context
REGISTER_HIP_OPERATOR(
    CopyFromCPUInput,
    CopyOp<HIPContext, HIPContext, CPUContext>);

// CopyGPUToCPU and CopyCPUToGPU should both be carried out in a hip context,
// since gpu code will be involved.
REGISTER_HIP_OPERATOR(
    CopyGPUToCPU,
    CopyOp<HIPContext, CPUContext, HIPContext>);
REGISTER_HIP_OPERATOR(
    CopyCPUToGPU,
    CopyOp<HIPContext, HIPContext, CPUContext>);
// If we only specify Copy, we assume that it is a gpu to gpu copy - maybe
// involving different GPUs.
REGISTER_HIP_OPERATOR(Copy, CopyOp<HIPContext, HIPContext, HIPContext>);

REGISTER_HIP_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<HIPContext, HIPContext, HIPContext>);

REGISTER_HIP_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<HIPContext>);

} // namespace caffe2
