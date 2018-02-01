#include "hip/hip_runtime.h"
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

#include <cmath>
#include "caffe2/core/context_hip.h"
#include "caffe2/operators/filler_op.h"

namespace caffe2 {

namespace {
__global__ void FillRangeKernel(const int n, float* data) {
  HIP_1D_KERNEL_LOOP(index, n) {
    data[index] = index;
  }
}

template <typename T>
__global__ void FillDiagonalKernel(
    const int num_diagonal_elements,
    const TIndex step_size,
    const T value,
    T* data) {
  HIP_1D_KERNEL_LOOP(index, num_diagonal_elements) {
    data[index * step_size] = value;
  }
}
}

template <>
bool RangeFillOp<float, HIPContext>::Fill(TensorCUDA* output) {
  int N = output->size();
  hipLaunchKernelGGL((FillRangeKernel), dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), N, output->mutable_data<float>());
  return true;
}

template <>
template <typename T>
bool DiagonalFillOp<HIPContext>::FillWithType(TensorCUDA* output) {
  VerifyOutputShape(output);
  auto* data = output->template mutable_data<T>();
  int size = output->size();
  // first fill everything with 0
  math::Set<T, HIPContext>(size, T(0), data, &context_);

  T value = OperatorBase::GetSingleArgument<T>("value", 0);
  TIndex step_size = GetStepSize(output);
  int num_diagonal_elements = ceil((float)size / step_size);

  hipLaunchKernelGGL((FillDiagonalKernel), dim3(CAFFE_GET_BLOCKS(num_diagonal_elements)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), num_diagonal_elements, step_size, value, data);
  return true;
}

REGISTER_HIP_OPERATOR(UniformFill, UniformFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(UniformIntFill, UniformFillOp<int, HIPContext>);
REGISTER_HIP_OPERATOR(ConstantFill, ConstantFillOp<HIPContext>);
REGISTER_HIP_OPERATOR(DiagonalFill, DiagonalFillOp<HIPContext>);
REGISTER_HIP_OPERATOR(GaussianFill, GaussianFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(XavierFill, XavierFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(MSRAFill, MSRAFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(RangeFill, RangeFillOp<float, HIPContext>);

} // namespace caffe2
