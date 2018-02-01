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

#include "caffe2/operators/half_float_ops.h"

#include "caffe2/core/context_hip.h"

#ifdef CAFFE_HAS_CUDA_FP16

namespace caffe2 {
namespace {
__global__ void FloatToHalfKernel(const int N, const float* X, half* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __float2half(X[i]);
  }
}

__global__ void HalfToFloatKernel(const int N, const half* X, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __half2float(X[i]);
  }
}
}

template <>
bool FloatToHalfOp<HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((FloatToHalfKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      X.size(),
      X.data<float>(),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
  return true;
}

template <>
bool HalfToFloatOp<HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  hipLaunchKernelGGL((HalfToFloatKernel), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      X.size(),
      reinterpret_cast<const half*>(X.data<float16>()),
      Y->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(FloatToHalf, FloatToHalfOp<HIPContext>);
REGISTER_HIP_OPERATOR(HalfToFloat, HalfToFloatOp<HIPContext>);
} // namespace caffe2

#endif // CAFFE_HAS_CUDA_FP16
