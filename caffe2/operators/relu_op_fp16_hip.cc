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

#include "caffe2/core/common_hip.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {
namespace {
__global__ void ReluKernelHalf(const int N, const __half* X, __half* Y) {
  const __half kZero = __float2half(0.0);
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __hgt(X[i], kZero) ? X[i] : kZero;
  }
}

__global__ void ReluKernelHalf2(const int N, const __half2* X, __half2* Y) {
  const __half2 kZero = __float2half2_rn(0.0);
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = __hmul2(__hgt2(X[i], kZero), X[i]);
  }
}

__global__ void ReluGradientKernelHalf(
    const int N, const __half* Y, const __half* dY, __half* dX) {
  const __half kZero = __float2half(0.0);
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = __hgt(Y[i], kZero) ? dY[i] : kZero;
  }
}
}  // namespace

template <>
bool ReluOp<float16, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
  if (X.size() % 2 == 0) {
    hipLaunchKernelGGL((ReluKernelHalf2), dim3(CAFFE_GET_BLOCKS(X.size() / 2)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(),
        X.size() / 2, reinterpret_cast<const __half2*>(X.data<float16>()),
        reinterpret_cast<__half2*>(Y->mutable_data<float16>()));
    return true;
  } else {
    hipLaunchKernelGGL((ReluKernelHalf), dim3(CAFFE_GET_BLOCKS(X.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(),
        X.size(), reinterpret_cast<const __half*>(X.data<float16>()),
        reinterpret_cast<__half*>(Y->mutable_data<float16>()));
    return true;
  }
}

template <>
bool ReluGradientOp<float16, HIPContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
  hipLaunchKernelGGL((ReluGradientKernelHalf), dim3(CAFFE_GET_BLOCKS(Y.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(),
      Y.size(), reinterpret_cast<const __half*>(Y.data<float16>()),
      reinterpret_cast<const __half*>(dY.data<float16>()),
      reinterpret_cast<__half*>(dX->mutable_data<float16>()));
  return true;
}

OPERATOR_SCHEMA(ReluFp16);
OPERATOR_SCHEMA(ReluFp16Gradient);

REGISTER_HIP_OPERATOR(ReluFp16, ReluOp<float16, HIPContext>);
REGISTER_HIP_OPERATOR(ReluFp16Gradient, ReluGradientOp<float16, HIPContext>);
}  // namespace caffe2

