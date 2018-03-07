#include "hip/hip_runtime.h"
/**
 * Copyright (c) 2018-present, Facebook, Inc.
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
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/operators/swish_op.h"

namespace caffe2 {

template <typename T>
__global__ void SwishKernel(const int N, const T* x, T* y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    y[i] = x[i] / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void
SwishGradientKernel(const int N, const T* x, const T* y, const T* dy, T* dx) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * (y[i] + (1. - y[i]) / (1. + exp(-x[i])));
  }
}

struct SwishHIPFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, HIPContext* device_context) {
    hipLaunchKernelGGL((SwishKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, x, y);
    return;
  }
};

template <>
template <typename T>
bool SwishGradientOp<HIPContext>::DoRunWithType() {
  auto& Xin = Input(X);
  auto& Yin = Input(Y);
  auto& DYin = Input(DY);
  auto* DXout = Output(DX);
  CAFFE_ENFORCE_EQ(Xin.size(), Yin.size());
  CAFFE_ENFORCE_EQ(DYin.size(), Yin.size());
  DXout->ResizeLike(Yin);

  const int n = Xin.size();
  const T* x = Xin.template data<T>();
  const T* y = Yin.template data<T>();
  const T* dy = DYin.template data<T>();
  T* dx = DXout->template mutable_data<T>();
  hipLaunchKernelGGL((SwishGradientKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), n, x, y, dy, dx);
  return true;
}

template <>
bool SwishGradientOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
}

REGISTER_HIP_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        HIPContext,
        SwishHIPFunctor>);
REGISTER_HIP_OPERATOR(SwishGradient, SwishGradientOp<HIPContext>);
} // namespace caffe2
