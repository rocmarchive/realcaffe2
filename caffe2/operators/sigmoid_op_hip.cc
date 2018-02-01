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
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

template <typename T>
__global__ void SigmoidKernel(const int N, const T* x, T* y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    y[i] = 1. / (1. + exp(-x[i]));
  }
}

template <typename T>
__global__ void SigmoidGradientKernel(const int N, const T* y, const T* dy,
                              T* dx) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dx[i] = dy[i] * y[i] * (1. - y[i]);
  }
}

struct SigmoidHIPFunctor {
  template <typename T>
  inline void operator()(const int n, const T* x,
                         T* y, HIPContext* device_context) {
    hipLaunchKernelGGL((SigmoidKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, x, y);
    return;
  }
};

struct SigmoidGradientHIPFunctor {
  template <typename T>
  inline void Run(const int n, const T* y, const T* dy,
                  T* dx, HIPContext* device_context) {
    hipLaunchKernelGGL((SigmoidGradientKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, y, dy, dx);
    return;
  }
};

REGISTER_HIP_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<TensorTypes<float>, HIPContext, SigmoidHIPFunctor>);
REGISTER_HIP_OPERATOR(
    SigmoidGradient, BinaryElementwiseOp<
        TensorTypes<float>, HIPContext,
        WithoutBroadcast<SigmoidGradientHIPFunctor>>);
}  // namespace caffe2
