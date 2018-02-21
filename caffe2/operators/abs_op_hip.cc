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
__global__ void AbsKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = fabs(X[i]);
  }
}

template <typename T>
__global__ void AbsGradientKernel(const int N, const T* X, const T* dY, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = X[i] == T(0) ? T(0) : (X[i] > T(0) ? dY[i] : -dY[i]);
  }
}

struct AbsHIPFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, HIPContext* device_context) {
    hipLaunchKernelGGL((AbsKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, x, y);
    return;
  }
};

struct AbsGradientHIPFunctor {
  template <typename T>
  inline void Run(
      const int n,
      const T* x,
      const T* dy,
      T* dx,
      HIPContext* device_context) {
    hipLaunchKernelGGL((AbsGradientKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, x, dy, dx);
    return;
  }
};

REGISTER_HIP_OPERATOR(
    Abs,
    UnaryElementwiseOp<TensorTypes<float>, HIPContext, AbsHIPFunctor>);
REGISTER_HIP_OPERATOR(
    AbsGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        WithoutBroadcast<AbsGradientHIPFunctor>>);
} // namespace caffe2
