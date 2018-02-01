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

#include "caffe2/core/context_hip.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/operators/logit_op.h"

namespace caffe2 {

template <typename T>
__global__ void LogitKernel(const int N, const T* X, const float eps, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = fminf(X[i], (1.0 - eps));
    Y[i] = fmaxf(Y[i], eps);
    Y[i] = logf(Y[i] / (1.0 - Y[i]));
  }
}

__global__ void LogitGradientKernel(
    const int N,
    const float* X,
    const float* dY,
    const float eps,
    float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = (X[i] < eps || X[i] > 1.0 - eps) ? 0 : (dY[i] / X[i] / (1 - X[i]));
  }
}

struct LogitHIPFunctor {
  explicit LogitHIPFunctor(OperatorBase& op)
      : eps_(op.GetSingleArgument<float>("eps", 1e-6)) {
    CAFFE_ENFORCE_GT(eps_, 0.0);
    CAFFE_ENFORCE_LT(eps_, 0.5);
  }
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, HIPContext* device_context) {
    hipLaunchKernelGGL((LogitKernel<T>), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, device_context->hip_stream(), n, x, eps_, y);
    return;
  }

 private:
  float eps_;
};

template <>
bool LogitGradientOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int n = X.size();
  hipLaunchKernelGGL((LogitGradientKernel), dim3(CAFFE_GET_BLOCKS(n)), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      n, X.data<float>(), dY.data<float>(), eps_, dX->mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        LogitHIPFunctor>);
REGISTER_HIP_OPERATOR(LogitGradient, LogitGradientOp<float, HIPContext>);
} // namespace caffe2
