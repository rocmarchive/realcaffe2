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
#include "caffe2/operators/recurrent_network_op.h"

namespace caffe2 {

namespace detail {

template <typename T, typename Context>
void initializeRecurrentInput(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    Context* context);

namespace {

template <typename T>
__global__
void initRecurrentInput_kernel(
    size_t stateSize,
    const T* input,
    T* state) {
  // index into appropriate target buffer
  const int block_id = blockIdx.x;
  T* state_local = state + block_id*stateSize;

  // copy
  for (int idx=threadIdx.x; idx < stateSize; idx+=blockDim.x) {
    state_local[idx] = input[idx];
  }
}


}; // namespace

template <>
void repeatCopy(
    size_t repeat_n,
    size_t n,
    const float* src,
    float* dst,
    HIPContext* context) {
    hipLaunchKernelGGL((initRecurrentInput_kernel<float>), dim3(repeat_n), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), 
        n, src, dst);
}
template <>
void repeatCopy(
    size_t repeat_n,
    size_t n,
    const float16* src,
    float16* dst,
    HIPContext* context) {
    hipLaunchKernelGGL((initRecurrentInput_kernel<float16>), dim3(repeat_n), dim3(CAFFE_HIP_NUM_THREADS), 0, context->hip_stream(), 
        n, src, dst);
}

}; // namespace detail

template <>
bool RecurrentNetworkOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

template <>
bool RecurrentNetworkGradientOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

template <>
bool AccumulateInputGradientOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(1));
}

template <>
bool RNNApplyLinkOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(1));
}

REGISTER_HIP_OPERATOR(
    RecurrentNetwork,
    RecurrentNetworkOp<HIPContext>);
REGISTER_HIP_OPERATOR(
    RecurrentNetworkGradient,
    RecurrentNetworkGradientOp<HIPContext>);
REGISTER_HIP_OPERATOR(
    rnn_internal_accumulate_gradient_input,
    AccumulateInputGradientOp<HIPContext>);
REGISTER_HIP_OPERATOR(
    rnn_internal_apply_link,
    RNNApplyLinkOp<HIPContext>);


} // namespace caffe2
