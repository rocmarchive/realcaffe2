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

#include "rmsprop_op.h"
#include "caffe2/core/common_hip.h"
#include "caffe2/core/context_hip.h"
#include "hip/hip_runtime.h"

namespace caffe2 {

__global__ void RmsPropUpdate(int N,
                              const float* g,
                              const float* ms,
                              const float* mom,
                              float* ng,
                              float* nms,
                              float* nmom,
                              float decay,
                              float momentum,
                              float epsilon,
                              const float* lr)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        // Update new mean square estimate
        nms[i] = ms[i] + (1.0f - decay) * (g[i] * g[i] - ms[i]);
        // Update momentum estimate
        nmom[i] = mom[i] * momentum + lr[0] * g[i] / sqrtf(epsilon + nms[i]);
        // New gradient is the momentum
        ng[i] = nmom[i];
    }
}

template <>
void rmsprop_update<HIPContext>(int N,
                                const float* g,
                                const float* ms,
                                const float* mom,
                                float* ng,
                                float* nms,
                                float* nmom,
                                float decay,
                                float momentum,
                                float epsilon,
                                const float* lr,
                                HIPContext* context)
{
    hipLaunchKernelGGL((RmsPropUpdate),
                       dim3(CAFFE_GET_BLOCKS(N)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context->hip_stream(),
                       N,
                       g,
                       ms,
                       mom,
                       ng,
                       nms,
                       nmom,
                       decay,
                       momentum,
                       epsilon,
                       lr);
}

REGISTER_HIP_OPERATOR(RmsProp, RmsPropOp<float, HIPContext>);
}
