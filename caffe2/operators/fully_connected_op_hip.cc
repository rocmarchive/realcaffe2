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
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

template <>
bool FullyConnectedOp<HIPContext>::RunOnDevice()
{
    if(Input(0).IsType<float>())
    {
        return DoRunWithType<float,    // X
                             float,    // W
                             float,    // B
                             float,    // Y
                             float>(); // Math
    }
    else if(Input(0).IsType<float16>())
    {
        if(float16_compute_)
        {
            return DoRunWithType<float16,    // X
                                 float16,    // W
                                 float16,    // B
                                 float16,    // Y
                                 float16>(); // Math
        }
        else
        {
            return DoRunWithType<float16,  // X
                                 float16,  // W
                                 float16,  // B
                                 float16,  // Y
                                 float>(); // Math
        }
    }
    else
    {
        CAFFE_THROW("Unsupported type");
    }
    return false;
}

template <>
bool FullyConnectedGradientOp<HIPContext>::RunOnDevice()
{
    if(Input(0).IsType<float>())
    {
        return DoRunWithType<float,    //  X
                             float,    //  W
                             float,    // dY
                             float,    //  B
                             float,    // dX
                             float,    // dW
                             float,    // dB
                             float>(); // Math
    }
    else if(Input(0).IsType<float16>())
    {
        if(float16_compute_)
        {
            return DoRunWithType<float16,    //  X
                                 float16,    //  W
                                 float16,    // dY
                                 float16,    //  B
                                 float16,    // dX
                                 float16,    // dW
                                 float16,    // dB
                                 float16>(); // Math
        }
        else
        {
            return DoRunWithType<float16,  //  X
                                 float16,  //  W
                                 float16,  // dY
                                 float16,  //  B
                                 float16,  // dX
                                 float16,  // dW
                                 float16,  // dB
                                 float>(); // Math
        }
    }
    else
    {
        CAFFE_THROW("Unsupported type");
    }
    return false;
}

REGISTER_HIP_OPERATOR(FC, FullyConnectedOp<HIPContext>);
REGISTER_HIP_OPERATOR(FCGradient, FullyConnectedGradientOp<HIPContext>);

} // namespace caffe2
