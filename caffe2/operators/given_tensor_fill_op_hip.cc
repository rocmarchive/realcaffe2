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
#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, HIPContext>);
REGISTER_HIP_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorBoolFill,
    GivenTensorFillOp<bool, HIPContext>);
}
