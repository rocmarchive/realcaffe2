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
#include "caffe2/core/operator.h"
#include "caffe2/operators/no_default_engine_op.h"

namespace caffe2 {
// Communication operators do not have default engines.
REGISTER_HIP_OPERATOR(CreateCommonWorld, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(CloneCommonWorld, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Broadcast, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Reduce, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Allgather, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Allreduce, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(SendTensor, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(ReceiveTensor, NoDefaultEngineOp<HIPContext>);

} // namespace caffe2
