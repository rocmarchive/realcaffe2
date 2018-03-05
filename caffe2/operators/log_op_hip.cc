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
#include "caffe2/operators/math_ops.h"

namespace caffe2 {

struct LogHIPFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, HIPContext* device_context) {
    math::Log<T, HIPContext>(n, x, y, device_context);
  }
};

REGISTER_HIP_OPERATOR(
    Log,
    UnaryElementwiseOp<TensorTypes<float>, HIPContext, LogHIPFunctor>);
}
