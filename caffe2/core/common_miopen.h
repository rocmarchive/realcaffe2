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
#ifndef CAFFE2_CORE_COMMON_MIOPEN_H_
#define CAFFE2_CORE_COMMON_MIOPEN_H_

#include <array>
#include <mutex>
#include "miopen/miopen.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

namespace internal {
/**
 * A helper function to obtain miopen error strings.
 */
inline const char* miopenGetErrorString(miopenStatus_t status) {
  switch (status) {
    case MIOPEN_STATUS_SUCCESS:
      return "MIOPEN_STATUS_SUCCESS";
    case MIOPEN_STATUS_NOT_INITIALIZED:
      return "MIOPEN_STATUS_NOT_INITIALIZED";
    case MIOPEN_STATUS_ALLOC_FAILED:
      return "MIOPEN_STATUS_ALLOC_FAILED";
    case MIOPEN_STATUS_BAD_PARAM:
      return "MIOPEN_STATUS_BAD_PARAM";
    case MIOPEN_STATUS_INTERNAL_ERROR:
      return "MIOPEN_STATUS_INTERNAL_ERROR";
    case MIOPEN_STATUS_INVALID_VALUE:
      return "MIOPEN_STATUS_INVALID_VALUE";
    case MIOPEN_STATUS_ARCH_MISMATCH:
      return "MIOPEN_STATUS_ARCH_MISMATCH";
    case MIOPEN_STATUS_MAPPING_ERROR:
      return "MIOPEN_STATUS_MAPPING_ERROR";
    case MIOPEN_STATUS_EXECUTION_FAILED:
      return "MIOPEN_STATUS_EXECUTION_FAILED";
    case MIOPEN_STATUS_NOT_SUPPORTED:
      return "MIOPEN_STATUS_NOT_SUPPORTED";
    case MIOPEN_STATUS_LICENSE_ERROR:
      return "MIOPEN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown MIOPEN error number";
  }
}
} // namespace internal

// A macro that wraps around a miopen statement so we can check if the miopen
// execution finishes or not.
#define MIOPEN_ENFORCE(condition)                          \
  do {                                                    \
    miopenStatus_t status = condition;                     \
    CAFFE_ENFORCE_EQ(                                     \
        status,                                           \
        MIOPEN_STATUS_SUCCESS,                             \
        ", Error at: ",                                   \
        __FILE__,                                         \
        ":",                                              \
        __LINE__,                                         \
        ": ",                                             \
        ::caffe2::internal::miopenGetErrorString(status)); \
  } while (0)
#define MIOPEN_CHECK(condition)                              \
  do {                                                      \
    miopenStatus_t status = condition;                       \
    CHECK(status == MIOPEN_STATUS_SUCCESS)                   \
        << ::caffe2::internal::miopenGetErrorString(status); \
  } while (0)

// report the version of cuDNN Caffe2 was compiled with
inline size_t miopenCompiledVersion() {
  return MIOPEN_VERSION;
}
// report the runtime version of cuDNN
inline size_t miopenRuntimeVersion() {
  return miopenGetVersion();
}
}

/**
 * miopenTypeWrapper is a wrapper class that allows us to refer to the miopen type
 * in a template function. The class is specialized explicitly for different
 * data types below.
 */
template <typename T>
class miopenTypeWrapper;

template <>
class miopenTypeWrapper<float> {
 public:
  static const miopenDataType_t type = MIOPEN_DATA_FLOAT;
  typedef const float ScalingParamType;
  typedef float BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static const ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class miopenTypeWrapper<double> {
 public:
  static const miopenDataType_t type = MIOPEN_DATA_DOUBLE;
  typedef const double ScalingParamType;
  typedef double BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class miopenTypeWrapper<float16> {
 public:
  static const miopenDataType_t type = MIOPEN_DATA_HALF;
  typedef const float ScalingParamType;
  typedef float BNParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

/**
 * A wrapper function to convert the Caffe storage order to miopen storage order
 * enum values.
 */
inline miopenTensorFormat_t GetCudnnTensorFormat(const StorageOrder& order) {
  switch (order) {
    case StorageOrder::NCHW:
      return MIOPEN_TENSOR_NCHW;
    default:
      LOG(FATAL) << "Unknown miopen equivalent for order: " << order;
  }
  // Just to suppress compiler warnings
  return MIOPEN_TENSOR_NCHW;
}

/**
 * miopenTensorDescWrapper is the placeholder that wraps around a
 * miopenTensorDescriptor_t, allowing us to do descriptor change as-needed during
 * runtime.
 */
class miopenTensorDescWrapper {
 public:
  miopenTensorDescWrapper() {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&desc_));
  }
  ~miopenTensorDescWrapper() noexcept {
    MIOPEN_CHECK(miopenDestroyTensorDescriptor(desc_));
  }

  inline miopenTensorDescriptor_t Descriptor(
      const miopenTensorFormat_t format,
      const miopenDataType_t type,
      const vector<int>& dims,
      bool* changed) {
    if (type_ == type && format_ == format && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed)
        *changed = false;
      return desc_;
    }
    CAFFE_ENFORCE_EQ(
        dims.size(), 4, "MIOPEN currently only support 4-dimensional tensor descriptor");

    CAFFE_ENFORCE_EQ(
        format, MIOPEN_TENSOR_NCHW, "MIOPEN currently only support tensor in NCHW format");

    format_ = format;
    type_ = type;
    dims_ = dims;
    MIOPEN_ENFORCE(miopenSetTensor4dDescriptor(
        desc_,
        format,
        type,
        dims_[0],
        dims_[1],
        dims_[2],
        dims_[3]);
    if (changed)
      *changed = true;
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t Descriptor(
      const StorageOrder& order,
      const vector<int>& dims) {
    return Descriptor(
        GetMIOPENTensorFormat(order), miopenTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  miopenTensorDescriptor_t desc_;
  miopenTensorFormat_t format_;
  miopenDataType_t type_;
  vector<int> dims_;
  DISABLE_COPY_AND_ASSIGN(miopenTensorDescWrapper);
};

class miopenFilterDescWrapper {
 public:
  miopenFilterDescWrapper() {
    MIOPEN_ENFORCE(miopenCreateFilterDescriptor(&desc_));
  }
  ~miopenFilterDescWrapper() noexcept {
    MIOPEN_CHECK(miopenDestroyFilterDescriptor(desc_));
  }

  inline miopenFilterDescriptor_t Descriptor(
      const StorageOrder& order,
      const miopenDataType_t type,
      const vector<int>& dims,
      bool* changed) {
    if (type_ == type && order_ == order && dims_ == dims) {
      // if not changed, simply return the current descriptor.
      if (changed)
        *changed = false;
      return desc_;
    }
    CAFFE_ENFORCE_EQ(
        dims.size(), 4, "Currently only 4-dimensional descriptor supported.");

    CAFFE_ENFORCE_EQ(
        order, StorageOrder::NCHW , "MIOPEN currently only support tensor in NCHW format");

    order_ = order;
    type_ = type;
    dims_ = dims;
    MIOPEN_ENFORCE(miopenSetFilter4dDescriptor(
        desc_,
        type,
        GetMIOPENTensorFormat(order),
        dims_[0],
        dims_[1],
        dims_[2],
        dims_[3]);
    if (changed)
      *changed = true;
    return desc_;
  }

  template <typename T>
  inline miopenFilterDescriptor_t Descriptor(
      const StorageOrder& order,
      const vector<int>& dims) {
    return Descriptor(order, miopenTypeWrapper<T>::type, dims, nullptr);
  }

 private:
  miopenFilterDescriptor_t desc_;
  StorageOrder order_;
  miopenDataType_t type_;
  vector<int> dims_;
  DISABLE_COPY_AND_ASSIGN(miopenFilterDescWrapper);
};


} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_MIOPEN_H_
