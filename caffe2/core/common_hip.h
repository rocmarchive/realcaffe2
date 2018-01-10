/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CAFFE2_CORE_COMMON_HIP_H_
#define CAFFE2_CORE_COMMON_HIP_H_

#define HIP_VERSION 1
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprng.h>
#include <rocblas.h>

#undef hipLaunchKernel
#define hipLaunchKernel hipLaunchKernelGGL
// This is a macro defined for hip fp16 support. In default, hip fp16 is
// supported by NVCC 7.5, but it is also included in the Tegra X1 platform with
// a (custom?) NVCC 7.0. As a result, we would normally just check the hip
// version here, but would also allow a use to pass in the flag
// CAFFE_HAS_HIP_FP16 manually.
#ifndef CAFFE_HAS_HIP_FP16
#if HIP_VERSION >= 7050
#define CAFFE_HAS_HIP_FP16
#endif // HIP_VERSION >= 7050
#endif // CAFFE_HAS_HIP_FP16
#ifdef CAFFE_HAS_HIP_FP16
#include <hip_fp16.h>
#endif
/**
 * The maximum number of GPUs that caffe2 recognizes.
 */
#define CAFFE2_COMPILE_TIME_MAX_GPUS 8
namespace caffe2 {
/**
 * A runtime function to report the hip version that Caffe2 is built with.
 */
inline int HipVersion() { return HIP_VERSION; }
/**
 * Returns the number of devices.
 */
int NumHipDevices();
/**
 * Check if the current running session has a hip gpu present.
 *
 * Note that this is different from having caffe2 built with hip. Building
 * Caffe2 with hip only guarantees that this function exists. If there are no
 * hip gpus present in the machine, or there are hardware configuration
 * problems like an insufficient driver, this function will still return false,
 * meaning that there is no usable GPU present.
 */
inline bool HasHipGPU() { return NumHipDevices() > 0; }
/**
 * Sets the default GPU id for Caffe2.
 *
 * If an operator is set to run on Hip GPU but no gpu id is given, we will use
 * the default gpu id to run the operator. Before this function is explicitly
 * called, GPU 0 will be the default GPU id.
 */
void SetDefaultGPUID(const int deviceid);
/**
 * Gets the default GPU id for Caffe2.
 */
int GetDefaultGPUID();
/**
 * Gets the current GPU id. This is a simple wrapper around hipGetDevice().
 */
int GetCurrentGPUID();
/**
 * Gets the GPU id that the current pointer is located at.
 */
int GetGPUIDForPointer(const void *ptr);
/**
 * Gets the device property for the given device.
 */
const hipDeviceProp_t &GetDeviceProperty(const int device);
/**
 * Runs a device query function and prints out the results to LOG(INFO).
 */
void DeviceQuery(const int deviceid);
/**
 * Return a peer access pattern by returning a matrix (in the format of a
 * nested vector) of boolean values specifying whether peer access is possible.
 *
 * This function returns false if anything wrong happens during the query of
 * the GPU access pattern.
 */
bool GetHipPeerAccessPattern(vector<vector<bool>> *pattern);
/**
 * Return a human readable rocblas error string.
 */
const char *rocblasGetErrorString(rocblas_status error);
/**
 * Return a human readable hiprng error string.
 */
const char *hiprngGetErrorString(hiprngStatus_t error);
// HIP: various checks for different function calls.
#define HIP_ENFORCE(condition)                                                 \
  do {                                                                         \
    hipError_t error = condition;                                              \
    CAFFE_ENFORCE_EQ(error, hipSuccess, "Error at: ", __FILE__, ":", __LINE__, \
                     ": ", hipGetErrorString(error));                          \
  } while (0)
#define HIP_CHECK(condition)                                                   \
  do {                                                                         \
    hipError_t error = condition;                                              \
    CHECK(error == hipSuccess) << hipGetErrorString(error);                    \
  } while (0)
#define HIP_DRIVERAPI_ENFORCE(condition)                                       \
  do {                                                                         \
    CUresult result = condition;                                               \
    if (result != HIP_SUCCESS) {                                               \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      CAFFE_THROW("Error at: ", __FILE__, ":", __LINE__, ": ", msg);           \
    }                                                                          \
  } while (0)
#define HIP_DRIVERAPI_CHECK(condition)                                         \
  do {                                                                         \
    CUresult result = condition;                                               \
    if (result != HIP_SUCCESS) {                                               \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "        \
                 << msg;                                                       \
    }                                                                          \
  } while (0)
#define ROCBLAS_ENFORCE(condition)                                             \
  do {                                                                         \
    rocblas_status status = condition;                                         \
    CAFFE_ENFORCE_EQ(status, rocblas_status_success, "Error at: ", __FILE__,   \
                     ":", __LINE__, ": ",                                      \
                     ::caffe2::rocblasGetErrorString(status));                 \
  } while (0)
#define ROCBLAS_CHECK(condition)                                               \
  do {                                                                         \
    rocblas_status status = condition;                                         \
    CHECK(status == rocblas_status_success)                                    \
        << ::caffe2::rocblasGetErrorString(status);                            \
  } while (0)
#define HIPRNG_ENFORCE(condition)                                              \
  do {                                                                         \
    hiprngStatus_t status = condition;                                         \
    CAFFE_ENFORCE_EQ(status, HIPRNG_STATUS_SUCCESS, "Error at: ", __FILE__,    \
                     ":", __LINE__, ": ",                                      \
                     ::caffe2::hiprngGetErrorString(status));                  \
  } while (0)
#define HIPRNG_CHECK(condition)                                                \
  do {                                                                         \
    hiprngStatus_t status = condition;                                         \
    CHECK(status == HIPRNG_STATUS_SUCCESS)                                     \
        << ::caffe2::hiprngGetErrorString(status);                             \
  } while (0)
#define HIP_1D_KERNEL_LOOP(i, n)                                               \
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < (n);        \
       i += hipBlockDim_x * hipGridDim_x)
#ifdef __APPLE__
#define HIP_KERNEL_ASSERT(...)
#else // __APPLE__
#define HIP_KERNEL_ASSERT(...) assert(__VA_ARGS__)
#endif // __APPLE__
// The following helper functions are here so that you can write a kernel call
// when you are not particularly interested in maxing out the kernels'
// performance. Usually, this will give you a reasonable speed, but if you
// really want to find the best performance, it is advised that you tune the
// size of the blocks and grids more reasonably.
// A legacy note: this is derived from the old good Caffe days, when I simply
// hard-coded the number of threads and wanted to keep backward compatibility
// for different computation capabilities.
// For more info on HIP compute capabilities, visit the NVidia website at:
//    http://docs.nvidia.com/hip/hip-c-programming-guide/index.html#compute-capabilities
// The number of hip threads to use. 512 is used for backward compatibility,
// and it is observed that setting it to 1024 usually does not bring much
// performance gain (which makes sense, because warp size being 32 means that
// blindly setting a huge block for a random kernel isn't optimal).
constexpr int CAFFE_HIP_NUM_THREADS = 512;
// The maximum number of blocks to use in the default kernel call. We set it to
// 4096 which would work for compute capability 2.x (where 65536 is the limit).
// This number is very carelessly chosen. Ideally, one would like to look at
// the hardware at runtime, and pick the number of blocks that makes most
// sense for the specific runtime environment. This is a todo item.
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
/**
 * @brief Compute the number of blocks needed to run N threads.
 */
inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min((N + CAFFE_HIP_NUM_THREADS - 1) / CAFFE_HIP_NUM_THREADS,
                  CAFFE_MAXIMUM_NUM_BLOCKS);
}
class DeviceGuard {
public:
  explicit DeviceGuard(int newDevice) : previous_(GetCurrentGPUID()) {
    if (previous_ != newDevice) {
      HIP_ENFORCE(hipSetDevice(newDevice));
    }
  }
  ~DeviceGuard() noexcept { HIP_CHECK(hipSetDevice(previous_)); }

private:
  int previous_;
};
} // namespace caffe2

#endif // CAFFE2_CORE_COMMON_HIP_H_
