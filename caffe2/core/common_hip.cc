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
#include "common_hip.h"
#include "caffe2/core/asan.h"
#include "caffe2/core/common_hip.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include <atomic>
#include <cstdlib>
#include <sstream>

namespace caffe2 {

int NumHipDevices() {
  if (getenv("CAFFE2_DEBUG_HIP_INIT_ORDER")) {
    static bool first = true;
    if (first) {
      first = false;
      std::cerr << "DEBUG: caffe2::NumHipDevices() invoked for the first time"
                << std::endl;
    }
  }
  static int count = -1;
  if (count < 0) {
    auto err = hipGetDeviceCount(&count);
    switch (err) {
    case hipSuccess:
      // Everything is good.
      break;
    case hipErrorNoDevice:
      count = 0;
      break;
    case hipErrorInsufficientDriver:
      LOG(WARNING) << "Insufficient hip driver. Cannot use hip.";
      count = 0;
      break;
    case hipErrorInitializationError:
      LOG(WARNING) << "Hip driver initialization failed, you might not "
                      "have a hip gpu.";
      count = 0;
      break;
    case hipErrorUnknown:
      LOG(ERROR) << "Found an unknown error - this may be due to an "
                    "incorrectly set up environment, e.g. changing env "
                    "variable HIP_VISIBLE_DEVICES after program start. "
                    "I will set the available devices to be zero.";
      count = 0;
      break;
    case hipErrorMemoryAllocation:
#if CAFFE2_ASAN_ENABLED
      // In ASAN mode, we know that a hipErrorMemoryAllocation error will
      // pop up.
      LOG(ERROR) << "It is known that HIP does not work well with ASAN. As "
                    "a result we will simply shut down HIP support. If you "
                    "would like to use GPUs, turn off ASAN.";
      count = 0;
      break;
#else  // CAFFE2_ASAN_ENABLED
      // If we are not in ASAN mode and we get hipErrorMemoryAllocation,
      // this means that something is wrong before NumHipDevices() call.
      LOG(FATAL) << "Unexpected error from hipGetDeviceCount(). Did you run "
                    "some hip functions before calling NumHipDevices() "
                    "that might have already set an error? Error: "
                 << err;
      break;
#endif // CAFFE2_ASAN_ENABLED
    default:
      LOG(FATAL) << "Unexpected error from hipGetDeviceCount(). Did you run "
                    "some hip functions before calling NumHipDevices() "
                    "that might have already set an error? Error: "
                 << err;
    }
  }
  return count;
}

namespace {
int gDefaultGPUID = 0;
} // namespace

void SetDefaultGPUID(const int deviceid) {
  CAFFE_ENFORCE_LT(
      deviceid, NumHipDevices(),
      "The default gpu id should be smaller than the number of gpus "
      "on this machine: ",
      deviceid, " vs ", NumHipDevices());
  gDefaultGPUID = deviceid;
}
int GetDefaultGPUID() { return gDefaultGPUID; }

int GetCurrentGPUID() {
  int gpu_id = 0;
  HIP_ENFORCE(hipGetDevice(&gpu_id));
  return gpu_id;
}

int GetGPUIDForPointer(const void *ptr) {
  hipPointerAttributes attr;
  HIP_ENFORCE(hipPointerGetAttributes(&attr, ptr));
  return attr.device;
}

const hipDeviceProp &GetDeviceProperty(const int deviceid) {
  static vector<hipDeviceProp> props;
  CAFFE_ENFORCE_LT(deviceid, NumHipDevices(),
                   "The gpu id should be smaller than the number of gpus ",
                   "on this machine: ", deviceid, " vs ", NumHipDevices());
  if (props.size() == 0) {
    props.resize(NumHipDevices());
    for (int i = 0; i < NumHipDevices(); ++i) {
      HIP_ENFORCE(hipGetDeviceProperties(&props[i], i));
    }
  }
  return props[deviceid];
}

void DeviceQuery(const int device) {
  const hipDeviceProp &prop = GetDeviceProperty(device);
  std::stringstream ss;
  ss << std::endl;
  ss << "Device id:                     " << device << std::endl;
  ss << "Major revision number:         " << prop.major << std::endl;
  ss << "Minor revision number:         " << prop.minor << std::endl;
  ss << "Name:                          " << prop.name << std::endl;
  ss << "Total global memory:           " << prop.totalGlobalMem << std::endl;
  ss << "Total shared memory per block: " << prop.sharedMemPerBlock
     << std::endl;
  ss << "Total registers per block:     " << prop.regsPerBlock << std::endl;
  ss << "Warp size:                     " << prop.warpSize << std::endl;
  ss << "Maximum memory pitch:          " << prop.memPitch << std::endl;
  ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock
     << std::endl;
  ss << "Maximum dimension of block:    " << prop.maxThreadsDim[0] << ", "
     << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
  ss << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", "
     << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
  ss << "Clock rate:                    " << prop.clockRate << std::endl;
  ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
  ss << "Texture alignment:             " << prop.textureAlignment << std::endl;
  ss << "Concurrent copy and execution: " << (prop.deviceOverlap ? "Yes" : "No")
     << std::endl;
  ss << "Number of multiprocessors:     " << prop.multiProcessorCount
     << std::endl;
  ss << "Kernel execution timeout:      "
     << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
  LOG(INFO) << ss.str();
  return;
}

bool GetHipPeerAccessPattern(vector<vector<bool>> *pattern) {
  int gpu_count;
  if (hipGetDeviceCount(&gpu_count) != hipSuccess)
    return false;
  pattern->clear();
  pattern->resize(gpu_count, vector<bool>(gpu_count, false));
  for (int i = 0; i < gpu_count; ++i) {
    for (int j = 0; j < gpu_count; ++j) {
      int can_access = true;
      if (i != j) {
        if (hipDeviceCanAccessPeer(&can_access, i, j) != hipSuccess) {
          return false;
        }
      }
      (*pattern)[i][j] = static_cast<bool>(can_access);
    }
  }
  return true;
}

const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if HIP_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#if HIP_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif // HIP_VERSION >= 6050
#endif // HIP_VERSION >= 6000
  }
  // To suppress compiler warning.
  return "Unrecognized cublas error string";
}

const char *curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  // To suppress compiler warning.
  return "Unrecognized curand error string";
}
} // namespace caffe2
