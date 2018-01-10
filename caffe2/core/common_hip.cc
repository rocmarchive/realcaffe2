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
#include "caffe2/core/common_hip.h"
#include "caffe2/core/asan.h"
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
  hipPointerAttribute_t attr;
  HIP_ENFORCE(hipPointerGetAttributes(&attr, ptr));
  return attr.device;
}
const hipDeviceProp_t &GetDeviceProperty(const int deviceid) {
  static vector<hipDeviceProp_t> props;
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
  const hipDeviceProp_t &prop = GetDeviceProperty(device);
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
  ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock
     << std::endl;
  ss << "Maximum dimension of block:    " << prop.maxThreadsDim[0] << ", "
     << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
  ss << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", "
     << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
  ss << "Clock rate:                    " << prop.clockRate << std::endl;
  ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
  ss << "Number of multiprocessors:     " << prop.multiProcessorCount
     << std::endl;
  ss << "Kernel execution timeout:      ";
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
const char *rocblasGetErrorString(rocblas_status error) {
  switch (error) {
  case rocblas_status_success:
    return "rocblas_status_success";
  case rocblas_status_invalid_handle:
    return "rocblas_status_invalid_handle";
  case rocblas_status_not_implemented:
    return "rocblas_status_not_implemented";
  case rocblas_status_invalid_pointer:
    return "rocblas_status_invalid_pointer";
  case rocblas_status_invalid_size:
    return "rocblas_status_invalid_size";
  case rocblas_status_memory_error:
    return "rocblas_status_memory_error";
  case rocblas_status_internal_error:
    return "rocblas_status_internal_error";
  }
  // To suppress compiler warning.
  return "Unrecognized rocblas error string";
}
const char *hiprngGetErrorString(hiprngStatus_t error) {
  switch (error) {
  case HIPRNG_STATUS_SUCCESS:
    return "HIPRNG_STATUS_SUCCESS";
  case HIPRNG_STATUS_VERSION_MISMATCH:
    return "HIPRNG_STATUS_VERSION_MISMATCH";
  case HIPRNG_STATUS_NOT_INITIALIZED:
    return "HIPRNG_STATUS_NOT_INITIALIZED";
  case HIPRNG_STATUS_ALLOCATION_FAILED:
    return "HIPRNG_STATUS_ALLOCATION_FAILED";
  case HIPRNG_STATUS_TYPE_ERROR:
    return "HIPRNG_STATUS_TYPE_ERROR";
  case HIPRNG_STATUS_OUT_OF_RANGE:
    return "HIPRNG_STATUS_OUT_OF_RANGE";
  case HIPRNG_STATUS_LENGTH_NOT_MULTIPLE:
    return "HIPRNG_STATUS_LENGTH_NOT_MULTIPLE";
  case HIPRNG_STATUS_LAUNCH_FAILURE:
    return "HIPRNG_STATUS_LAUNCH_FAILURE";
  case HIPRNG_STATUS_PREEXISTING_FAILURE:
    return "HIPRNG_STATUS_PREEXISTING_FAILURE";
  case HIPRNG_STATUS_INITIALIZATION_FAILED:
    return "HIPRNG_STATUS_INITIALIZATION_FAILED";
  case HIPRNG_STATUS_ARCH_MISMATCH:
    return "HIPRNG_STATUS_ARCH_MISMATCH";
  case HIPRNG_FUNCTION_NOT_IMPLEMENTED:
    return "HIPRNG_FUNCTION_NOT_IMPLEMENTED";
  case HIPRNG_INVALID_SEED:
    return "HIPRNG_INVALID_SEED";
  case HIPRNG_INVALID_STREAM_CREATOR:
    return "HIPRNG_INVALID_STREAM_CREATOR";
  case HIPRNG_INVALID_VALUE:
    return "HIPRNG_INVALID_VALUE";
  case HIPRNG_STATUS_INTERNAL_ERROR:
    return "HIPRNG_STATUS_INTERNAL_ERROR";
  }
  // To suppress compiler warning.
  return "Unrecognized hiprng error string";
}
} // namespace caffe2