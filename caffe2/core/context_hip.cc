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
#include "context_hip.h"
#include "cub/util_allocator.cuh"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_map>
#ifdef CAFFE2_USE_CNMEM
#include "cnmem.h"
#endif // CAFFE2_USE_CNMEM

#include "caffe2/core/asan.h"
#include "caffe2/core/common_hip.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"

#define CNMEM_CHECK(condition)                                                 \
  do {                                                                         \
    cnmemStatus_t error = condition;                                           \
    CAFFE_ENFORCE_EQ(error, CNMEM_STATUS_SUCCESS, cnmemGetErrorString(error)); \
  } while (0)

CAFFE2_DEFINE_string(caffe2_hip_memory_pool, "",
                     "Sets the memory pool used by caffe2. Possible values are "
                     "none, cnmen and cub.");
CAFFE2_DEFINE_double(
    caffe2_cnmem_reserve, 0.8,
    "Sets the proportion of memory pre-allocated by the memory "
    "pool if you use cnmem.");
CAFFE2_DEFINE_string(
    caffe2_cnmem_gpus, "",
    "A comma separated list containing the index of gpus that "
    "we will set the memory pool on. If not set, we will set "
    "up the memory pool on all available GPUs. This only applies "
    "to cnmem.");
// TODO(jiayq): Figure out the best default values for the params below.
// Currently we are using the setting copied from caffe.
CAFFE2_DEFINE_int(
    caffe2_cub_bin_growth, 2,
    "If using cub as the memory allocator, sets the growth of bins "
    "used by the cub pool.");
CAFFE2_DEFINE_int(
    caffe2_cub_min_bin, 6,
    "If using cub as the memory allocator, sets the min number of "
    "bins.");
CAFFE2_DEFINE_int(
    caffe2_cub_max_bin, 16,
    "If using cub as the memory allocator, sets the max number of "
    "bins.");

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<HIPContext>);

thread_local ThreadLocalHIPObjects HIPContext::hip_objects_;

// TODO(jiayq): these variables shouldn't be currently accessed during static
// initialization. We should consider moving them to a Mayer's singleton to
// be totally safe against SIOF.

// Static global variables for setting up the memory pool.
HipMemoryPoolType g_hip_memory_pool_type;
#ifdef CAFFE2_USE_CNMEM
// For cnmem allocator
vector<bool> g_cnmem_available_for_device;
#endif // CAFFE2_USE_CNMEM
       // For cub allocator
unique_ptr<cub::CachingDeviceAllocator> g_cub_allocator;
// an unordered map that holds the map from the hip memory pointer to the
// device id that it is allocated from. This is used in the hip memory pool
// cases, where we need the device id to carry out the deletion.
// Note(jiayq): an alternate approach is to use hipGetPointerAttributes, but
// that is usually quite slow. We might want to benchmark the speed difference
// though.
// Note(jiayq): another alternate approach is to augment the Tensor class that
// would allow one to record the device id. However, this does not address any
// non-tensor allocation and deallocation.
// Ideally, a memory pool should already have the device id information, as
// long as we are using UVA (as of HIP 5 and later) so the addresses are
// unique.
static std::unordered_map<void *, uint8_t> g_hip_device_affiliation;

HipMemoryPoolType GetHipMemoryPoolType() { return g_hip_memory_pool_type; }

///////////////////////////////////////////////////////////////////////////////
// A wrapper to allow us to lazily initialize all hip environments that Caffe
// uses. This gets done the first time a caffe2::HIPContext::New() gets called
// which is probably the decisive indication that this caffe2 run is going to
// use GPUs. We avoid hip initialization with core/init.h functionalities so
// that we have minimal resource impact in case we will need to run multiple
// caffe2 instances on a GPU machine.
///////////////////////////////////////////////////////////////////////////////

static void Caffe2InitializeHip() {
  // If the current run does not have any hip devices, do nothing.
  if (!HasHipGPU()) {
    VLOG(1) << "No hip gpu present. Skipping.";
    return;
  }
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  CAFFE_ENFORCE_LE(
      NumHipDevices(), CAFFE2_COMPILE_TIME_MAX_GPUS,
      "Number of HIP devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      CAFFE2_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile the caffe binary.");
  // Save the current device so we can restore it after moving across
  // different devices.
  int init_device;
  HIP_ENFORCE(hipGetDevice(&init_device));

  for (int i = 0; i < NumHipDevices(); ++i) {
    auto err = hipSetDevice(i);
    if (err != hipSuccess) {
      LOG(WARNING) << "Cannot use device " << i
                   << "due to the following error: " << hipGetErrorString(err);
      continue;
    }
    // Enable peer access.
    for (int j = 0; j < NumHipDevices(); ++j) {
      if (i == j)
        continue;
      int can_access;
      HIP_ENFORCE(hipDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        VLOG(1) << "Enabling peer access from " << i << " to " << j;
        // Note: just for future reference, the 0 here is not a gpu id, it is
        // a reserved flag for hipDeviceEnablePeerAccess that should always be
        // zero currently.
        HIP_ENFORCE(hipDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  // Restore the current device.
  HIP_ENFORCE(hipSetDevice(init_device));

  RegisterTypeCallFunction(TypeMeta::Id<Tensor<HIPContext>>(),
                           GetTensorType<HIPContext>);

  RegisterShapeCallFunction(TypeMeta::Id<Tensor<HIPContext>>(),
                            GetTensorShape<HIPContext>);

  // Check the versions of cuDNN that were compiled and linked with are
  // compatible
  CheckCuDNNVersions();
}

#ifdef CAFFE2_USE_CNMEM
static void SetUpCNMEM() {
  g_cnmem_available_for_device.assign(NumHipDevices(), false);
  VLOG(1) << "Setting up cnmem memory pool.";
  vector<int> device_ids;
  // If the cnmem gpus are not set, set up all gpus.
  if (FLAGS_caffe2_cnmem_gpus.size() == 0) {
    device_ids.resize(NumHipDevices());
    for (int i = 0; i < device_ids.size(); ++i) {
      device_ids[i] = i;
    }
  } else {
    vector<string> device_ids_str = split(',', FLAGS_caffe2_cnmem_gpus);
    for (const string &id_str : device_ids_str) {
      int id = 0;
      try {
        id = std::stoi(id_str);
      } catch (...) {
        CAFFE_THROW("Cannot parse device id ", id_str,
                    " to a valid int number.");
      }
      device_ids.push_back(id);
    }
  }
  CAFFE_ENFORCE(FLAGS_caffe2_cnmem_reserve >= 0 &&
                    FLAGS_caffe2_cnmem_reserve < 1.0,
                "caffe2_cnmem_reserve number must be in [0, 1)");
  vector<cnmemDevice_t> cnmem_devs(device_ids.size());
  for (int i = 0; i < device_ids.size(); ++i) {
    const int id = device_ids[i];
    CAFFE_ENFORCE(id >= 0 && id < NumHipDevices(), "GPU id ", id,
                  " out of the range of available GPUs.");
    DeviceGuard guard(id);
    size_t free, used;
    HIP_ENFORCE(hipMemGetInfo(&free, &used));
    VLOG(1) << "Reserving " << FLAGS_caffe2_cnmem_reserve * 100
            << " percent of the free memory (total " << free << ") on device "
            << id;
    // Note: we create a dummy non-null stream for memory allocations, so that
    // any malloc can be called from any hip stream, since caffe2 uses a lot of
    // non-default streams for computation. We will allocate all the reserved
    // memory to that non-null stream.
    cnmem_devs[i].device = id;
    cnmem_devs[i].size = size_t(FLAGS_caffe2_cnmem_reserve * free);
    cnmem_devs[i].numStreams = 0;
    cnmem_devs[i].streamSizes = nullptr;
    g_cnmem_available_for_device[id] = true;
  }
  CNMEM_CHECK(
      cnmemInit(cnmem_devs.size(), cnmem_devs.data(), CNMEM_FLAGS_DEFAULT));
  VLOG(1) << "Done setting up cnmem memory pool.";
}
#endif // CAFFE2_USE_CNMEM

static void SetUpCub() {
  VLOG(1) << "Setting up cub memory pool.";
  const bool k_cub_debug =
#ifdef NDEBUG
      false;
#else
      true;
#endif
  // Sets up the cub memory pool
  try {
    g_cub_allocator.reset(new cub::CachingDeviceAllocator(
        FLAGS_caffe2_cub_bin_growth, FLAGS_caffe2_cub_min_bin,
        FLAGS_caffe2_cub_max_bin, static_cast<size_t>(-1), false, k_cub_debug));
  } catch (...) {
    CAFFE_THROW("Some error happened at cub initialization.");
  }
  VLOG(1) << "Done setting up cub memory pool.";
}

static void Caffe2SetHIPMemoryPool() {
  if (FLAGS_caffe2_hip_memory_pool == "" ||
      FLAGS_caffe2_hip_memory_pool == "none") {
    g_hip_memory_pool_type = HipMemoryPoolType::NONE;
  } else if (FLAGS_caffe2_hip_memory_pool == "cnmem") {
#ifdef CAFFE2_USE_CNMEM
    // sets up cnmem.
    g_hip_memory_pool_type = HipMemoryPoolType::CNMEM;
    SetUpCNMEM();
#else
    CAFFE_THROW("This caffe2 is not built with cnmem support, so you should "
                "not use the cnmem memory pool type.");
#endif // CAFFE2_USE_CNMEM
  } else if (FLAGS_caffe2_hip_memory_pool == "cub") {
    // Sets up cub.
    g_hip_memory_pool_type = HipMemoryPoolType::CUB;
    SetUpCub();
  } else {
    CAFFE_THROW("Unrecognized hip memory pool type: ",
                FLAGS_caffe2_hip_memory_pool);
  }
}

// An initialization function that sets the CPU side to use pinned cpu
// allocator.
void Caffe2UsePinnedCPUAllocator() {
#if CAFFE2_ASAN_ENABLED
  // Note(jiayq): for more details, see
  //     https://github.com/google/sanitizers/issues/629
  LOG(WARNING) << "There are known issues between address sanitizer and "
                  "hipMallocHost. As a result, caffe2 will not enable pinned "
                  "memory allocation in asan mode. If you are expecting any "
                  "behavior that depends on asan, be advised that it is not "
                  "turned on.";
#else
  if (!HasHipGPU()) {
    VLOG(1) << "No GPU present. I won't use pinned allocator then.";
    return;
  }
  VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
  SetCPUAllocator(new PinnedCPUAllocator());
#endif
}

// Caffe2HipInitializerHelper is a minimal struct whose sole purpose is to
// detect the first hint that this Caffe2 run is going to use GPU: either
// HIPContext is initialized or HIPContext::New is called. It then runs
// all the related hip initialization functions.
namespace {
struct Caffe2HipInitializerHelper {
  Caffe2HipInitializerHelper() {
    // We cannot use bool because nvcc changes bool to __nv_bool which does
    // not have a std::atomic instantiation.
    static std::atomic<char> first_call(1);
    if (first_call.fetch_and((char)0)) {
      Caffe2InitializeHip();
      Caffe2SetHIPMemoryPool();
      Caffe2UsePinnedCPUAllocator();
    }
  }
};
} // namespace

HIPContext::HIPContext(const int gpu_id)
    : gpu_id_(gpu_id == -1 ? GetDefaultGPUID() : gpu_id),
      random_seed_(math::randomNumberSeed()) {
  static Caffe2HipInitializerHelper g_hip_initializer_;
}

HIPContext::HIPContext(const DeviceOption &option)
    : gpu_id_(option.has_hip_gpu_id() ? option.hip_gpu_id()
                                      : GetDefaultGPUID()),
      random_seed_(option.has_random_seed() ? option.random_seed()
                                            : math::randomNumberSeed()) {
  static Caffe2HipInitializerHelper g_hip_initializer_;
  DCHECK_EQ(option.device_type(), HIP);
}

// shared mutex to lock out alloc / free during NCCL launches
std::mutex &HIPContext::mutex() {
  static std::mutex m;
  return m;
}

void *HIPContext::New(size_t nbytes) {
  // Lock the mutex
  std::lock_guard<std::mutex> lock(HIPContext::mutex());
  // A one-time caffe2 hip initializer.
  static Caffe2HipInitializerHelper g_hip_initializer_;
  void *ptr = nullptr;
  switch (g_hip_memory_pool_type) {
  case HipMemoryPoolType::NONE:
    HIP_ENFORCE(hipMalloc(&ptr, nbytes));
    return ptr;
  case HipMemoryPoolType::CNMEM: {
#ifdef CAFFE2_USE_CNMEM
    auto gpuId = GetCurrentGPUID();
    CAFFE_ENFORCE(gpuId < g_cnmem_available_for_device.size() &&
                      g_cnmem_available_for_device[gpuId],
                  "Trying to allocate on device ", gpuId,
                  " but cnmem pool is not set up for it.");
    CNMEM_CHECK(cnmemMalloc(&ptr, nbytes, nullptr));
    g_hip_device_affiliation[ptr] = GetCurrentGPUID();
    VLOG(2) << "CNMEM allocating pointer " << ptr << " on device "
            << GetCurrentGPUID();
    return ptr;
#else
    CAFFE_THROW("This caffe2 is not built with cnmem support, so you should "
                "not use the cnmem memory pool type.");
#endif // CAFFE2_USE_CNMEM
  }
  case HipMemoryPoolType::CUB:
    HIP_ENFORCE(g_cub_allocator->DeviceAllocate(&ptr, nbytes));
    g_hip_device_affiliation[ptr] = GetCurrentGPUID();
    VLOG(2) << "CUB allocating pointer " << ptr << " on device "
            << GetCurrentGPUID();
    return ptr;
  }
}

void HIPContext::Delete(void *ptr) {
  // lock the mutex
  std::lock_guard<std::mutex> lock(HIPContext::mutex());

  switch (g_hip_memory_pool_type) {
  case HipMemoryPoolType::NONE: {
    // If memory pool is not set up, use simple hipFree.
    hipError_t error = hipFree(ptr);
    // For some reason, in Python runtime we sometimes delete a data pointer
    // after the hip runtime exits - this is odd but is probably caused by
    // a static workspace that pycaffe2 uses, and the destruction got
    // entangled in some race condition. Anyway, since hip runtime is exiting
    // anyway, we will not need to worry about memory leak, so we basically
    // ignore it. This is definitely not ideal but works for now.
    if (error != hipSuccess && error != hipErrorHiprtUnloading) {
      LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "
                 << hipGetErrorString(error);
    }
    break;
  }
  case HipMemoryPoolType::CNMEM: {
#ifdef CAFFE2_USE_CNMEM
    auto it = g_hip_device_affiliation.find(ptr);
    DCHECK(it != g_hip_device_affiliation.end());
    DeviceGuard guard(it->second);
    VLOG(2) << "CNMEM freeing pointer " << ptr << " on device " << it->second;
    CNMEM_CHECK(cnmemFree(ptr, nullptr));
    g_hip_device_affiliation.erase(it);
    break;
#else
    CAFFE_THROW("This caffe2 is not built with cnmem support, so you should "
                "not use the cnmem memory pool type.");
#endif // CAFFE2_USE_CNMEM
  }
  case HipMemoryPoolType::CUB: {
    auto it = g_hip_device_affiliation.find(ptr);
    DCHECK(it != g_hip_device_affiliation.end());
    VLOG(2) << "CUB freeing pointer " << ptr << " on device " << it->second;
    HIP_ENFORCE(g_cub_allocator->DeviceFree(it->second, ptr));
    g_hip_device_affiliation.erase(it);
    break;
  }
  }
}
} // namespace caffe2