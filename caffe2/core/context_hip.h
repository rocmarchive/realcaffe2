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

#ifndef CAFFE2_CORE_CONTEXT_HIP_H_
#define CAFFE2_CORE_CONTEXT_HIP_H_
#include "caffe2/core/common_hip.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2.pb.h"
#include <ctime>
#include <mutex>
namespace caffe2 {
enum class HipMemoryPoolType {
  NONE = 0,
  //  CNMEM = 1,
  //  CUB = 2,
};
/**
 * Gets the current memory pool type used by Caffe2.
 *
 * The memory pool is set up during caffe2's global initialization time.
 */
HipMemoryPoolType GetHipMemoryPoolType();
/**
 * A struct to host thread-local hip objects.
 *
 * In Caffe2, each thread has its own non-default hip stream as well as
 * related objects such as rocblas and hiprng handles. This is achieved by
 * having the ThreadLocalHIPObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside HIPContext and should not be used externally.
 */
class ThreadLocalHIPObjects {
  friend class HIPContext;

private:
  ThreadLocalHIPObjects() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      hip_streams_[i] = vector<hipStream_t>();
      rocblas_handles_[i] = vector<rocblas_handle>();
    }
  }
  hipStream_t GetStream(int gpu, int stream_id) {
    vector<hipStream_t> &gpu_streams = hip_streams_[gpu];
    if (gpu_streams.size() <= stream_id) {
      gpu_streams.resize(stream_id + 1, nullptr);
    }
    if (!gpu_streams[stream_id]) {
      DeviceGuard guard(gpu);
      HIP_ENFORCE(hipStreamCreateWithFlags(&gpu_streams[stream_id],
                                           hipStreamNonBlocking));
    }
    return gpu_streams[stream_id];
  }
  rocblas_handle GetHandle(int gpu, int stream_id) {
    DeviceGuard guard(gpu);
    vector<rocblas_handle> &gpu_handles = rocblas_handles_[gpu];
    if (gpu_handles.size() <= stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (!gpu_handles[stream_id]) {
      ROCBLAS_ENFORCE(rocblas_create_handle(&gpu_handles[stream_id]));
      // The default is ROCBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the rocblas handle, but do that with
      // caution.
      ROCBLAS_ENFORCE(rocblas_set_stream(gpu_handles[stream_id],
                                         GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }
  ~ThreadLocalHIPObjects() noexcept {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_GPUS; ++i) {
      for (auto &handle : rocblas_handles_[i]) {
        if (handle) {
          ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        }
      }
      for (auto &stream : hip_streams_[i]) {
        if (stream) {
          HIP_CHECK(hipStreamDestroy(stream));
        }
      }
    }
  }
  vector<hipStream_t> hip_streams_[CAFFE2_COMPILE_TIME_MAX_GPUS];
  vector<rocblas_handle> rocblas_handles_[CAFFE2_COMPILE_TIME_MAX_GPUS];
};
class HIPContext final {
public:
  // The default hip context constructor.
  explicit HIPContext(const int gpu_id = -1);
  explicit HIPContext(const DeviceOption &option);
  ~HIPContext() {
    if (hiprng_generator_) {
      HIPRNG_ENFORCE(hiprngDestroyGenerator(hiprng_generator_));
    }
    CAFFE_ENFORCE(FinishDeviceComputation());
  }
  inline void SwitchToDevice(int stream_id) {
    set_stream_id(stream_id);
    HIP_ENFORCE(hipSetDevice(gpu_id_));
  }
  inline void SwitchToDevice() { SwitchToDevice(0); }
  bool FinishDeviceComputation() {
    hipStreamSynchronize(hip_objects_.GetStream(gpu_id_, stream_id_));
    hipError_t error = hipGetLastError();
    if (error == hipSuccess) {
      return true;
    } else {
      LOG(ERROR) << "Encountered HIP error: " << hipGetErrorString(error);
      return false;
    }
  }
  inline int hip_gpu_id() const { return gpu_id_; }
  inline hipStream_t hip_stream() { return hip_stream(gpu_id_, stream_id_); }
  inline hipStream_t hip_stream() const {
    return hip_stream(gpu_id_, stream_id_);
  }
  static hipStream_t hip_stream(int gpu_id, int stream_id) {
    return hip_objects_.GetStream(gpu_id, stream_id);
  }
  rocblas_handle get_rocblas_handle() {
    return hip_objects_.GetHandle(gpu_id_, stream_id_);
  }
  hiprngGenerator_t &hiprng_generator() {
    if (!hiprng_generator_) {
      DeviceGuard guard(gpu_id_);
      HIPRNG_ENFORCE(
          hiprngCreateGenerator(&hiprng_generator_, HIPRNG_RNG_PSEUDO_DEFAULT));
      HIPRNG_ENFORCE(
          hiprngSetPseudoRandomGeneratorSeed(hiprng_generator_, random_seed_));
      CHECK_NOTNULL(hiprng_generator_);
    }
    HIPRNG_ENFORCE(hiprngSetStream(hiprng_generator_, hip_stream()));
    return hiprng_generator_;
  }
  static void *New(size_t nbytes);
  static void Delete(void *data);
  // Get a mutex to lock out hipMalloc / hipFree calls when
  // NCCL kernels are being launched. Should remove threat of
  // deadlocks
  static std::mutex &mutex();
  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void *src, void *dst) {
    HIP_ENFORCE(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault,
                               hip_objects_.GetStream(gpu_id_, stream_id_)));
  }
  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T *src, T *dst) {
    CopyBytes<SrcContext, DstContext>(n * sizeof(T),
                                      static_cast<const void *>(src),
                                      static_cast<void *>(dst));
  }
  template <class SrcContext, class DstContext>
  inline void CopyItems(const TypeMeta &meta, size_t n, const void *src,
                        void *dst) {
    CAFFE_ENFORCE(!meta.copy(), "HIPContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }
  void set_stream_id(int stream_id) { stream_id_ = stream_id; }

protected:
  int gpu_id_;
  int stream_id_ = 0;
  int random_seed_;
  hiprngGenerator_t hiprng_generator_{nullptr};
  static thread_local ThreadLocalHIPObjects hip_objects_;
};
// For the CPU context, we also allow a (probably expensive) function
// to copy the data from a hip context. Inside the function, we create
// a temporary HIPContext object to carry out the copy. From the caller's
// side, these functions are synchronous with respect to the host, similar
// to a normal CPUContext::CopyBytes<CPUContext, CPUContext> call.
template <>
inline void CPUContext::CopyBytes<HIPContext, CPUContext>(size_t nbytes,
                                                          const void *src,
                                                          void *dst) {
  HIPContext context(GetGPUIDForPointer(src));
  context.CopyBytes<HIPContext, CPUContext>(nbytes, src, dst);
}
template <>
inline void CPUContext::CopyBytes<CPUContext, HIPContext>(size_t nbytes,
                                                          const void *src,
                                                          void *dst) {
  HIPContext context(GetGPUIDForPointer(dst));
  context.CopyBytes<CPUContext, HIPContext>(nbytes, src, dst);
}
/**
 * An allocator that does the CPU memory allocation with pinned memory.
 *
 * This is needed because if we want to do any asynchronous hip memcpy,
 * the underlying CPU memory also needs to be allocated into pinned memory
 * space. As a result, whenever Caffe2 is built with GPU and there is
 * GPU present during runtime, at global initialization time we will set
 * the CPU memory allocator to allocate pinned memory.
 */
struct PinnedCPUAllocator final : CPUAllocator {
  PinnedCPUAllocator() {}
  ~PinnedCPUAllocator() {}
  void *New(size_t nbytes) override {
    void *data;
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    HIP_ENFORCE(hipHostMalloc(&data, nbytes));
    memset(data, 0, nbytes);
    return data;
  }
  void Delete(void *data) override {
    // Caffe2 uses a lazy way to figure out if one is actually going to use GPUs
    // or not. If a HIPContext::New() call is made, inside the HIPContext
    // function we will switch the cpu side allocator to a PinnedCPUAllocator.
    // But, if one calls CPUContext::New() before any hip allocations,
    // PinnedCPUAllocator can still delete the corresponding memory.
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    hipError_t err = hipHostFree(data);
    if (err == hipErrorInvalidValue) {
      free(data);
      // Calling hipGetLastError will reset the hip error.
      hipGetLastError();
    } else {
      // For all other errors, still do a hip check.
      HIP_ENFORCE(err);
    }
  }
};
// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
typedef Tensor<HIPContext> TensorHIP;
} // namespace caffe2
#endif // CAFFE2_CONTEXT_HIP_H
