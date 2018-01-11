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
#include "caffe2/core/context_hip.h"
#include "caffe2/proto/caffe2.pb.h"
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <random>
#include <thread>

namespace caffe2 {
TEST(HIPContextTest, TestAllocDealloc) {
  if (!HasHipGPU())
    return;
  HIPContext context(0);
  context.SwitchToDevice();
  float *data = static_cast<float *>(HIPContext::New(10 * sizeof(float)));
  EXPECT_NE(data, nullptr);
  HIPContext::Delete(data);
}
TEST(HIPContextTest, MemoryPoolAllocateDealloc) {
  if (!HasHipGPU())
    return;
  if (GetHipMemoryPoolType() == HipMemoryPoolType::NONE) {
    LOG(ERROR) << "Choose a memory type that is not none to test memory pool.";
    return;
  }
  const int nbytes = 1048576;
  for (int i = 0; i < NumHipDevices(); ++i) {
    LOG(INFO) << "Device " << i << " of " << NumHipDevices();
    DeviceGuard guard(i);
    void *allocated = HIPContext::New(nbytes);
    EXPECT_NE(allocated, nullptr);
    hipPointerAttribute_t attr;
    HIP_ENFORCE(hipPointerGetAttributes(&attr, allocated));
    EXPECT_EQ(attr.memoryType, hipMemoryTypeDevice);
    EXPECT_EQ(attr.device, i);
    HIPContext::Delete(allocated);
    void *new_allocated = HIPContext::New(nbytes);
    // With a pool, the above allocation should yield the same address.
    EXPECT_EQ(new_allocated, allocated);
    // But, if we are allocating something larger, we will have a different
    // chunk of memory.
    void *larger_allocated = HIPContext::New(nbytes * 2);
    EXPECT_NE(larger_allocated, new_allocated);
    HIPContext::Delete(new_allocated);
    HIPContext::Delete(larger_allocated);
  }
}
hipStream_t getStreamForHandle(rocblas_handle handle) {
  hipStream_t stream = nullptr;
  ROCBLAS_ENFORCE(rocblas_get_stream(handle, &stream));
  CHECK_NOTNULL(stream);
  return stream;
}
TEST(HIPContextTest, TestSameThreadSameObject) {
  if (!HasHipGPU())
    return;
  HIPContext context_a(0);
  HIPContext context_b(0);
  EXPECT_EQ(context_a.hip_stream(), context_b.hip_stream());
  EXPECT_EQ(context_a.get_rocblas_handle(), context_b.get_rocblas_handle());
  EXPECT_EQ(context_a.hip_stream(),
            getStreamForHandle(context_b.get_rocblas_handle()));
  // CuRAND generators are context-local.
  //  EXPECT_NE(context_a.curand_generator(), context_b.curand_generator());
}
TEST(HIPContextTest, TestSameThreadDifferntObjectIfDifferentDevices) {
  if (NumHipDevices() > 1) {
    HIPContext context_a(0);
    HIPContext context_b(1);
    EXPECT_NE(context_a.hip_stream(), context_b.hip_stream());
    EXPECT_NE(context_a.get_rocblas_handle(), context_b.get_rocblas_handle());
    EXPECT_NE(context_a.hip_stream(),
              getStreamForHandle(context_b.get_rocblas_handle()));
    //    EXPECT_NE(context_a.curand_generator(), context_b.curand_generator());
  }
}
namespace {
// A test function to return a stream address from a temp HIP context. You
// should not use that stream though, because the actual stream is destroyed
// after thread exit.
void TEST_GetStreamAddress(hipStream_t *ptr) {
  HIPContext context(0);
  *ptr = context.hip_stream();
  // Sleep for a while so we have concurrent thread executions
  std::this_thread::sleep_for(std::chrono::seconds(1));
}
} // namespace
TEST(HIPContextTest, TestDifferntThreadDifferentobject) {
  if (!HasHipGPU())
    return;
  std::array<hipStream_t, 2> temp = {0};
  // Same thread
  TEST_GetStreamAddress(&temp[0]);
  TEST_GetStreamAddress(&temp[1]);
  EXPECT_TRUE(temp[0] != nullptr);
  EXPECT_TRUE(temp[1] != nullptr);
  EXPECT_EQ(temp[0], temp[1]);
  // Different threads
  std::thread thread_a(TEST_GetStreamAddress, &temp[0]);
  std::thread thread_b(TEST_GetStreamAddress, &temp[1]);
  thread_a.join();
  thread_b.join();
  EXPECT_TRUE(temp[0] != nullptr);
  EXPECT_TRUE(temp[1] != nullptr);
  EXPECT_NE(temp[0], temp[1]);
}
} // namespace caffe2
