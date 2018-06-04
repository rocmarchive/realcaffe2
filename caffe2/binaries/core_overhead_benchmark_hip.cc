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

#include "benchmark/benchmark.h"

#include "caffe2/core/context.h"
#include "caffe2/core/context_hip.h"
#include "caffe2/core/operator.h"

#define CAFFE2_SKIP_IF_NO_GPU                                         \
    if(!caffe2::NumHipDevices())                                      \
    {                                                                 \
        state.SkipWithError("No HIP available, skipping benchmark."); \
        return;                                                       \
    }

using namespace caffe2;

static void BM_HIPContextCreation(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    volatile HIPContext context_so_we_do_initialization_work;
    while(state.KeepRunning())
    {
        volatile HIPContext context;
    }
}
BENCHMARK(BM_HIPContextCreation);

static void BM_HIPContextStreamAccess(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    HIPContext context;
    while(state.KeepRunning())
    {
        volatile hipStream_t stream = context.hip_stream();
    }
}
BENCHMARK(BM_HIPContextStreamAccess);

static void BM_hipGetDevice(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    int id;
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipGetDevice(&id));
    }
}
BENCHMARK(BM_hipGetDevice);

static void BM_hipSetDevice(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    int total = NumHipDevices();
    int i     = 0;
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipSetDevice((i++) % total));
    }
}
BENCHMARK(BM_hipSetDevice);

static void BM_hipSetAndGetDevice(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    int total = NumHipDevices();
    int i     = 0;
    int id;
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipSetDevice((i++) % total));
        HIP_ENFORCE(hipGetDevice(&id));
    }
}
BENCHMARK(BM_hipSetAndGetDevice);

static void BM_hipSetSameDevice(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipSetDevice(0));
    }
}
BENCHMARK(BM_hipSetSameDevice);

static void BM_hipStreamCreateSyncDelete(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    hipStream_t stream;
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipStreamCreate(&stream));
        HIP_ENFORCE(hipStreamSynchronize(stream));
        HIP_ENFORCE(hipStreamDestroy(stream));
    }
}
BENCHMARK(BM_hipStreamCreateSyncDelete);

static void BM_hipStreamSynchronize(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    hipStream_t stream;
    HIP_ENFORCE(hipStreamCreate(&stream));
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipStreamSynchronize(stream));
    }
}
BENCHMARK(BM_hipStreamSynchronize);

static void BM_hipEventRecord(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    hipStream_t stream;
    hipEvent_t event;
    HIP_ENFORCE(hipStreamCreate(&stream));
    HIP_ENFORCE(hipEventCreateWithFlags(&event, hipEventDefault | hipEventDisableTiming));
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipEventRecord(event, stream));
    }
}
BENCHMARK(BM_hipEventRecord);

static void BM_hipStreamWaitEventThenStreamSynchronize(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    hipStream_t stream;
    hipEvent_t event;
    HIP_ENFORCE(hipStreamCreate(&stream));
    HIP_ENFORCE(hipEventCreateWithFlags(&event, hipEventDefault | hipEventDisableTiming));
    HIP_ENFORCE(hipEventRecord(event, stream));
    HIP_ENFORCE(hipStreamWaitEvent(stream, event, 0));
    HIP_ENFORCE(hipStreamSynchronize(stream));
    while(state.KeepRunning())
    {
        HIP_ENFORCE(hipStreamWaitEvent(stream, event, 0));
        HIP_ENFORCE(hipStreamSynchronize(stream));
    }
}
BENCHMARK(BM_hipStreamWaitEventThenStreamSynchronize);

static void BM_HipPointerAffinity(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    TensorHIP tensor(vector<TIndex>{1, 2, 3, 4});
    float* ptr = tensor.mutable_data<float>();
    while(state.KeepRunning())
    {
        volatile int id = GetGPUIDForPointer(ptr);
    }
}
BENCHMARK(BM_HipPointerAffinity);

namespace {
template <class Context>
class DummyEmptyOp : public Operator<Context>
{
    public:
    DummyEmptyOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {}

    bool RunOnDevice() final { return true; }
};

REGISTER_CPU_OPERATOR(DummyEmpty, DummyEmptyOp<CPUContext>);
REGISTER_HIP_OPERATOR(DummyEmpty, DummyEmptyOp<HIPContext>);
OPERATOR_SCHEMA(DummyEmpty);
} // namespace

static void BM_OperatorCreationCPU(benchmark::State& state)
{
    std::unique_ptr<OperatorBase> op;
    OperatorDef def;
    Workspace ws;
    def.set_type("DummyEmpty");
    def.mutable_device_option()->set_device_type(CPU);
    while(state.KeepRunning())
    {
        op = CreateOperator(def, &ws);
    }
}
BENCHMARK(BM_OperatorCreationCPU);

static void BM_OperatorCreationHIP(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    std::unique_ptr<OperatorBase> op;
    OperatorDef def;
    Workspace ws;
    def.set_type("DummyEmpty");
    def.mutable_device_option()->set_device_type(HIP);
    while(state.KeepRunning())
    {
        op = CreateOperator(def, &ws);
    }
}
BENCHMARK(BM_OperatorCreationHIP);

static void BM_RawAllocDeallocCPU(benchmark::State& state)
{
    while(state.KeepRunning())
    {
        // Allocating only 1 byte in order to measure the overhead.
        auto ptr_and_deleter = GetCPUAllocator()->New(1);
        // Deallocate.
        ptr_and_deleter.second(ptr_and_deleter.first);
    }
}
BENCHMARK(BM_RawAllocDeallocCPU);

static void BM_TensorAllocDeallocCPU(benchmark::State& state)
{
    Tensor<CPUContext> tensor;
    // small allocation
    tensor.Resize(32, 32);
    while(state.KeepRunning())
    {
        CHECK(tensor.mutable_data<float>());
        tensor.FreeMemory();
    }
}
BENCHMARK(BM_TensorAllocDeallocCPU);

static void BM_TensorAllocDeallocHIP(benchmark::State& state)
{
    CAFFE2_SKIP_IF_NO_GPU;
    Tensor<HIPContext> tensor;
    // small allocation
    tensor.Resize(32, 32);
    while(state.KeepRunning())
    {
        CHECK(tensor.mutable_data<float>());
        tensor.FreeMemory();
    }
}
BENCHMARK(BM_TensorAllocDeallocHIP);

BENCHMARK_MAIN()
