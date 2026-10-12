#include "runtime.hpp"

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/__event/timed_event.h>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda/std/span>
#include <cooperative_groups.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace tileinfer {
namespace {

void check(cudaError_t status, const char* stage) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(status));
}

__device__ float gelu_tanh(float x) {
    constexpr float scale = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(scale * (x + 0.044715f * x * x * x)));
}

__device__ void execute_gemv(const Task& task, const float* weights, const float* input,
                             float* output, cub::WarpReduce<float>::TempStorage* reductions) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (unsigned row = task.row_begin + warp; row < task.row_end; row += kThreads / 32) {
        float sum = 0.0f;
        const float* matrix_row = weights + task.weight_offset + row * task.width;
        for (unsigned column = lane; column < task.width; column += 32)
            sum = fmaf(matrix_row[column], input[column], sum);
        sum = cub::WarpReduce<float>(reductions[warp]).Sum(sum);
        if (lane == 0) {
            float value = sum + weights[task.bias_offset + row];
            if (task.opcode == static_cast<unsigned>(Opcode::gemv_bias_gelu_f32)) value = gelu_tanh(value);
            output[row] = value;
        }
        __syncwarp();
    }
}

__global__ void worker(cuda::std::span<const Task> tasks, cuda::std::span<const Range> ranges,
                       cuda::std::span<const float> weights, cuda::std::span<const float> input,
                       cuda::std::span<float> scratch, cuda::std::span<float> output, int waves, int workers) {
    __shared__ cub::WarpReduce<float>::TempStorage reductions[kThreads / 32];
    for (int wave = 0; wave < waves; ++wave) {
        const Range range = ranges[wave * workers + blockIdx.x];
        for (unsigned i = range.begin; i < range.begin + range.count; ++i) {
            const Task task = tasks[i];
            const float* source = task.input_buffer == static_cast<unsigned>(Buffer::input) ? input.data() : scratch.data();
            float* destination = task.output_buffer == static_cast<unsigned>(Buffer::scratch) ? scratch.data() : output.data();
            switch (static_cast<Opcode>(task.opcode)) {
                case Opcode::gemv_bias_gelu_f32:
                case Opcode::gemv_bias_f32:
                    execute_gemv(task, weights.data(), source, destination, reductions);
                    break;
            }
        }
        if (wave + 1 < waves) cooperative_groups::this_grid().sync();
    }
}

} // namespace

Target describe_target() {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device query");
    check(cudaSetDevice(0), "select device");
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, worker), "compiled worker attributes");
    int occupancy = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, worker, kThreads, 0), "compiled worker occupancy");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    return {props.name, props.major, props.minor, props.multiProcessorCount,
            static_cast<bool>(props.cooperativeLaunch), occupancy, attributes.numRegs,
            static_cast<int>(attributes.sharedSizeBytes), props.maxThreadsPerMultiProcessor,
            props.totalGlobalMem, driver, runtime};
}

struct Executor::Impl {
    cuda::device_ref device{0};
    cuda::stream stream{device};
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(device);
    cuda::device_buffer<std::uint8_t> weights;
    cuda::device_buffer<Task> tasks;
    cuda::device_buffer<Range> ranges;
    cuda::device_buffer<float> input;
    cuda::device_buffer<float> scratch;
    cuda::device_buffer<float> output;
    int workers;
    int waves;
    int d;
    int h;
    int o;

    explicit Impl(const Plan& plan)
        : weights(stream, pool, plan.weights.size(), cuda::no_init),
          tasks(stream, pool, plan.tasks.size(), cuda::no_init),
          ranges(stream, pool, plan.ranges.size(), cuda::no_init),
          input(stream, pool, plan.d, cuda::no_init),
          scratch(stream, pool, plan.h, cuda::no_init),
          output(stream, pool, plan.o, cuda::no_init),
          workers(plan.workers), waves(plan.wave_count), d(plan.d), h(plan.h), o(plan.o) {
        cuda::copy_bytes(stream, cuda::std::span<const std::uint8_t>{plan.weights.data(), plan.weights.size()}, weights);
        cuda::copy_bytes(stream, cuda::std::span<const Task>{plan.tasks.data(), plan.tasks.size()}, tasks);
        cuda::copy_bytes(stream, cuda::std::span<const Range>{plan.ranges.data(), plan.ranges.size()}, ranges);
        stream.sync();
    }

    void launch() {
        auto config = cuda::make_config(cuda::block_dims<kThreads>(), cuda::grid_dims(workers), cuda::cooperative_launch{});
        cuda::launch(stream, config, worker,
                     cuda::std::span<const Task>{tasks.data(), tasks.size()},
                     cuda::std::span<const Range>{ranges.data(), ranges.size()},
                     cuda::std::span<const float>{reinterpret_cast<const float*>(weights.data()), weights.size() / sizeof(float)},
                     cuda::std::span<const float>{input.data(), input.size()},
                     cuda::std::span<float>{scratch.data(), scratch.size()},
                     cuda::std::span<float>{output.data(), output.size()}, waves, workers);
    }
};

Executor::Executor(const Plan& plan) : impl_(std::make_unique<Impl>(plan)) {}
Executor::~Executor() = default;

void Executor::upload_input(const float* data) {
    cuda::copy_bytes(impl_->stream, cuda::std::span<const float>{data, static_cast<size_t>(impl_->d)}, impl_->input);
}

void Executor::launch() { impl_->launch(); }
void Executor::synchronize() { impl_->stream.sync(); }

void Executor::run(const float* source, float* destination, float* hidden) {
    upload_input(source);
    launch();
    cuda::copy_bytes(impl_->stream, impl_->output, cuda::std::span<float>{destination, static_cast<size_t>(impl_->o)});
    if (hidden) cuda::copy_bytes(impl_->stream, impl_->scratch, cuda::std::span<float>{hidden, static_cast<size_t>(impl_->h)});
    synchronize();
}

double Executor::timed_batch(int iterations) {
    cuda::timed_event start{impl_->stream}, end{impl_->stream};
    start.record(impl_->stream);
    for (int i = 0; i < iterations; ++i) launch();
    end.record(impl_->stream);
    end.sync();
    return cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count() / iterations;
}

} // namespace tileinfer
