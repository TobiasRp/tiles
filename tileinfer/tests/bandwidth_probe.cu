// Bandwidth roofline and row-dot microbenchmarks for the decoder worker.
//
//   bandwidth_probe stream BYTES   streaming-read sweep over BYTES of device memory
//   bandwidth_probe rowdot         the worker's BF16 GEMV routine on decoder shapes
//
// Both print JSON. Timings are CUDA-event averages over repeated work and are
// diagnostic: they separate attainable DRAM bandwidth, per-worker bandwidth,
// and wave-quantization costs from the scheduled decoder.

#include "detail/cuda_check.hpp"
#include "ops.cuh"

#include <cooperative_groups.h>
#include <cuda/__event/timed_event.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace tileinfer;
using json = nlohmann::json;
using detail::check;
using Bf16 = __nv_bfloat16;

// Reads count 16-byte vectors with Unroll independent loads in flight per
// thread; the XOR result is published only to keep the loads alive.
template <int Unroll>
__global__ void stream_vectors(const uint4* data, std::size_t count, unsigned* sink)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned value = 0;
    for (; index + (Unroll - 1) * stride < count; index += Unroll * stride) {
        uint4 loaded[Unroll];
#pragma unroll
        for (int i = 0; i < Unroll; ++i)
            loaded[i] = data[index + i * stride];
#pragma unroll
        for (int i = 0; i < Unroll; ++i)
            value ^= loaded[i].x ^ loaded[i].y ^ loaded[i].z ^ loaded[i].w;
    }
    for (; index < count; index += stride)
        value ^= data[index].x ^ data[index].y ^ data[index].z ^ data[index].w;
    if (value == 0x9e3779b9u)
        sink[0] = value;
}

// Reads BF16 elements one per lane, the access width of the worker's row dot.
__global__ void stream_elements(const Bf16* data, std::size_t count, unsigned* sink)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float value = 0.0f;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count; index += stride)
        value += static_cast<float>(data[index]);
    if (value == 1.2345f)
        sink[0] = 1;
}

// One GEMV per iteration using the worker's ops::gemv, followed by a grid
// barrier as in a decoder wave. Blocks at or above `active` only join the
// barriers. tile == 0 splits rows evenly over active blocks; otherwise tiles
// of `tile` rows go round-robin, matching the planner's equal-work greedy
// assignment. Iterations rotate through matrix copies to defeat L2 reuse.
__global__ __launch_bounds__(kThreads, 1) void rowdot(const Bf16* matrices,
    std::size_t matrix_elements, int copies, const TensorRecord* records, Bf16* input, Bf16* output,
    TaskRecord task, int rows, int width, int active, int tile, int iterations)
{
    __shared__ ops::SharedStorage shared;
    NumericConfig numeric{};
    const ops::BlockShape shape{width, width, 1, 2, 0, 1, numeric, nullptr};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const ops::Context<Bf16> context{records,
            {input, output, nullptr, nullptr,
                matrices + static_cast<std::size_t>(iteration % copies) * matrix_elements, nullptr,
                nullptr, nullptr},
            shape, &shared};
        if (static_cast<int>(blockIdx.x) < active) {
            if (tile == 0) {
                task.begin = static_cast<std::uint32_t>(
                    static_cast<std::int64_t>(rows) * blockIdx.x / active);
                task.end = static_cast<std::uint32_t>(
                    static_cast<std::int64_t>(rows) * (blockIdx.x + 1) / active);
                if (task.end > task.begin)
                    ops::gemv(task, context);
                __syncthreads();
            } else {
                for (int first = blockIdx.x * tile; first < rows; first += active * tile) {
                    task.begin = first;
                    task.end = first + tile < rows ? first + tile : rows;
                    ops::gemv(task, context);
                    __syncthreads();
                }
            }
        }
        cooperative_groups::this_grid().sync();
    }
}

std::size_t parse_bytes(std::string_view text)
{
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < sizeof(uint4))
        throw std::runtime_error("invalid byte count: " + std::string(text));
    return value;
}

double elapsed_us(const cuda::timed_event& start, const cuda::timed_event& end)
{
    return cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count();
}

json stream_sweep(std::size_t bytes, int sm_count)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    const std::size_t vectors = bytes / sizeof(uint4);
    cuda::device_buffer<uint4> data(stream, pool, vectors, cuda::no_init);
    cuda::device_buffer<unsigned> sink(stream, pool, 1, cuda::no_init);
    check(cudaMemsetAsync(data.data(), 1, vectors * sizeof(uint4), stream.get()), "fill");
    stream.sync();
    json rows = json::array();
    auto measure = [&](const std::string& name, int grid, int block, auto kernel, auto... args) {
        const auto config = cuda::make_config(cuda::block_dims(block), cuda::grid_dims(grid));
        cuda::launch(stream, config, kernel, args...);
        constexpr int repeats = 5;
        cuda::timed_event start{stream};
        start.record(stream);
        for (int i = 0; i < repeats; ++i)
            cuda::launch(stream, config, kernel, args...);
        cuda::timed_event end{stream};
        end.record(stream);
        end.sync();
        const double us = elapsed_us(start, end) / repeats;
        rows.push_back(
            {{"kernel", name}, {"grid", grid}, {"block", block}, {"bytes", vectors * sizeof(uint4)},
                {"us", us}, {"gbps", vectors * sizeof(uint4) / us / 1e3}});
    };
    for (int grid : {sm_count, 2 * sm_count, 4 * sm_count, 8 * sm_count, 32 * sm_count})
        for (int block : {256, 1024}) {
            measure(
                "uint4-unroll1", grid, block, stream_vectors<1>, data.data(), vectors, sink.data());
            measure(
                "uint4-unroll4", grid, block, stream_vectors<4>, data.data(), vectors, sink.data());
            measure(
                "uint4-unroll8", grid, block, stream_vectors<8>, data.data(), vectors, sink.data());
        }
    // The decoder worker's shape and access width: 2-byte BF16 element loads.
    measure("bf16-elements", sm_count, kThreads, stream_elements,
        reinterpret_cast<const Bf16*>(data.data()), vectors * 8, sink.data());
    return rows;
}

json rowdot_sweep(int sm_count)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    struct Shape
    {
        const char* name;
        int rows;
        int width;
    };
    // Llama 3.2 1B projections; Q, K and V tiles share one wave, as do gate and up.
    const std::vector<Shape> shapes{{"qkv", 3072, 2048}, {"wo", 2048, 2048},
        {"gate_up", 16384, 2048}, {"down", 2048, 8192}, {"lm_head", 128256, 2048}};
    constexpr std::size_t kCopyBytes = std::size_t{256} << 20;
    constexpr std::size_t kTargetBytes = std::size_t{2} << 30;

    std::vector<TensorRecord> records(kTensorCount, {StorageClass::scratch, 0});
    records[tensor_index(TensorId::x)] = {StorageClass::input, 0};
    records[tensor_index(TensorId::Wq)] = {StorageClass::weight, 0};
    records[tensor_index(TensorId::q)] = {StorageClass::output, 0};
    cuda::device_buffer<TensorRecord> d_records(stream, pool, records.size(), cuda::no_init);
    cuda::copy_bytes(stream, records, d_records);
    const int max_width = 8192, max_rows = 128256;
    cuda::device_buffer<Bf16> input(stream, pool, max_width, cuda::no_init);
    cuda::device_buffer<Bf16> output(stream, pool, max_rows, cuda::no_init);
    check(cudaMemsetAsync(input.data(), 0, max_width * sizeof(Bf16), stream.get()), "fill");
    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(sm_count), cuda::cooperative_launch{});
    // A Wq-like projection binding: input x, weight matrix, output q.
    TaskRecord task{};
    task.opcode = Opcode::gemv;
    task.a = tensor_index(TensorId::x);
    task.b = tensor_index(TensorId::Wq);
    task.c = tensor_index(TensorId::q);
    task.d = kNoTensor;

    json rows = json::array();
    auto time = [&](const Bf16* matrices, std::size_t elements, int copies, int matrix_rows,
                    int width, int active, int tile, int iterations) {
        cuda::launch(stream, config, rowdot, matrices, elements, copies, d_records.data(),
            input.data(), output.data(), task, matrix_rows, width, active, tile, 1);
        cuda::timed_event start{stream};
        start.record(stream);
        cuda::launch(stream, config, rowdot, matrices, elements, copies, d_records.data(),
            input.data(), output.data(), task, matrix_rows, width, active, tile, iterations);
        cuda::timed_event end{stream};
        end.record(stream);
        end.sync();
        return elapsed_us(start, end) / iterations;
    };
    {
        // A wave with no work: the cost of one grid barrier.
        cuda::device_buffer<Bf16> unused(stream, pool, 1, cuda::no_init);
        rows.push_back({{"shape", "barrier"}, {"active", 0},
            {"us", time(unused.data(), 1, 1, 0, 1, 0, 0, 2000)}});
    }
    for (const auto& shape : shapes) {
        const std::size_t elements = static_cast<std::size_t>(shape.rows) * shape.width;
        const std::size_t matrix_bytes = elements * sizeof(Bf16);
        const int copies = static_cast<int>(std::max<std::size_t>(1, kCopyBytes / matrix_bytes));
        const int iterations =
            static_cast<int>(std::clamp<std::size_t>(kTargetBytes / matrix_bytes, 8, 2000));
        cuda::device_buffer<Bf16> matrices(stream, pool, elements * copies, cuda::no_init);
        check(cudaMemsetAsync(matrices.data(), 0, elements * copies * sizeof(Bf16), stream.get()),
            "fill");
        auto record = [&](int active, int tile) {
            const double us = time(matrices.data(), elements, copies, shape.rows, shape.width,
                active, tile, iterations);
            rows.push_back({{"shape", shape.name}, {"rows", shape.rows}, {"width", shape.width},
                {"active", active}, {"tile", tile}, {"us", us}, {"gbps", matrix_bytes / us / 1e3}});
        };
        for (int active = 1; active <= sm_count; ++active)
            record(active, 0);
        record(sm_count, 64);
    }
    return rows;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, 0), "device query");
        const std::string_view mode = argc >= 2 ? argv[1] : "";
        json result{{"device", props.name}, {"sm_count", props.multiProcessorCount},
            {"l2_bytes", props.l2CacheSize}};
        if (mode == "stream" && argc == 3)
            result["stream"] = stream_sweep(parse_bytes(argv[2]), props.multiProcessorCount);
        else if (mode == "rowdot" && argc == 2)
            result["rowdot"] = rowdot_sweep(props.multiProcessorCount);
        else
            throw std::runtime_error("usage: bandwidth_probe stream BYTES | rowdot");
        std::cout << result.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "bandwidth_probe: " << error.what() << '\n';
        return 1;
    }
}
