// Bandwidth roofline and row-dot microbenchmarks for the decoder worker.
//
//   bandwidth_probe stream BYTES   streaming-read sweep over BYTES of device memory
//   bandwidth_probe rowdot         the worker's BF16 GEMV routine on decoder shapes
//   bandwidth_probe ring           GEMV waves through a cp.async weight ring, with
//                                  and without prefetch across grid barriers
//
// All print JSON. Timings are CUDA-event averages over repeated work and are
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
#include <cuda/pipeline>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// One GEMV per iteration using the worker's ops::project_rows, followed by a grid
// barrier as in a decoder wave. Blocks at or above `active` only join the
// barriers. tile == 0 splits rows evenly over active blocks; otherwise tiles
// of `tile` rows go round-robin, matching the planner's equal-work greedy
// assignment. Iterations rotate through matrix copies to defeat L2 reuse.
__global__ __launch_bounds__(kThreads, 1) void rowdot(const Bf16* matrices,
    std::size_t matrix_elements, int copies, const Bf16* input, Bf16* output, int rows, int width,
    int active, int tile, int iterations)
{
    __shared__ ops::SharedStorage shared;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const Bf16* matrix =
            matrices + static_cast<std::size_t>(iteration % copies) * matrix_elements;
        // The zero output doubles as the residual, as in the decoder's projections.
        auto project = [&](int begin, int end) {
            ops::project_rows(
                begin, end, matrix, input, width, output, output, nullptr, nullptr, shared);
            __syncthreads();
        };
        if (static_cast<int>(blockIdx.x) < active) {
            if (tile == 0)
                project(static_cast<int>(static_cast<std::int64_t>(rows) * blockIdx.x / active),
                    static_cast<int>(static_cast<std::int64_t>(rows) * (blockIdx.x + 1) / active));
            else
                for (int first = blockIdx.x * tile; first < rows; first += active * tile)
                    project(first, first + tile < rows ? first + tile : rows);
        }
        cooperative_groups::this_grid().sync();
    }
}

// One wave of a ring program: each worker projects an even share of `rows`.
struct RingWave
{
    const Bf16* matrix;
    Bf16* output;
    int rows;
    int width;
};

enum class Prefetch : int
{
    direct,          // ops::project_rows, loads straight from global memory
    within_task,     // ring drained at each task end, refilled after the barrier
    across_barriers, // ring runs ahead into the next task before the barrier
    l2,              // project_rows, after an L2 prefetch of the next task's first bytes
};

// Waves run in order, `iterations` times, with a grid barrier and `gap_cycles`
// of busy waiting after each wave. The gap stands in for a prologue or a short
// attention wave, during which a worker loads no weights of its own.
struct RingProgram
{
    RingWave waves[4];
    int wave_count;
    int iterations;
    const Bf16* input;
    const Bf16* residual;
    long long gap_cycles;
    Prefetch prefetch;
    int l2_bytes; // per worker, for Prefetch::l2
};

inline constexpr int kRingRows = ops::kWarpsPerBlock; // a stage holds one row slice per warp

__device__ inline void spin(long long cycles)
{
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

__device__ inline void worker_rows(const RingWave& wave, int& begin, int& end)
{
    begin = static_cast<int>(static_cast<std::int64_t>(wave.rows) * blockIdx.x / gridDim.x);
    end = static_cast<int>(static_cast<std::int64_t>(wave.rows) * (blockIdx.x + 1) / gridDim.x);
}

// A ring stage is a K-slice of kRingRows consecutive rows: kColumns elements of
// each. Lane l of the warp that owns a row still sums columns l, l + 32, ... in
// order across all slices, so results match ops::project_rows bit for bit.
template <int Depth, int StageBytes>
struct Ring
{
    static constexpr int kColumns = StageBytes / (kRingRows * static_cast<int>(sizeof(Bf16)));
    static constexpr int kCopies = StageBytes / static_cast<int>(sizeof(uint4));
    static constexpr int kCopiesPerRow = kCopies / kRingRows;
    static constexpr int kCopiesPerThread = (kCopies + kThreads - 1) / kThreads;
    static constexpr int kRowsPerPass =
        kThreads / kCopiesPerRow; // row offset between a thread's copies
    static_assert(Depth >= 2 && Depth - 2 <= 8, "cp.async.wait_group allows at most 8 pending");
    static_assert(kCopiesPerRow >= 1 && kCopies % kRingRows == 0 && kThreads % kCopiesPerRow == 0);

    // The next stage to load. Every stage costs a few adds; the divisions that
    // place a task's rows run once per task, in start().
    struct Cursor
    {
        int task;            // iteration * wave_count + wave
        int slices_left;     // in the current row group, block-uniform
        int groups_left;     // after the current one, block-uniform
        int rows_left;       // rows of the task from this thread's first copy on
        int row_vectors;     // uint4 vectors per matrix row
        const uint4* source; // this thread's first copy of the next stage
    };

    const RingProgram& program;
    uint4* slots;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    Cursor next;
    int issued = 0;   // stages loaded so far, which picks the next slot
    int consumed = 0; // stages computed so far

    __device__ Ring(const RingProgram& program, uint4* slots)
        : program(program)
        , slots(slots)
    {
        start(0);
    }

    __device__ void start(int task)
    {
        next.task = task;
        if (task >= program.iterations * program.wave_count)
            return;
        const RingWave& wave = program.waves[task % program.wave_count];
        int begin, end;
        worker_rows(wave, begin, end);
        const int row = threadIdx.x / kCopiesPerRow;
        next.slices_left = wave.width / kColumns;
        next.groups_left = (end - begin + kRingRows - 1) / kRingRows - 1;
        next.rows_left = end - begin - row;
        next.row_vectors =
            wave.width * static_cast<int>(sizeof(Bf16)) / static_cast<int>(sizeof(uint4));
        next.source = reinterpret_cast<const uint4*>(wave.matrix) +
                      static_cast<std::size_t>(begin + row) * next.row_vectors +
                      threadIdx.x % kCopiesPerRow;
    }

    // Loads the next stage if its task is below `limit`; either way commits one
    // group, so a stage is always Depth - 2 groups behind the newest one.
    __device__ void issue(int limit)
    {
        if (next.task < limit) {
            uint4* slot = slots + (issued % Depth) * kCopies + threadIdx.x;
#pragma unroll
            for (int i = 0; i < kCopiesPerThread; ++i)
                if ((kCopies % kThreads == 0 || threadIdx.x + i * kThreads < kCopies) &&
                    next.rows_left > i * kRowsPerPass)
                    cuda::memcpy_async(slot + i * kThreads,
                        next.source + i * kRowsPerPass * next.row_vectors,
                        cuda::aligned_size_t<sizeof(uint4)>(sizeof(uint4)), pipe);
            ++issued;
            next.source += kCopiesPerRow;
            if (--next.slices_left == 0) {
                if (next.groups_left-- == 0) {
                    start(next.task + 1);
                } else {
                    next.slices_left = next.row_vectors / kCopiesPerRow;
                    next.source += (kRingRows - 1) * next.row_vectors;
                    next.rows_left -= kRingRows;
                }
            }
        }
        pipe.producer_commit();
    }

    __device__ void fill(int limit)
    {
        for (int i = 0; i < Depth - 1; ++i)
            issue(limit);
    }

    // Waits for the oldest stage, refills the slot every thread finished
    // reading in the previous call, and returns this warp's row slice.
    __device__ const Bf16* front(int limit)
    {
        cuda::pipeline_consumer_wait_prior<Depth - 2>(pipe);
        __syncthreads();
        issue(limit);
        const auto* stage = reinterpret_cast<const Bf16*>(slots + (consumed++ % Depth) * kCopies);
        return stage + (threadIdx.x / ops::kWarpSize) * kColumns;
    }
};

template <int Depth, int StageBytes>
__global__ __launch_bounds__(kThreads, 1) void ring_rowdot(
    __grid_constant__ const RingProgram program)
{
    namespace cg = cooperative_groups;
    __shared__ ops::SharedStorage shared;
    extern __shared__ uint4 slots[];
    const int tasks = program.iterations * program.wave_count;
    const int warp = threadIdx.x / ops::kWarpSize;
    const int lane = threadIdx.x % ops::kWarpSize;

    if (program.prefetch == Prefetch::direct || program.prefetch == Prefetch::l2) {
        for (int task = 0; task < tasks; ++task) {
            const RingWave& wave = program.waves[task % program.wave_count];
            int begin, end;
            worker_rows(wave, begin, end);
            ops::project_rows(begin, end, wave.matrix, program.input, wave.width, program.residual,
                wave.output, nullptr, nullptr, shared);
            // A worker's rows are contiguous, so the next task's first bytes are one range.
            if (program.prefetch == Prefetch::l2 && task + 1 < tasks) {
                const RingWave& next = program.waves[(task + 1) % program.wave_count];
                int next_begin, next_end;
                worker_rows(next, next_begin, next_end);
                const auto* first = reinterpret_cast<const char*>(
                    next.matrix + static_cast<std::size_t>(next_begin) * next.width);
                const int bytes = min(program.l2_bytes,
                    (next_end - next_begin) * next.width * static_cast<int>(sizeof(Bf16)));
                for (int offset = threadIdx.x * 128; offset < bytes; offset += kThreads * 128)
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(first + offset));
            }
            __syncthreads();
            cg::this_grid().sync();
            spin(program.gap_cycles);
        }
        return;
    }

    using Stages = Ring<Depth, StageBytes>;
    const bool across = program.prefetch == Prefetch::across_barriers;
    Stages ring(program, slots);
    ring.fill(across ? tasks : 1);
    for (int task = 0; task < tasks; ++task) {
        const int limit = across ? tasks : task + 1;
        if (!across && task > 0)
            ring.fill(limit);
        const RingWave& wave = program.waves[task % program.wave_count];
        int begin, end;
        worker_rows(wave, begin, end);
        for (int group = begin; group < end; group += kRingRows) {
            const int row = group + warp;
            float value = 0.0f;
            for (int slice = 0; slice < wave.width / Stages::kColumns; ++slice) {
                const Bf16* weights = ring.front(limit);
                const Bf16* input = program.input + slice * Stages::kColumns;
                if (row < end)
#pragma unroll
                    for (int column = lane; column < Stages::kColumns; column += ops::kWarpSize)
                        value +=
                            static_cast<float>(weights[column]) * static_cast<float>(input[column]);
            }
            if (row >= end)
                continue;
            const float sum = ops::WarpSum(shared.warp[warp]).Sum(value);
            __syncwarp();
            if (lane == 0) {
                const Bf16 projected = static_cast<Bf16>(sum);
                wave.output[row] = static_cast<Bf16>(
                    static_cast<float>(projected) + static_cast<float>(program.residual[row]));
            }
        }
        __syncthreads();
        cg::this_grid().sync();
        spin(program.gap_cycles);
    }
    cuda::pipeline_consumer_wait_prior<0>(ring.pipe);
}

// Deterministic BF16 values in [-1, 1).
__global__ void fill_hashed(Bf16* data, std::size_t count, unsigned seed)
{
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
        i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        unsigned h = (static_cast<unsigned>(i) ^ seed) * 0x9e3779b1u;
        h ^= h >> 15;
        h *= 0x85ebca77u;
        h ^= h >> 13;
        data[i] = static_cast<Bf16>(static_cast<float>(h & 0xffffu) / 32768.0f - 1.0f);
    }
}

__global__ void spin_kernel(long long cycles)
{
    spin(cycles);
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

    const int max_width = 8192, max_rows = 128256;
    cuda::device_buffer<Bf16> input(stream, pool, max_width, cuda::no_init);
    cuda::device_buffer<Bf16> output(stream, pool, max_rows, cuda::no_init);
    check(cudaMemsetAsync(input.data(), 0, max_width * sizeof(Bf16), stream.get()), "fill");
    check(cudaMemsetAsync(output.data(), 0, max_rows * sizeof(Bf16), stream.get()), "fill");
    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(sm_count), cuda::cooperative_launch{});

    json rows = json::array();
    auto time = [&](const Bf16* matrices, std::size_t elements, int copies, int matrix_rows,
                    int width, int active, int tile, int iterations) {
        cuda::launch(stream, config, rowdot, matrices, elements, copies, input.data(),
            output.data(), matrix_rows, width, active, tile, 1);
        cuda::timed_event start{stream};
        start.record(stream);
        cuda::launch(stream, config, rowdot, matrices, elements, copies, input.data(),
            output.data(), matrix_rows, width, active, tile, iterations);
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

using RingKernel = void (*)(RingProgram);

struct RingConfig
{
    int depth;
    int stage_bytes;
    RingKernel kernel;
};

template <int Depth, int StageBytes>
RingConfig ring_config()
{
    return {Depth, StageBytes, ring_rowdot<Depth, StageBytes>};
}

// Times wave programs on decoder GEMV shapes for ops::project_rows and for
// each ring geometry, with and without prefetch across barriers, and checks
// that every ring run reproduces project_rows' outputs bit for bit.
json ring_sweep(int sm_count)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    constexpr double kRooflineGbps = 212.9; // S0's measured streaming roofline
    // The S2 decoder worker's static shared memory, reserved after the ring so
    // the L1/shared carveout matches the decoder's.
    constexpr int kWorkerSharedBytes = 23'760;
    struct Shape
    {
        const char* name;
        int rows;
        int width;
    };
    const std::vector<Shape> shapes{{"qkv", 3072, 2048}, {"wo", 2048, 2048},
        {"gate_up", 16384, 2048}, {"down", 2048, 8192}, {"lm_head", 128256, 2048}};
    auto filled = [&](std::size_t count, unsigned seed) {
        cuda::device_buffer<Bf16> buffer(stream, pool, count, cuda::no_init);
        cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
            fill_hashed, buffer.data(), count, seed);
        return buffer;
    };
    std::vector<cuda::device_buffer<Bf16>> matrices, outputs;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        matrices.push_back(
            filled(static_cast<std::size_t>(shapes[i].rows) * shapes[i].width, 17 + i));
        outputs.push_back(filled(shapes[i].rows, 0));
    }
    const auto input = filled(8192, 3);
    const auto residual = filled(128256, 5);

    // SM cycles per microsecond, for the busy-wait gaps.
    double cycles_per_us = 0.0;
    {
        constexpr long long cycles = 50'000'000;
        const auto config = cuda::make_config(cuda::block_dims<1>(), cuda::grid_dims(1));
        cuda::launch(stream, config, spin_kernel, cycles);
        cuda::timed_event start{stream};
        start.record(stream);
        cuda::launch(stream, config, spin_kernel, cycles);
        cuda::timed_event end{stream};
        end.record(stream);
        end.sync();
        cycles_per_us = cycles / elapsed_us(start, end);
    }

    const std::vector<RingConfig> configs{ring_config<2, 16384>(), ring_config<2, 32768>(),
        ring_config<3, 16384>(), ring_config<4, 16384>(), ring_config<4, 8192>(),
        ring_config<6, 8192>(), ring_config<8, 8192>(), ring_config<9, 8192>(),
        ring_config<8, 4096>()};

    json result{{"cycles_per_us", cycles_per_us}, {"roofline_gbps", kRooflineGbps}};
    json kernels = json::array();
    for (const auto& config : configs) {
        cudaFuncAttributes attributes{};
        check(cudaFuncGetAttributes(&attributes, config.kernel), "ring attributes");
        kernels.push_back({{"depth", config.depth}, {"stage_bytes", config.stage_bytes},
            {"registers", attributes.numRegs}, {"local_bytes", attributes.localSizeBytes},
            {"static_shared", attributes.sharedSizeBytes}});
    }
    result["kernels"] = kernels;

    // Median of three timed launches after a warm-up, per program iteration.
    auto time = [&](RingKernel kernel, int ring_bytes, const RingProgram& program) {
        auto run = [&] {
            const auto dims = cuda::make_config(cuda::block_dims<kThreads>(),
                cuda::grid_dims(sm_count), cuda::cooperative_launch{},
                cuda::dynamic_shared_memory<uint4[]>(
                    (ring_bytes + kWorkerSharedBytes) / sizeof(uint4), cuda::non_portable));
            cuda::launch(stream, dims, kernel, program);
        };
        run();
        std::vector<double> samples;
        for (int i = 0; i < 3; ++i) {
            cuda::timed_event start{stream};
            start.record(stream);
            run();
            cuda::timed_event end{stream};
            end.record(stream);
            end.sync();
            samples.push_back(elapsed_us(start, end) / program.iterations);
        }
        std::ranges::sort(samples);
        return samples[1];
    };
    auto read_outputs = [&](const std::vector<int>& waves) {
        std::vector<std::vector<Bf16>> values;
        for (int wave : waves) {
            values.emplace_back(shapes[wave].rows);
            cuda::copy_bytes(stream, outputs[wave], values.back());
        }
        stream.sync();
        return values;
    };
    auto same_bits = [](const std::vector<std::vector<Bf16>>& a,
                         const std::vector<std::vector<Bf16>>& b) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(Bf16)) != 0)
                return false;
        return true;
    };

    struct Program
    {
        const char* name;
        std::vector<int> waves;
        int iterations;
        std::vector<double> gaps_us;
    };
    // One decoder layer's GEMVs; a chain of small Wo-sized waves, where barrier
    // restarts weigh most; and the lm_head as one long wave.
    const std::vector<Program> programs{{"layer", {0, 1, 2, 3}, 16, {0, 2, 5, 10}},
        {"wo_chain", {1}, 64, {0, 2, 5, 10}}, {"lm_head", {4}, 2, {0}}};
    json rows = json::array();
    for (const auto& spec : programs) {
        RingProgram program{};
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < spec.waves.size(); ++i) {
            const Shape& shape = shapes[spec.waves[i]];
            program.waves[i] = {matrices[spec.waves[i]].data(), outputs[spec.waves[i]].data(),
                shape.rows, shape.width};
            bytes += static_cast<std::size_t>(shape.rows) * shape.width * sizeof(Bf16);
        }
        program.wave_count = static_cast<int>(spec.waves.size());
        program.iterations = spec.iterations;
        program.input = input.data();
        program.residual = residual.data();
        const double ideal_us = bytes / kRooflineGbps / 1e3;
        for (double gap : spec.gaps_us) {
            program.gap_cycles = static_cast<long long>(gap * cycles_per_us);
            auto record = [&](const char* prefetch, int depth, int stage_bytes, double us,
                              bool equal) {
                rows.push_back({{"program", spec.name}, {"prefetch", prefetch}, {"depth", depth},
                    {"stage_bytes", stage_bytes}, {"ring_bytes", depth * stage_bytes},
                    {"gap_us", gap}, {"waves", spec.waves.size()}, {"us", us},
                    {"ideal_us", ideal_us}, {"excess_us", us - ideal_us},
                    {"gbps", bytes / us / 1e3}, {"bitwise_equal", equal}});
            };
            program.prefetch = Prefetch::direct;
            const double direct_us = time(configs.front().kernel, 0, program);
            const auto reference = read_outputs(spec.waves);
            record("direct", 0, 0, direct_us, true);
            for (int l2_bytes : {16384, 32768, 65536, 98304}) {
                program.prefetch = Prefetch::l2;
                program.l2_bytes = l2_bytes;
                const double us = time(configs.front().kernel, 0, program);
                record("l2", 0, l2_bytes, us, same_bits(reference, read_outputs(spec.waves)));
            }
            for (const auto& config : configs)
                for (Prefetch mode : {Prefetch::within_task, Prefetch::across_barriers}) {
                    program.prefetch = mode;
                    for (int wave : spec.waves)
                        check(cudaMemsetAsync(outputs[wave].data(), 0,
                                  shapes[wave].rows * sizeof(Bf16), stream.get()),
                            "clear outputs");
                    const double us =
                        time(config.kernel, config.depth * config.stage_bytes, program);
                    record(mode == Prefetch::within_task ? "within_task" : "across_barriers",
                        config.depth, config.stage_bytes, us,
                        same_bits(reference, read_outputs(spec.waves)));
                }
        }
    }
    result["runs"] = rows;
    return result;
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
        else if (mode == "ring" && argc == 2)
            result["ring"] = ring_sweep(props.multiProcessorCount);
        else
            throw std::runtime_error("usage: bandwidth_probe stream BYTES | rowdot | ring");
        std::cout << result.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "bandwidth_probe: " << error.what() << '\n';
        return 1;
    }
}
