// Bandwidth roofline and row-dot microbenchmarks for the decoder worker.
//
//   bandwidth_probe stream BYTES   streaming-read sweep over BYTES of device memory
//   bandwidth_probe rowdot         the worker's BF16 GEMV routine on decoder shapes
//   bandwidth_probe ring           GEMV waves through a cp.async weight ring, with
//                                  and without prefetch across grid barriers
//   bandwidth_probe qkv            the QKV waves of 16 layers as one-head tasks or
//                                  as one even row range per worker (S2c gate)
//   bandwidth_probe ffn            the FFN waves of 16 layers split by a barrier or
//                                  fused per worker with split-K down (S4 gate)
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
#include <cstdlib>
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
    within_task,     // ring drained at each task end, refilled after the gap
    after_barrier,   // ring drained at each task end, refilled before the gap
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
    const bool early = program.prefetch == Prefetch::after_barrier;
    Stages ring(program, slots);
    ring.fill(across ? tasks : 1);
    for (int task = 0; task < tasks; ++task) {
        const int limit = across ? tasks : task + 1;
        if (program.prefetch == Prefetch::within_task && task > 0)
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
        if (early)
            ring.fill(limit + 1);
        spin(program.gap_cycles);
    }
    cuda::pipeline_consumer_wait_prior<0>(ring.pipe);
}

// The QKV waves of `layers` decoder layers, each followed by a grid barrier.
// Every layer reads its own copy of [Wq; Wk; Wv], so the waves stream from
// DRAM as in the decoder.
struct QkvProgram
{
    const Bf16* weights; // layers x [Wq; Wk; Wv], row-major
    const Bf16* x;
    const Bf16* norm;
    Bf16* q;     // unrotated Q, even modes
    Bf16* k;     // unrotated K, even modes
    Bf16* q_rot; // rotated Q, head mode
    Bf16* key_cache;
    Bf16* value_cache;
    ops::Shape shape;
    int layers;
    int iterations;
};

enum class QkvMode : int
{
    barriers, // only the grid barriers
    heads,    // S2b's one-head tasks, round-robin as its planner assigned them
    even,     // ops::qkv: one even row range of [Q; K; V] per worker, one norm, no RoPE
    even_raw, // as even, but projecting x without the norm prologue (diagnostic)
};

// S2b's QKV task, kept as the reference: one head of [Q; K; V] with its own
// norm prologue, then RoPE for Q and K and the cache write.
__device__ inline void qkv_head(
    int begin, const ops::Layer<Bf16>& layer, const ops::Shape& shape, ops::SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int dh = shape.head_dim;
    const int kv_width = shape.kv_heads * dh;
    Bf16* input = ops::shared_vector<Bf16>(shared);
    ops::normalize(layer.x, layer.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
    __syncthreads();
    const Bf16* matrix = layer.Wq;
    int row = begin;
    if (begin >= d + kv_width) {
        matrix = layer.Wv;
        row = begin - d - kv_width;
    } else if (begin >= d) {
        matrix = layer.Wk;
        row = begin - d;
    }
    Bf16* head = reinterpret_cast<Bf16*>(shared.rotated);
    for (int i = threadIdx.x / ops::kWarpSize; i < dh; i += ops::kWarpsPerBlock) {
        const float value =
            ops::warp_dot(matrix + static_cast<std::size_t>(row + i) * d, input, d, shared);
        if (threadIdx.x % ops::kWarpSize == 0)
            head[i] = static_cast<Bf16>(value);
    }
    __syncthreads();
    const int cache_row = shape.position * kv_width + row;
    if (matrix == layer.Wv) {
        for (int i = threadIdx.x; i < dh; i += blockDim.x)
            layer.value_cache[cache_row + i] = head[i];
        return;
    }
    Bf16* rotated = matrix == layer.Wq ? layer.q_rot + row : layer.key_cache + cache_row;
    for (int pair = threadIdx.x; pair < dh / 2; pair += blockDim.x)
        ops::rotate_pair(head, rotated, pair, shape);
}

// Without the norm prologue (diagnostic): the projection of x itself, with
// the rows and stores of ops::qkv.
__device__ inline void qkv_raw(int begin, int end, const ops::Layer<Bf16>& layer,
    const ops::Shape& shape, ops::SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int kv_width = shape.kv_heads * shape.head_dim;
    Bf16* input = ops::shared_vector<Bf16>(shared);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        input[i] = layer.x[i];
    __syncthreads();
    for (int row = begin + threadIdx.x / ops::kWarpSize; row < end; row += ops::kWarpsPerBlock) {
        const Bf16* weights = row < d ? layer.Wq + static_cast<std::size_t>(row) * d
                              : row < d + kv_width
                                  ? layer.Wk + static_cast<std::size_t>(row - d) * d
                                  : layer.Wv + static_cast<std::size_t>(row - d - kv_width) * d;
        const float value = ops::warp_dot(weights, input, d, shared);
        if (threadIdx.x % ops::kWarpSize != 0)
            continue;
        Bf16* output = row < d ? layer.q + row
                       : row < d + kv_width
                           ? layer.k + row - d
                           : layer.value_cache + shape.position * kv_width + row - d - kv_width;
        *output = static_cast<Bf16>(value);
    }
}

__global__ __launch_bounds__(kThreads, 1) void qkv_waves(
    __grid_constant__ const QkvProgram program, QkvMode mode)
{
    __shared__ ops::SharedStorage shared;
    const auto& shape = program.shape;
    const int d = shape.hidden_size;
    const int kv_width = shape.kv_heads * shape.head_dim;
    const int rows = d + 2 * kv_width;
    const int heads = rows / shape.head_dim;
    for (int wave = 0; wave < program.iterations * program.layers; ++wave) {
        const Bf16* weights =
            program.weights + static_cast<std::size_t>(wave % program.layers) * rows * d;
        const ops::Layer<Bf16> layer{program.x, nullptr, program.q, program.k, program.q_rot,
            nullptr, nullptr, nullptr, program.key_cache, program.value_cache, nullptr, nullptr,
            nullptr, nullptr, program.norm, weights, weights + static_cast<std::size_t>(d) * d,
            weights + static_cast<std::size_t>(d + kv_width) * d, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr};
        const int begin =
            static_cast<int>(static_cast<std::int64_t>(rows) * blockIdx.x / gridDim.x);
        const int end =
            static_cast<int>(static_cast<std::int64_t>(rows) * (blockIdx.x + 1) / gridDim.x);
        if (mode == QkvMode::heads)
            for (int head = blockIdx.x; head < heads; head += gridDim.x) {
                qkv_head(head * shape.head_dim, layer, shape, shared);
                __syncthreads();
            }
        else if (mode == QkvMode::even)
            ops::qkv(begin, end, layer, shape, shared);
        else if (mode == QkvMode::even_raw)
            qkv_raw(begin, end, layer, shape, shared);
        __syncthreads();
        cooperative_groups::this_grid().sync();
    }
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
        std::vector<double> samples = {};
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
                for (Prefetch mode :
                    {Prefetch::within_task, Prefetch::after_barrier, Prefetch::across_barriers}) {
                    program.prefetch = mode;
                    for (int wave : spec.waves)
                        check(cudaMemsetAsync(outputs[wave].data(), 0,
                                  shapes[wave].rows * sizeof(Bf16), stream.get()),
                            "clear outputs");
                    const double us =
                        time(config.kernel, config.depth * config.stage_bytes, program);
                    const char* name = mode == Prefetch::within_task     ? "within_task"
                                       : mode == Prefetch::after_barrier ? "after_barrier"
                                                                         : "across_barriers";
                    record(name, config.depth, config.stage_bytes, us,
                        same_bits(reference, read_outputs(spec.waves)));
                }
        }
    }
    result["runs"] = rows;
    return result;
}
// Times the QKV waves of 16 Llama 3.2 1B layers in each QkvMode and reports
// per-layer time; the barriers-only program is the floor of both. V rows are
// stored unrotated in both modes and must agree bit for bit.
json qkv_sweep(int sm_count)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    constexpr double kRooflineGbps = 212.9;
    constexpr int d = 2048, kv_heads = 8, head_dim = 64, layers = 16, context = 4096;
    constexpr int kv_width = kv_heads * head_dim;
    constexpr int rows = d + 2 * kv_width;
    auto filled = [&](std::size_t count, unsigned seed) {
        cuda::device_buffer<Bf16> buffer(stream, pool, count, cuda::no_init);
        cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
            fill_hashed, buffer.data(), count, seed);
        return buffer;
    };
    const auto weights = filled(static_cast<std::size_t>(layers) * rows * d, 11);
    const auto x = filled(d, 3);
    const auto norm = filled(d, 7);
    auto q = filled(d, 0);
    auto k = filled(kv_width, 0);
    auto q_rot = filled(d, 0);
    auto key_cache = filled(static_cast<std::size_t>(context) * kv_width, 0);
    auto value_cache = filled(static_cast<std::size_t>(context) * kv_width, 0);
    std::vector<float> host_frequencies(head_dim / 2);
    for (int pair = 0; pair < head_dim / 2; ++pair)
        host_frequencies[pair] = std::pow(500000.0f, -2.0f * pair / head_dim);
    cuda::device_buffer<float> frequencies(stream, pool, head_dim / 2, cuda::no_init);
    cuda::copy_bytes(stream, host_frequencies, frequencies);

    QkvProgram program{weights.data(), x.data(), norm.data(), q.data(), k.data(), q_rot.data(),
        key_cache.data(), value_cache.data(),
        {d, 8192, d / head_dim, kv_heads, head_dim, 0, context, 2, 2048, 384, sm_count,
            {1e-5f, 500000.0f, 32.0f, 1.0f, 4.0f, 8192.0f}, frequencies.data()},
        layers, 8};
    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(sm_count), cuda::cooperative_launch{});
    // Median of five timed launches after a warm-up, per layer.
    auto time = [&](QkvMode mode) {
        cuda::launch(stream, config, qkv_waves, program, mode);
        std::vector<double> samples = {};
        for (int i = 0; i < 5; ++i) {
            cuda::timed_event start{stream};
            start.record(stream);
            cuda::launch(stream, config, qkv_waves, program, mode);
            cuda::timed_event end{stream};
            end.record(stream);
            end.sync();
            samples.push_back(elapsed_us(start, end) / (program.iterations * layers));
        }
        std::ranges::sort(samples);
        return samples[2];
    };
    auto values = [&] {
        std::vector<Bf16> result(kv_width);
        cuda::copy_bytes(stream,
            cuda::std::span<const Bf16>(
                value_cache.data() + program.shape.position * kv_width, kv_width),
            result);
        stream.sync();
        return result;
    };

    const double ideal_us = static_cast<double>(rows) * d * sizeof(Bf16) / kRooflineGbps / 1e3;
    json runs = json::array();
    for (int position : {0, 255, 4095}) {
        program.shape.position = position;
        const double barrier_us = time(QkvMode::barriers);
        const double heads_us = time(QkvMode::heads);
        const auto heads_values = values();
        const double even_us = time(QkvMode::even);
        const bool equal =
            std::memcmp(heads_values.data(), values().data(), kv_width * sizeof(Bf16)) == 0;
        const double raw_us = time(QkvMode::even_raw);
        for (const auto& [mode, us] : {std::pair{"barriers", barrier_us}, {"heads", heads_us},
                 {"even", even_us}, {"even_raw", raw_us}})
            runs.push_back(
                {{"position", position}, {"mode", mode}, {"layer_us", us}, {"ideal_us", ideal_us},
                    {"excess_us", us - barrier_us - ideal_us}, {"v_bitwise_equal", equal}});
        runs.push_back({{"position", position}, {"mode", "saving"},
            {"layer_us", heads_us - even_us}, {"token_ms", (heads_us - even_us) * layers / 1e3}});
    }
    return {{"roofline_gbps", kRooflineGbps}, {"runs", runs}};
}
// The FFN waves of `layers` decoder layers (S4 gate). Every layer reads its
// own [Wgate; Wup; Wdown], and Wdown also exists transposed, so that a
// worker's column chunk is contiguous. Each wave starts with the next consumer's prologue, standing in
// for the next layer's QKV norm: in the split modes it normalizes y, in the
// fused modes it first reduces the previous wave's down partials into y.
struct FfnProgram
{
    const Bf16* weights;     // layers x [Wgate (f x d); Wup (f x d); Wdown (d x f)]
    const Bf16* down_chunks; // layers x Wdown transposed, [f x d]
    const Bf16* r;           // FFN input and residual
    const Bf16* ffn_norm;
    const Bf16* attn_norm; // the consumer's norm scale
    Bf16* product;
    Bf16* y;
    float* partials; // 2 x [workers x d], alternating by wave
    ops::Shape shape;
    int layers;
    int iterations;
};

enum class FfnMode : int
{
    barriers_split,     // only the split modes' two barriers per layer
    barriers_fused,     // only the fused modes' one barrier per layer
    split,              // S2c: gate/up rows, barrier, down rows, barrier
    fused_rows,         // gate/up chunk, then its down partial from row-major Wdown
    fused_rows_plain,   // fused_rows, but the consumer only normalizes (no reduction)
    fused_chunks,       // gate/up chunk, then its down partial from a transposed chunk
    fused_chunks_plain, // fused_chunks, but the consumer only normalizes (no reduction)
    fused_chunks_slice, // fused_chunks; workers reduce their slices of y, then a barrier
};

__host__ __device__ constexpr bool fused(FfnMode mode)
{
    return mode != FfnMode::barriers_split && mode != FfnMode::split &&
           mode != FfnMode::barriers_fused;
}

__host__ __device__ constexpr bool reduces(FfnMode mode)
{
    return mode == FfnMode::fused_rows || mode == FfnMode::fused_chunks;
}

__device__ inline int worker_begin(int rows)
{
    return static_cast<int>(static_cast<std::int64_t>(rows) * blockIdx.x / gridDim.x);
}

__device__ inline int worker_end(int rows)
{
    return static_cast<int>(static_cast<std::int64_t>(rows) * (blockIdx.x + 1) / gridDim.x);
}

// The next consumer's y: the worker partials summed in worker order, rounded,
// plus the residual, rounded, as ops::ffn_reduce rounds the projection.
// With `local` set, every worker reduces all of y into shared memory and
// stores its own slice; without it, a worker reduces only its slice.
__device__ inline void reduce_partials(
    const float* partials, const Bf16* residual, Bf16* y, Bf16* local, int d)
{
    const int begin = worker_begin(d);
    const int end = worker_end(d);
    for (int i = (local == nullptr ? begin : 0) + threadIdx.x; i < (local == nullptr ? end : d);
        i += blockDim.x) {
        float total = 0.0f;
#pragma unroll 4
        for (int part = 0; part < static_cast<int>(gridDim.x); ++part)
            total += partials[part * d + i];
        const Bf16 projected = static_cast<Bf16>(total);
        const Bf16 sum =
            static_cast<Bf16>(static_cast<float>(projected) + static_cast<float>(residual[i]));
        if (local != nullptr)
            local[i] = sum;
        if (i >= begin && i < end)
            y[i] = sum;
    }
    __syncthreads();
}

// Worker blockIdx.x's chunk of the fused FFN: the gate/up rows of its product
// chunk, then that chunk's down partial over all d outputs.
template <FfnMode Mode>
__device__ inline void fused_ffn(const ops::Layer<Bf16>& layer, const Bf16* chunks, float* partials,
    const ops::Shape& shape, ops::SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int f = shape.intermediate_size;
    const int begin = worker_begin(f);
    const int columns = worker_end(f) - begin;
    ops::gate_up(begin, begin + columns, layer, shape, shared);
    __syncthreads();
    Bf16* product = reinterpret_cast<Bf16*>(shared.rotated);
    for (int j = threadIdx.x; j < columns; j += blockDim.x)
        product[j] = layer.product[begin + j];
    __syncthreads();
    float* output = partials + static_cast<std::size_t>(blockIdx.x) * d;
    if constexpr (Mode == FfnMode::fused_rows || Mode == FfnMode::fused_rows_plain) {
        for (int row = threadIdx.x / ops::kWarpSize; row < d; row += ops::kWarpsPerBlock) {
            const float value = ops::warp_dot(
                layer.Wdown + static_cast<std::size_t>(row) * f + begin, product, columns, shared);
            if (threadIdx.x % ops::kWarpSize == 0)
                output[row] = value;
        }
    } else {
        // Chunk element (j, i) is Wdown[i, begin + j]; a thread owns two
        // adjacent outputs and sums its columns in order.
        const auto* block =
            reinterpret_cast<const __nv_bfloat162*>(chunks + static_cast<std::size_t>(begin) * d);
        for (int pair = threadIdx.x; pair < d / 2; pair += blockDim.x) {
            float low = 0.0f;
            float high = 0.0f;
#pragma unroll 8
            for (int j = 0; j < columns; ++j) {
                const float2 w =
                    __bfloat1622float2(block[static_cast<std::size_t>(j) * (d / 2) + pair]);
                const float p = static_cast<float>(product[j]);
                low += w.x * p;
                high += w.y * p;
            }
            output[2 * pair] = low;
            output[2 * pair + 1] = high;
        }
    }
}

template <FfnMode Mode>
__global__ __launch_bounds__(kThreads, 1) void ffn_waves(__grid_constant__ const FfnProgram program)
{
    namespace cg = cooperative_groups;
    __shared__ ops::SharedStorage shared;
    const auto& shape = program.shape;
    const int d = shape.hidden_size;
    const int f = shape.intermediate_size;
    const std::size_t matrix = static_cast<std::size_t>(f) * d;
    const int waves = program.iterations * program.layers;
    // One trailing consumer step reduces the last wave's partials.
    for (int wave = 0; wave <= waves; ++wave) {
        Bf16* input = ops::shared_vector<Bf16>(shared);
        float* previous =
            program.partials + static_cast<std::size_t>((wave + 1) % 2) * gridDim.x * d;
        if constexpr (Mode == FfnMode::fused_chunks_slice) {
            reduce_partials(previous, program.r, program.y, nullptr, d);
            cg::this_grid().sync();
            ops::normalize(
                program.y, program.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
            __syncthreads();
        } else if constexpr (reduces(Mode)) {
            reduce_partials(previous, program.r, program.y, input, d);
            ops::normalize(
                input, program.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
            __syncthreads();
        } else if constexpr (Mode != FfnMode::barriers_split && Mode != FfnMode::barriers_fused) {
            ops::normalize(
                program.y, program.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
            __syncthreads();
        }
        if (wave == waves)
            break;
        const Bf16* weights = program.weights + (wave % program.layers) * 3 * matrix;
        const ops::Layer<Bf16> layer{nullptr, program.y, nullptr, nullptr, nullptr, nullptr,
            const_cast<Bf16*>(program.r), program.product, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, program.ffn_norm,
            weights, weights + matrix, weights + 2 * matrix, nullptr, nullptr};
        if constexpr (Mode == FfnMode::split) {
            ops::gate_up(worker_begin(f), worker_end(f), layer, shape, shared);
            __syncthreads();
            cg::this_grid().sync();
            // S2c's down task, on this probe's row-major Wdown.
            ops::project_rows(worker_begin(d), worker_end(d), layer.Wdown, layer.product, f,
                layer.r, layer.y, nullptr, nullptr, shared);
            __syncthreads();
        } else if constexpr (Mode == FfnMode::barriers_split) {
            cg::this_grid().sync();
        } else if constexpr (fused(Mode)) {
            float* current = program.partials + static_cast<std::size_t>(wave % 2) * gridDim.x * d;
            fused_ffn<Mode>(layer, program.down_chunks + (wave % program.layers) * matrix, current,
                shape, shared);
            __syncthreads();
        }
        cg::this_grid().sync();
    }
}

// Stores Wdown [d x f] transposed, [f x d]. Its rows [begin_k, end_k) are
// worker k's column chunk, so the layout does not depend on the worker count.
__global__ void transpose_down(const Bf16* down, Bf16* transposed, int d, int f)
{
    const std::size_t count = static_cast<std::size_t>(d) * f;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        transposed[index] = down[index % d * f + index / d];
}

// Times every FfnMode over 16 Llama 3.2 1B FFN layers in alternating rounds
// and reports per-layer medians and the paired saving of each fused mode
// against split. y after each launch is compared with split's.
json ffn_sweep(int sm_count)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    constexpr double kRooflineGbps = 212.9;
    constexpr int d = 2048, f = 8192, layers = 16, rounds = 7;
    constexpr std::size_t matrix = static_cast<std::size_t>(f) * d;
    auto filled = [&](std::size_t count, unsigned seed) {
        cuda::device_buffer<Bf16> buffer(stream, pool, count, cuda::no_init);
        cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
            fill_hashed, buffer.data(), count, seed);
        return buffer;
    };
    const auto weights = filled(layers * 3 * matrix, 11);
    auto chunks = filled(layers * matrix, 0);
    for (int layer = 0; layer < layers; ++layer)
        cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
            transpose_down, weights.data() + (3 * layer + 2) * matrix,
            chunks.data() + layer * matrix, d, f);
    const auto r = filled(d, 3);
    const auto ffn_norm = filled(d, 5);
    const auto attn_norm = filled(d, 7);
    auto product = filled(f, 0);
    auto y = filled(d, 9);
    const std::size_t partial_count = 2 * static_cast<std::size_t>(sm_count) * d;
    cuda::device_buffer<float> partials(stream, pool, partial_count, cuda::no_init);
    check(cudaMemsetAsync(partials.data(), 0, partial_count * sizeof(float), stream.get()),
        "zero partials");

    FfnProgram program{weights.data(), chunks.data(), r.data(), ffn_norm.data(), attn_norm.data(),
        product.data(), y.data(), partials.data(),
        {d, f, 32, 8, 64, 0, 4096, 2, 2048, 384, sm_count,
            {1e-5f, 500000.0f, 32.0f, 1.0f, 4.0f, 8192.0f}, nullptr},
        layers, 4};
    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(sm_count), cuda::cooperative_launch{});

    struct Arm
    {
        const char* name;
        void (*kernel)(FfnProgram);
        std::vector<double> samples = {};
        std::vector<Bf16> y = {};
    };
    std::vector<Arm> arms{{"barriers_split", ffn_waves<FfnMode::barriers_split>},
        {"barriers_fused", ffn_waves<FfnMode::barriers_fused>},
        {"split", ffn_waves<FfnMode::split>}, {"fused_rows", ffn_waves<FfnMode::fused_rows>},
        {"fused_rows_plain", ffn_waves<FfnMode::fused_rows_plain>},
        {"fused_chunks", ffn_waves<FfnMode::fused_chunks>},
        {"fused_chunks_plain", ffn_waves<FfnMode::fused_chunks_plain>},
        {"fused_chunks_slice", ffn_waves<FfnMode::fused_chunks_slice>}};
    auto launch = [&](const Arm& arm) { cuda::launch(stream, config, arm.kernel, program); };
    // Warm-up: every arm twice, so clocks settle before the first round.
    for (int i = 0; i < 2; ++i)
        for (const auto& arm : arms)
            launch(arm);
    stream.sync();
    for (int round = 0; round < rounds; ++round)
        for (std::size_t n = 0; n < arms.size(); ++n) {
            auto& arm = arms[round % 2 == 0 ? n : arms.size() - 1 - n];
            cuda::timed_event start{stream};
            start.record(stream);
            launch(arm);
            cuda::timed_event end{stream};
            end.record(stream);
            end.sync();
            arm.samples.push_back(elapsed_us(start, end) / (program.iterations * layers));
            if (round == 0) {
                arm.y.resize(d);
                cuda::copy_bytes(stream, y, arm.y);
                stream.sync();
            }
        }

    auto median = [](std::vector<double> values) {
        std::ranges::sort(values);
        return values[values.size() / 2];
    };
    const double ideal_us = 3.0 * matrix * sizeof(Bf16) / kRooflineGbps / 1e3;
    const Arm& split = arms[2];
    json runs = json::array();
    for (const auto& arm : arms) {
        std::vector<double> saving;
        for (int round = 0; round < rounds; ++round)
            saving.push_back(split.samples[round] - arm.samples[round]);
        int differing = 0;
        int max_ulps = 0;
        for (int i = 0; i < d; ++i) {
            std::uint16_t a, b;
            std::memcpy(&a, &split.y[i], 2);
            std::memcpy(&b, &arm.y[i], 2);
            if (a != b) {
                ++differing;
                // Same-sign BF16 values differ by their bit patterns in units.
                max_ulps = std::max(max_ulps, (a ^ b) & 0x8000 ? 0xffff : std::abs(a - b));
            }
        }
        const double us = median(arm.samples);
        runs.push_back({{"mode", arm.name}, {"layer_us", us}, {"ideal_us", ideal_us},
            {"samples", arm.samples}, {"saving_vs_split_layer_us", median(saving)},
            {"saving_vs_split_token_ms", median(saving) * layers / 1e3},
            {"y_elements_differing_from_split", differing}, {"y_max_ulps", max_ulps}});
    }
    return {{"roofline_gbps", kRooflineGbps}, {"rounds", rounds}, {"runs", runs}};
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
        else if (mode == "qkv" && argc == 2)
            result["qkv"] = qkv_sweep(props.multiProcessorCount);
        else if (mode == "ffn" && argc == 2)
            result["ffn"] = ffn_sweep(props.multiProcessorCount);
        else
            throw std::runtime_error(
                "usage: bandwidth_probe stream BYTES | rowdot | ring | qkv | ffn");
        std::cout << result.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "bandwidth_probe: " << error.what() << '\n';
        return 1;
    }
}
