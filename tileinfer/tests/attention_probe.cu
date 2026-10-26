// Attention microbenchmark for the decoder worker.
//
//   attention_probe [POSITIONS [CHUNKS]]
//
// POSITIONS and CHUNKS are comma-separated lists (default 0,63,255,1023,2047,4095
// and 5). For each position and chunk count it runs the Llama 3.2 1B attention
// waves (scores, grid barrier, values, grid barrier) over 16 layers with one
// block per SM and prints JSON. Variants:
//
//   empty           barriers only
//   production      S2's attention tasks (s2_scores / s2_values), the reference
//   worker          the worker's ops::attention_scores / ops::attention_values,
//                   with the planner's default chunk rule (one fused task up to
//                   384 steps, then at least two chunks of at most 2,048) and
//                   the barrier skip; runs once per position, whatever CHUNKS says
//   instr           copy of production with clock64 stamps between phases
//   instr_noreduce  as instr, without the warp reductions of the dot products
//   instr_konly     as instr, K loads only (no q loads, no reductions)
//   wide            16-byte K/V loads, four rows per warp instruction, warp-level
//                   statistics; wide_scores and wide_values swap in only one half
//   wide_hm         wide with a head-major K/V layout (timing only)
//   wide_u2, wide_u8  wide with two or eight rows in flight per lane
//   fused           chunks == 1: scores, softmax and P*V in one task, one wave
//   stream, stream_hm  row reads only, as a bandwidth floor for each layout
//
// The wide variants assume head_dim 64 and four query heads per K/V head, and
// match production up to FP32 summation order, as does the worker variant.
//
// Timings are CUDA-event averages and diagnostic: they split the S2 attention
// waves into K/V reads, reductions and block synchronizations.

#include "detail/cuda_check.hpp"
#include "ops.cuh"

#include <cooperative_groups.h>
#include <cuda/__event/timed_event.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tileinfer;
using json = nlohmann::json;
using detail::check;
using Bf16 = __nv_bfloat16;

constexpr int kLayers = 16;
constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kMaxContext = 4096;
// The planner's defaults for this model: the BF16 worker's capacity at four
// heads per K/V head, and the fused-task limit.
constexpr int kWorkerChunkSteps = 2048;
constexpr int kWorkerFusedSteps = 384;

// S2's attention tasks (commit 87d8152) and their shared scratch, kept as the
// probe's baseline. The wide candidates below share its even split of the
// prefix over attention_chunks; the worker's own ops split by chunk steps.
using BlockSum = cub::BlockReduce<float, kThreads>;

struct FloatMax
{
    __device__ float operator()(float a, float b) const { return cuda::std::fmax(a, b); }
};

struct S2Shared
{
    ops::WarpSum::TempStorage warp[ops::kWarpsPerBlock];
    BlockSum::TempStorage block;
    float softmax_max;
    float softmax_sum;
    float slices[kThreads];
};

__device__ inline void legacy_range(int chunk, const ops::Shape& shape, int& begin, int& end)
{
    const int length = shape.position + 1;
    begin = static_cast<int>(static_cast<long long>(length) * chunk / shape.attention_chunks);
    end = static_cast<int>(static_cast<long long>(length) * (chunk + 1) / shape.attention_chunks);
}

/// Scores of every query head of one K/V head over one context chunk, plus each
/// head's chunk maximum and sum of exp(score - maximum). Each warp owns whole
/// time steps and reads K[t] once for all heads of the group. QK and the scaling
/// are separate BF16 rounding boundaries, as in eager attention.
template <typename Scalar>
__device__ inline void s2_scores(int kv_head, int chunk, const ops::Layer<Scalar>& layer,
    const ops::Shape& shape, S2Shared& shared)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    const int first_head = kv_head * group;
    const int lane = threadIdx.x % ops::kWarpSize;
    const int warp = threadIdx.x / ops::kWarpSize;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(dh));
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);

    for (int time = begin + warp; time < end; time += ops::kWarpsPerBlock) {
        const Scalar* key = layer.key_cache + (time * shape.kv_heads + kv_head) * dh;
        float dots[kMaxGroup] = {};
        for (int i = lane; i < dh; i += ops::kWarpSize) {
            const float k = static_cast<float>(key[i]);
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g)
                if (g < group)
                    dots[g] += static_cast<float>(layer.q_rot[(first_head + g) * dh + i]) * k;
        }
        for (int g = 0; g < group; ++g) {
            const float dot = ops::WarpSum(shared.warp[warp]).Sum(dots[g]);
            __syncwarp();
            if (lane == 0)
                layer.scores[(first_head + g) * shape.max_context + time] =
                    static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(dot)) * scale);
        }
    }
    __syncthreads();

    for (int g = 0; g < group; ++g) {
        const Scalar* scores = layer.scores + (first_head + g) * shape.max_context;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            maximum = cuda::std::fmax(maximum, static_cast<float>(scores[time]));
        maximum = BlockSum(shared.block).Reduce(maximum, FloatMax());
        if (threadIdx.x == 0)
            shared.softmax_max = maximum;
        __syncthreads();
        float sum = 0.0f;
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            sum += cuda::std::exp(static_cast<float>(scores[time]) - shared.softmax_max);
        sum = BlockSum(shared.block).Sum(sum);
        if (threadIdx.x == 0) {
            float* stats = layer.stats + (chunk * shape.query_heads + first_head + g) * 2;
            stats[0] = shared.softmax_max;
            stats[1] = sum;
        }
        __syncthreads();
    }
}

/// Probabilities and partial P*V of one K/V head's query heads over one chunk.
/// The softmax maximum and denominator combine every chunk's statistics in
/// chunk order. Probabilities round once to Scalar before P*V, as eager
/// attention does; the FP32 partial sums are combined by output_projection.
template <typename Scalar>
__device__ inline void s2_values(int kv_head, int chunk, const ops::Layer<Scalar>& layer,
    const ops::Shape& shape, S2Shared& shared)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);
    // Threads split the chunk's time steps into slices for each output coordinate.
    const int slices = kThreads / dh;
    const int coordinate = threadIdx.x % dh;
    const int slice = threadIdx.x / dh;

    for (int g = 0; g < group; ++g) {
        const int head = kv_head * group + g;
        if (threadIdx.x == 0) {
            float maximum = -cuda::std::numeric_limits<float>::infinity();
            for (int c = 0; c < shape.attention_chunks; ++c)
                maximum = cuda::std::fmax(maximum, layer.stats[(c * shape.query_heads + head) * 2]);
            float sum = 0.0f;
            for (int c = 0; c < shape.attention_chunks; ++c) {
                const float* stats = layer.stats + (c * shape.query_heads + head) * 2;
                sum += stats[1] * cuda::std::exp(stats[0] - maximum);
            }
            shared.softmax_max = maximum;
            shared.softmax_sum = sum;
        }
        __syncthreads();
        Scalar* probabilities = layer.scores + head * shape.max_context;
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            probabilities[time] = static_cast<Scalar>(
                cuda::std::exp(static_cast<float>(probabilities[time]) - shared.softmax_max) /
                shared.softmax_sum);
        __syncthreads();

        float sum = 0.0f;
        if (slice < slices)
            for (int time = begin + slice; time < end; time += slices)
                sum += static_cast<float>(probabilities[time]) *
                       static_cast<float>(
                           layer.value_cache[(time * shape.kv_heads + kv_head) * dh + coordinate]);
        shared.slices[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x < dh) {
            float total = 0.0f;
            for (int s = 0; s < slices; ++s)
                total += shared.slices[s * dh + threadIdx.x];
            layer.partials[(chunk * shape.query_heads + head) * dh + threadIdx.x] = total;
        }
        __syncthreads();
    }
}

enum Variant : int
{
    v_empty,
    v_production,
    v_instr,
    v_instr_noreduce,
    v_instr_konly,
    v_wide,        // wide scores and wide values
    v_wide_scores, // wide scores, production values
    v_wide_values, // production scores, wide values
    v_wide_hm,     // wide, head-major K/V layout (timing only)
    v_wide_u2,     // wide with two rows in flight per lane
    v_wide_u8,     // wide with eight rows in flight per lane
    v_fused,       // chunks == 1: scores, softmax and P*V in one task, one wave
    v_stream,      // K then V row reads only, production layout
    v_stream_hm,   // K then V row reads only, head-major layout
    v_worker,      // the worker's attention ops, chunk rule and barrier skip
    kVariants,
};
constexpr const char* kVariantNames[kVariants] = {"empty", "production", "instr", "instr_noreduce",
    "instr_konly", "wide", "wide_scores", "wide_values", "wide_hm", "wide_u2", "wide_u8", "fused",
    "stream", "stream_hm", "worker"};
// Variants whose attention vector must match production up to rounding order.
constexpr bool compares_with_production(int variant)
{
    return variant == v_production || variant == v_instr || variant == v_wide ||
           variant == v_wide_scores || variant == v_wide_values || variant == v_fused ||
           variant == v_wide_u2 || variant == v_wide_u8 || variant == v_worker;
}

enum Phase : int
{
    p_dot,     // scores: K reads, dot products, warp reductions, score stores
    p_max,     // scores: chunk maximum
    p_sum,     // scores: sum of exp, statistics store
    p_combine, // values: combine chunk statistics
    p_prob,    // values: probabilities
    p_pv,      // values: P*V loop
    p_final,   // values: slice reduction and partial store
    kPhases,
};
constexpr const char* kPhaseNames[kPhases] = {"scores_dot", "scores_max", "scores_sum",
    "values_combine", "values_prob", "values_pv", "values_final"};

__device__ inline unsigned long long global_timer_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Per-phase cycle accumulation by thread 0, taken after the block's own
// synchronization so each phase includes the slowest warp.
struct Stamps
{
    long long last = 0;
    long long total[kPhases] = {};
    __device__ void start()
    {
        if (threadIdx.x == 0)
            last = clock64();
    }
    template <int P>
    __device__ void mark()
    {
        if (threadIdx.x == 0) {
            const long long now = clock64();
            total[P] += now - last;
            last = now;
        }
    }
};

// Copy of ops::attention_scores. Mode 0 is identical; mode 1 skips the warp
// reductions (lane 0 stores its own partial dot); mode 2 also skips the q loads.
template <int Mode>
__device__ inline void scores_instrumented(int kv_head, int chunk, const ops::Layer<Bf16>& layer,
    const ops::Shape& shape, S2Shared& shared, Stamps& stamps)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    const int first_head = kv_head * group;
    const int lane = threadIdx.x % ops::kWarpSize;
    const int warp = threadIdx.x / ops::kWarpSize;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(dh));
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);
    stamps.start();

    for (int time = begin + warp; time < end; time += ops::kWarpsPerBlock) {
        const Bf16* key = layer.key_cache + (time * shape.kv_heads + kv_head) * dh;
        float dots[kMaxGroup] = {};
        for (int i = lane; i < dh; i += ops::kWarpSize) {
            const float k = static_cast<float>(key[i]);
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g)
                if (g < group) {
                    if (Mode == 2)
                        dots[g] += k;
                    else
                        dots[g] += static_cast<float>(layer.q_rot[(first_head + g) * dh + i]) * k;
                }
        }
        for (int g = 0; g < group; ++g) {
            const float dot = Mode == 0 ? ops::WarpSum(shared.warp[warp]).Sum(dots[g]) : dots[g];
            if (Mode == 0)
                __syncwarp();
            if (lane == 0)
                layer.scores[(first_head + g) * shape.max_context + time] =
                    static_cast<Bf16>(static_cast<float>(static_cast<Bf16>(dot)) * scale);
        }
    }
    __syncthreads();
    stamps.mark<p_dot>();

    for (int g = 0; g < group; ++g) {
        const Bf16* scores = layer.scores + (first_head + g) * shape.max_context;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            maximum = cuda::std::fmax(maximum, static_cast<float>(scores[time]));
        maximum = BlockSum(shared.block).Reduce(maximum, FloatMax());
        if (threadIdx.x == 0)
            shared.softmax_max = maximum;
        __syncthreads();
        stamps.mark<p_max>();
        float sum = 0.0f;
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            sum += cuda::std::exp(static_cast<float>(scores[time]) - shared.softmax_max);
        sum = BlockSum(shared.block).Sum(sum);
        if (threadIdx.x == 0) {
            float* stats = layer.stats + (chunk * shape.query_heads + first_head + g) * 2;
            stats[0] = shared.softmax_max;
            stats[1] = sum;
        }
        __syncthreads();
        stamps.mark<p_sum>();
    }
}

// Copy of ops::attention_values with phase stamps.
__device__ inline void values_instrumented(int kv_head, int chunk, const ops::Layer<Bf16>& layer,
    const ops::Shape& shape, S2Shared& shared, Stamps& stamps)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);
    const int slices = kThreads / dh;
    const int coordinate = threadIdx.x % dh;
    const int slice = threadIdx.x / dh;
    stamps.start();

    for (int g = 0; g < group; ++g) {
        const int head = kv_head * group + g;
        if (threadIdx.x == 0) {
            float maximum = -cuda::std::numeric_limits<float>::infinity();
            for (int c = 0; c < shape.attention_chunks; ++c)
                maximum = cuda::std::fmax(maximum, layer.stats[(c * shape.query_heads + head) * 2]);
            float sum = 0.0f;
            for (int c = 0; c < shape.attention_chunks; ++c) {
                const float* stats = layer.stats + (c * shape.query_heads + head) * 2;
                sum += stats[1] * cuda::std::exp(stats[0] - maximum);
            }
            shared.softmax_max = maximum;
            shared.softmax_sum = sum;
        }
        __syncthreads();
        stamps.mark<p_combine>();
        Bf16* probabilities = layer.scores + head * shape.max_context;
        for (int time = begin + threadIdx.x; time < end; time += blockDim.x)
            probabilities[time] = static_cast<Bf16>(
                cuda::std::exp(static_cast<float>(probabilities[time]) - shared.softmax_max) /
                shared.softmax_sum);
        __syncthreads();
        stamps.mark<p_prob>();

        float sum = 0.0f;
        if (slice < slices)
            for (int time = begin + slice; time < end; time += slices)
                sum += static_cast<float>(probabilities[time]) *
                       static_cast<float>(
                           layer.value_cache[(time * shape.kv_heads + kv_head) * dh + coordinate]);
        shared.slices[threadIdx.x] = sum;
        __syncthreads();
        stamps.mark<p_pv>();
        if (threadIdx.x < dh) {
            float total = 0.0f;
            for (int s = 0; s < slices; ++s)
                total += shared.slices[s * dh + threadIdx.x];
            layer.partials[(chunk * shape.query_heads + head) * dh + threadIdx.x] = total;
        }
        __syncthreads();
        stamps.mark<p_final>();
    }
}

// ---- Wide candidates: 16-byte K/V loads, four rows per warp instruction, Unroll
// independent loads in flight per lane, q in registers, warp-level statistics.
// They assume head_dim 64 and four query heads per K/V head, and reduce in a
// different FP32 order than production, so they match it up to rounding only.

// Element offset of K/V row `time` of one K/V head. Layout 0 is the decoder's
// [time][kv_head][head_dim]; layout 1 is head-major [kv_head][time][head_dim].
template <int Layout>
__device__ inline std::size_t row_offset(int time, int kv_head, const ops::Shape& shape)
{
    return Layout == 0 ? (static_cast<std::size_t>(time) * shape.kv_heads + kv_head) * kHeadDim
                       : (static_cast<std::size_t>(kv_head) * shape.max_context + time) * kHeadDim;
}

__device__ inline float bf16_low(unsigned word)
{
    return __uint_as_float(word << 16);
}

__device__ inline float bf16_high(unsigned word)
{
    return __uint_as_float(word & 0xffff0000u);
}

// Element j in [0, 8) of a 16-byte vector of BF16 values.
__device__ inline float element(const uint4& raw, int j)
{
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    return (j & 1) ? bf16_high(words[j >> 1]) : bf16_low(words[j >> 1]);
}

__device__ inline uint4 load_row_part(const Bf16* base, bool valid)
{
    return valid ? *reinterpret_cast<const uint4*>(base) : uint4{0, 0, 0, 0};
}

template <int Unroll, int Layout>
__device__ inline void scores_wide(
    int kv_head, int chunk, const ops::Layer<Bf16>& layer, const ops::Shape& shape)
{
    constexpr int group = 4;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int sub = lane / 8, v = lane % 8;
    const int first_head = kv_head * group;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(kHeadDim));
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);

    uint4 q[group];
#pragma unroll
    for (int g = 0; g < group; ++g)
        q[g] = *reinterpret_cast<const uint4*>(layer.q_rot + (first_head + g) * kHeadDim + v * 8);

    constexpr int rows = ops::kWarpsPerBlock * 4 * Unroll;
    for (int first = begin; first < end; first += rows) {
        uint4 raw[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * ops::kWarpsPerBlock + warp) * 4 + sub;
            raw[u] = load_row_part(
                layer.key_cache + row_offset<Layout>(t, kv_head, shape) + v * 8, t < end);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * ops::kWarpsPerBlock + warp) * 4 + sub;
            float dots[group] = {};
#pragma unroll
            for (int g = 0; g < group; ++g) {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    dots[g] += element(q[g], j) * element(raw[u], j);
#pragma unroll
                for (int offset = 4; offset > 0; offset >>= 1)
                    dots[g] += __shfl_xor_sync(0xffffffffu, dots[g], offset);
                if (v == 0 && t < end)
                    layer.scores[(first_head + g) * shape.max_context + t] =
                        static_cast<Bf16>(static_cast<float>(static_cast<Bf16>(dots[g])) * scale);
            }
        }
    }
    __syncthreads();

    // One warp per query head: chunk maximum and sum of exp(score - maximum).
    if (warp < group) {
        const Bf16* scores = layer.scores + (first_head + warp) * shape.max_context;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int t = begin + lane; t < end; t += 32)
            maximum = cuda::std::fmax(maximum, static_cast<float>(scores[t]));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            maximum = cuda::std::fmax(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        float sum = 0.0f;
        for (int t = begin + lane; t < end; t += 32)
            sum += cuda::std::exp(static_cast<float>(scores[t]) - maximum);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            float* stats = layer.stats + (chunk * shape.query_heads + first_head + warp) * 2;
            stats[0] = maximum;
            stats[1] = sum;
        }
    }
}

// `reduce` holds 4 heads x 8 warps x 64 floats.
template <int Unroll, int Layout>
__device__ inline void values_wide(int kv_head, int chunk, const ops::Layer<Bf16>& layer,
    const ops::Shape& shape, ops::SharedStorage& shared, float* reduce)
{
    constexpr int group = 4;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int sub = lane / 8, v = lane % 8;
    const int first_head = kv_head * group;
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);
    const int length = end - begin;
    if (group * length * static_cast<int>(sizeof(Bf16)) > static_cast<int>(sizeof(shared.vector)))
        __trap();

    // Same chunk-order combine as production, one thread per head.
    float* combined = shared.norm_partials; // [0, 4): maximum, [4, 8): denominator
    if (threadIdx.x < group) {
        const int head = first_head + threadIdx.x;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int c = 0; c < shape.attention_chunks; ++c)
            maximum = cuda::std::fmax(maximum, layer.stats[(c * shape.query_heads + head) * 2]);
        float sum = 0.0f;
        for (int c = 0; c < shape.attention_chunks; ++c) {
            const float* stats = layer.stats + (c * shape.query_heads + head) * 2;
            sum += stats[1] * cuda::std::exp(stats[0] - maximum);
        }
        combined[threadIdx.x] = maximum;
        combined[group + threadIdx.x] = sum;
    }
    __syncthreads();
    Bf16* probabilities = reinterpret_cast<Bf16*>(shared.vector); // [group][length]
    for (int index = threadIdx.x; index < group * length; index += blockDim.x) {
        const int g = index / length, t = index - g * length;
        const Bf16* scores = layer.scores + (first_head + g) * shape.max_context + begin;
        probabilities[index] = static_cast<Bf16>(
            cuda::std::exp(static_cast<float>(scores[t]) - combined[g]) / combined[group + g]);
    }
    __syncthreads();

    // Warps 8g..8g+7 accumulate head g over the whole chunk.
    const int g = warp / 8, w8 = warp % 8;
    const Bf16* p = probabilities + g * length;
    float acc[8] = {};
    constexpr int rows = 8 * 4 * Unroll;
    for (int first = 0; first < length; first += rows) {
        uint4 raw[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * 8 + w8) * 4 + sub;
            raw[u] = load_row_part(
                layer.value_cache + row_offset<Layout>(begin + t, kv_head, shape) + v * 8,
                t < length);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * 8 + w8) * 4 + sub;
            const float probability = t < length ? static_cast<float>(p[t]) : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] += probability * element(raw[u], j);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 8);
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 16);
    }
    if (sub == 0)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reduce[(g * 8 + w8) * kHeadDim + v * 8 + j] = acc[j];
    __syncthreads();
    if (threadIdx.x < group * kHeadDim) {
        const int g2 = threadIdx.x / kHeadDim, c = threadIdx.x % kHeadDim;
        float total = 0.0f;
        for (int w = 0; w < 8; ++w)
            total += reduce[(g2 * 8 + w) * kHeadDim + c];
        layer.partials[(chunk * shape.query_heads + first_head + g2) * kHeadDim + c] = total;
    }
}

// One task per K/V head over the whole prefix (attention_chunks == 1): scores,
// softmax and P*V without a grid barrier between them. Scores and
// probabilities live in shared memory.
template <int Unroll, int Layout>
__device__ inline void attention_fused(int kv_head, const ops::Layer<Bf16>& layer,
    const ops::Shape& shape, ops::SharedStorage& shared, float* reduce)
{
    constexpr int group = 4;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int sub = lane / 8, v = lane % 8;
    const int first_head = kv_head * group;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(kHeadDim));
    const int length = shape.position + 1;
    if (group * length * static_cast<int>(sizeof(Bf16)) > static_cast<int>(sizeof(shared.vector)))
        __trap();
    Bf16* scores = reinterpret_cast<Bf16*>(shared.vector); // [group][length], then probabilities

    uint4 q[group];
#pragma unroll
    for (int g = 0; g < group; ++g)
        q[g] = *reinterpret_cast<const uint4*>(layer.q_rot + (first_head + g) * kHeadDim + v * 8);
    constexpr int rows = ops::kWarpsPerBlock * 4 * Unroll;
    for (int first = 0; first < length; first += rows) {
        uint4 raw[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * ops::kWarpsPerBlock + warp) * 4 + sub;
            raw[u] = load_row_part(
                layer.key_cache + row_offset<Layout>(t, kv_head, shape) + v * 8, t < length);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * ops::kWarpsPerBlock + warp) * 4 + sub;
            float dots[group] = {};
#pragma unroll
            for (int g = 0; g < group; ++g) {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    dots[g] += element(q[g], j) * element(raw[u], j);
#pragma unroll
                for (int offset = 4; offset > 0; offset >>= 1)
                    dots[g] += __shfl_xor_sync(0xffffffffu, dots[g], offset);
                if (v == 0 && t < length)
                    scores[g * length + t] =
                        static_cast<Bf16>(static_cast<float>(static_cast<Bf16>(dots[g])) * scale);
            }
        }
    }
    __syncthreads();

    float* combined = shared.norm_partials; // [0, 4): maximum, [4, 8): denominator
    if (warp < group) {
        const Bf16* own = scores + warp * length;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int t = lane; t < length; t += 32)
            maximum = cuda::std::fmax(maximum, static_cast<float>(own[t]));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            maximum = cuda::std::fmax(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        float sum = 0.0f;
        for (int t = lane; t < length; t += 32)
            sum += cuda::std::exp(static_cast<float>(own[t]) - maximum);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (lane == 0) {
            combined[warp] = maximum;
            combined[group + warp] = sum;
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < group * length; index += blockDim.x) {
        const int g = index / length;
        scores[index] = static_cast<Bf16>(
            cuda::std::exp(static_cast<float>(scores[index]) - combined[g]) / combined[group + g]);
    }
    __syncthreads();

    const int g = warp / 8, w8 = warp % 8;
    const Bf16* p = scores + g * length;
    float acc[8] = {};
    constexpr int value_rows = 8 * 4 * Unroll;
    for (int first = 0; first < length; first += value_rows) {
        uint4 raw[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * 8 + w8) * 4 + sub;
            raw[u] = load_row_part(
                layer.value_cache + row_offset<Layout>(t, kv_head, shape) + v * 8, t < length);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * 8 + w8) * 4 + sub;
            const float probability = t < length ? static_cast<float>(p[t]) : 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] += probability * element(raw[u], j);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 8);
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 16);
    }
    if (sub == 0)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reduce[(g * 8 + w8) * kHeadDim + v * 8 + j] = acc[j];
    __syncthreads();
    if (threadIdx.x < group * kHeadDim) {
        const int g2 = threadIdx.x / kHeadDim, c = threadIdx.x % kHeadDim;
        float total = 0.0f;
        for (int w = 0; w < 8; ++w)
            total += reduce[(g2 * 8 + w) * kHeadDim + c];
        layer.partials[(first_head + g2) * kHeadDim + c] = total;
    }
}

// Reads one chunk's K or V rows the way the wide kernels do, with no arithmetic.
template <int Unroll, int Layout>
__device__ inline void stream_rows(
    const Bf16* base, int kv_head, int chunk, const ops::Shape& shape, S2Shared& shared)
{
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int sub = lane / 8, v = lane % 8;
    int begin = 0, end = 0;
    legacy_range(chunk, shape, begin, end);
    unsigned bits = 0;
    constexpr int rows = ops::kWarpsPerBlock * 4 * Unroll;
    for (int first = begin; first < end; first += rows) {
        uint4 raw[Unroll];
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int t = first + (u * ops::kWarpsPerBlock + warp) * 4 + sub;
            raw[u] = load_row_part(base + row_offset<Layout>(t, kv_head, shape) + v * 8, t < end);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u)
            bits ^= raw[u].x ^ raw[u].y ^ raw[u].z ^ raw[u].w;
    }
    shared.slices[threadIdx.x] = __uint_as_float(bits);
}

struct Program
{
    Bf16* key;   // [layers][max_context][kv_heads][head_dim]
    Bf16* value; // same
    Bf16* q;     // [query_heads][head_dim]
    Bf16* k;     // [layers][kv_heads][head_dim]: each layer's K row at `position`
    Bf16* q_rot; // the worker's rotated Q
    Bf16* scores;
    float* partials;
    float* stats;
    ops::Shape shape;
    int layers;
    int iterations;
    int variant;
    long long* cycles;             // [grid][kPhases]
    unsigned long long* wave_info; // block 0: scores ns, values ns, scores cycles, values cycles
};

__global__ __launch_bounds__(kThreads, 1) void attention_program(Program program)
{
    __shared__ ops::SharedStorage shared;
    __shared__ S2Shared s2;
    __shared__ float reduce[4 * 8 * kHeadDim];
    const auto& shape = program.shape;
    const int tasks = shape.kv_heads * shape.attention_chunks;
    const std::size_t layer_elements =
        static_cast<std::size_t>(shape.max_context) * shape.kv_heads * shape.head_dim;
    Stamps stamps;
    unsigned long long scores_ns = 0, values_ns = 0, scores_cycles = 0, values_cycles = 0;
    auto grid = cooperative_groups::this_grid();

    for (int iteration = 0; iteration < program.iterations; ++iteration)
        for (int l = 0; l < program.layers; ++l) {
            ops::Layer<Bf16> layer{};
            // S2's kernels read Q as rotated and the current K row from the
            // cache; the worker's rotates both itself. Zero rotary
            // frequencies make that rotation exact, so all see the same values.
            layer.q = program.q;
            layer.k = program.k + l * shape.kv_heads * shape.head_dim;
            layer.q_rot = program.q;
            if (program.variant == v_worker)
                layer.q_rot = program.q_rot;
            layer.key_cache = program.key + l * layer_elements;
            layer.value_cache = program.value + l * layer_elements;
            layer.scores = program.scores;
            layer.partials = program.partials;
            layer.stats = program.stats;

            const unsigned long long t0 = global_timer_ns();
            const long long c0 = clock64();
            for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
                const int kv_head = task / shape.attention_chunks;
                const int chunk = task % shape.attention_chunks;
                switch (program.variant) {
                case v_production:
                    s2_scores(kv_head, chunk, layer, shape, s2);
                    break;
                case v_worker:
                    ops::attention_scores(kv_head, chunk, layer, shape, shared);
                    break;
                case v_instr:
                    scores_instrumented<0>(kv_head, chunk, layer, shape, s2, stamps);
                    break;
                case v_instr_noreduce:
                    scores_instrumented<1>(kv_head, chunk, layer, shape, s2, stamps);
                    break;
                case v_instr_konly:
                    scores_instrumented<2>(kv_head, chunk, layer, shape, s2, stamps);
                    break;
                case v_wide:
                case v_wide_scores:
                    scores_wide<4, 0>(kv_head, chunk, layer, shape);
                    break;
                case v_wide_values:
                    s2_scores(kv_head, chunk, layer, shape, s2);
                    break;
                case v_wide_hm:
                    scores_wide<4, 1>(kv_head, chunk, layer, shape);
                    break;
                case v_wide_u2:
                    scores_wide<2, 0>(kv_head, chunk, layer, shape);
                    break;
                case v_wide_u8:
                    scores_wide<8, 0>(kv_head, chunk, layer, shape);
                    break;
                case v_fused:
                    attention_fused<4, 0>(kv_head, layer, shape, shared, reduce);
                    break;
                case v_stream:
                    stream_rows<4, 0>(layer.key_cache, kv_head, chunk, shape, s2);
                    break;
                case v_stream_hm:
                    stream_rows<4, 1>(layer.key_cache, kv_head, chunk, shape, s2);
                    break;
                default:
                    break;
                }
                __syncthreads();
            }
            // The worker skips this barrier while one chunk is active.
            const bool fused_worker = program.variant == v_worker && ops::active_chunks(shape) == 1;
            if (!fused_worker)
                grid.sync();
            const unsigned long long t1 = global_timer_ns();
            const long long c1 = clock64();
            for (int task = blockIdx.x; task < tasks && program.variant != v_fused;
                task += gridDim.x) {
                const int kv_head = task / shape.attention_chunks;
                const int chunk = task % shape.attention_chunks;
                switch (program.variant) {
                case v_production:
                    s2_values(kv_head, chunk, layer, shape, s2);
                    break;
                case v_worker:
                    ops::attention_values(kv_head, chunk, layer, shape, shared);
                    break;
                case v_instr:
                case v_instr_noreduce:
                case v_instr_konly:
                    values_instrumented(kv_head, chunk, layer, shape, s2, stamps);
                    break;
                case v_wide:
                case v_wide_values:
                    values_wide<4, 0>(kv_head, chunk, layer, shape, shared, reduce);
                    break;
                case v_wide_scores:
                    s2_values(kv_head, chunk, layer, shape, s2);
                    break;
                case v_wide_hm:
                    values_wide<4, 1>(kv_head, chunk, layer, shape, shared, reduce);
                    break;
                case v_wide_u2:
                    values_wide<2, 0>(kv_head, chunk, layer, shape, shared, reduce);
                    break;
                case v_wide_u8:
                    values_wide<8, 0>(kv_head, chunk, layer, shape, shared, reduce);
                    break;
                case v_stream:
                    stream_rows<4, 0>(layer.value_cache, kv_head, chunk, shape, s2);
                    break;
                case v_stream_hm:
                    stream_rows<4, 1>(layer.value_cache, kv_head, chunk, shape, s2);
                    break;
                default:
                    break;
                }
                __syncthreads();
            }
            if (program.variant != v_fused)
                grid.sync();
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                scores_ns += t1 - t0;
                values_ns += global_timer_ns() - t1;
                scores_cycles += c1 - c0;
                values_cycles += clock64() - c1;
            }
        }
    if (threadIdx.x == 0) {
        for (int p = 0; p < kPhases; ++p)
            program.cycles[blockIdx.x * kPhases + p] = stamps.total[p];
        if (blockIdx.x == 0) {
            program.wave_info[0] = scores_ns;
            program.wave_info[1] = values_ns;
            program.wave_info[2] = scores_cycles;
            program.wave_info[3] = values_cycles;
        }
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

double elapsed_us(const cuda::timed_event& start, const cuda::timed_event& end)
{
    return cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count();
}

std::vector<int> parse_list(const std::string& text)
{
    std::vector<int> values;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');)
        values.push_back(std::stoi(item));
    return values;
}

json attention_sweep(
    int sm_count, const std::vector<int>& positions, const std::vector<int>& chunk_counts)
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    auto& pool = cuda::device_default_memory_pool(device);
    const std::size_t layer_elements = static_cast<std::size_t>(kMaxContext) * kKvHeads * kHeadDim;
    auto filled = [&](std::size_t count, unsigned seed) {
        cuda::device_buffer<Bf16> buffer(stream, pool, count, cuda::no_init);
        cuda::launch(stream, cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024)),
            fill_hashed, buffer.data(), count, seed);
        return buffer;
    };
    auto key = filled(layer_elements * kLayers, 11);
    auto value = filled(layer_elements * kLayers, 23);
    auto q = filled(kQueryHeads * kHeadDim, 5);
    cuda::device_buffer<Bf16> q_rot(stream, pool, kQueryHeads * kHeadDim, cuda::no_init);
    cuda::device_buffer<Bf16> k(
        stream, pool, static_cast<std::size_t>(kLayers) * kKvHeads * kHeadDim, cuda::no_init);
    const std::vector<float> zero_frequencies(kHeadDim / 2, 0.0f);
    cuda::device_buffer<float> frequencies(stream, pool, kHeadDim / 2, cuda::no_init);
    cuda::copy_bytes(stream, zero_frequencies, frequencies);
    cuda::device_buffer<Bf16> scores(
        stream, pool, static_cast<std::size_t>(kQueryHeads) * kMaxContext, cuda::no_init);
    constexpr int kMaxChunks = 64;
    cuda::device_buffer<float> partials(
        stream, pool, static_cast<std::size_t>(kMaxChunks) * kQueryHeads * kHeadDim, cuda::no_init);
    cuda::device_buffer<float> stats(
        stream, pool, static_cast<std::size_t>(kMaxChunks) * kQueryHeads * 2, cuda::no_init);
    cuda::device_buffer<long long> cycles(
        stream, pool, static_cast<std::size_t>(sm_count) * kPhases, cuda::no_init);
    cuda::device_buffer<unsigned long long> wave_info(stream, pool, 4, cuda::no_init);
    stream.sync();

    const auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(sm_count), cuda::cooperative_launch{});
    json rows = json::array();
    for (int chunks : chunk_counts)
        for (int position : positions) {
            for (int l = 0; l < kLayers; ++l)
                check(
                    cudaMemcpyAsync(k.data() + static_cast<std::size_t>(l) * kKvHeads * kHeadDim,
                        key.data() + l * layer_elements +
                            static_cast<std::size_t>(position) * kKvHeads * kHeadDim,
                        kKvHeads * kHeadDim * sizeof(Bf16), cudaMemcpyDeviceToDevice, stream.get()),
                    "copy current K rows");
            std::vector<float> reference;
            for (int variant = 0; variant < kVariants; ++variant) {
                const bool needs_shared_probabilities =
                    variant == v_wide || variant == v_wide_values || variant == v_wide_hm ||
                    variant == v_wide_u2 || variant == v_wide_u8 || variant == v_fused;
                const int length = (position + chunks) / chunks;
                if (needs_shared_probabilities && length > 2048)
                    continue;
                if (variant == v_fused && chunks != 1)
                    continue;
                if (variant == v_worker && chunks != chunk_counts.front())
                    continue;
                // The worker's rule, as in ops::active_chunks.
                const bool worker = variant == v_worker;
                const int program_chunks = worker ? kMaxContext / kWorkerChunkSteps : chunks;
                const int active = !worker                             ? chunks
                                   : position + 1 <= kWorkerFusedSteps ? 1
                                                                       : program_chunks;
                Program program{};
                program.key = key.data();
                program.value = value.data();
                program.q = q.data();
                program.k = k.data();
                program.q_rot = q_rot.data();
                program.scores = scores.data();
                program.partials = partials.data();
                program.stats = stats.data();
                program.shape = ops::Shape{2048, 8192, kQueryHeads, kKvHeads, kHeadDim, position,
                    kMaxContext, program_chunks, worker ? kWorkerChunkSteps : kMaxContext,
                    worker ? kWorkerFusedSteps : kMaxContext, 0, NumericConfig{},
                    frequencies.data()};
                program.layers = kLayers;
                program.variant = variant;
                program.cycles = cycles.data();
                program.wave_info = wave_info.data();
                program.iterations = 1;
                cuda::launch(stream, config, attention_program, program);
                program.iterations = position > 1000 ? 8 : 24;
                check(cudaMemsetAsync(cycles.data(), 0,
                          static_cast<std::size_t>(sm_count) * kPhases * sizeof(long long),
                          stream.get()),
                    "clear");
                cuda::timed_event start{stream};
                start.record(stream);
                cuda::launch(stream, config, attention_program, program);
                cuda::timed_event end{stream};
                end.record(stream);
                end.sync();
                const double layer_us = elapsed_us(start, end) / (program.iterations * kLayers);

                std::vector<long long> host_cycles(static_cast<std::size_t>(sm_count) * kPhases);
                unsigned long long host_wave[4];
                std::vector<float> host_partials(
                    static_cast<std::size_t>(active) * kQueryHeads * kHeadDim);
                check(cudaMemcpy(host_cycles.data(), cycles.data(),
                          host_cycles.size() * sizeof(long long), cudaMemcpyDeviceToHost),
                    "copy cycles");
                check(cudaMemcpy(
                          host_wave, wave_info.data(), sizeof(host_wave), cudaMemcpyDeviceToHost),
                    "copy waves");
                check(cudaMemcpy(host_partials.data(), partials.data(),
                          host_partials.size() * sizeof(float), cudaMemcpyDeviceToHost),
                    "copy partials");

                const double evals = static_cast<double>(program.iterations) * kLayers;
                const double mhz = host_wave[2] + host_wave[3] > 0
                                       ? (host_wave[2] + host_wave[3]) * 1e3 /
                                             static_cast<double>(host_wave[0] + host_wave[1])
                                       : 0.0;
                json row{{"variant", kVariantNames[variant]}, {"position", position},
                    {"context", position + 1}, {"chunks", active}, {"layer_us", layer_us},
                    {"scores_wave_us", host_wave[0] / evals / 1e3},
                    {"values_wave_us", host_wave[1] / evals / 1e3}, {"sm_mhz", mhz}};
                if (variant >= v_instr) {
                    // Critical block: mean over blocks of per-layer cycles in each phase.
                    json phases = json::object();
                    for (int p = 0; p < kPhases; ++p) {
                        double sum = 0, worst = 0;
                        for (int b = 0; b < sm_count; ++b) {
                            const double c = host_cycles[b * kPhases + p] / evals;
                            sum += c;
                            worst = std::max(worst, c);
                        }
                        phases[kPhaseNames[p]] = {{"mean_cycles", sum / sm_count},
                            {"max_cycles", worst},
                            {"mean_us", mhz > 0 ? sum / sm_count / mhz : 0.0}};
                    }
                    row["phases"] = phases;
                }
                // Fixed-order chunk sum per head coordinate, as output_projection does.
                std::vector<float> combined(kQueryHeads * kHeadDim, 0.0f);
                for (int c = 0; c < active; ++c)
                    for (int i = 0; i < kQueryHeads * kHeadDim; ++i)
                        combined[i] += host_partials[c * kQueryHeads * kHeadDim + i];
                if (variant == v_production)
                    reference = combined;
                if (compares_with_production(variant)) {
                    double worst = 0;
                    for (int i = 0; i < kQueryHeads * kHeadDim; ++i)
                        worst = std::max(worst, std::fabs(double(combined[i]) - reference[i]));
                    row["max_abs_diff_vs_production"] = worst;
                }
                rows.push_back(row);
            }
        }
    return rows;
}
} // namespace

int main(int argc, char** argv)
{
    try {
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, 0), "device query");
        const std::vector<int> positions =
            parse_list(argc >= 2 ? argv[1] : "0,63,255,1023,2047,4095");
        const std::vector<int> chunks = parse_list(argc >= 3 ? argv[2] : "5");
        json result{{"device", props.name}, {"sm_count", props.multiProcessorCount}};
        result["attention"] = attention_sweep(props.multiProcessorCount, positions, chunks);
        std::cout << result.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "attention_probe: " << error.what() << '\n';
        return 1;
    }
}
