#pragma once

/**
 * @file
 * @brief Device arithmetic for a validated decoder task program.
 *
 * Scalar is float or __nv_bfloat16 for weights, activations and K/V; reductions
 * remain FP32. Block collectives require a one-dimensional kThreads-thread block
 * with block-uniform task/shape arguments and block-private SharedStorage.
 * Tensor IDs, dimensions, head indices and task ranges are host-validated.
 * Scalar/warp helpers and grid operations state their participation separately.
 * No routine performs a grid barrier. Unless an exit barrier is documented,
 * the caller must synchronize before another thread consumes outputs or reuses
 * collective scratch; cross-block dependencies require a grid/stream boundary.
 */

#include "plan.hpp"

#include <cuda_bf16.h>
#include <cuda/std/type_traits>

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace tileinfer::ops {

/// Borrowed storage with a uniform Scalar dtype; optional ffn_trace widens diagnostics to FP32.
template <typename Scalar>
struct TensorViews
{
    Scalar* input;
    Scalar* output;
    Scalar* scratch;
    Scalar* kv;
    const Scalar* weights;
    Scalar* attention_scores;
    float* ffn_trace;
};

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = kThreads / kWarpSize;
inline constexpr int kNormReductionThreads = 512;
using WarpSum = cub::WarpReduce<float>;
using BlockSum = cub::BlockReduce<float, kThreads>;

/// FP32 reduction operator for the softmax maximum.
struct FloatMax
{
    __device__ float operator()(float a, float b) const { return cuda::std::fmax(a, b); }
};

/// Block-private scratch; each warp owns one warp slot, and block collectives share block.
struct SharedStorage
{
    WarpSum::TempStorage warp[kWarpsPerBlock];
    BlockSum::TempStorage block;
    float inverse_rms;
    float norm_partials[kNormReductionThreads];
    float softmax_max;
    float softmax_sum;
    float previous_scale;
    float key_weight;
};

/// Validated decoder dimensions, current cache position and FP32 numerical constants.
struct BlockShape
{
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    int position;
    int max_context;
    NumericConfig numeric;
    const float* rotary_frequencies;
};

/// One layer's tensor bindings and the calling block's collective scratch.
template <typename Scalar>
struct Context
{
    const TensorRecord* tensors;
    TensorViews<Scalar> views;
    BlockShape shape;
    SharedStorage* shared;
};

/// Resolve a validated activation binding using element offsets, not byte offsets.
/// @param context Borrowed buffers and the layer's tensor-record table.
/// @param id In-range tensor ID; kNoTensor is not accepted.
/// @return Its storage pointer plus offset, or nullptr for a weight record.
template <typename Scalar>
__device__ inline Scalar* activation_tensor(const Context<Scalar>& context, cuda::std::uint32_t id)
{
    const TensorRecord record = context.tensors[id];
    Scalar* base = nullptr;
    switch (record.storage) {
    case StorageClass::input:
        base = context.views.input;
        break;
    case StorageClass::output:
        base = context.views.output;
        break;
    case StorageClass::scratch:
        base = context.views.scratch;
        break;
    case StorageClass::kv:
        base = context.views.kv;
        break;
    case StorageClass::weight:
        return nullptr;
    }
    return base + record.offset;
}

/// Resolve a weight binding without device-side storage/bounds checks.
/// @param context Immutable weight base and validated tensor records.
/// @param id In-range ID already verified to belong to weight storage.
/// @return Borrowed pointer to the tensor's first Scalar element.
template <typename Scalar>
__device__ inline const Scalar* weight_tensor(
    const Context<Scalar>& context, cuda::std::uint32_t id)
{
    return context.views.weights + context.tensors[id].offset;
}

/// Rotate one split-half pair; BF16 coefficients and products round before the sum.
/// @param source,destination Head storage; disjoint or exactly in-place, never partly overlapping.
/// @param source_base,destination_base Element offsets of the corresponding head starts.
/// @param pair Pair index in [0, shape.head_dim / 2), owned by this thread.
/// @param shape Even head width, position and FP32 inverse-frequency table.
/// @post Only destination[destination_base+pair] and its partner at +head_dim/2 change.
/// This scalar helper has no collective or synchronization requirement.
template <typename Scalar>
__device__ inline void rotate_pair(const Scalar* source, Scalar* destination, int source_base,
    int destination_base, int pair, const BlockShape& shape)
{
    const float angle = shape.position * shape.rotary_frequencies[pair];
    const float sine = static_cast<float>(static_cast<Scalar>(cuda::std::sin(angle)));
    const float cosine = static_cast<float>(static_cast<Scalar>(cuda::std::cos(angle)));
    const int half = shape.head_dim / 2;
    const float first = source[source_base + pair];
    const float second = source[source_base + pair + half];
    destination[destination_base + pair] =
        static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(first * cosine)) -
                            static_cast<float>(static_cast<Scalar>(second * sine)));
    destination[destination_base + pair + half] =
        static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(second * cosine)) +
                            static_cast<float>(static_cast<Scalar>(first * sine)));
}

/// Compute inverse RMS in FP32; only thread 0 writes the result.
/// @param input,width Contiguous vector and positive element count, uniform across the block.
/// @param epsilon Positive FP32 stabilizer, added after the mean of squares.
/// @param[out] inverse_rms One block-shared float, disjoint from input and reduction_scratch.
/// @param[out] reduction_scratch At least kNormReductionThreads block-shared floats; unused for FP32.
/// @post An exit block barrier makes *inverse_rms readable by every participating thread.
/// BF16 follows the pinned PyTorch mean reduction; FP32 retains the legacy serial order.
template <typename Scalar>
__device__ inline void rms_norm_factor(
    const Scalar* input, int width, float epsilon, float* inverse_rms, float* reduction_scratch)
{
    if constexpr (cuda::std::is_same_v<Scalar, __nv_bfloat16>) {
        // Match the pinned PyTorch contiguous FP32 mean: four independent
        // accumulators, a power-of-two block tree, then decreasing warp offsets.
        const bool contiguous_groups = width >= 128;
        int reduction_threads = 1;
        const int available_groups = contiguous_groups ? width / 4 : width;
        while (
            reduction_threads < kNormReductionThreads && reduction_threads * 2 <= available_groups)
            reduction_threads *= 2;
        float sums[4] = {};
        if (threadIdx.x < reduction_threads) {
            if (contiguous_groups) {
                for (int group = threadIdx.x; group * 4 + 3 < width; group += reduction_threads)
                    for (int i = 0; i < 4; ++i) {
                        const float value = static_cast<float>(input[group * 4 + i]);
                        sums[i] += __fmul_rn(value, value);
                    }
                const int tail = width - width % 4 + threadIdx.x;
                if (tail < width) {
                    const float value = static_cast<float>(input[tail]);
                    sums[0] += __fmul_rn(value, value);
                }
            } else {
                for (int base = threadIdx.x; base < width; base += reduction_threads * 4)
                    for (int i = 0; i < 4; ++i) {
                        const int index = base + i * reduction_threads;
                        if (index < width) {
                            const float value = static_cast<float>(input[index]);
                            sums[i] += __fmul_rn(value, value);
                        }
                    }
            }
            reduction_scratch[threadIdx.x] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
        }
        __syncthreads();
        for (int offset = reduction_threads / 2; offset >= kWarpSize; offset /= 2) {
            if (threadIdx.x < offset)
                reduction_scratch[threadIdx.x] += reduction_scratch[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x < kWarpSize) {
            float value = threadIdx.x < reduction_threads ? reduction_scratch[threadIdx.x] : 0.0f;
            for (int offset = (reduction_threads < kWarpSize ? reduction_threads : kWarpSize) / 2;
                offset > 0; offset /= 2)
                value += __shfl_down_sync(0xffffffff, value, offset);
            if (threadIdx.x == 0) {
                const float mean = __fmul_rn(value, 1.0f / static_cast<float>(width));
                *inverse_rms = rsqrtf(__fadd_rn(mean, epsilon));
            }
        }
    } else if (threadIdx.x == 0) {
        float squares = 0.0f;
        for (int index = 0; index < width; ++index)
            squares += input[index] * input[index];
        *inverse_rms = 1.0f / cuda::std::sqrt(squares / width + epsilon);
    }
    __syncthreads();
}

/// Cooperatively reduce one row dot product using FP32 partial sums.
/// @param row,input,width Read-only vectors and their positive common length, uniform within the warp.
/// @param shared This block's scratch; only the calling warp's slot is accessed.
/// @pre All 32 lanes of the physical warp participate with the same arguments.
/// @return The dot product in lane 0 only; other lanes' return values are unspecified.
/// The exit warp barrier permits that warp to reuse its scratch immediately.
template <typename Scalar>
__device__ inline float warp_dot(
    const Scalar* row, const Scalar* input, int width, SharedStorage& shared)
{
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    float value = 0.0f;
    for (int column = lane; column < width; column += kWarpSize)
        value += static_cast<float>(row[column]) * static_cast<float>(input[column]);
    const float sum = WarpSum(shared.warp[warp]).Sum(value);
    __syncwarp();
    return sum;
}

/// Normalize one hidden vector, rounding the normalized BF16 value before scale multiplication.
/// @param task Bindings a=input, b=scale, c=output; the entire hidden vector is owned by this block.
/// @param context Hidden width, epsilon, tensor bindings and block-private norm scratch.
/// @pre Input/output may alias exactly; partial overlap and aliasing scale/scratch are unsupported.
/// @post Every output element is written once; there is no exit block barrier.
template <typename Scalar>
__device__ inline void rms_norm(const TaskRecord& task, const Context<Scalar>& context)
{
    const Scalar* input = activation_tensor(context, task.a);
    const Scalar* scale = weight_tensor(context, task.b);
    Scalar* output = activation_tensor(context, task.c);
    rms_norm_factor(input, context.shape.hidden_size, context.shape.numeric.rms_norm_eps,
        &context.shared->inverse_rms, context.shared->norm_partials);
    for (int index = threadIdx.x; index < context.shape.hidden_size; index += blockDim.x)
        output[index] = static_cast<Scalar>(
            static_cast<float>(static_cast<Scalar>(
                static_cast<float>(input[index]) * context.shared->inverse_rms)) *
            static_cast<float>(scale[index]));
}

/// Commit one owned row, rounding the projection before any residual addition.
/// @param task,context Identify Wdown and an optional FP32 pre-residual diagnostic trace.
/// @param output,residual Row vectors; residual is nullptr, disjoint from output, or aliases it exactly.
/// @param row Output index exclusively owned by this thread.
/// @param accumulation Complete FP32 dot product; no partial-K result is accepted.
/// @post Writes output[row] and, for Wdown when requested, the rounded projection trace.
/// This scalar epilogue has no collective or synchronization requirement.
template <typename Scalar>
__device__ inline void store_projection(const TaskRecord& task, const Context<Scalar>& context,
    Scalar* output, const Scalar* residual, int row, float accumulation)
{
    const Scalar projected = static_cast<Scalar>(accumulation);
    if (context.views.ffn_trace && task.b == static_cast<cuda::std::uint32_t>(TensorId::Wdown))
        context.views.ffn_trace[row] = static_cast<float>(projected);
    output[row] = residual ? static_cast<Scalar>(
                                 static_cast<float>(projected) + static_cast<float>(residual[row]))
                           : projected;
}

/// Compute complete dot products for the task's exclusive [begin,end) output rows.
/// @param task Bindings a=input, b=matrix, c=output, d=optional residual (kNoTensor disables it).
/// @param context Wdown uses intermediate_size columns; other projections use hidden_size.
/// @pre Output is disjoint from input/matrix; a residual may exactly alias output.
/// @post Warp leaders own the stores, one warp per row. No exit barrier.
template <typename Scalar>
__device__ inline void gemv(const TaskRecord& task, const Context<Scalar>& context)
{
    const Scalar* input = activation_tensor(context, task.a);
    const Scalar* matrix = weight_tensor(context, task.b);
    Scalar* output = activation_tensor(context, task.c);
    const Scalar* residual = task.d == kNoTensor ? nullptr : activation_tensor(context, task.d);
    const int width = task.b == static_cast<cuda::std::uint32_t>(TensorId::Wdown)
                          ? context.shape.intermediate_size
                          : context.shape.hidden_size;
    const int warp = threadIdx.x / kWarpSize;
    for (cuda::std::uint32_t row = task.begin + warp; row < task.end; row += kWarpsPerBlock) {
        const float value = warp_dot(
            matrix + static_cast<cuda::std::size_t>(row) * width, input, width, *context.shared);
        if (threadIdx.x % kWarpSize == 0)
            store_projection(task, context, output, residual, row, value);
    }
}

/// Rotate only the selected query head, using the precision boundaries of rotate_pair().
/// @param task Bindings a=Q, b=rotated Q, and the owned query-head index.
/// @param context Valid even head width, position and FP32 inverse frequencies.
/// @pre Source/destination are disjoint or exactly in-place; this block owns the head.
/// @post Other heads are untouched; there is no exit block barrier.
template <typename Scalar>
__device__ inline void rope_query(const TaskRecord& task, const Context<Scalar>& context)
{
    const Scalar* input = activation_tensor(context, task.a);
    Scalar* output = activation_tensor(context, task.b);
    for (int pair = threadIdx.x; pair < context.shape.head_dim / 2; pair += blockDim.x) {
        const int base = task.head * context.shape.head_dim;
        rotate_pair(input, output, base, base, pair, context.shape);
    }
}

/// Append rotated K and unchanged V at one cache position for one KV head.
/// @param task Bindings a=K, b=V, c=K cache, d=V cache, and the owned KV-head index.
/// @param context Caches use [max_context,kv_heads,head_dim]; position is in range.
/// @pre Source vectors and cache buffers are disjoint; this block exclusively owns the appended head.
/// @post Earlier positions and other heads are untouched. Consumers need the caller's barrier.
template <typename Scalar>
__device__ inline void rope_append_kv(const TaskRecord& task, const Context<Scalar>& context)
{
    const Scalar* key = activation_tensor(context, task.a);
    const Scalar* value = activation_tensor(context, task.b);
    Scalar* key_cache = activation_tensor(context, task.c);
    Scalar* value_cache = activation_tensor(context, task.d);
    const int source = task.kv_head * context.shape.head_dim;
    const int destination =
        context.shape.position * context.shape.kv_heads * context.shape.head_dim + source;
    for (int pair = threadIdx.x; pair < context.shape.head_dim / 2; pair += blockDim.x)
        rotate_pair(key, key_cache, source, destination, pair, context.shape);
    for (int index = threadIdx.x; index < context.shape.head_dim; index += blockDim.x)
        value_cache[destination + index] = value[source + index];
}

/// Compute one FP32 query head using constant-workspace online softmax.
/// @param task Bindings a=Q, b=K cache, c=V cache, d=output, plus query/KV-head indices.
/// @param context Valid prefix [0,position] inclusive and block-private reduction/softmax scratch.
/// @pre That prefix is populated; caches are [max_context,kv_heads,head_dim]. Output is disjoint from Q/K/V.
/// @post Only this head's output changes; unused cache capacity is never read. No exit barrier.
/// Thread 0 publishes the running maximum and rescaling factors; threads own output coordinates.
/// Rescaling both numerator and denominator avoids storing a score matrix.
__device__ inline void attention_prefix_f32(const TaskRecord& task, const Context<float>& context)
{
    const float* query = activation_tensor(context, task.a) + task.head * context.shape.head_dim;
    const float* keys = activation_tensor(context, task.b);
    const float* values = activation_tensor(context, task.c);
    float* output = activation_tensor(context, task.d) + task.head * context.shape.head_dim;
    const BlockShape& shape = context.shape;
    SharedStorage& shared = *context.shared;
    for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
        output[coordinate] = 0.0f;
    if (threadIdx.x == 0) {
        shared.softmax_max = -cuda::std::numeric_limits<float>::infinity();
        shared.softmax_sum = 0.0f;
    }
    __syncthreads();
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(shape.head_dim));
    for (int time = 0; time <= shape.position; ++time) {
        const int base = (time * shape.kv_heads + task.kv_head) * shape.head_dim;
        float dot = 0.0f;
        for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
            dot += query[coordinate] * keys[base + coordinate];
        const float score = BlockSum(shared.block).Sum(dot) * scale;
        if (threadIdx.x == 0) {
            // Shift the previous numerator/denominator onto the new maximum before merging this key.
            const float maximum = cuda::std::fmax(shared.softmax_max, score);
            shared.previous_scale = cuda::std::exp(shared.softmax_max - maximum);
            shared.key_weight = cuda::std::exp(score - maximum);
            shared.softmax_sum = shared.softmax_sum * shared.previous_scale + shared.key_weight;
            shared.softmax_max = maximum;
        }
        __syncthreads();
        for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
            output[coordinate] = output[coordinate] * shared.previous_scale +
                                 values[base + coordinate] * shared.key_weight;
        // Every coordinate must consume these factors before thread 0 advances to the next key.
        __syncthreads();
    }
    for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
        output[coordinate] /= shared.softmax_sum;
}

/// Compute one BF16 query head with eager score/probability rounding and FP32 reductions.
/// @param task Bindings a=Q, b=K cache, c=V cache, d=output, plus query/KV-head indices.
/// @param context Populated prefix [0,position], cache layout [max_context,kv_heads,head_dim], and scratch.
/// @pre This block owns its head's [max_context] attention_scores slice, disjoint from Q/K/V/output.
/// @pre Output is disjoint from Q/K/V; all inputs remain immutable during the call.
/// @post The valid workspace prefix contains BF16 probabilities; its tail and other heads are untouched.
/// Only the owned output head is written, after the complete FP32 PV sum. No exit barrier.
__device__ inline void attention_prefix_bf16(
    const TaskRecord& task, const Context<__nv_bfloat16>& context)
{
    const auto& shape = context.shape;
    auto& shared = *context.shared;
    const auto* query = activation_tensor(context, task.a) + task.head * shape.head_dim;
    const auto* keys = activation_tensor(context, task.b);
    const auto* values = activation_tensor(context, task.c);
    auto* output = activation_tensor(context, task.d) + task.head * shape.head_dim;
    auto* scores = context.views.attention_scores + task.head * shape.max_context;
    const int length = shape.position + 1;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(shape.head_dim));
    // QK and scaling are separate BF16 rounding boundaries; each warp owns whole scores.
    for (int time = warp; time < length; time += kWarpsPerBlock) {
        const int base = (time * shape.kv_heads + task.kv_head) * shape.head_dim;
        float dot = 0.0f;
        for (int coordinate = lane; coordinate < shape.head_dim; coordinate += kWarpSize)
            dot +=
                static_cast<float>(query[coordinate]) * static_cast<float>(keys[base + coordinate]);
        const float sum = WarpSum(shared.warp[warp]).Sum(dot);
        if (lane == 0)
            scores[time] =
                __float2bfloat16_rn(static_cast<float>(__float2bfloat16_rn(sum)) * scale);
        __syncwarp();
    }
    __syncthreads();
    // Only thread 0 publishes the FP32 maximum/denominator; barriers protect CUB scratch reuse.
    float maximum = -cuda::std::numeric_limits<float>::infinity();
    for (int time = threadIdx.x; time < length; time += blockDim.x)
        maximum = cuda::std::fmax(maximum, static_cast<float>(scores[time]));
    maximum = BlockSum(shared.block).Reduce(maximum, FloatMax());
    if (threadIdx.x == 0)
        shared.softmax_max = maximum;
    __syncthreads();
    float denominator = 0.0f;
    for (int time = threadIdx.x; time < length; time += blockDim.x)
        denominator += cuda::std::exp(static_cast<float>(scores[time]) - shared.softmax_max);
    denominator = BlockSum(shared.block).Sum(denominator);
    if (threadIdx.x == 0)
        shared.softmax_sum = denominator;
    __syncthreads();
    // Overwrite scores only after the denominator is known; rounded probabilities need not sum to one.
    for (int time = threadIdx.x; time < length; time += blockDim.x)
        scores[time] = __float2bfloat16_rn(
            cuda::std::exp(static_cast<float>(scores[time]) - shared.softmax_max) /
            shared.softmax_sum);
    __syncthreads();
    // Publish all probabilities before PV; each coordinate keeps its full prefix sum in FP32.
    for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x) {
        float sum = 0.0f;
        for (int time = 0; time < length; ++time) {
            const int base = (time * shape.kv_heads + task.kv_head) * shape.head_dim;
            sum += static_cast<float>(scores[time]) * static_cast<float>(values[base + coordinate]);
        }
        output[coordinate] = __float2bfloat16_rn(sum);
    }
}

/// Apply SiLU(gate)*up, preserving the intermediate BF16 SiLU rounding.
/// @param task Bindings a=gate, b=up, c=output and the block-owned [begin,end) element range.
/// @param context Tensor bindings with the uniform storage dtype.
/// @pre Output may exactly alias either input; partial overlap is unsupported.
/// @post Each owned element is written once; no other range is touched and no exit barrier is issued.
template <typename Scalar>
__device__ inline void silu_product(const TaskRecord& task, const Context<Scalar>& context)
{
    const Scalar* gate = activation_tensor(context, task.a);
    const Scalar* up = activation_tensor(context, task.b);
    Scalar* output = activation_tensor(context, task.c);
    for (cuda::std::uint32_t index = task.begin + threadIdx.x; index < task.end;
        index += blockDim.x)
        output[index] = static_cast<Scalar>(
            static_cast<float>(
                static_cast<Scalar>(static_cast<float>(gate[index]) /
                                    (1.0f + cuda::std::exp(-static_cast<float>(gate[index]))))) *
            static_cast<float>(up[index]));
}

/// Dispatch one loader-validated task uniformly across the calling block.
/// @param task Opcode, bindings and owned output range; its dtype must agree with Scalar.
/// @param context Layer bindings, current prefix shape and block-private collective scratch.
/// @post Output ownership follows the selected operation; the caller must finish the task's barriers.
/// Device dispatch does not validate malformed opcode/dtype combinations.
template <typename Scalar>
__device__ inline void execute(const TaskRecord& task, const Context<Scalar>& context)
{
    switch (task.opcode) {
    case Opcode::rms_norm_f32:
    case Opcode::rms_norm_bf16:
        rms_norm(task, context);
        break;
    case Opcode::gemv_f32:
    case Opcode::gemv_bf16:
        gemv(task, context);
        break;
    case Opcode::rope_query_f32:
    case Opcode::rope_query_bf16:
        rope_query(task, context);
        break;
    case Opcode::rope_cache_f32:
    case Opcode::rope_cache_bf16:
        rope_append_kv(task, context);
        break;
    case Opcode::attention_prefix_f32:
        if constexpr (cuda::std::is_same_v<Scalar, float>)
            attention_prefix_f32(task, context);
        break;
    case Opcode::attention_prefix_bf16:
        if constexpr (cuda::std::is_same_v<Scalar, __nv_bfloat16>)
            attention_prefix_bf16(task, context);
        break;
    case Opcode::silu_product_f32:
    case Opcode::silu_product_bf16:
        silu_product(task, context);
        break;
    }
}

/// Copy a token embedding with unique element ownership across a one-dimensional grid.
/// @param token_id Validated vocabulary index; bounds are not checked here.
/// @param hidden_size Positive embedding row width.
/// @param embedding,hidden Row-major embedding table and a disjoint hidden-vector destination.
/// @pre All threads of a one-dimensional grid and blocks participate; weights are already visible.
/// @post The hidden vector is copied without arithmetic conversion; readers need a grid/stream boundary.
template <typename Scalar>
__device__ inline void embedding_lookup(
    TokenId token_id, int hidden_size, const Scalar* embedding, Scalar* hidden)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        hidden[index] = embedding[token_id * hidden_size + index];
}

/// Normalize the final hidden vector, partitioning output elements across the grid.
/// @param hidden,scale,normalized Hidden input, RMS scale and disjoint output vectors of hidden_size elements.
/// @param hidden_size,epsilon Positive vector width and FP32 stabilizer.
/// @param shared_inverse_rms,reduction_scratch Block-private shared float and kNormReductionThreads floats.
/// @pre All threads of a one-dimensional grid of kThreads-thread blocks read the same input.
/// @pre Normalized output must not alias hidden or scale; block scratch is disjoint from the vectors.
/// Each block computes its own inverse RMS to avoid inter-block synchronization; BF16 rounds before scaling.
/// @post The caller must publish the completed output with a grid barrier before computing logits.
template <typename Scalar>
__device__ inline void final_rms_norm(const Scalar* hidden, const Scalar* scale, Scalar* normalized,
    int hidden_size, float epsilon, float* shared_inverse_rms, float* reduction_scratch)
{
    rms_norm_factor(hidden, hidden_size, epsilon, shared_inverse_rms, reduction_scratch);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        normalized[index] = static_cast<Scalar>(
            static_cast<float>(
                static_cast<Scalar>(static_cast<float>(hidden[index]) * *shared_inverse_rms)) *
            static_cast<float>(scale[index]));
}

/// Compute vocabulary logits with warp-owned rows across the one-dimensional grid.
/// @param hidden,lm_head Globally visible normalized input and row-major [vocab_size,hidden_size] weights.
/// @param[out] output Disjoint float32 API buffer; BF16 mathematical logits round before widening.
/// @param hidden_size,vocab_size Positive matrix dimensions, uniform across the grid.
/// @param shared Block-private reduction scratch; warps use separate slots.
/// @pre All threads in a one-dimensional grid of kThreads-thread blocks participate.
/// @post Every logit has one writer, lane 0 of its warp; consumers require a grid/stream boundary.
template <typename Scalar>
__device__ inline void logits(const Scalar* hidden, const Scalar* lm_head, float* output,
    int hidden_size, int vocab_size, SharedStorage& shared)
{
    const int warp = threadIdx.x / kWarpSize;
    for (int row = blockIdx.x * kWarpsPerBlock + warp; row < vocab_size;
        row += gridDim.x * kWarpsPerBlock) {
        const float value = warp_dot(lm_head + static_cast<cuda::std::size_t>(row) * hidden_size,
            hidden, hidden_size, shared);
        if (threadIdx.x % kWarpSize == 0)
            output[row] = static_cast<float>(static_cast<Scalar>(value));
    }
}

} // namespace tileinfer::ops
