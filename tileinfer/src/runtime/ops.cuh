#pragma once

/**
 * @file
 * @brief Device arithmetic for the decoder's fused tile tasks.
 *
 * Scalar is float or __nv_bfloat16 for weights, activations and K/V. Reductions
 * stay FP32, and BF16 values round at the same operator boundaries as eager
 * PyTorch: after each projection, before a residual, after SiLU, and so on.
 * Every routine runs on one kThreads-thread block with block-uniform arguments
 * and does not synchronize the grid: a consumer in another block needs the
 * worker program's barrier. Routines end with their block-local stores; the
 * caller synchronizes the block before another task reuses shared scratch.
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

/// Block-private scratch. Each warp owns one warp slot; block collectives share block.
struct SharedStorage
{
    WarpSum::TempStorage warp[kWarpsPerBlock];
    BlockSum::TempStorage block;
    float inverse_rms;
    float norm_partials[kNormReductionThreads];
    float softmax_max;
    float softmax_sum;
    float slices[kThreads]; // one partial sum per thread
    // A task's input vector and one projected head, typed by the caller.
    alignas(16) unsigned char vector[kMaxHidden * sizeof(float)];
    alignas(16) unsigned char head[kMaxHeadDim * sizeof(float)];
};

template <typename Scalar>
__device__ inline Scalar* shared_vector(SharedStorage& shared)
{
    return reinterpret_cast<Scalar*>(shared.vector);
}

/// Model dimensions, the current position and FP32 constants for one launch.
struct Shape
{
    int hidden_size;
    int intermediate_size;
    int query_heads;
    int kv_heads;
    int head_dim;
    int position;
    int max_context;
    int attention_chunks;
    NumericConfig numeric;
    const float* rotary_frequencies;
};

/// One layer's buffers, already offset to that layer.
template <typename Scalar>
struct Layer
{
    const Scalar* x; // layer input
    Scalar* y;       // layer output
    Scalar* q_rot;
    Scalar* attention;
    Scalar* r; // attention block output, the FFN's input and residual
    Scalar* product;
    Scalar* key_cache;   // [max_context, kv_heads, head_dim]
    Scalar* value_cache; // [max_context, kv_heads, head_dim]
    Scalar* scores;      // [query_heads, max_context]: scores, then probabilities
    float* partials;     // [chunks, query_heads, head_dim]: partial P*V
    float* stats;        // [chunks, query_heads, 2]: chunk maximum and sum of exp
    const Scalar* attn_norm;
    const Scalar* Wq;
    const Scalar* Wk;
    const Scalar* Wv;
    const Scalar* Wo;
    const Scalar* ffn_norm;
    const Scalar* Wgate;
    const Scalar* Wup;
    const Scalar* Wdown;
    float* ffn_trace;   // optional: Wdown projection before the residual
    float* block_trace; // optional: this layer's output
};

/// Rotate one split-half pair; BF16 coefficients and products round before the sum.
/// @param source,destination Head vectors of shape.head_dim elements.
/// @param pair Pair index in [0, shape.head_dim / 2), owned by this thread.
template <typename Scalar>
__device__ inline void rotate_pair(
    const Scalar* source, Scalar* destination, int pair, const Shape& shape)
{
    const float angle = shape.position * shape.rotary_frequencies[pair];
    const float sine = static_cast<float>(static_cast<Scalar>(cuda::std::sin(angle)));
    const float cosine = static_cast<float>(static_cast<Scalar>(cuda::std::cos(angle)));
    const int half = shape.head_dim / 2;
    const float first = source[pair];
    const float second = source[pair + half];
    destination[pair] =
        static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(first * cosine)) -
                            static_cast<float>(static_cast<Scalar>(second * sine)));
    destination[pair + half] =
        static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(second * cosine)) +
                            static_cast<float>(static_cast<Scalar>(first * sine)));
}

/// Compute inverse RMS in FP32 into *inverse_rms, visible to the whole block on return.
/// BF16 follows the pinned PyTorch mean reduction; FP32 keeps the serial order.
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

/// Write elements [begin,end) of RMSNorm(input) * scale; BF16 rounds before the scale.
/// The statistic always covers all `width` inputs.
template <typename Scalar>
__device__ inline void normalize(const Scalar* input, const Scalar* scale, Scalar* output,
    int begin, int end, int width, float epsilon, SharedStorage& shared)
{
    rms_norm_factor(input, width, epsilon, &shared.inverse_rms, shared.norm_partials);
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x)
        output[index] = static_cast<Scalar>(
            static_cast<float>(
                static_cast<Scalar>(static_cast<float>(input[index]) * shared.inverse_rms)) *
            static_cast<float>(scale[index]));
}

/// One row dot product reduced across a warp in FP32.
/// @pre All 32 lanes participate with the same arguments.
/// @return The dot product in lane 0 only.
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

/// Project rows [begin,end) and add the residual; the projection rounds before the addition.
/// One warp owns each row. Optional traces record the rounded projection and the output.
template <typename Scalar>
__device__ inline void project_rows(int begin, int end, const Scalar* matrix, const Scalar* input,
    int width, const Scalar* residual, Scalar* output, float* projection_trace, float* output_trace,
    SharedStorage& shared)
{
    for (int row = begin + threadIdx.x / kWarpSize; row < end; row += kWarpsPerBlock) {
        const float value =
            warp_dot(matrix + static_cast<cuda::std::size_t>(row) * width, input, width, shared);
        if (threadIdx.x % kWarpSize != 0)
            continue;
        const Scalar projected = static_cast<Scalar>(value);
        const Scalar sum =
            static_cast<Scalar>(static_cast<float>(projected) + static_cast<float>(residual[row]));
        output[row] = sum;
        if (projection_trace)
            projection_trace[row] = static_cast<float>(projected);
        if (output_trace)
            output_trace[row] = static_cast<float>(sum);
    }
}

/// One head of the concatenated [Q; K; V] projection: normalize the layer input,
/// project the head's rows, then write rotated Q, rotated K, or V into its cache row.
/// @param begin,end Rows of [Q; K; V] covering exactly one head.
template <typename Scalar>
__device__ inline void qkv(
    int begin, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int dh = shape.head_dim;
    const int kv_width = shape.kv_heads * dh;
    Scalar* input = shared_vector<Scalar>(shared);
    normalize(layer.x, layer.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
    __syncthreads();

    const Scalar* matrix = layer.Wq;
    int row = begin;
    if (begin >= d + kv_width) {
        matrix = layer.Wv;
        row = begin - d - kv_width;
    } else if (begin >= d) {
        matrix = layer.Wk;
        row = begin - d;
    }
    Scalar* head = reinterpret_cast<Scalar*>(shared.head);
    for (int i = threadIdx.x / kWarpSize; i < dh; i += kWarpsPerBlock) {
        const float value =
            warp_dot(matrix + static_cast<cuda::std::size_t>(row + i) * d, input, d, shared);
        if (threadIdx.x % kWarpSize == 0)
            head[i] = static_cast<Scalar>(value);
    }
    __syncthreads();

    const int cache_row = shape.position * kv_width + row;
    if (matrix == layer.Wv) {
        for (int i = threadIdx.x; i < dh; i += blockDim.x)
            layer.value_cache[cache_row + i] = head[i];
        return;
    }
    Scalar* rotated = matrix == layer.Wq ? layer.q_rot + row : layer.key_cache + cache_row;
    for (int pair = threadIdx.x; pair < dh / 2; pair += blockDim.x)
        rotate_pair(head, rotated, pair, shape);
}

/// The time steps of the visible prefix [0, position] that a context chunk covers.
__device__ inline void chunk_range(int chunk, const Shape& shape, int& begin, int& end)
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
__device__ inline void attention_scores(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    const int first_head = kv_head * group;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(dh));
    int begin = 0, end = 0;
    chunk_range(chunk, shape, begin, end);

    for (int time = begin + warp; time < end; time += kWarpsPerBlock) {
        const Scalar* key = layer.key_cache + (time * shape.kv_heads + kv_head) * dh;
        float dots[kMaxGroup] = {};
        for (int i = lane; i < dh; i += kWarpSize) {
            const float k = static_cast<float>(key[i]);
#pragma unroll
            for (int g = 0; g < kMaxGroup; ++g)
                if (g < group)
                    dots[g] += static_cast<float>(layer.q_rot[(first_head + g) * dh + i]) * k;
        }
        for (int g = 0; g < group; ++g) {
            const float dot = WarpSum(shared.warp[warp]).Sum(dots[g]);
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
__device__ inline void attention_values(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int dh = shape.head_dim;
    const int group = shape.query_heads / shape.kv_heads;
    int begin = 0, end = 0;
    chunk_range(chunk, shape, begin, end);
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

/// Output projection rows: sum each head's chunk partials into the attention
/// vector (rounded once), project rows [begin,end), and add the layer input.
/// The task owning row 0 also stores the attention vector for diagnostics.
template <typename Scalar>
__device__ inline void output_projection(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    Scalar* input = shared_vector<Scalar>(shared);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float total = 0.0f;
        for (int c = 0; c < shape.attention_chunks; ++c)
            total += layer.partials[c * d + i];
        input[i] = static_cast<Scalar>(total);
        if (begin == 0)
            layer.attention[i] = input[i];
    }
    __syncthreads();
    project_rows(begin, end, layer.Wo, input, d, layer.x, layer.r, nullptr, nullptr, shared);
}

/// Gate and up rows [begin,end) of the normalized attention output, combined as
/// SiLU(gate) * up. Gate, up and SiLU each round to Scalar, as in eager PyTorch.
template <typename Scalar>
__device__ inline void gate_up(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    Scalar* input = shared_vector<Scalar>(shared);
    normalize(layer.r, layer.ffn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
    __syncthreads();
    for (int row = begin + threadIdx.x / kWarpSize; row < end; row += kWarpsPerBlock) {
        const auto offset = static_cast<cuda::std::size_t>(row) * d;
        const float gate = static_cast<float>(
            static_cast<Scalar>(warp_dot(layer.Wgate + offset, input, d, shared)));
        const float up =
            static_cast<float>(static_cast<Scalar>(warp_dot(layer.Wup + offset, input, d, shared)));
        if (threadIdx.x % kWarpSize == 0)
            layer.product[row] = static_cast<Scalar>(
                static_cast<float>(static_cast<Scalar>(gate / (1.0f + cuda::std::exp(-gate)))) *
                up);
    }
}

/// Down projection rows [begin,end) plus the attention block output as residual.
template <typename Scalar>
__device__ inline void down(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    project_rows(begin, end, layer.Wdown, layer.product, shape.intermediate_size, layer.r, layer.y,
        layer.ffn_trace, layer.block_trace, shared);
}

/// Copy elements [begin,end) of a token's embedding row; the token is host-validated.
template <typename Scalar>
__device__ inline void embedding_lookup(
    TokenId token_id, int begin, int end, int hidden_size, const Scalar* embedding, Scalar* hidden)
{
    for (int index = begin + threadIdx.x; index < end; index += blockDim.x)
        hidden[index] = embedding[token_id * hidden_size + index];
}

/// Vocabulary logits [begin,end), one warp per row; BF16 logits round before widening.
template <typename Scalar>
__device__ inline void logits(const Scalar* hidden, const Scalar* lm_head, float* output, int begin,
    int end, int hidden_size, SharedStorage& shared)
{
    for (int row = begin + threadIdx.x / kWarpSize; row < end; row += kWarpsPerBlock) {
        const float value = warp_dot(lm_head + static_cast<cuda::std::size_t>(row) * hidden_size,
            hidden, hidden_size, shared);
        if (threadIdx.x % kWarpSize == 0)
            output[row] = static_cast<float>(static_cast<Scalar>(value));
    }
}

} // namespace tileinfer::ops
