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

#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace tileinfer::ops {

inline constexpr int kWarpSize = 32;
inline constexpr int kWarpsPerBlock = kThreads / kWarpSize;
inline constexpr int kNormReductionThreads = 512;
using WarpSum = cub::WarpReduce<float>;

/// Block-private scratch. Each warp owns one warp slot.
struct SharedStorage
{
    WarpSum::TempStorage warp[kWarpsPerBlock];
    float inverse_rms;
    float norm_partials[kNormReductionThreads];
    // A task's input vector, or an attention task's scores, probabilities and
    // P*V reduction; typed by the caller.
    alignas(16) unsigned char vector[kAttentionSharedBytes];
    // An attention task's rotated query heads, then its rotated current K row.
    alignas(16) unsigned char rotated[(kMaxGroup + 1) * kMaxHeadDim * sizeof(float)];
};
static_assert(kAttentionSharedBytes >= kMaxHidden * sizeof(float));
static_assert(kAttentionSharedBytes >= kWarpsPerBlock * kMaxHeadDim * sizeof(float));

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
    int attention_chunks;      // chunk tasks per K/V head in the program
    int attention_chunk_steps; // most time steps one chunk task covers
    int attention_fused_steps; // longest prefix that one fused task per K/V head handles
    int ffn_chunks;            // FFN product chunks, each with one FP32 down partial
    NumericConfig numeric;
    const float* rotary_frequencies;
};

/// One layer's buffers, already offset to that layer.
template <typename Scalar>
struct Layer
{
    const Scalar* x; // layer input
    Scalar* y;       // layer output
    Scalar* q;       // projected Q before RoPE
    Scalar* k;       // projected K of the current position before RoPE
    Scalar* q_rot;   // rotated Q, kept for diagnostics
    Scalar* attention;
    Scalar* r; // attention block output, the FFN's input and residual
    Scalar* product;
    Scalar* key_cache;   // [max_context, kv_heads, head_dim]
    Scalar* value_cache; // [max_context, kv_heads, head_dim]
    Scalar* scores;      // [query_heads, max_context]: scores, then probabilities
    float* partials;     // [chunks, query_heads, head_dim]: partial P*V
    float* stats;        // [chunks, query_heads, 2]: chunk maximum and sum of exp
    float* ffn_partials; // [ffn_chunks, hidden_size]: FP32 down partial of each product chunk
    const Scalar* attn_norm;
    const Scalar* Wq;
    const Scalar* Wk;
    const Scalar* Wv;
    const Scalar* Wo;
    const Scalar* ffn_norm;
    const Scalar* Wgate;
    const Scalar* Wup;
    const Scalar* Wdown; // stored transposed, [intermediate_size, hidden_size]
    float* ffn_trace;    // optional: Wdown projection before the residual
    float* block_trace;  // optional: this layer's output
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

/// Rows [begin,end) of the concatenated [Q; K; V] projection of the normalized
/// layer input, one warp per row. Q and K round to Scalar and stay unrotated in
/// their scratch vectors; the attention task of their K/V head applies RoPE. V
/// rows go to the cache row of the current position.
template <typename Scalar>
__device__ inline void qkv(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int kv_width = shape.kv_heads * shape.head_dim;
    Scalar* input = shared_vector<Scalar>(shared);
    normalize(layer.x, layer.attn_norm, input, 0, d, d, shape.numeric.rms_norm_eps, shared);
    __syncthreads();
    for (int row = begin + threadIdx.x / kWarpSize; row < end; row += kWarpsPerBlock) {
        const Scalar* weights =
            row < d ? layer.Wq + static_cast<cuda::std::size_t>(row) * d
            : row < d + kv_width
                ? layer.Wk + static_cast<cuda::std::size_t>(row - d) * d
                : layer.Wv + static_cast<cuda::std::size_t>(row - d - kv_width) * d;
        const float value = warp_dot(weights, input, d, shared);
        if (threadIdx.x % kWarpSize != 0)
            continue;
        Scalar* output = row < d ? layer.q + row
                         : row < d + kv_width
                             ? layer.k + row - d
                             : layer.value_cache + shape.position * kv_width + row - d - kv_width;
        *output = static_cast<Scalar>(value);
    }
}

/// Chunk tasks with work at this position. A prefix of at most
/// attention_fused_steps time steps is one chunk, and its scores task finishes
/// attention by itself. A longer prefix splits into the fewest chunks of at
/// most attention_chunk_steps steps, but at least two: the S2b probe found two
/// half-length tasks faster than one beyond about 384 steps.
__device__ inline int active_chunks(const Shape& shape)
{
    const int length = shape.position + 1;
    if (length <= shape.attention_fused_steps)
        return 1;
    const int chunks = (length + shape.attention_chunk_steps - 1) / shape.attention_chunk_steps;
    return chunks > 2 ? chunks : 2;
}

/// The time steps of the visible prefix that active chunk `chunk` covers; the
/// active chunks split the prefix evenly.
__device__ inline void chunk_range(int chunk, const Shape& shape, int& begin, int& end)
{
    const int length = shape.position + 1;
    const int chunks = active_chunks(shape);
    begin = static_cast<int>(static_cast<long long>(length) * chunk / chunks);
    end = static_cast<int>(static_cast<long long>(length) * (chunk + 1) / chunks);
}

/// Scalar elements in one 16-byte q or K/V load.
template <typename Scalar>
inline constexpr int kVectorElements = 16 / static_cast<int>(sizeof(Scalar));
/// Independent K/V loads in flight per lane; the S2b probe found two fastest.
inline constexpr int kRowsInFlight = 2;

/// Element j of a 16-byte vector of Scalar values, widened to FP32.
template <typename Scalar>
__device__ inline float vector_element(const uint4& raw, int j)
{
    const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
    if constexpr (cuda::std::is_same_v<Scalar, __nv_bfloat16>)
        return __uint_as_float((j & 1) ? words[j >> 1] & 0xffff0000u : words[j >> 1] << 16);
    else
        return __uint_as_float(words[j]);
}

__device__ inline uint4 load_vector(const void* address, bool valid)
{
    return valid ? *static_cast<const uint4*>(address) : uint4{0, 0, 0, 0};
}

/// Element offset of time step `time` of one K/V head in a layer's cache.
__device__ inline cuda::std::size_t kv_row(int time, int kv_head, const Shape& shape)
{
    return (static_cast<cuda::std::size_t>(time) * shape.kv_heads + kv_head) * shape.head_dim;
}

/// How a warp reads K/V rows: `count` lanes load one head_dim row as 16-byte
/// vectors, so one warp instruction covers `rows` rows. Lanes > 0 fixes the
/// count at compile time, which saves about 1 us per layer of run-time
/// geometry; Lanes == 0 reads it from head_dim (see attention_scores).
template <typename Scalar, int Lanes>
struct RowLanes
{
    int count;
    int rows;
    int sub;    // this lane's row in the warp instruction
    int vector; // this lane's vector within the row

    __device__ explicit RowLanes(const Shape& shape)
        : count(Lanes > 0 ? Lanes : shape.head_dim / kVectorElements<Scalar>)
        , rows(kWarpSize / count)
        , sub(threadIdx.x % kWarpSize / count)
        , vector(threadIdx.x % kWarpSize % count)
    {
    }
};

/// The head width compiled with a fixed lane count: Llama 3.2 1B's.
inline constexpr int kFixedLaneHeadDim = 64;

/// Query heads per K/V head: Group, or the shape's when Group == 0.
template <int Group>
__device__ inline int group_size(const Shape& shape)
{
    return Group > 0 ? Group : shape.query_heads / shape.kv_heads;
}

/// One K/V row's dot product with a query head, reduced over the row's lanes.
template <typename Scalar>
__device__ inline float row_dot(const uint4& query, const uint4& row, int lanes_per_row)
{
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorElements<Scalar>; ++j)
        dot += vector_element<Scalar>(query, j) * vector_element<Scalar>(row, j);
#pragma unroll
    for (int offset = lanes_per_row / 2; offset > 0; offset /= 2)
        dot += __shfl_xor_sync(0xffffffffu, dot, offset);
    return dot;
}

/// Scores of a K/V head's query heads over time steps [begin,end):
/// scores[g * stride + time - begin]. `rotated` holds the group's rotated
/// query heads; `current`, when set, is the rotated K row of time end - 1,
/// which the cache may not hold yet. QK and the scaling are separate BF16
/// rounding boundaries, as in eager attention. With a fixed Group, lanes keep
/// q in registers; Group == 0 reloads it from shared memory for each row.
template <typename Scalar, int Group, int Lanes>
__device__ inline void score_rows(int kv_head, int begin, int end, const Scalar* rotated,
    const Scalar* current, const Layer<Scalar>& layer, const Shape& shape, Scalar* scores,
    int stride)
{
    constexpr int kVector = kVectorElements<Scalar>;
    const RowLanes<Scalar, Lanes> lanes(shape);
    const int warp = threadIdx.x / kWarpSize;
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(shape.head_dim));
    const int group = group_size<Group>(shape);
    const Scalar* query = rotated + lanes.vector * kVector;
    const int cached = current == nullptr ? end : end - 1; // rows read from the cache
    uint4 q[Group > 0 ? Group : 1];
    if constexpr (Group > 0)
#pragma unroll
        for (int g = 0; g < Group; ++g)
            q[g] = load_vector(query + g * shape.head_dim, true);
    // Every lane joins the row reduction; lanes of a valid row's first vector store.
    auto store = [&](int g, int time, bool valid, float dot) {
        if (lanes.vector == 0 && valid)
            scores[g * stride + time - begin] =
                static_cast<Scalar>(static_cast<float>(static_cast<Scalar>(dot)) * scale);
    };
    auto score = [&](int time, bool valid, const uint4& row) {
        if constexpr (Group > 0) {
#pragma unroll
            for (int g = 0; g < Group; ++g)
                store(g, time, valid, row_dot<Scalar>(q[g], row, lanes.count));
        } else {
            for (int g = 0; g < group; ++g)
                store(g, time, valid,
                    row_dot<Scalar>(
                        load_vector(query + g * shape.head_dim, true), row, lanes.count));
        }
    };

    const int rows = kWarpsPerBlock * lanes.rows * kRowsInFlight;
    for (int first = begin; first < cached; first += rows) {
        uint4 raw[kRowsInFlight];
#pragma unroll
        for (int u = 0; u < kRowsInFlight; ++u) {
            const int time = first + (u * kWarpsPerBlock + warp) * lanes.rows + lanes.sub;
            raw[u] =
                load_vector(layer.key_cache + kv_row(time < cached ? time : begin, kv_head, shape) +
                                lanes.vector * kVector,
                    time < cached);
        }
#pragma unroll
        for (int u = 0; u < kRowsInFlight; ++u) {
            const int time = first + (u * kWarpsPerBlock + warp) * lanes.rows + lanes.sub;
            score(time, time < cached, raw[u]);
        }
    }
    // The current row, by the first row slot of the last warp: the same lanes
    // and reduction as a cached row, so its scores match bit for bit.
    if (current != nullptr && warp == kWarpsPerBlock - 1)
        score(
            cached, lanes.sub == 0, load_vector(current + lanes.vector * kVector, lanes.sub == 0));
}

/// Softmax statistics of `length` scores by one warp: the maximum and the sum
/// of exp(score - maximum), returned in every lane.
template <typename Scalar>
__device__ inline float2 warp_softmax_statistics(const Scalar* scores, int length)
{
    const int lane = threadIdx.x % kWarpSize;
    float maximum = -cuda::std::numeric_limits<float>::infinity();
    for (int time = lane; time < length; time += kWarpSize)
        maximum = cuda::std::fmax(maximum, static_cast<float>(scores[time]));
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        maximum = cuda::std::fmax(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    float sum = 0.0f;
    for (int time = lane; time < length; time += kWarpSize)
        sum += cuda::std::exp(static_cast<float>(scores[time]) - maximum);
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    return {maximum, sum};
}

/// FP32 partial P*V of a K/V head's query heads over time steps
/// [begin, begin + length), stored to partials[g * head_dim + i]. The
/// probabilities [group][length] live in shared.vector, which the cross-warp
/// reduction reuses after a block synchronization. Each head gets
/// kWarpsPerBlock / group warps; a lane accumulates its vector's elements over
/// its rows, then rows and warps are summed in a fixed order.
template <typename Scalar, int Group, int Lanes>
__device__ inline void value_rows(int kv_head, int begin, int length, const Scalar* probabilities,
    const Layer<Scalar>& layer, const Shape& shape, float* partials, SharedStorage& shared)
{
    constexpr int kVector = kVectorElements<Scalar>;
    const int group = group_size<Group>(shape);
    const int warps_per_head = kWarpsPerBlock / group;
    const RowLanes<Scalar, Lanes> lanes(shape);
    const int dh = lanes.count * kVector;
    const int warp = threadIdx.x / kWarpSize;
    const int member = warp % warps_per_head;
    const Scalar* own = probabilities + warp / warps_per_head * length;
    float sums[kVector] = {};
    const int rows = warps_per_head * lanes.rows * kRowsInFlight;
    for (int first = 0; first < length; first += rows) {
        uint4 raw[kRowsInFlight];
#pragma unroll
        for (int u = 0; u < kRowsInFlight; ++u) {
            const int time = first + (u * warps_per_head + member) * lanes.rows + lanes.sub;
            raw[u] = load_vector(layer.value_cache +
                                     kv_row(begin + (time < length ? time : 0), kv_head, shape) +
                                     lanes.vector * kVector,
                time < length);
        }
#pragma unroll
        for (int u = 0; u < kRowsInFlight; ++u) {
            const int time = first + (u * warps_per_head + member) * lanes.rows + lanes.sub;
            const float probability = time < length ? static_cast<float>(own[time]) : 0.0f;
#pragma unroll
            for (int j = 0; j < kVector; ++j)
                sums[j] += probability * vector_element<Scalar>(raw[u], j);
        }
    }
    // Lanes holding the same vector of different rows.
#pragma unroll
    for (int offset = lanes.count; offset < kWarpSize; offset *= 2)
#pragma unroll
        for (int j = 0; j < kVector; ++j)
            sums[j] += __shfl_xor_sync(0xffffffffu, sums[j], offset);
    __syncthreads(); // every warp is done with the probabilities
    float* reduction = reinterpret_cast<float*>(shared.vector); // [warp][head_dim]
    if (lanes.sub == 0)
#pragma unroll
        for (int j = 0; j < kVector; ++j)
            reduction[warp * dh + lanes.vector * kVector + j] = sums[j];
    __syncthreads();
    for (int index = threadIdx.x; index < group * dh; index += blockDim.x) {
        const int head = index / dh;
        const int i = index - head * dh;
        float total = 0.0f;
        for (int m = 0; m < warps_per_head; ++m)
            total += reduction[(head * warps_per_head + m) * dh + i];
        partials[index] = total;
    }
}

/// Rotate a K/V head's query heads and its current K row into
/// shared.rotated: [group][head_dim] query heads, then the K row. Chunk 0
/// also stores them, rotated Q for diagnostics and K into the cache row of
/// the current position, which no other task reads during this token.
template <typename Scalar>
__device__ inline Scalar* rotate_group(int kv_head, int chunk, int group,
    const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int dh = shape.head_dim;
    const int first_head = kv_head * group;
    Scalar* rotated = reinterpret_cast<Scalar*>(shared.rotated);
    for (int index = threadIdx.x; index < (group + 1) * dh / 2; index += blockDim.x) {
        const int head = index / (dh / 2);
        const Scalar* source =
            head < group ? layer.q + (first_head + head) * dh : layer.k + kv_head * dh;
        rotate_pair(source, rotated + head * dh, index - head * (dh / 2), shape);
    }
    __syncthreads();
    if (chunk == 0) {
        Scalar* key = layer.key_cache + kv_row(shape.position, kv_head, shape);
        for (int index = threadIdx.x; index < (group + 1) * dh; index += blockDim.x)
            if (index < group * dh)
                layer.q_rot[first_head * dh + index] = rotated[index];
            else
                key[index - group * dh] = rotated[index];
    }
    return rotated;
}

/// Scores task of one K/V head and context chunk. With several active chunks it
/// stores the chunk's scores and each head's chunk statistics for the values
/// task. With one active chunk it keeps scores and probabilities in shared
/// memory and also computes the softmax and P*V, so the values wave has no work.
/// Every active task rotates its query heads itself; the task whose chunk ends
/// at the current position takes that K row from shared memory, not the cache.
template <typename Scalar, int Group, int Lanes>
__device__ inline void attention_scores_group(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int chunks = active_chunks(shape);
    if (chunk >= chunks)
        return;
    int begin = 0, end = 0;
    chunk_range(chunk, shape, begin, end);
    const int group = group_size<Group>(shape);
    const int first_head = kv_head * group;
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const Scalar* rotated = rotate_group(kv_head, chunk, group, layer, shape, shared);
    const Scalar* current = end == shape.position + 1 ? rotated + group * shape.head_dim : nullptr;
    if (chunks > 1) {
        Scalar* scores = layer.scores + first_head * shape.max_context + begin;
        score_rows<Scalar, Group, Lanes>(
            kv_head, begin, end, rotated, current, layer, shape, scores, shape.max_context);
        __syncthreads();
        if (warp < group) {
            const float2 statistics =
                warp_softmax_statistics(scores + warp * shape.max_context, end - begin);
            if (lane == 0) {
                float* stats = layer.stats + (chunk * shape.query_heads + first_head + warp) * 2;
                stats[0] = statistics.x;
                stats[1] = statistics.y;
            }
        }
        return;
    }

    const int length = end;
    Scalar* scores = shared_vector<Scalar>(shared); // [group][length], then probabilities
    score_rows<Scalar, Group, Lanes>(
        kv_head, 0, length, rotated, current, layer, shape, scores, length);
    __syncthreads();
    float* statistics = shared.norm_partials; // [0, group): maximum, [group, 2 group): sum
    if (warp < group) {
        const float2 own = warp_softmax_statistics(scores + warp * length, length);
        if (lane == 0) {
            statistics[warp] = own.x;
            statistics[group + warp] = own.y;
        }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < group * length; index += blockDim.x) {
        const int g = index / length;
        scores[index] =
            static_cast<Scalar>(cuda::std::exp(static_cast<float>(scores[index]) - statistics[g]) /
                                statistics[group + g]);
    }
    __syncthreads();
    value_rows<Scalar, Group, Lanes>(kv_head, 0, length, scores, layer, shape,
        layer.partials + first_head * shape.head_dim, shared);
}

/// Values task of one K/V head and chunk when several chunks are active:
/// probabilities and partial P*V. The softmax maximum and denominator combine
/// every active chunk's statistics in chunk order. Probabilities round once to
/// Scalar before P*V, as eager attention does; output_projection combines the
/// FP32 partials.
template <typename Scalar, int Group, int Lanes>
__device__ inline void attention_values_group(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int chunks = active_chunks(shape);
    if (chunks == 1 || chunk >= chunks)
        return;
    int begin = 0, end = 0;
    chunk_range(chunk, shape, begin, end);
    const int length = end - begin;
    const int group = group_size<Group>(shape);
    const int first_head = kv_head * group;
    float* combined = shared.norm_partials; // [0, group): maximum, [group, 2 group): sum
    if (threadIdx.x < group) {
        const int head = first_head + threadIdx.x;
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int c = 0; c < chunks; ++c)
            maximum = cuda::std::fmax(maximum, layer.stats[(c * shape.query_heads + head) * 2]);
        float sum = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            const float* stats = layer.stats + (c * shape.query_heads + head) * 2;
            sum += stats[1] * cuda::std::exp(stats[0] - maximum);
        }
        combined[threadIdx.x] = maximum;
        combined[group + threadIdx.x] = sum;
    }
    __syncthreads();
    Scalar* probabilities = shared_vector<Scalar>(shared); // [group][length]
    for (int index = threadIdx.x; index < group * length; index += blockDim.x) {
        const int g = index / length;
        const Scalar score =
            layer.scores[(first_head + g) * shape.max_context + begin + index - g * length];
        probabilities[index] = static_cast<Scalar>(
            cuda::std::exp(static_cast<float>(score) - combined[g]) / combined[group + g]);
    }
    __syncthreads();
    value_rows<Scalar, Group, Lanes>(kv_head, begin, length, probabilities, layer, shape,
        layer.partials + (chunk * shape.query_heads + first_head) * shape.head_dim, shared);
}

/// The attention shape compiled with a fixed group and lane count: Llama 3.2
/// 1B's four query heads per K/V head of width kFixedLaneHeadDim. Every other
/// shape runs one instantiation that reads both at run time. One
/// instantiation per group and lane count made the worker ten times larger
/// and its GEMV tasks slower; one per group, four times and about 0.1 ms per
/// token.
inline constexpr int kFixedGroup = 4;

__device__ inline bool fixed_attention_shape(const Shape& shape)
{
    return shape.query_heads / shape.kv_heads == kFixedGroup && shape.head_dim == kFixedLaneHeadDim;
}

/// Scores task: see attention_scores_group. The loader admits power-of-two
/// groups up to kMaxGroup and head_dim a power-of-two multiple of one 16-byte
/// vector, at most a warp of lanes per row.
template <typename Scalar>
__device__ inline void attention_scores(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    constexpr int kLanes = kFixedLaneHeadDim / kVectorElements<Scalar>;
    if (fixed_attention_shape(shape))
        return attention_scores_group<Scalar, kFixedGroup, kLanes>(
            kv_head, chunk, layer, shape, shared);
    attention_scores_group<Scalar, 0, 0>(kv_head, chunk, layer, shape, shared);
}

/// Values task: see attention_values_group.
template <typename Scalar>
__device__ inline void attention_values(
    int kv_head, int chunk, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    constexpr int kLanes = kFixedLaneHeadDim / kVectorElements<Scalar>;
    if (fixed_attention_shape(shape))
        return attention_values_group<Scalar, kFixedGroup, kLanes>(
            kv_head, chunk, layer, shape, shared);
    attention_values_group<Scalar, 0, 0>(kv_head, chunk, layer, shape, shared);
}

/// Output projection rows: sum each head's active-chunk partials into the
/// attention vector (rounded once), project rows [begin,end), and add the layer
/// input. The task owning row 0 also stores the attention vector for diagnostics.
template <typename Scalar>
__device__ inline void output_projection(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int chunks = active_chunks(shape);
    Scalar* input = shared_vector<Scalar>(shared);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float total = 0.0f;
        for (int c = 0; c < chunks; ++c)
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

/// Two adjacent elements of a row, widened to FP32.
/// @pre The address is aligned to two elements.
template <typename Scalar>
__device__ inline float2 load_pair(const Scalar* address)
{
    if constexpr (cuda::std::is_same_v<Scalar, __nv_bfloat16>)
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(address));
    else
        return *reinterpret_cast<const float2*>(address);
}

/// The FP32 down partial of product rows [begin,end) over all outputs, into
/// ffn_partials[chunk]. Wdown is stored transposed, so the chunk's rows are
/// contiguous: a thread owns two adjacent outputs and sums the chunk's rows in
/// order, with no warp reductions.
/// @pre product[begin,end) was written by this block, end - begin fits the
/// shared vector, and hidden_size is even (the loader requires an even head_dim).
template <typename Scalar>
__device__ inline void down_partial(int begin, int end, int chunk, const Layer<Scalar>& layer,
    const Shape& shape, SharedStorage& shared)
{
    const int d = shape.hidden_size;
    const int rows = end - begin;
    Scalar* product = shared_vector<Scalar>(shared);
    for (int j = threadIdx.x; j < rows; j += blockDim.x)
        product[j] = layer.product[begin + j];
    __syncthreads();
    const Scalar* weights = layer.Wdown + static_cast<cuda::std::size_t>(begin) * d;
    float* output = layer.ffn_partials + static_cast<cuda::std::size_t>(chunk) * d;
    for (int pair = threadIdx.x; pair < d / 2; pair += blockDim.x) {
        float low = 0.0f;
        float high = 0.0f;
#pragma unroll 8
        for (int j = 0; j < rows; ++j) {
            const float2 w = load_pair(weights + static_cast<cuda::std::size_t>(j) * d + 2 * pair);
            const float p = static_cast<float>(product[j]);
            low += w.x * p;
            high += w.y * p;
        }
        output[2 * pair] = low;
        output[2 * pair + 1] = high;
    }
}

/// One FFN chunk: the gate/up rows [begin,end) of the product, then their
/// down partial. A worker reads only its own product, so no barrier separates
/// the two; ffn_reduce combines the chunks after the next one.
template <typename Scalar>
__device__ inline void ffn(int begin, int end, int chunk, const Layer<Scalar>& layer,
    const Shape& shape, SharedStorage& shared)
{
    gate_up(begin, end, layer, shape, shared);
    __syncthreads();
    down_partial(begin, end, chunk, layer, shape, shared);
}

/// Down projection rows [begin,end) plus the attention block output as
/// residual: the chunks' FP32 partials summed in chunk order, rounded before
/// the residual as in project_rows. Optional traces record the rounded
/// projection and the output.
template <typename Scalar>
__device__ inline void ffn_reduce(
    int begin, int end, const Layer<Scalar>& layer, const Shape& shape)
{
    const int d = shape.hidden_size;
    for (int row = begin + threadIdx.x; row < end; row += blockDim.x) {
        float total = 0.0f;
#pragma unroll 4
        for (int chunk = 0; chunk < shape.ffn_chunks; ++chunk)
            total += layer.ffn_partials[static_cast<cuda::std::size_t>(chunk) * d + row];
        const Scalar projected = static_cast<Scalar>(total);
        const Scalar sum =
            static_cast<Scalar>(static_cast<float>(projected) + static_cast<float>(layer.r[row]));
        layer.y[row] = sum;
        if (layer.ffn_trace)
            layer.ffn_trace[row] = static_cast<float>(projected);
        if (layer.block_trace)
            layer.block_trace[row] = static_cast<float>(sum);
    }
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
