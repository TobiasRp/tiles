#pragma once

#include "plan.hpp"

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numbers>

namespace tileinfer::ops {

template <typename Weight>
struct TensorViews
{
    float* input;
    float* output;
    float* scratch;
    float* kv;
    const Weight* weights;
};

inline constexpr int kWarpSize = 32;
inline constexpr int kWarps = kThreads / kWarpSize;
using WarpSum = cub::WarpReduce<float>;
using BlockSum = cub::BlockReduce<float, kThreads>;

struct SharedStorage
{
    WarpSum::TempStorage warp[kWarps];
    BlockSum::TempStorage block;
    float norm_factor;
    float maximum;
    float denominator;
    float old_scale;
    float new_scale;
};

struct BlockShape
{
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    int position;
    NumericConfig numeric;
    const float* rotary_frequencies;
};

template <typename Weight>
struct Context
{
    const TensorRecord* tensors;
    TensorViews<Weight> views;
    BlockShape shape;
    SharedStorage* shared;
};

template <typename Weight>
__device__ inline float* activation_tensor(const Context<Weight>& context, cuda::std::uint32_t id)
{
    const TensorRecord record = context.tensors[id];
    float* base = nullptr;
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

template <typename Weight>
__device__ inline const Weight* weight_tensor(
    const Context<Weight>& context, cuda::std::uint32_t id)
{
    return context.views.weights + context.tensors[id].offset;
}

__device__ inline void rotate_pair(const float* source, float* destination, int source_base,
    int destination_base, int pair, const BlockShape& shape)
{
    const float angle = shape.position * shape.rotary_frequencies[pair];
    const float sine = cuda::std::sin(angle);
    const float cosine = cuda::std::cos(angle);
    const int half = shape.head_dim / 2;
    const float first = source[source_base + pair];
    const float second = source[source_base + pair + half];
    destination[destination_base + pair] = first * cosine - second * sine;
    destination[destination_base + pair + half] = second * cosine + first * sine;
}

__device__ inline void normalization_factor(
    const float* input, int size, float epsilon, float* factor)
{
    if (threadIdx.x == 0) {
        float squares = 0.0f;
        for (int index = 0; index < size; ++index)
            squares += input[index] * input[index];
        *factor = 1.0f / cuda::std::sqrt(squares / size + epsilon);
    }
    __syncthreads();
}

template <typename Weight>
__device__ inline float row_dot(
    const Weight* row, const float* input, int width, SharedStorage& shared)
{
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    float value = 0.0f;
    for (int column = lane; column < width; column += kWarpSize)
        value += static_cast<float>(row[column]) * input[column];
    const float sum = WarpSum(shared.warp[warp]).Sum(value);
    __syncwarp();
    return sum;
}

template <typename Weight>
__device__ inline void rms_norm(const TaskRecord& task, const Context<Weight>& context)
{
    const float* x = activation_tensor(context, task.a);
    const Weight* scale = weight_tensor(context, task.b);
    float* y = activation_tensor(context, task.c);
    normalization_factor(x, context.shape.hidden_size, context.shape.numeric.rms_norm_eps,
        &context.shared->norm_factor);
    for (int index = threadIdx.x; index < context.shape.hidden_size; index += blockDim.x)
        y[index] = x[index] * context.shared->norm_factor * static_cast<float>(scale[index]);
}

template <typename Weight>
__device__ inline void gemv(const TaskRecord& task, const Context<Weight>& context)
{
    const float* x = activation_tensor(context, task.a);
    const Weight* weight = weight_tensor(context, task.b);
    float* y = activation_tensor(context, task.c);
    const float* addend = task.d == kNoTensor ? nullptr : activation_tensor(context, task.d);
    const int width = task.b == static_cast<cuda::std::uint32_t>(TensorId::Wdown)
                          ? context.shape.intermediate_size
                          : context.shape.hidden_size;
    const int warp = threadIdx.x / kWarpSize;
    for (cuda::std::uint32_t row = task.begin + warp; row < task.end; row += kWarps) {
        const float value = row_dot(
            weight + static_cast<cuda::std::size_t>(row) * width, x, width, *context.shared);
        if (threadIdx.x % kWarpSize == 0)
            y[row] = value + (addend ? addend[row] : 0.0f);
    }
}

template <typename Weight>
__device__ inline void rope_query(const TaskRecord& task, const Context<Weight>& context)
{
    const float* x = activation_tensor(context, task.a);
    float* y = activation_tensor(context, task.b);
    for (int pair = threadIdx.x; pair < context.shape.head_dim / 2; pair += blockDim.x) {
        const int base = task.head * context.shape.head_dim;
        rotate_pair(x, y, base, base, pair, context.shape);
    }
}

template <typename Weight>
__device__ inline void rope_cache(const TaskRecord& task, const Context<Weight>& context)
{
    const float* key = activation_tensor(context, task.a);
    const float* value = activation_tensor(context, task.b);
    float* key_cache = activation_tensor(context, task.c);
    float* value_cache = activation_tensor(context, task.d);
    const int source = task.kv_head * context.shape.head_dim;
    const int destination =
        context.shape.position * context.shape.kv_heads * context.shape.head_dim + source;
    for (int pair = threadIdx.x; pair < context.shape.head_dim / 2; pair += blockDim.x)
        rotate_pair(key, key_cache, source, destination, pair, context.shape);
    for (int index = threadIdx.x; index < context.shape.head_dim; index += blockDim.x)
        value_cache[destination + index] = value[source + index];
}

template <typename Weight>
__device__ inline void attention_prefix(const TaskRecord& task, const Context<Weight>& context)
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
        shared.maximum = -cuda::std::numeric_limits<float>::infinity();
        shared.denominator = 0.0f;
    }
    __syncthreads();
    const float scale = 1.0f / cuda::std::sqrt(static_cast<float>(shape.head_dim));
    // Online softmax uses constant shared storage for any context/head length.
    // Each score is reduced once; output coordinates share its rescaling factors.
    for (int time = 0; time <= shape.position; ++time) {
        const int base = (time * shape.kv_heads + task.kv_head) * shape.head_dim;
        float dot = 0.0f;
        for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
            dot += query[coordinate] * keys[base + coordinate];
        const float score = BlockSum(shared.block).Sum(dot) * scale;
        if (threadIdx.x == 0) {
            const float maximum = cuda::std::fmax(shared.maximum, score);
            shared.old_scale = cuda::std::exp(shared.maximum - maximum);
            shared.new_scale = cuda::std::exp(score - maximum);
            shared.denominator = shared.denominator * shared.old_scale + shared.new_scale;
            shared.maximum = maximum;
        }
        __syncthreads();
        for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
            output[coordinate] = output[coordinate] * shared.old_scale +
                                 values[base + coordinate] * shared.new_scale;
        __syncthreads();
    }
    for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x)
        output[coordinate] /= shared.denominator;
}

template <typename Weight>
__device__ inline void silu_product(const TaskRecord& task, const Context<Weight>& context)
{
    const float* gate = activation_tensor(context, task.a);
    const float* up = activation_tensor(context, task.b);
    float* y = activation_tensor(context, task.c);
    for (cuda::std::uint32_t index = task.begin + threadIdx.x; index < task.end;
        index += blockDim.x)
        y[index] = (gate[index] / (1.0f + cuda::std::exp(-gate[index]))) * up[index];
}

template <typename Weight>
__device__ inline void execute(const TaskRecord& task, const Context<Weight>& context)
{
    switch (task.opcode) {
    case Opcode::rms_norm_f32:
        rms_norm(task, context);
        break;
    case Opcode::gemv_f32:
        gemv(task, context);
        break;
    case Opcode::rope_query_f32:
        rope_query(task, context);
        break;
    case Opcode::rope_cache_f32:
        rope_cache(task, context);
        break;
    case Opcode::attention_prefix_f32:
        attention_prefix(task, context);
        break;
    case Opcode::silu_product_f32:
        silu_product(task, context);
        break;
    }
}

template <typename Weight>
__device__ inline void embedding_lookup(
    TokenId token_id, int hidden_size, const Weight* embedding, float* hidden)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        hidden[index] = static_cast<float>(embedding[token_id * hidden_size + index]);
}

template <typename Weight>
__device__ inline void final_norm(const float* hidden, const Weight* scale, float* normalized,
    int hidden_size, float epsilon, float* shared_norm_factor)
{
    normalization_factor(hidden, hidden_size, epsilon, shared_norm_factor);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        normalized[index] = hidden[index] * *shared_norm_factor * static_cast<float>(scale[index]);
}

template <typename Weight>
__device__ inline void logits(const float* hidden, const Weight* lm_head, float* output,
    int hidden_size, int vocab_size, SharedStorage& shared)
{
    const int warp = threadIdx.x / kWarpSize;
    for (int row = blockIdx.x * kWarps + warp; row < vocab_size; row += gridDim.x * kWarps) {
        const float value = row_dot(lm_head + static_cast<cuda::std::size_t>(row) * hidden_size,
            hidden, hidden_size, shared);
        if (threadIdx.x % kWarpSize == 0)
            output[row] = value;
    }
}

} // namespace tileinfer::ops
