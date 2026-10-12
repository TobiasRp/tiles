#pragma once

#include "runtime.hpp"

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

struct BlockShape
{
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    int position;
    NumericConfig numeric;
};

template <typename Weight>
struct Context
{
    const TensorRecord* tensors;
    TensorViews<Weight> views;
    BlockShape shape;
    float* shared_norm_factor;
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

__device__ inline float rotary_frequency(int pair, const BlockShape& shape)
{
    const NumericConfig& cfg = shape.numeric;
    const float inverse = 1.0f / cuda::std::pow(cfg.rope_theta, 2.0f * pair / shape.head_dim);
    const float wavelength = 2.0f * cuda::std::numbers::pi_v<float> / inverse;
    const float low = cfg.rope_original_context / cfg.rope_low_freq_factor;
    const float high = cfg.rope_original_context / cfg.rope_high_freq_factor;
    if (wavelength > low)
        return inverse / cfg.rope_factor;
    if (wavelength < high)
        return inverse;
    const float smooth = (cfg.rope_original_context / wavelength - cfg.rope_low_freq_factor) /
                         (cfg.rope_high_freq_factor - cfg.rope_low_freq_factor);
    return inverse * ((1.0f - smooth) / cfg.rope_factor + smooth);
}

__device__ inline void rotate_pair(const float* source, float* destination, int source_base,
    int destination_base, int pair, const BlockShape& shape)
{
    const float angle = shape.position * rotary_frequency(pair, shape);
    const float sine = cuda::std::sin(angle);
    const float cosine = cuda::std::cos(angle);
    const int half = shape.head_dim / 2;
    const float first = source[source_base + pair];
    const float second = source[source_base + pair + half];
    destination[destination_base + pair] = first * cosine - second * sine;
    destination[destination_base + pair + half] = second * cosine + first * sine;
}

template <typename Weight>
__device__ inline void rms_norm(const TaskRecord& task, const Context<Weight>& context)
{
    const float* x = activation_tensor(context, task.a);
    const Weight* scale = weight_tensor(context, task.b);
    float* y = activation_tensor(context, task.c);
    if (threadIdx.x == 0) {
        float squares = 0.0f;
        for (int index = 0; index < context.shape.hidden_size; ++index)
            squares += x[index] * x[index];
        *context.shared_norm_factor = 1.0f / cuda::std::sqrt(squares / context.shape.hidden_size +
                                                             context.shape.numeric.rms_norm_eps);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < context.shape.hidden_size; index += blockDim.x)
        y[index] = x[index] * *context.shared_norm_factor * static_cast<float>(scale[index]);
}

template <typename Weight>
__device__ inline void gemv(const TaskRecord& task, const Context<Weight>& context)
{
    const float* x = activation_tensor(context, task.a);
    const Weight* weight = weight_tensor(context, task.b);
    float* y = activation_tensor(context, task.c);
    const float* addend = task.d == cuda::std::numeric_limits<cuda::std::uint32_t>::max()
                              ? nullptr
                              : activation_tensor(context, task.d);
    const int width = task.b == static_cast<cuda::std::uint32_t>(TensorId::Wdown)
                          ? context.shape.intermediate_size
                          : context.shape.hidden_size;
    for (cuda::std::uint32_t row = task.begin + threadIdx.x; row < task.end; row += blockDim.x) {
        float value = 0.0f;
        for (int column = 0; column < width; ++column)
            value += static_cast<float>(weight[row * width + column]) * x[column];
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

__device__ inline float attention_score(
    const float* query, const float* key, int head, int kv_head, int time, const BlockShape& shape)
{
    float dot = 0.0f;
    const int query_base = head * shape.head_dim;
    const int key_base = (time * shape.kv_heads + kv_head) * shape.head_dim;
    for (int index = 0; index < shape.head_dim; ++index)
        dot += query[query_base + index] * key[key_base + index];
    return dot / cuda::std::sqrt(static_cast<float>(shape.head_dim));
}

template <typename Weight>
__device__ inline void attention_prefix(const TaskRecord& task, const Context<Weight>& context)
{
    const float* query = activation_tensor(context, task.a);
    const float* key_cache = activation_tensor(context, task.b);
    const float* value_cache = activation_tensor(context, task.c);
    float* y = activation_tensor(context, task.d);
    const BlockShape& shape = context.shape;
    for (int coordinate = threadIdx.x; coordinate < shape.head_dim; coordinate += blockDim.x) {
        float maximum = -cuda::std::numeric_limits<float>::infinity();
        for (int time = 0; time <= shape.position; ++time)
            maximum = cuda::std::fmax(
                maximum, attention_score(query, key_cache, task.head, task.kv_head, time, shape));
        float denominator = 0.0f;
        float numerator = 0.0f;
        for (int time = 0; time <= shape.position; ++time) {
            const float probability = cuda::std::exp(
                attention_score(query, key_cache, task.head, task.kv_head, time, shape) - maximum);
            denominator += probability;
            numerator +=
                probability *
                value_cache[(time * shape.kv_heads + task.kv_head) * shape.head_dim + coordinate];
        }
        y[task.head * shape.head_dim + coordinate] = numerator / denominator;
    }
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
    int token_id, int hidden_size, const Weight* embedding, float* hidden)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        hidden[index] = static_cast<float>(embedding[token_id * hidden_size + index]);
}

template <typename Weight>
__device__ inline void final_norm(const float* hidden, const Weight* scale, float* normalized,
    int hidden_size, float epsilon, float* shared_norm_factor)
{
    if (threadIdx.x == 0) {
        float squares = 0.0f;
        for (int index = 0; index < hidden_size; ++index)
            squares += hidden[index] * hidden[index];
        *shared_norm_factor = 1.0f / cuda::std::sqrt(squares / hidden_size + epsilon);
    }
    __syncthreads();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
        index += gridDim.x * blockDim.x)
        normalized[index] = hidden[index] * *shared_norm_factor * static_cast<float>(scale[index]);
}

template <typename Weight>
__device__ inline void logits(
    const float* hidden, const Weight* lm_head, float* output, int hidden_size, int vocab_size)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < vocab_size;
        row += gridDim.x * blockDim.x) {
        float value = 0.0f;
        for (int column = 0; column < hidden_size; ++column)
            value += static_cast<float>(lm_head[row * hidden_size + column]) * hidden[column];
        output[row] = value;
    }
}

} // namespace tileinfer::ops
