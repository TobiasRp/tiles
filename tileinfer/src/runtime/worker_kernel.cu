#include "detail/cuda_check.hpp"
#include "detail/launch.cuh"
#include "ops.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/launch>
#include <cuda/stream>

#include <algorithm>

namespace tileinfer {
using detail::check;
namespace {

__device__ inline cuda::std::uint64_t global_timer_ns()
{
    cuda::std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Everything one worker launch reads, passed by value.
template <typename Scalar>
struct Launch
{
    const TaskRecord* instructions;
    const WorkerTaskRange* streams;
    LayerLayout layout;
    ops::Shape shape;
    StackConfig stack;
    TokenId token_id;             // used when device_tokens is null
    const TokenId* device_tokens; // the token is device_tokens[token_index] when set
    int token_index;
    Scalar* input;  // hidden state of even layers
    Scalar* output; // hidden state of odd layers
    Scalar* scratch;
    Scalar* kv;
    const Scalar* weights;
    Scalar* attention_scores;
    float* attention_partials;
    float* logits;
    float* block_trace;
    float* ffn_trace;
    cuda::std::uint64_t* timeline;

    // Layer l reads hidden(l) and writes hidden(l + 1). A conditional, not an
    // indexed array, keeps both pointers in the global address space; a local
    // array made activation reads generic loads.
    __device__ Scalar* hidden(int index) const { return index % 2 == 0 ? input : output; }

    __device__ ops::Layer<Scalar> layer(int index) const
    {
        const Scalar* w = weights + stack.layer_weights_offset + index * stack.layer_weight_stride;
        Scalar* cache = kv + index * stack.layer_kv_stride;
        float* partials = attention_partials;
        float* stats = partials + shape.attention_chunks * shape.hidden_size;
        return {hidden(index), hidden(index + 1), scratch + layout.q_rot,
            scratch + layout.attention, scratch + layout.r, scratch + layout.product,
            cache + layout.key, cache + layout.value, attention_scores, partials, stats,
            w + layout.attn_norm, w + layout.Wq, w + layout.Wk, w + layout.Wv, w + layout.Wo,
            w + layout.ffn_norm, w + layout.Wgate, w + layout.Wup, w + layout.Wdown, ffn_trace,
            block_trace == nullptr ? nullptr : block_trace + index * shape.hidden_size};
    }
};

// Executes this block's validated instruction stream for one token. Barrier
// instructions appear at the same stream positions in every worker.
template <typename Scalar>
__global__ __launch_bounds__(kThreads, 1) void decoder_worker(Launch<Scalar> launch)
{
    namespace cg = cooperative_groups;
    __shared__ ops::SharedStorage shared;
    const auto& shape = launch.shape;
    const auto& stack = launch.stack;

    const TokenId token = launch.device_tokens == nullptr
                              ? launch.token_id
                              : launch.device_tokens[launch.token_index];
    // Every block takes the same EOS skip decision, so no barrier is left waiting.
    if (stack.full_stack && token < 0)
        return;

    const WorkerTaskRange stream = launch.streams[blockIdx.x];
    for (cuda::std::uint32_t index = stream.begin; index < stream.begin + stream.count; ++index) {
        const TaskRecord task = launch.instructions[index];
        // Store start times at once: values kept live across a task would
        // compete for the worker's 64 registers per thread and cause spills.
        if (launch.timeline != nullptr && threadIdx.x == 0) {
            auto* record = launch.timeline + detail::kTimelineFields * index;
            record[0] = clock64();
            record[2] = global_timer_ns();
        }
        const int begin = static_cast<int>(task.begin);
        const int end = static_cast<int>(task.end);
        switch (task.opcode) {
        case Opcode::barrier:
            if (shape.position + 1 > begin)
                cg::this_grid().sync();
            break;
        case Opcode::qkv:
            ops::qkv(begin, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::attention_scores:
            ops::attention_scores(
                task.kv_head, task.chunk, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::attention_values:
            ops::attention_values(
                task.kv_head, task.chunk, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::output_projection:
            ops::output_projection(begin, end, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::gate_up:
            ops::gate_up(begin, end, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::down:
            ops::down(begin, end, launch.layer(task.layer), shape, shared);
            break;
        case Opcode::embedding:
            ops::embedding_lookup(token, begin, end, shape.hidden_size,
                launch.weights + stack.embedding_offset, launch.hidden(0));
            break;
        case Opcode::final_rms_norm:
            ops::normalize(launch.hidden(stack.layers), launch.weights + stack.final_norm_offset,
                launch.hidden(stack.layers + 1), begin, end, shape.hidden_size,
                shape.numeric.rms_norm_eps, shared);
            break;
        case Opcode::logits:
            ops::logits(launch.hidden(stack.layers + 1), launch.weights + stack.lm_head_offset,
                launch.logits, begin, end, shape.hidden_size, shared);
            break;
        }
        if (task.opcode != Opcode::barrier)
            __syncthreads();
        if (launch.timeline != nullptr && threadIdx.x == 0)
            launch.timeline[detail::kTimelineFields * index + 1] = clock64();
    }
}

} // namespace

namespace detail {

template <typename Scalar>
void enqueue_worker_typed(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token)
{
    const Launch<Scalar> launch{program.instructions, program.streams, program.layout,
        {program.hidden_size, program.intermediate_size, program.query_heads, program.kv_heads,
            program.head_dim, position, program.max_context, program.attention_chunks,
            program.attention_chunk_steps, program.attention_fused_steps, program.numeric,
            program.rotary_frequencies},
        program.stack, token.immediate, token.ids, token.index, static_cast<Scalar*>(buffers.input),
        static_cast<Scalar*>(buffers.output), static_cast<Scalar*>(buffers.scratch),
        static_cast<Scalar*>(buffers.kv), static_cast<const Scalar*>(program.weights),
        static_cast<Scalar*>(buffers.attention_scores), buffers.attention_partials, buffers.logits,
        buffers.block_trace, buffers.ffn_trace, buffers.timeline};
    auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(program.workers), cuda::cooperative_launch{});
    cuda::launch(stream, config, decoder_worker<Scalar>, launch);
}

void enqueue_worker(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token)
{
    if (program.weight_dtype == WeightDtype::bfloat16)
        enqueue_worker_typed<__nv_bfloat16>(stream, program, buffers, position, token);
    else
        enqueue_worker_typed<float>(stream, program, buffers, position, token);
}

} // namespace detail

Target describe_target()
{
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device query");
    check(cudaSetDevice(0), "select device");
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, decoder_worker<float>), "FP32 worker attributes");
    cudaFuncAttributes bf16_attributes{};
    check(cudaFuncGetAttributes(&bf16_attributes, decoder_worker<__nv_bfloat16>),
        "BF16 worker attributes");
    int f32_occupancy = 0;
    int bf16_occupancy = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &f32_occupancy, decoder_worker<float>, kThreads, 0),
        "FP32 worker occupancy");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &bf16_occupancy, decoder_worker<__nv_bfloat16>, kThreads, 0),
        "BF16 worker occupancy");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    return {props.name, props.major, props.minor, props.multiProcessorCount,
        static_cast<bool>(props.cooperativeLaunch), std::min(f32_occupancy, bf16_occupancy),
        std::max(attributes.numRegs, bf16_attributes.numRegs),
        std::max(static_cast<int>(attributes.sharedSizeBytes),
            static_cast<int>(bf16_attributes.sharedSizeBytes)),
        props.maxThreadsPerMultiProcessor, props.totalGlobalMem, driver, runtime,
        {{"float32-simt", attributes.numRegs, static_cast<int>(attributes.sharedSizeBytes),
             f32_occupancy},
            {"bfloat16-simt", bf16_attributes.numRegs,
                static_cast<int>(bf16_attributes.sharedSizeBytes), bf16_occupancy}}};
}

} // namespace tileinfer
