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

// Executes this block's validated instruction stream for one token. Barrier
// instructions appear at the same stream positions in every worker.
template <typename Scalar>
__global__ __launch_bounds__(kThreads, 1) void decoder_worker(const TaskRecord* instructions,
    const WorkerTaskRange* streams, const TensorRecord* tensors, int position, int hidden_size,
    int intermediate_size, int kv_heads, int head_dim, NumericConfig numeric, StackConfig stack,
    TokenId token_id, const TokenId* device_tokens, int token_index, void* raw_input,
    void* raw_output, void* raw_scratch, void* raw_kv, const Scalar* weights, float* logits,
    float* block_trace, const float* rotary_frequencies, int max_context, void* attention_scores,
    float* ffn_trace, cuda::std::uint64_t* timeline)
{
    // Layer l reads hidden(l) and writes hidden(l + 1). Selecting with a
    // conditional, not an indexed array, keeps both pointers in the global
    // address space; a local array made activation reads generic loads.
    auto* const input = static_cast<Scalar*>(raw_input);
    auto* const output = static_cast<Scalar*>(raw_output);
    const auto hidden = [&](int index) { return index % 2 == 0 ? input : output; };
    auto* scratch = static_cast<Scalar*>(raw_scratch);
    auto* kv = static_cast<Scalar*>(raw_kv);
    namespace cg = cooperative_groups;
    __shared__ ops::SharedStorage shared;

    const TokenId current_token = device_tokens == nullptr ? token_id : device_tokens[token_index];
    // Every block takes the same EOS skip decision, so no barrier is left waiting.
    if (stack.full_stack && current_token < 0)
        return;

    const WorkerTaskRange stream = streams[blockIdx.x];
    for (cuda::std::uint32_t index = stream.begin; index < stream.begin + stream.count; ++index) {
        const TaskRecord task = instructions[index];
        // Store start times immediately: values kept live across a task would
        // compete for the worker's 64 registers per thread and cause spills.
        if (timeline != nullptr && threadIdx.x == 0) {
            auto* record = timeline + detail::kTimelineFields * index;
            record[0] = clock64();
            record[2] = global_timer_ns();
        }
        switch (task.opcode) {
        case Opcode::barrier:
            cg::this_grid().sync();
            break;
        case Opcode::embedding:
            ops::embedding_lookup(current_token, task.begin, task.end, hidden_size,
                weights + stack.embedding_offset, hidden(0));
            break;
        case Opcode::final_rms_norm:
            ops::final_rms_norm(hidden(stack.layers), weights + stack.final_norm_offset,
                hidden(stack.layers + 1), task.begin, task.end, hidden_size, numeric.rms_norm_eps,
                &shared.inverse_rms, shared.norm_partials);
            break;
        case Opcode::logits:
            ops::logits(hidden(stack.layers + 1), weights + stack.lm_head_offset, logits,
                task.begin, task.end, hidden_size, shared);
            break;
        default: {
            const int layer = static_cast<int>(task.layer);
            const ops::Context<Scalar> context{
                tensors,
                {hidden(layer), hidden(layer + 1), scratch, kv + layer * stack.layer_kv_stride,
                    weights + stack.layer_weights_offset + layer * stack.layer_weight_stride,
                    static_cast<Scalar*>(attention_scores), ffn_trace,
                    block_trace == nullptr ? nullptr : block_trace + layer * hidden_size},
                {hidden_size, intermediate_size, kv_heads, head_dim, position, max_context, numeric,
                    rotary_frequencies},
                &shared,
            };
            ops::execute(task, context);
        }
        }
        if (task.opcode != Opcode::barrier)
            __syncthreads();
        if (timeline != nullptr && threadIdx.x == 0)
            timeline[detail::kTimelineFields * index + 1] = clock64();
    }
}

} // namespace

namespace detail {

template <typename Scalar>
void enqueue_worker_typed(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token)
{
    auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(program.workers), cuda::cooperative_launch{});
    cuda::launch(stream, config, decoder_worker<Scalar>, program.instructions, program.streams,
        program.tensors, position, program.hidden_size, program.intermediate_size, program.kv_heads,
        program.head_dim, program.numeric, program.stack, token.immediate, token.ids, token.index,
        buffers.input, buffers.output, buffers.scratch, buffers.kv,
        static_cast<const Scalar*>(program.weights), buffers.logits, buffers.block_trace,
        program.rotary_frequencies, program.max_context, buffers.attention_scores,
        buffers.ffn_trace, buffers.timeline);
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
