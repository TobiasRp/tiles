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

template <typename Weight>
__global__ void decoder_worker(const TaskRecord* tasks, const WorkerTaskRange* ranges,
    const TensorRecord* tensors, int waves, int workers, int position, int hidden_size,
    int intermediate_size, int kv_heads, int head_dim, NumericConfig numeric, StackConfig stack,
    TokenId token_id, const TokenId* device_tokens, int token_index, float* input, float* output,
    float* scratch, float* kv, const Weight* weights, float* logits, float* block_trace)
{
    namespace cg = cooperative_groups;
    __shared__ float norm_factor;

    const TokenId current_token = device_tokens == nullptr ? token_id : device_tokens[token_index];
    if (stack.full_stack && current_token < 0)
        return;

    if (stack.full_stack) {
        ops::embedding_lookup(current_token, hidden_size, weights + stack.embedding_offset, input);
        cg::this_grid().sync();
    }

    for (int layer = 0; layer < stack.layers; ++layer) {
        float* current_input = layer % 2 == 0 ? input : output;
        float* current_output = layer % 2 == 0 ? output : input;
        float* layer_kv = kv + layer * stack.layer_kv_stride;
        const Weight* layer_weights =
            weights + stack.layer_weights_offset + layer * stack.layer_weight_stride;
        const ops::Context<Weight> context{
            tensors,
            {current_input, current_output, scratch, layer_kv, layer_weights},
            {hidden_size, intermediate_size, kv_heads, head_dim, position, numeric},
            &norm_factor,
        };

        for (int wave = 0; wave < waves; ++wave) {
            const WorkerTaskRange range = ranges[wave * workers + blockIdx.x];
            for (cuda::std::uint32_t index = range.begin; index < range.begin + range.count;
                ++index) {
                ops::execute(tasks[index], context);
                __syncthreads();
            }
            if (wave + 1 < waves)
                cg::this_grid().sync();
        }
        cg::this_grid().sync();
        if (block_trace != nullptr) {
            for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_size;
                index += gridDim.x * blockDim.x)
                block_trace[layer * hidden_size + index] = current_output[index];
            cg::this_grid().sync();
        }
    }

    if (stack.full_stack) {
        const float* last_hidden = stack.layers % 2 == 0 ? input : output;
        float* final_hidden = stack.layers % 2 == 0 ? output : input;
        ops::final_norm(last_hidden, weights + stack.final_norm_offset, final_hidden, hidden_size,
            numeric.rms_norm_eps, &norm_factor);
        cg::this_grid().sync();
        ops::logits(
            final_hidden, weights + stack.lm_head_offset, logits, hidden_size, stack.vocab_size);
    }
}

} // namespace

namespace detail {

template <typename Weight>
void enqueue_worker_typed(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token)
{
    auto config = cuda::make_config(
        cuda::block_dims<kThreads>(), cuda::grid_dims(program.workers), cuda::cooperative_launch{});
    cuda::launch(stream, config, decoder_worker<Weight>, program.tasks, program.ranges,
        program.tensors, program.waves, program.workers, position, program.hidden_size,
        program.intermediate_size, program.kv_heads, program.head_dim, program.numeric,
        program.stack, token.immediate, token.ids, token.index, buffers.input, buffers.output,
        buffers.scratch, buffers.kv, static_cast<const Weight*>(program.weights), buffers.logits,
        buffers.block_trace);
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
        props.maxThreadsPerMultiProcessor, props.totalGlobalMem, driver, runtime};
}

} // namespace tileinfer
