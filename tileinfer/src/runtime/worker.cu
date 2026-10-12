#include "ops.cuh"
#include "runtime.hpp"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tileinfer {
namespace {

void check(cudaError_t status, const char* stage)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(status));
}

template <typename Weight>
__global__ void decoder_worker(const TaskRecord* tasks, const Range* ranges,
    const TensorRecord* tensors, int waves, int workers, int position, int hidden_size,
    int intermediate_size, int kv_heads, int head_dim, NumericConfig numeric, StackConfig stack,
    int token_id, float* input, float* output, float* scratch, float* kv, const Weight* weights,
    float* logits, float* block_trace)
{
    namespace cg = cooperative_groups;
    __shared__ float norm_factor;

    if (stack.full_stack) {
        ops::embedding_lookup(token_id, hidden_size, weights + stack.embedding_offset, input);
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
            const Range range = ranges[wave * workers + blockIdx.x];
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

struct SequenceContext::Impl
{
    cuda::device_ref device{0};
    cuda::stream stream{device};
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(device);
    cuda::device_buffer<float> input;
    cuda::device_buffer<float> output;
    cuda::device_buffer<float> scratch;
    cuda::device_buffer<float> kv;
    cuda::device_buffer<std::uint8_t> weights;
    cuda::device_buffer<float> logits;
    cuda::device_buffer<float> block_trace;
    cuda::device_buffer<TaskRecord> tasks;
    cuda::device_buffer<Range> ranges;
    cuda::device_buffer<TensorRecord> tensors;
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    int workers;
    int waves;
    NumericConfig numeric;
    StackConfig stack;
    WeightDtype weight_dtype;
    int max_context;
    int position = 0;
    bool valid = true;
    bool in_flight = false;
    bool capture_blocks = false;

    explicit Impl(const LoadedPlan& plan)
        : input(stream, pool, plan.hidden_size, cuda::no_init)
        , output(stream, pool, plan.hidden_size, cuda::no_init)
        , scratch(stream, pool, plan.scratch_bytes / sizeof(float), cuda::no_init)
        , kv(stream, pool, plan.kv_bytes / sizeof(float), cuda::no_init)
        , weights(stream, pool, plan.weights.size(), cuda::no_init)
        , logits(stream, pool, std::max(1, plan.stack.vocab_size), cuda::no_init)
        , block_trace(stream, pool, plan.stack.layers * plan.hidden_size, cuda::no_init)
        , tasks(stream, pool, plan.tasks.size(), cuda::no_init)
        , ranges(stream, pool, plan.ranges.size(), cuda::no_init)
        , tensors(stream, pool, plan.tensors.size(), cuda::no_init)
        , hidden_size(plan.hidden_size)
        , intermediate_size(plan.intermediate_size)
        , kv_heads(plan.kv_heads)
        , head_dim(plan.head_dim)
        , workers(plan.workers)
        , waves(plan.wave_count)
        , numeric(plan.numeric)
        , stack(plan.stack)
        , weight_dtype(plan.weight_dtype)
        , max_context(plan.max_context)
    {
        stream.sync();
        check(cudaMemcpyAsync(weights.data(), plan.weights.data(), plan.weights.size(),
                  cudaMemcpyHostToDevice, stream.get()),
            "copy weights");
        check(cudaMemcpyAsync(tasks.data(), plan.tasks.data(),
                  plan.tasks.size() * sizeof(TaskRecord), cudaMemcpyHostToDevice, stream.get()),
            "copy tasks");
        check(cudaMemcpyAsync(ranges.data(), plan.ranges.data(), plan.ranges.size() * sizeof(Range),
                  cudaMemcpyHostToDevice, stream.get()),
            "copy ranges");
        check(cudaMemcpyAsync(tensors.data(), plan.tensors.data(),
                  plan.tensors.size() * sizeof(TensorRecord), cudaMemcpyHostToDevice, stream.get()),
            "copy tensor addresses");
        stream.sync();
    }

    template <typename Weight>
    void launch_typed(int position, int token_id)
    {
        auto config = cuda::make_config(
            cuda::block_dims<kThreads>(), cuda::grid_dims(workers), cuda::cooperative_launch{});
        cuda::launch(stream, config, decoder_worker<Weight>, tasks.data(), ranges.data(),
            tensors.data(), waves, workers, position, hidden_size, intermediate_size, kv_heads,
            head_dim, numeric, stack, token_id, input.data(), output.data(), scratch.data(),
            kv.data(), reinterpret_cast<const Weight*>(weights.data()), logits.data(),
            capture_blocks ? block_trace.data() : nullptr);
    }

    void launch(int position, int token_id)
    {
        switch (weight_dtype) {
        case WeightDtype::float32:
            launch_typed<float>(position, token_id);
            break;
        case WeightDtype::bfloat16:
            launch_typed<__nv_bfloat16>(position, token_id);
            break;
        }
    }
};

SequenceContext::SequenceContext(const LoadedPlan& plan)
    : impl_(std::make_unique<Impl>(plan))
{
}

SequenceContext::~SequenceContext() = default;

int SequenceContext::begin_token()
{
    if (!impl_->valid)
        throw std::runtime_error("sequence context is invalid; reset required");
    if (impl_->in_flight)
        throw std::runtime_error("sequence context already has an in-flight token");
    if (impl_->position >= impl_->max_context)
        throw std::runtime_error("sequence context capacity reached");
    impl_->in_flight = true;
    return impl_->position;
}

void SequenceContext::complete_token(bool success)
{
    if (!impl_->in_flight)
        throw std::runtime_error("sequence context has no in-flight token");
    impl_->in_flight = false;
    if (success)
        ++impl_->position;
    else
        impl_->valid = false;
}

void SequenceContext::reset()
{
    impl_->stream.sync();
    impl_->position = 0;
    impl_->valid = true;
    impl_->in_flight = false;
}

int SequenceContext::position() const
{
    return impl_->position;
}

bool SequenceContext::valid() const
{
    return impl_->valid;
}

std::vector<float> SequenceContext::run_hidden(const std::vector<float>& hidden)
{
    if (impl_->stack.full_stack)
        throw std::runtime_error("stack package requires a token ID");
    if (hidden.size() != static_cast<size_t>(impl_->hidden_size))
        throw std::runtime_error("hidden input has wrong size");
    const int p = begin_token();
    try {
        check(cudaMemcpyAsync(impl_->input.data(), hidden.data(), hidden.size() * sizeof(float),
                  cudaMemcpyHostToDevice, impl_->stream.get()),
            "copy hidden input");
        impl_->launch(p, -1);
        impl_->stream.sync();
        std::vector<float> result(impl_->hidden_size);
        check(cudaMemcpyAsync(result.data(), impl_->output.data(), result.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, impl_->stream.get()),
            "copy hidden output");
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
        complete_token(false);
        throw;
    }
}

std::vector<float> SequenceContext::run_token(int token_id)
{
    if (!impl_->stack.full_stack)
        throw std::runtime_error("block package requires a hidden vector");
    if (token_id < 0 || token_id >= impl_->stack.vocab_size)
        throw std::runtime_error("token ID outside vocabulary");
    const int p = begin_token();
    try {
        impl_->launch(p, token_id);
        impl_->stream.sync();
        std::vector<float> result(impl_->stack.vocab_size);
        check(cudaMemcpyAsync(result.data(), impl_->logits.data(), result.size() * sizeof(float),
                  cudaMemcpyDeviceToHost, impl_->stream.get()),
            "copy logits");
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
        complete_token(false);
        throw;
    }
}

void SequenceContext::enable_block_trace()
{
    impl_->capture_blocks = true;
}

std::vector<float> SequenceContext::block_trace_snapshot() const
{
    if (!impl_->capture_blocks)
        throw std::runtime_error("block trace is not enabled");
    impl_->stream.sync();
    std::vector<float> result(impl_->block_trace.size());
    check(cudaMemcpyAsync(result.data(), impl_->block_trace.data(), result.size() * sizeof(float),
              cudaMemcpyDeviceToHost, impl_->stream.get()),
        "copy block trace");
    impl_->stream.sync();
    return result;
}

std::vector<float> SequenceContext::scratch_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->scratch.size());
    check(cudaMemcpyAsync(result.data(), impl_->scratch.data(), result.size() * sizeof(float),
              cudaMemcpyDeviceToHost, impl_->stream.get()),
        "copy scratch");
    impl_->stream.sync();
    return result;
}

std::vector<float> SequenceContext::kv_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->kv.size());
    check(cudaMemcpyAsync(result.data(), impl_->kv.data(), result.size() * sizeof(float),
              cudaMemcpyDeviceToHost, impl_->stream.get()),
        "copy kv");
    impl_->stream.sync();
    return result;
}

} // namespace tileinfer
