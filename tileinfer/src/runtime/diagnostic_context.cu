#include "detail/cleanup.hpp"
#include "detail/context_buffers.cuh"
#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "diagnostic.hpp"

#include <cuda/__event/timed_event.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tileinfer {
using detail::check;
namespace {

// Deterministic values in [-1, 1) for a synthetic K/V prefix.
template <typename Scalar>
__global__ void fill_synthetic(Scalar* data, std::size_t count)
{
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
        i += stride) {
        unsigned hash = static_cast<unsigned>(i) * 0x9e3779b1u;
        hash ^= hash >> 15;
        hash *= 0x85ebca77u;
        hash ^= hash >> 13;
        data[i] = static_cast<Scalar>(static_cast<float>(hash & 0xffffu) / 32768.0f - 1.0f);
    }
}

} // namespace

struct DiagnosticContext::Impl
{
    detail::UploadedPlan model;
    cuda::stream stream{model.device};
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(model.device);
    detail::ContextBuffers buffers;
    cuda::device_buffer<float> ffn_trace;
    int position = 0;
    bool valid = true;
    bool in_flight = false;
    bool capture_blocks = false;
    // Fixed-position benchmarks: K/V rows [0, prefix) hold prefix_token's
    // prefix, or synthetic values for every position.
    int prefix = 0;
    TokenId prefix_token = -1;
    bool synthetic_prefix = false;

    explicit Impl(const LoadedPlan& plan)
        : model(plan)
        , buffers(stream, pool, model)
        , ffn_trace(stream, pool, model.shape.hidden_size, cuda::no_init)
    {
        stream.sync();
    }

    ~Impl()
    {
        detail::cleanup_noexcept([&] { stream.sync(); });
    }

    void launch(int position, TokenId token_id, std::uint64_t* timeline = nullptr)
    {
        detail::WorkerBuffers worker = buffers.worker_buffers(capture_blocks);
        worker.ffn_trace = ffn_trace.data();
        worker.timeline = timeline;
        detail::enqueue_worker(stream, model.view(), worker, position, {token_id, nullptr, 1});
    }

    // Fills the visible prefix and warms up outside any timed region. Repeated
    // launches at position overwrite only K/V[position], so every later launch
    // sees the same context length and all earlier cache entries remain valid.
    // Every prefix row holds the same token's value, so a later benchmark with
    // that token extends the rows already filled instead of refilling them.
    void prepare_fixed_position(int position, TokenId token_id, int warmup)
    {
        if (!model.stack.full_stack)
            throw std::runtime_error("benchmark requires a decoder stack package");
        if (position < 0 || position >= model.shape.max_context || token_id < 0 ||
            token_id >= model.stack.vocab_size || warmup < 0)
            throw std::runtime_error("invalid benchmark arguments");
        if (this->position != 0 || !valid || in_flight)
            throw std::runtime_error("benchmark requires a fresh context");
        if (token_id != prefix_token && !synthetic_prefix)
            prefix = 0;
        prefix_token = token_id;
        if (synthetic_prefix && prefix < position) {
            const auto config = cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims(1024));
            if (model.weight_dtype == WeightDtype::bfloat16)
                cuda::launch(stream, config, fill_synthetic<__nv_bfloat16>,
                    static_cast<__nv_bfloat16*>(buffers.kv.data()), buffers.kv.size());
            else
                cuda::launch(stream, config, fill_synthetic<float>,
                    static_cast<float*>(buffers.kv.data()), buffers.kv.size());
            prefix = model.shape.max_context;
        }
        for (int p = prefix; p < position; ++p)
            launch(p, token_id);
        prefix = std::max(prefix, position);
        stream.sync();
        for (int i = 0; i < warmup; ++i)
            launch(position, token_id);
        stream.sync();
    }
};

DiagnosticContext::DiagnosticContext(const LoadedPlan& plan)
    : impl_(std::make_unique<Impl>(plan))
{
}

DiagnosticContext::~DiagnosticContext() = default;

int DiagnosticContext::begin_token()
{
    if (!impl_->valid)
        throw std::runtime_error("sequence context is invalid; reset required");
    if (impl_->in_flight)
        throw std::runtime_error("sequence context already has an in-flight token");
    if (impl_->position >= impl_->model.shape.max_context)
        throw std::runtime_error("sequence context capacity reached");
    impl_->in_flight = true;
    return impl_->position;
}

void DiagnosticContext::complete_token(bool success)
{
    if (!impl_->in_flight)
        throw std::runtime_error("sequence context has no in-flight token");
    impl_->in_flight = false;
    if (success)
        ++impl_->position;
    else
        impl_->valid = false;
}

void DiagnosticContext::reset()
{
    impl_->stream.sync();
    impl_->position = 0;
    impl_->valid = true;
    impl_->in_flight = false;
    impl_->prefix = 0;
}

void DiagnosticContext::use_synthetic_prefix()
{
    impl_->synthetic_prefix = true;
    impl_->prefix = 0;
}

int DiagnosticContext::position() const
{
    return impl_->position;
}

bool DiagnosticContext::valid() const
{
    return impl_->valid;
}

std::vector<float> DiagnosticContext::run_hidden(const std::vector<float>& hidden)
{
    if (impl_->model.stack.full_stack)
        throw std::runtime_error("stack package requires a token ID");
    if (hidden.size() != static_cast<size_t>(impl_->model.shape.hidden_size))
        throw std::runtime_error("hidden input has wrong size");
    std::vector<float> result(impl_->model.shape.hidden_size);
    const int p = begin_token();
    try {
        impl_->buffers.input.upload(hidden);
        impl_->launch(p, -1);
        impl_->stream.sync();
        impl_->buffers.output.copy_to(impl_->stream, result);
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
        detail::cleanup_noexcept([&] { impl_->stream.sync(); });
        complete_token(false);
        throw;
    }
}

std::vector<float> DiagnosticContext::run_token(TokenId token_id)
{
    if (!impl_->model.stack.full_stack)
        throw std::runtime_error("block package requires a hidden vector");
    if (token_id < 0 || token_id >= impl_->model.stack.vocab_size)
        throw std::runtime_error("token ID outside vocabulary");
    std::vector<float> result(impl_->model.stack.vocab_size);
    const int p = begin_token();
    try {
        impl_->launch(p, token_id);
        impl_->stream.sync();
        cuda::copy_bytes(impl_->stream, impl_->buffers.logits, result);
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
        detail::cleanup_noexcept([&] { impl_->stream.sync(); });
        complete_token(false);
        throw;
    }
}

BenchmarkResult DiagnosticContext::benchmark_token(
    int position, TokenId token_id, int warmup, int iterations, int rounds, bool profile)
{
    if (iterations <= 0 || rounds <= 0)
        throw std::runtime_error("invalid benchmark arguments");
    try {
        impl_->prepare_fixed_position(position, token_id, warmup);

        std::vector<double> samples;
        samples.reserve(rounds);
        if (profile)
            check(cudaProfilerStart(), "start profiler");
        for (int round = 0; round < rounds; ++round) {
            cuda::timed_event start{impl_->stream}, end{impl_->stream};
            start.record(impl_->stream);
            for (int i = 0; i < iterations; ++i)
                impl_->launch(position, token_id);
            end.record(impl_->stream);
            end.sync();
            samples.push_back(
                cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count() /
                iterations);
        }
        if (profile)
            check(cudaProfilerStop(), "stop profiler");
        std::vector<float> logits(impl_->model.stack.vocab_size);
        cuda::copy_bytes(impl_->stream, impl_->buffers.logits, logits);
        impl_->stream.sync();
        const auto top = std::max_element(logits.begin(), logits.end());
        return {std::move(samples), static_cast<int>(top - logits.begin()), *top};
    } catch (...) {
        impl_->valid = false;
        throw;
    }
}

TimelineResult DiagnosticContext::timeline_token(
    int position, TokenId token_id, int warmup, int samples)
{
    if (samples <= 0)
        throw std::runtime_error("invalid timeline sample count");
    try {
        impl_->prepare_fixed_position(position, token_id, warmup);
        const std::size_t words = detail::kTimelineFields * impl_->model.instructions.size();
        cuda::device_buffer<std::uint64_t> timeline(
            impl_->stream, impl_->pool, words, cuda::no_init);
        TimelineResult result;
        for (int sample = 0; sample < samples; ++sample) {
            cuda::timed_event start{impl_->stream}, end{impl_->stream};
            start.record(impl_->stream);
            impl_->launch(position, token_id, timeline.data());
            end.record(impl_->stream);
            end.sync();
            result.kernel_us.push_back(
                cuda::std::chrono::duration<double, cuda::std::micro>(end - start).count());
            auto& records = result.records.emplace_back(words);
            cuda::copy_bytes(impl_->stream, timeline, records);
            impl_->stream.sync();
        }
        return result;
    } catch (...) {
        impl_->valid = false;
        throw;
    }
}

void DiagnosticContext::enable_block_trace()
{
    impl_->buffers.enable_block_trace();
    impl_->capture_blocks = true;
}

std::vector<float> DiagnosticContext::block_trace_snapshot() const
{
    if (!impl_->capture_blocks)
        throw std::runtime_error("block trace is not enabled");
    impl_->stream.sync();
    std::vector<float> result(impl_->buffers.block_trace->size());
    cuda::copy_bytes(impl_->stream, *impl_->buffers.block_trace, result);
    impl_->stream.sync();
    return result;
}

std::vector<float> DiagnosticContext::scratch_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->buffers.scratch.size());
    // Copy only tensor payloads; alignment padding has no producer.
    const auto& model = impl_->model;
    // The validated scratch layout is retained on the host for exact slices.
    for (const auto& [offset, count] : model.scratch_regions) {
        cuda::device_buffer<float> widened(impl_->stream, impl_->pool, count, cuda::no_init);
        const auto* source = static_cast<const std::uint8_t*>(impl_->buffers.scratch.data()) +
                             offset * element_bytes(model.weight_dtype);
        detail::enqueue_widen(impl_->stream, source, widened.data(), count, model.weight_dtype);
        cuda::copy_bytes(
            impl_->stream, widened, cuda::std::span<float>(result.data() + offset, count));
    }
    impl_->stream.sync();
    return result;
}

std::vector<float> DiagnosticContext::ffn_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->model.shape.hidden_size);
    cuda::copy_bytes(impl_->stream, impl_->ffn_trace, result);
    impl_->stream.sync();
    return result;
}

std::vector<float> DiagnosticContext::kv_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->buffers.kv.size());
    detail::enqueue_zero_unused_kv(impl_->stream, impl_->buffers.kv.data(),
        impl_->model.weight_dtype, impl_->model.kv_count, impl_->model.stack.layers,
        impl_->model.stack.layer_kv_stride,
        impl_->model.shape.kv_heads * impl_->model.shape.head_dim, impl_->model.shape.max_context,
        impl_->position, nullptr);
    impl_->buffers.kv.copy_to(impl_->stream, result);
    impl_->stream.sync();
    return result;
}

} // namespace tileinfer
