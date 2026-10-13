#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "diagnostic.hpp"

#include <cuda/__event/timed_event.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tileinfer {
using detail::check;

struct DiagnosticContext::Impl
{
    detail::UploadedPlan model;
    cuda::stream stream{model.device};
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(model.device);
    cuda::device_buffer<float> input;
    cuda::device_buffer<float> output;
    cuda::device_buffer<float> scratch;
    cuda::device_buffer<float> kv;
    cuda::device_buffer<float> logits;
    cuda::device_buffer<float> block_trace;
    int position = 0;
    bool valid = true;
    bool in_flight = false;
    bool capture_blocks = false;

    explicit Impl(const LoadedPlan& plan)
        : model(plan)
        , input(stream, pool, model.hidden_size, cuda::no_init)
        , output(stream, pool, model.hidden_size, cuda::no_init)
        , scratch(stream, pool, model.scratch_count, cuda::no_init)
        , kv(stream, pool, model.kv_count, cuda::no_init)
        , logits(stream, pool, std::max(1, model.stack.vocab_size), cuda::no_init)
        , block_trace(stream, pool, model.stack.layers * model.hidden_size, cuda::no_init)
    {
        stream.sync();
    }

    void launch(int position, TokenId token_id)
    {
        const detail::WorkerBuffers buffers{input.data(), output.data(), scratch.data(), kv.data(),
            logits.data(), capture_blocks ? block_trace.data() : nullptr};
        detail::enqueue_worker(stream, model.view(), buffers, position, {token_id, nullptr, 0});
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
    if (impl_->position >= impl_->model.max_context)
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
    if (hidden.size() != static_cast<size_t>(impl_->model.hidden_size))
        throw std::runtime_error("hidden input has wrong size");
    const int p = begin_token();
    try {
        cuda::copy_bytes(impl_->stream, hidden, impl_->input);
        impl_->launch(p, -1);
        impl_->stream.sync();
        std::vector<float> result(impl_->model.hidden_size);
        cuda::copy_bytes(impl_->stream, impl_->output, result);
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
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
    const int p = begin_token();
    try {
        impl_->launch(p, token_id);
        impl_->stream.sync();
        std::vector<float> result(impl_->model.stack.vocab_size);
        cuda::copy_bytes(impl_->stream, impl_->logits, result);
        impl_->stream.sync();
        complete_token(true);
        return result;
    } catch (...) {
        complete_token(false);
        throw;
    }
}

BenchmarkResult DiagnosticContext::benchmark_token(
    int position, TokenId token_id, int warmup, int iterations, int rounds, bool profile)
{
    if (!impl_->model.stack.full_stack)
        throw std::runtime_error("benchmark requires a decoder stack package");
    if (position < 0 || position >= impl_->model.max_context || token_id < 0 ||
        token_id >= impl_->model.stack.vocab_size || warmup < 0 || iterations <= 0 || rounds <= 0)
        throw std::runtime_error("invalid benchmark arguments");
    if (impl_->position != 0 || !impl_->valid || impl_->in_flight)
        throw std::runtime_error("benchmark requires a fresh context");

    // Fill the visible prefix outside the timed region. Repeated launches at
    // position overwrite only K/V[position], so every sample sees the same
    // context length and all earlier cache entries remain valid.
    try {
        for (int p = 0; p < position; ++p)
            impl_->launch(p, token_id);
        impl_->stream.sync();
        for (int i = 0; i < warmup; ++i)
            impl_->launch(position, token_id);
        impl_->stream.sync();

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
        cuda::copy_bytes(impl_->stream, impl_->logits, logits);
        impl_->stream.sync();
        const auto top = std::max_element(logits.begin(), logits.end());
        return {std::move(samples), static_cast<int>(top - logits.begin()), *top};
    } catch (...) {
        impl_->valid = false;
        throw;
    }
}

void DiagnosticContext::enable_block_trace()
{
    impl_->capture_blocks = true;
}

std::vector<float> DiagnosticContext::block_trace_snapshot() const
{
    if (!impl_->capture_blocks)
        throw std::runtime_error("block trace is not enabled");
    impl_->stream.sync();
    std::vector<float> result(impl_->block_trace.size());
    cuda::copy_bytes(impl_->stream, impl_->block_trace, result);
    impl_->stream.sync();
    return result;
}

std::vector<float> DiagnosticContext::scratch_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->scratch.size());
    cuda::copy_bytes(impl_->stream, impl_->scratch, result);
    impl_->stream.sync();
    return result;
}

std::vector<float> DiagnosticContext::kv_snapshot() const
{
    impl_->stream.sync();
    std::vector<float> result(impl_->kv.size());
    cuda::copy_bytes(impl_->stream, impl_->kv, result);
    impl_->stream.sync();
    return result;
}

} // namespace tileinfer
