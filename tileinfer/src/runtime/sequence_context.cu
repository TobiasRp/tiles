#include "detail/cuda_check.hpp"
#include "detail/input_ring_cursor.hpp"
#include "detail/ticket_state.cuh"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tileinfer {
using detail::check;

namespace {
std::size_t input_ring_size(int max_context)
{
    // One maximum prompt and 64 additional queued one-token inputs.
    const auto maximum = static_cast<std::size_t>(max_context);
    if (maximum > std::numeric_limits<std::size_t>::max() - 64 ||
        maximum + 64 > std::numeric_limits<std::size_t>::max() / sizeof(TokenId))
        throw std::runtime_error("input ring size overflow");
    return maximum + 64;
}
} // namespace

struct SequenceContext::Impl
{
    std::shared_ptr<detail::UploadedPlan> model;
    cuda::stream stream;
    cuda::device_memory_pool_ref& pool;
    cuda::device_buffer<float> input;
    cuda::device_buffer<float> output;
    cuda::device_buffer<float> scratch;
    cuda::device_buffer<float> kv;
    cuda::device_buffer<float> logits;
    cuda::device_buffer<float> block_trace;
    cuda::device_buffer<TokenId> tokens;
    cuda::device_buffer<TokenId> generated;
    cuda::device_buffer<int> status;
    cuda::host_buffer<TokenId> input_ring;
    detail::InputRingCursor ring_cursor;
    int queued_position = 0;
    std::shared_ptr<Ticket::State> pending_position;
    bool valid = true;
    std::vector<std::shared_ptr<Ticket::State>> outstanding;

    explicit Impl(std::shared_ptr<detail::UploadedPlan> value)
        : model(std::move(value))
        , stream(model->device)
        , pool(cuda::device_default_memory_pool(model->device))
        , input(stream, pool, model->hidden_size, cuda::no_init)
        , output(stream, pool, model->hidden_size, cuda::no_init)
        , scratch(stream, pool, model->scratch_count, cuda::no_init)
        , kv(stream, pool, model->kv_count, cuda::no_init)
        , logits(stream, pool, model->stack.vocab_size, cuda::no_init)
        , block_trace(stream, pool, model->stack.layers * model->hidden_size, cuda::no_init)
        , tokens(stream, pool, model->max_context, cuda::no_init)
        , generated(stream, pool, model->max_context, cuda::no_init)
        , status(stream, pool, 2, cuda::no_init)
        , input_ring(stream, cuda::pinned_default_memory_pool(),
              input_ring_size(model->max_context), cuda::no_init)
        , ring_cursor(input_ring.size())
    {
    }

    ~Impl() { stream.sync(); }

    void require_ready() const
    {
        if (!valid)
            throw std::runtime_error("sequence context is invalid; reset required");
        if (pending_position)
            throw std::runtime_error("collect generated result before appending more tokens");
    }

    void validate_token(TokenId token_id) const
    {
        if (token_id < 0 || token_id >= model->stack.vocab_size)
            throw std::runtime_error("token ID outside vocabulary");
    }

    void validate_capacity(int count) const
    {
        if (count < 0 || count > model->max_context - queued_position)
            throw std::runtime_error("sequence context capacity reached");
    }

    void launch(int position, const TokenId* token_ids, int token_index, bool trace)
    {
        const detail::WorkerBuffers buffers{input.data(), output.data(), scratch.data(), kv.data(),
            logits.data(), trace ? block_trace.data() : nullptr};
        detail::enqueue_worker(
            stream, model->view(), buffers, position, {-1, token_ids, token_index});
    }

    // Stream order makes the first unfinished event the ring's reclamation
    // boundary.
    void reclaim_completed()
    {
        while (!outstanding.empty()) {
            const auto& item = outstanding.front();
            const cudaError_t event_status = cudaEventQuery(item->event);
            if (event_status == cudaErrorNotReady)
                break;
            if (event_status != cudaSuccess) {
                valid = false;
                check(event_status, "query outstanding ticket");
            }
            ring_cursor.release(item->input_count);
            outstanding.erase(outstanding.begin());
        }
    }

    std::shared_ptr<Ticket::State> new_ticket(Ticket::State::Kind kind)
    {
        reclaim_completed();
        auto result = std::make_shared<Ticket::State>(kind);
        result->owner = this;
        return result;
    }

    std::size_t reserve_input(const std::shared_ptr<Ticket::State>& ticket, std::size_t count)
    {
        const std::size_t begin = ring_cursor.reserve(count);
        ticket->input_begin = begin;
        ticket->input_count = count;
        return begin;
    }

    void copy_input(std::size_t begin, std::size_t count)
    {
        const std::size_t tail = std::min(count, input_ring.size() - begin);
        cuda::copy_bytes(
            stream, cuda::std::span<const TokenId>(input_ring.data() + begin, tail), tokens);
        if (tail < count)
            cuda::copy_bytes(stream,
                cuda::std::span<const TokenId>(input_ring.data(), count - tail),
                cuda::std::span<TokenId>(tokens.data() + tail, count - tail));
    }

    void failed_input_enqueue(const std::shared_ptr<Ticket::State>& ticket)
    {
        stream.sync();
        reclaim_completed();
        ring_cursor.release(ticket->input_count);
        ticket->input_count = 0;
    }

    void finish_ticket(const std::shared_ptr<Ticket::State>& ticket, DiagnosticRequest request)
    {
        if (request.logits || request.kv || request.block_outputs)
            ticket->diagnostic.emplace();
        if (request.logits) {
            auto& host = ticket->diagnostic->logits.emplace(
                stream, cuda::pinned_default_memory_pool(), model->stack.vocab_size, cuda::no_init);
            cuda::copy_bytes(stream, logits, host);
        }
        if (request.kv) {
            auto& host = ticket->diagnostic->kv.emplace(
                stream, cuda::pinned_default_memory_pool(), model->kv_count, cuda::no_init);
            cuda::copy_bytes(stream, kv, host);
        }
        if (request.block_outputs) {
            auto& host =
                ticket->diagnostic->blocks.emplace(stream, cuda::pinned_default_memory_pool(),
                    model->stack.layers * model->hidden_size, cuda::no_init);
            cuda::copy_bytes(stream, block_trace, host);
        }
        check(cudaEventRecord(ticket->event, stream.get()), "record ticket");
        ticket->recorded = true;
        outstanding.push_back(ticket);
    }
};

SequenceContext::SequenceContext(std::shared_ptr<detail::UploadedPlan> model)
    : impl_(std::make_unique<Impl>(std::move(model)))
{
}
SequenceContext::~SequenceContext() = default;
SequenceContext::SequenceContext(SequenceContext&&) noexcept = default;
SequenceContext& SequenceContext::operator=(SequenceContext&&) noexcept = default;

Ticket SequenceContext::enqueue_prefill(
    std::span<const TokenId> token_ids, DiagnosticRequest diagnostics)
{
    impl_->require_ready();
    if (token_ids.empty())
        throw std::runtime_error("prefill requires at least one token");
    if (token_ids.size() > static_cast<std::size_t>(impl_->model->max_context))
        throw std::runtime_error("sequence context capacity reached");
    impl_->validate_capacity(static_cast<int>(token_ids.size()));
    for (const TokenId id : token_ids)
        impl_->validate_token(id);
    auto ticket = impl_->new_ticket(Ticket::State::Kind::prefill);
    const int start_position = impl_->queued_position;
    ticket->end_position = start_position + static_cast<int>(token_ids.size());
    const std::size_t begin = impl_->reserve_input(ticket, token_ids.size());
    try {
        const std::size_t tail = std::min(token_ids.size(), impl_->input_ring.size() - begin);
        std::memcpy(impl_->input_ring.data() + begin, token_ids.data(), tail * sizeof(TokenId));
        if (tail < token_ids.size())
            std::memcpy(impl_->input_ring.data(), token_ids.data() + tail,
                (token_ids.size() - tail) * sizeof(TokenId));
        impl_->copy_input(begin, token_ids.size());
        for (size_t i = 0; i < token_ids.size(); ++i)
            impl_->launch(start_position + static_cast<int>(i), impl_->tokens.data(),
                static_cast<int>(i), diagnostics.block_outputs && i + 1 == token_ids.size());
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        impl_->failed_input_enqueue(ticket);
        throw;
    }
    return Ticket(std::move(ticket));
}

Ticket SequenceContext::enqueue_decode(TokenId token_id, DiagnosticRequest diagnostics)
{
    impl_->require_ready();
    impl_->validate_token(token_id);
    impl_->validate_capacity(1);
    auto ticket = impl_->new_ticket(Ticket::State::Kind::decode);
    const int start_position = impl_->queued_position;
    ticket->end_position = start_position + 1;
    const std::size_t begin = impl_->reserve_input(ticket, 1);
    try {
        impl_->input_ring.data()[begin] = token_id;
        impl_->copy_input(begin, 1);
        impl_->launch(start_position, impl_->tokens.data(), 0, diagnostics.block_outputs);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        impl_->failed_input_enqueue(ticket);
        throw;
    }
    return Ticket(std::move(ticket));
}

Ticket SequenceContext::enqueue_generate(GenerateOptions options, DiagnosticRequest diagnostics)
{
    impl_->require_ready();
    if (impl_->queued_position == 0)
        throw std::runtime_error("generation requires a nonempty prefix");
    if (options.max_new_tokens <= 0)
        throw std::runtime_error("generation requires a positive token count");
    impl_->validate_capacity(options.max_new_tokens);
    if (options.eos_id)
        impl_->validate_token(*options.eos_id);
    auto ticket = impl_->new_ticket(Ticket::State::Kind::generate);
    ticket->end_position = impl_->queued_position + options.max_new_tokens;
    try {
        auto& generation = ticket->generation.emplace(
            impl_->stream, impl_->queued_position, options.max_new_tokens);
        check(cudaMemsetAsync(impl_->status.data(), 0, 2 * sizeof(int), impl_->stream.get()),
            "reset selection state");
        generation.compute_start.record(impl_->stream);
        const TokenId eos = options.eos_id.value_or(-1);
        for (int step = 0; step < generation.requested; ++step) {
            detail::enqueue_greedy_sampling(impl_->stream, impl_->logits.data(),
                impl_->model->stack.vocab_size, impl_->generated.data(), impl_->status.data(), step,
                eos);
            impl_->launch(generation.start_position + step, impl_->generated.data(), step,
                diagnostics.block_outputs);
        }
        generation.compute_end.record(impl_->stream);
        cuda::copy_bytes(impl_->stream,
            cuda::std::span<const TokenId>(impl_->generated.data(), generation.requested),
            generation.output);
        cuda::copy_bytes(impl_->stream, impl_->status, generation.status);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
        if (options.eos_id)
            impl_->pending_position = ticket;
    } catch (...) {
        impl_->valid = false;
        impl_->stream.sync();
        throw;
    }
    return Ticket(std::move(ticket));
}

Ticket SequenceContext::enqueue_reset()
{
    if (impl_->pending_position)
        throw std::runtime_error("collect generated result before reset");
    auto ticket = impl_->new_ticket(Ticket::State::Kind::reset);
    ticket->end_position = 0;
    try {
        check(cudaMemsetAsync(impl_->status.data(), 0, 2 * sizeof(int), impl_->stream.get()),
            "reset selection state");
        impl_->finish_ticket(ticket, {});
        impl_->queued_position = 0;
        impl_->valid = true;
    } catch (...) {
        impl_->valid = false;
        impl_->stream.sync();
        throw;
    }
    return Ticket(std::move(ticket));
}

bool SequenceContext::ready(const Ticket& ticket) const
{
    if (!ticket.state_ || ticket.state_->owner != impl_.get())
        throw std::runtime_error("ticket belongs to a different context");
    const cudaError_t status = cudaEventQuery(ticket.state_->event);
    if (status == cudaErrorNotReady)
        return false;
    if (status != cudaSuccess) {
        impl_->valid = false;
        check(status, "query ticket");
    }
    return true;
}

Completion SequenceContext::wait(const Ticket& ticket)
{
    if (!ticket.state_ || ticket.state_->owner != impl_.get())
        throw std::runtime_error("ticket belongs to a different context");
    try {
        check(cudaEventSynchronize(ticket.state_->event), "wait for ticket");
    } catch (...) {
        impl_->valid = false;
        throw;
    }
    Completion result{};
    result.position = ticket.state_->end_position;
    if (ticket.state_->generation) {
        const auto& generation = *ticket.state_->generation;
        result.compute_us = cuda::std::chrono::duration<double, cuda::std::micro>(
            generation.compute_end - generation.compute_start)
                                .count();
        const int count = generation.status.data()[0];
        if (count < 0 || count > generation.requested)
            throw std::runtime_error("invalid generated token count");
        result.generated.assign(generation.output.data(), generation.output.data() + count);
        result.position = generation.start_position + count;
        result.stop_reason = generation.status.data()[1] ? StopReason::eos : StopReason::limit;
        if (impl_->pending_position == ticket.state_) {
            impl_->queued_position = result.position;
            impl_->pending_position.reset();
        }
    }
    return result;
}

Diagnostics SequenceContext::collect(const Ticket& ticket)
{
    wait(ticket);
    Diagnostics result;
    if (ticket.state_->diagnostic) {
        const auto& diagnostic = *ticket.state_->diagnostic;
        if (diagnostic.logits)
            result.logits.assign(
                diagnostic.logits->data(), diagnostic.logits->data() + diagnostic.logits->size());
        if (diagnostic.kv)
            result.kv.assign(diagnostic.kv->data(), diagnostic.kv->data() + diagnostic.kv->size());
        if (diagnostic.blocks)
            result.block_outputs.assign(
                diagnostic.blocks->data(), diagnostic.blocks->data() + diagnostic.blocks->size());
    }
    return result;
}

} // namespace tileinfer
