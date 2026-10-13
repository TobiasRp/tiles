#include "detail/cuda_check.hpp"
#include "detail/ticket_state.cuh"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tileinfer {
using detail::check;

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
    {
        stream.sync();
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

    std::shared_ptr<Ticket::State> new_ticket(Ticket::State::Kind kind)
    {
        auto result = std::make_shared<Ticket::State>(kind);
        result->owner = this;
        std::erase_if(outstanding, [this](const auto& item) {
            const cudaError_t status = cudaEventQuery(item->event);
            if (status == cudaSuccess)
                return true;
            if (status == cudaErrorNotReady)
                return false;
            valid = false;
            check(status, "query outstanding ticket");
            return false;
        });
        return result;
    }

    void finish_ticket(const std::shared_ptr<Ticket::State>& ticket, DiagnosticRequest diagnostics)
    {
        if (diagnostics.logits) {
            ticket->logits_host.allocate(model->stack.vocab_size);
            check(cudaMemcpyAsync(ticket->logits_host.data(), logits.data(),
                      ticket->logits_host.size() * sizeof(float), cudaMemcpyDeviceToHost,
                      stream.get()),
                "copy diagnostic logits");
        }
        if (diagnostics.kv) {
            ticket->kv_host.allocate(model->kv_count);
            check(cudaMemcpyAsync(ticket->kv_host.data(), kv.data(),
                      ticket->kv_host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.get()),
                "copy diagnostic cache");
        }
        if (diagnostics.block_outputs) {
            ticket->blocks_host.allocate(model->stack.layers * model->hidden_size);
            check(cudaMemcpyAsync(ticket->blocks_host.data(), block_trace.data(),
                      ticket->blocks_host.size() * sizeof(float), cudaMemcpyDeviceToHost,
                      stream.get()),
                "copy diagnostic blocks");
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
    impl_->validate_capacity(static_cast<int>(token_ids.size()));
    for (const TokenId id : token_ids)
        impl_->validate_token(id);
    auto ticket = impl_->new_ticket(Ticket::State::Kind::prefill);
    ticket->start_position = impl_->queued_position;
    ticket->end_position = impl_->queued_position + static_cast<int>(token_ids.size());
    try {
        ticket->input_host.allocate(token_ids.size());
        std::memcpy(ticket->input_host.data(), token_ids.data(), token_ids.size_bytes());
        check(cudaMemcpyAsync(impl_->tokens.data(), ticket->input_host.data(),
                  token_ids.size_bytes(), cudaMemcpyHostToDevice, impl_->stream.get()),
            "copy prefill tokens");
        for (size_t i = 0; i < token_ids.size(); ++i)
            impl_->launch(ticket->start_position + static_cast<int>(i), impl_->tokens.data(),
                static_cast<int>(i), diagnostics.block_outputs && i + 1 == token_ids.size());
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        impl_->stream.sync();
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
    ticket->start_position = impl_->queued_position;
    ticket->end_position = impl_->queued_position + 1;
    try {
        ticket->input_host.allocate(1);
        ticket->input_host[0] = token_id;
        check(cudaMemcpyAsync(impl_->tokens.data(), ticket->input_host.data(), sizeof(TokenId),
                  cudaMemcpyHostToDevice, impl_->stream.get()),
            "copy decode token");
        impl_->launch(ticket->start_position, impl_->tokens.data(), 0, diagnostics.block_outputs);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        impl_->stream.sync();
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
    ticket->start_position = impl_->queued_position;
    ticket->end_position = impl_->queued_position + options.max_new_tokens;
    ticket->requested = options.max_new_tokens;
    try {
        ticket->output_host.allocate(ticket->requested);
        ticket->status_host.allocate(2);
        check(cudaMemsetAsync(impl_->status.data(), 0, 2 * sizeof(int), impl_->stream.get()),
            "reset selection state");
        ticket->compute_start = std::make_unique<cuda::timed_event>(impl_->stream);
        ticket->compute_end = std::make_unique<cuda::timed_event>(impl_->stream);
        ticket->compute_start->record(impl_->stream);
        const TokenId eos = options.eos_id.value_or(-1);
        for (int step = 0; step < ticket->requested; ++step) {
            detail::enqueue_greedy(impl_->stream, impl_->logits.data(),
                impl_->model->stack.vocab_size, impl_->generated.data(), impl_->status.data(), step,
                eos);
            impl_->launch(ticket->start_position + step, impl_->generated.data(), step,
                diagnostics.block_outputs);
        }
        ticket->compute_end->record(impl_->stream);
        check(cudaMemcpyAsync(ticket->output_host.data(), impl_->generated.data(),
                  ticket->requested * sizeof(TokenId), cudaMemcpyDeviceToHost, impl_->stream.get()),
            "copy generated IDs");
        check(cudaMemcpyAsync(ticket->status_host.data(), impl_->status.data(), 2 * sizeof(int),
                  cudaMemcpyDeviceToHost, impl_->stream.get()),
            "copy generation status");
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
    if (ticket.state_->kind == Ticket::State::Kind::generate) {
        result.compute_us = cuda::std::chrono::duration<double, cuda::std::micro>(
            *ticket.state_->compute_end - *ticket.state_->compute_start)
                                .count();
        const int count = ticket.state_->status_host[0];
        if (count < 0 || count > ticket.state_->requested)
            throw std::runtime_error("invalid generated token count");
        result.generated.assign(
            ticket.state_->output_host.data(), ticket.state_->output_host.data() + count);
        result.position = ticket.state_->start_position + count;
        result.stop_reason = ticket.state_->status_host[1] ? StopReason::eos : StopReason::limit;
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
    if (ticket.state_->logits_host.data())
        result.logits.assign(ticket.state_->logits_host.data(),
            ticket.state_->logits_host.data() + ticket.state_->logits_host.size());
    if (ticket.state_->kv_host.data())
        result.kv.assign(ticket.state_->kv_host.data(),
            ticket.state_->kv_host.data() + ticket.state_->kv_host.size());
    if (ticket.state_->blocks_host.data())
        result.block_outputs.assign(ticket.state_->blocks_host.data(),
            ticket.state_->blocks_host.data() + ticket.state_->blocks_host.size());
    return result;
}

} // namespace tileinfer
