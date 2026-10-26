#include "detail/activation_buffer.cuh"
#include "detail/cleanup.hpp"
#include "detail/cuda_check.hpp"
#include "detail/input_ring_cursor.hpp"
#include "detail/sequence_test_access.cuh"
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
#include <deque>
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
    detail::ActivationBuffer input;
    detail::ActivationBuffer output;
    detail::ActivationBuffer scratch;
    detail::ActivationBuffer kv;
    detail::ActivationBuffer attention_scores;
    cuda::device_buffer<float> attention_partials;
    cuda::device_buffer<float> logits;
    cuda::device_buffer<Candidate> candidates; // the last token's greedy candidates
    std::optional<cuda::device_buffer<float>> block_trace;
    cuda::device_buffer<TokenId> tokens;
    cuda::device_buffer<TokenId> generated;
    cuda::device_buffer<int> status;
    cuda::host_buffer<TokenId> input_ring;
    detail::InputRingCursor ring_cursor;
    int queued_position = 0;
    std::shared_ptr<Ticket::State> pending_position;
    bool valid = true;
    std::deque<std::shared_ptr<Ticket::State>> outstanding;

    explicit Impl(std::shared_ptr<detail::UploadedPlan> value)
        : model(std::move(value))
        , stream(model->device)
        , pool(cuda::device_default_memory_pool(model->device))
        , input(stream, pool, model->hidden_size, model->weight_dtype)
        , output(stream, pool, model->hidden_size, model->weight_dtype)
        , scratch(stream, pool, model->scratch_count, model->weight_dtype)
        , kv(stream, pool, model->kv_count, model->weight_dtype)
        , attention_scores(stream, pool, model->attention_count, model->weight_dtype)
        , attention_partials(stream, pool, model->partial_count, cuda::no_init)
        , logits(stream, pool, model->stack.vocab_size, cuda::no_init)
        , candidates(stream, pool, model->selection_slots, cuda::no_init)
        , tokens(stream, pool, model->max_context, cuda::no_init)
        , generated(stream, pool, model->max_context, cuda::no_init)
        , status(stream, pool, detail::kStatusWords, cuda::no_init)
        , input_ring(stream, cuda::pinned_default_memory_pool(),
              input_ring_size(model->max_context), cuda::no_init)
        , ring_cursor(input_ring.size())
    {
    }

    ~Impl()
    {
        detail::cleanup_noexcept([&] { stream.sync(); });
    }

    void require_ready()
    {
        if (!valid)
            throw std::runtime_error("sequence context is invalid; reset required");
        resolve_pending();
    }

    // Only the device knows where a generation stopped: at EOS, or earlier
    // at a device error. A later request waits for it here, so a caller that
    // dropped or never waited on its ticket can still continue the sequence.
    void resolve_pending()
    {
        if (!pending_position)
            return;
        const auto ticket = pending_position;
        try {
            check(cudaEventSynchronize(ticket->event), "wait for pending generation");
            queued_position = generation_end(*ticket->generation);
        } catch (...) {
            valid = false;
            throw;
        }
        pending_position.reset();
    }

    // The position after a completed generation. Throws for a device error
    // that stopped it.
    static int generation_end(const Ticket::State::Generation& generation)
    {
        const int* status = generation.status.data();
        switch (status[detail::kStatusError]) {
        case detail::no_error:
            break;
        case detail::non_finite_logits:
            throw std::runtime_error("generation stopped: non-finite logits");
        case detail::token_out_of_range:
            throw std::runtime_error("generation stopped: selected token outside vocabulary");
        default:
            throw std::runtime_error("generation stopped: unknown device error");
        }
        const int count = status[detail::kStatusCount];
        if (count < 0 || count > generation.requested)
            throw std::runtime_error("invalid generated token count");
        return generation.start_position + count;
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

    // Decodes `steps` positions from `position` in launches of at most
    // kMaxLaunchSteps. Step s decodes token_ids[s], or generates when
    // selection is set; a trace keeps the last step's block outputs.
    void launch(int position, const TokenId* token_ids, int steps, bool trace,
        detail::Selection selection = {})
    {
        if (trace && !block_trace)
            block_trace.emplace(stream, pool,
                static_cast<std::size_t>(model->stack.layers) * model->hidden_size, cuda::no_init);
        const detail::WorkerBuffers buffers{input.data(), output.data(), scratch.data(), kv.data(),
            logits.data(), trace ? block_trace->data() : nullptr, attention_scores.data(),
            attention_partials.data(), candidates.data()};
        for (int first = 0; first < steps; first += detail::kMaxLaunchSteps) {
            const int count = std::min(detail::kMaxLaunchSteps, steps - first);
            selection.first = first;
            detail::enqueue_worker(stream, model->view(), buffers, position + first,
                {-1, token_ids == nullptr ? nullptr : token_ids + first, count}, selection);
        }
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
            outstanding.pop_front();
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
            const auto& generation = ticket->generation;
            detail::enqueue_zero_unused_kv(stream, kv.data(), model->weight_dtype, model->kv_count,
                model->stack.layers, model->stack.layer_kv_stride,
                model->kv_heads * model->head_dim, model->max_context,
                generation ? generation->start_position : ticket->end_position,
                generation ? status.data() : nullptr);
            kv.copy_to(stream, host);
        }
        if (request.block_outputs) {
            auto& host =
                ticket->diagnostic->blocks.emplace(stream, cuda::pinned_default_memory_pool(),
                    model->stack.layers * model->hidden_size, cuda::no_init);
            cuda::copy_bytes(stream, *block_trace, host);
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
std::size_t detail::SequenceTestAccess::input_capacity(const SequenceContext& context)
{
    return context.impl_->input_ring.size();
}

cuda::stream_ref detail::SequenceTestAccess::stream(SequenceContext& context)
{
    return context.impl_->stream;
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
        impl_->launch(start_position, impl_->tokens.data(), static_cast<int>(token_ids.size()),
            diagnostics.block_outputs);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        detail::cleanup_noexcept([&] { impl_->failed_input_enqueue(ticket); });
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
        impl_->launch(start_position, impl_->tokens.data(), 1, diagnostics.block_outputs);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
    } catch (...) {
        impl_->valid = false;
        detail::cleanup_noexcept([&] { impl_->failed_input_enqueue(ticket); });
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
        check(cudaMemsetAsync(
                  impl_->status.data(), 0, detail::kStatusWords * sizeof(int), impl_->stream.get()),
            "reset selection state");
        // Steps after a stop write no token; their slots are still copied back.
        check(cudaMemsetAsync(impl_->generated.data(), 0xff,
                  static_cast<std::size_t>(generation.requested) * sizeof(TokenId),
                  impl_->stream.get()),
            "reset generated tokens");
        generation.compute_start.record(impl_->stream);
        impl_->launch(generation.start_position, nullptr, generation.requested,
            diagnostics.block_outputs,
            {impl_->generated.data(), impl_->status.data(), 0, options.eos_id.value_or(-1)});
        generation.compute_end.record(impl_->stream);
        cuda::copy_bytes(impl_->stream,
            cuda::std::span<const TokenId>(impl_->generated.data(), generation.requested),
            generation.output);
        cuda::copy_bytes(impl_->stream, impl_->status, generation.status);
        impl_->finish_ticket(ticket, diagnostics);
        impl_->queued_position = ticket->end_position;
        impl_->pending_position = ticket;
    } catch (...) {
        impl_->valid = false;
        detail::cleanup_noexcept([&] { impl_->stream.sync(); });
        throw;
    }
    return Ticket(std::move(ticket));
}

Ticket SequenceContext::enqueue_reset()
{
    // A reset discards the sequence, so a pending EOS position no longer matters.
    impl_->pending_position.reset();
    auto ticket = impl_->new_ticket(Ticket::State::Kind::reset);
    ticket->end_position = 0;
    try {
        check(cudaMemsetAsync(
                  impl_->status.data(), 0, detail::kStatusWords * sizeof(int), impl_->stream.get()),
            "reset selection state");
        impl_->finish_ticket(ticket, {});
        impl_->queued_position = 0;
        impl_->valid = true;
    } catch (...) {
        impl_->valid = false;
        detail::cleanup_noexcept([&] { impl_->stream.sync(); });
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
        const bool pending = impl_->pending_position == ticket.state_;
        try {
            result.position = Impl::generation_end(generation);
        } catch (...) {
            // A stopped sequence needs a reset, unless one already discarded it.
            if (pending)
                impl_->valid = false;
            throw;
        }
        const int count = result.position - generation.start_position;
        result.generated.assign(generation.output.data(), generation.output.data() + count);
        result.stop_reason =
            generation.status.data()[detail::kStatusEos] ? StopReason::eos : StopReason::limit;
        if (pending) {
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
