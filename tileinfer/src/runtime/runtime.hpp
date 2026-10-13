#pragma once

#include "plan.hpp"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace tileinfer {

namespace detail {
class UploadedPlan;
struct SequenceTestAccess;
} // namespace detail

// Specifies values to retain before the next token overwrites live buffers.
struct DiagnosticRequest
{
    bool logits = false;
    bool kv = false;
    bool block_outputs = false;
};

// Host copies of optional final-step diagnostic values.
struct Diagnostics
{
    std::vector<float> logits;
    std::vector<float> kv;
    std::vector<float> block_outputs;
};

// Bounds one generation request and selects its token policy.
struct GenerateOptions
{
    int max_new_tokens;
    std::optional<TokenId> eos_id;
};

// Explains why the device stopped selecting new tokens.
enum class StopReason
{
    none,
    limit,
    eos,
};

// Contains the compact result after a queued request completes.
struct Completion
{
    std::vector<TokenId> generated;
    int position = 0;
    StopReason stop_reason = StopReason::none;
    // CUDA time for selection plus decode, excluding prompt setup and readback.
    double compute_us = 0.0;
};

// Retains one queued operation's event and result storage until completion.
// Its originating context must outlive the ticket, including after completion.
class Ticket
{
  public:
    Ticket() = default;
    ~Ticket();
    Ticket(Ticket&&) noexcept;
    Ticket& operator=(Ticket&&) noexcept;
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;

  private:
    struct State;
    std::shared_ptr<State> state_;
    explicit Ticket(std::shared_ptr<State> state);
    friend class SequenceContext;
};

class SequenceContext;

// Uploads one validated model for use by one or more sequence contexts.
class DecoderModel
{
  public:
    explicit DecoderModel(const LoadedPlan& plan);
    ~DecoderModel();
    DecoderModel(DecoderModel&&) noexcept;
    DecoderModel& operator=(DecoderModel&&) noexcept;
    DecoderModel(const DecoderModel&) = delete;
    DecoderModel& operator=(const DecoderModel&) = delete;
    SequenceContext create_context() const;

  private:
    std::shared_ptr<detail::UploadedPlan> impl_;
    friend class SequenceContext;
};

// Queues work for one sequence while retaining its stream and K/V cache.
// Calls on the same context must be serialized by the caller.
class SequenceContext
{
  public:
    ~SequenceContext();
    SequenceContext(SequenceContext&&) noexcept;
    SequenceContext& operator=(SequenceContext&&) noexcept;
    SequenceContext(const SequenceContext&) = delete;
    SequenceContext& operator=(const SequenceContext&) = delete;

    Ticket enqueue_prefill(std::span<const TokenId> token_ids, DiagnosticRequest diagnostics = {});
    Ticket enqueue_decode(TokenId token_id, DiagnosticRequest diagnostics = {});
    Ticket enqueue_generate(GenerateOptions options, DiagnosticRequest diagnostics = {});
    Ticket enqueue_reset();
    [[nodiscard]] bool ready(const Ticket& ticket) const;
    Completion wait(const Ticket& ticket);
    Diagnostics collect(const Ticket& ticket);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit SequenceContext(std::shared_ptr<detail::UploadedPlan> model);
    friend class DecoderModel;
    friend struct detail::SequenceTestAccess;
};

} // namespace tileinfer
