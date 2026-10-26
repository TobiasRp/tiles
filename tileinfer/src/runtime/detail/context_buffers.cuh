#pragma once

#include "detail/activation_buffer.cuh"
#include "detail/uploaded_plan.cuh"

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <optional>

namespace tileinfer::detail {

// One sequence's mutable device storage for worker launches. A block package
// has no vocabulary, so its logits and candidates hold one unused element.
class ContextBuffers
{
  public:
    ContextBuffers(
        cuda::stream& stream, cuda::device_memory_pool_ref& pool, const UploadedPlan& model)
        : input(stream, pool, model.shape.hidden_size, model.weight_dtype)
        , output(stream, pool, model.shape.hidden_size, model.weight_dtype)
        , scratch(stream, pool, model.scratch_count, model.weight_dtype)
        , kv(stream, pool, model.kv_count, model.weight_dtype)
        , attention_scores(stream, pool, model.attention_count, model.weight_dtype)
        , attention_partials(stream, pool, model.partial_count, cuda::no_init)
        , logits(stream, pool, std::max(1, model.stack.vocab_size), cuda::no_init)
        , candidates(stream, pool, std::max(1, model.selection_slots), cuda::no_init)
        , stream_(stream)
        , pool_(pool)
        , block_trace_count_(static_cast<std::size_t>(model.stack.layers) * model.shape.hidden_size)
    {
    }

    // Allocates the per-layer block outputs on first use.
    void enable_block_trace()
    {
        if (!block_trace)
            block_trace.emplace(stream_, pool_, block_trace_count_, cuda::no_init);
    }

    // The buffers of one launch; trace_blocks requires enable_block_trace().
    [[nodiscard]] WorkerBuffers worker_buffers(bool trace_blocks)
    {
        return {input.data(), output.data(), scratch.data(), kv.data(), logits.data(),
            trace_blocks ? block_trace->data() : nullptr, attention_scores.data(),
            attention_partials.data(), candidates.data()};
    }

    ActivationBuffer input;
    ActivationBuffer output;
    ActivationBuffer scratch;
    ActivationBuffer kv;
    ActivationBuffer attention_scores;
    cuda::device_buffer<float> attention_partials; // FP32 attention partials, then FFN partials
    cuda::device_buffer<float> logits;
    cuda::device_buffer<Candidate> candidates; // the last token's greedy candidates
    std::optional<cuda::device_buffer<float>> block_trace;

  private:
    cuda::stream& stream_;
    cuda::device_memory_pool_ref& pool_;
    std::size_t block_trace_count_;
};

} // namespace tileinfer::detail
