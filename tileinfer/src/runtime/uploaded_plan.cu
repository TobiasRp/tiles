#include "detail/cleanup.hpp"
#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>
#include <cuda/algorithm>

#include <memory>
#include <stdexcept>

namespace tileinfer::detail {

UploadedPlan::UploadedPlan(const LoadedPlan& plan)
    : weights(stream, pool, plan.weights.size(), cuda::no_init)
    , instructions(stream, pool, plan.instructions.size(), cuda::no_init)
    , streams(stream, pool, plan.streams.size(), cuda::no_init)
    , rotary_frequencies(stream, pool, plan.rotary_frequencies.size(), cuda::no_init)
    , layout(plan.layout)
    , hidden_size(plan.hidden_size)
    , intermediate_size(plan.intermediate_size)
    , query_heads(plan.query_heads)
    , kv_heads(plan.kv_heads)
    , head_dim(plan.head_dim)
    , workers(plan.workers)
    , max_context(plan.max_context)
    , attention_chunks(plan.attention_chunks)
    , scratch_count(plan.scratch_bytes / element_bytes(plan.weight_dtype))
    , kv_count(plan.kv_bytes / element_bytes(plan.weight_dtype))
    , attention_count(plan.attention_bytes / element_bytes(plan.weight_dtype))
    , attention_partial_count(plan.attention_partial_bytes / sizeof(float))
    , numeric(plan.numeric)
    , stack(plan.stack)
    , weight_dtype(plan.weight_dtype)
{
    for (const auto id : {TensorId::q_rot, TensorId::attention, TensorId::r, TensorId::product}) {
        const std::size_t count = id == TensorId::product ? intermediate_size : hidden_size;
        scratch_regions.emplace_back(plan.tensors[tensor_index(id)].offset, count);
    }
    try {
        cuda::copy_bytes(stream, plan.weights, weights);
        cuda::copy_bytes(stream, plan.instructions, instructions);
        cuda::copy_bytes(stream, plan.streams, streams);
        cuda::copy_bytes(stream, plan.rotary_frequencies, rotary_frequencies);
        stream.sync();
    } catch (...) {
        cleanup_noexcept([&] { stream.sync(); });
        throw;
    }
}

WorkerProgramView UploadedPlan::view() const
{
    return {instructions.data(), streams.data(), layout, weights.data(), rotary_frequencies.data(),
        workers, hidden_size, intermediate_size, query_heads, kv_heads, head_dim, max_context,
        attention_chunks, numeric, stack, weight_dtype};
}

} // namespace tileinfer::detail

namespace tileinfer {

DecoderModel::DecoderModel(const LoadedPlan& plan)
{
    if (!plan.stack.full_stack)
        throw std::runtime_error("generation requires a decoder stack package");
    impl_ = std::make_shared<detail::UploadedPlan>(plan);
}

DecoderModel::~DecoderModel() = default;
DecoderModel::DecoderModel(DecoderModel&&) noexcept = default;
DecoderModel& DecoderModel::operator=(DecoderModel&&) noexcept = default;

SequenceContext DecoderModel::create_context() const
{
    return SequenceContext(impl_);
}

} // namespace tileinfer
