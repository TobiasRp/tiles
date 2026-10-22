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
    , tasks(stream, pool, plan.tasks.size(), cuda::no_init)
    , ranges(stream, pool, plan.ranges.size(), cuda::no_init)
    , tensors(stream, pool, plan.tensors.size(), cuda::no_init)
    , rotary_frequencies(stream, pool, plan.rotary_frequencies.size(), cuda::no_init)
    , hidden_size(plan.hidden_size)
    , intermediate_size(plan.intermediate_size)
    , kv_heads(plan.kv_heads)
    , head_dim(plan.head_dim)
    , workers(plan.workers)
    , waves(plan.wave_count)
    , max_context(plan.max_context)
    , scratch_count(plan.scratch_bytes / element_bytes(plan.weight_dtype))
    , kv_count(plan.kv_bytes / element_bytes(plan.weight_dtype))
    , attention_count(plan.attention_bytes / element_bytes(plan.weight_dtype))
    , numeric(plan.numeric)
    , stack(plan.stack)
    , weight_dtype(plan.weight_dtype)
{
    for (const auto id :
        {TensorId::u, TensorId::q, TensorId::k, TensorId::v, TensorId::q_rot, TensorId::attention,
            TensorId::r, TensorId::z, TensorId::gate, TensorId::up, TensorId::product}) {
        const std::size_t count =
            id == TensorId::k || id == TensorId::v ? kv_heads * head_dim
            : id == TensorId::gate || id == TensorId::up || id == TensorId::product
                ? intermediate_size
                : hidden_size;
        scratch_regions.emplace_back(plan.tensors[tensor_index(id)].offset, count);
    }
    try {
        cuda::copy_bytes(stream, plan.weights, weights);
        cuda::copy_bytes(stream, plan.tasks, tasks);
        cuda::copy_bytes(stream, plan.ranges, ranges);
        cuda::copy_bytes(stream, plan.tensors, tensors);
        cuda::copy_bytes(stream, plan.rotary_frequencies, rotary_frequencies);
        stream.sync();
    } catch (...) {
        cleanup_noexcept([&] { stream.sync(); });
        throw;
    }
}

WorkerProgramView UploadedPlan::view() const
{
    return {tasks.data(), ranges.data(), tensors.data(), weights.data(), rotary_frequencies.data(),
        waves, workers, hidden_size, intermediate_size, kv_heads, head_dim, numeric, stack,
        weight_dtype, max_context};
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
