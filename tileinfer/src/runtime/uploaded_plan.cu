#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>
#include <cuda/algorithm>

#include <memory>
#include <stdexcept>

namespace tileinfer::detail {

UploadedPlan::UploadedPlan(const LoadedPlan& plan)
    : weights(stream, pool, plan.data().weights.size(), cuda::no_init)
    , tasks(stream, pool, plan.data().tasks.size(), cuda::no_init)
    , ranges(stream, pool, plan.data().ranges.size(), cuda::no_init)
    , tensors(stream, pool, plan.data().tensors.size(), cuda::no_init)
    , rotary_frequencies(stream, pool, plan.data().rotary_frequencies.size(), cuda::no_init)
    , hidden_size(plan.data().hidden_size)
    , intermediate_size(plan.data().intermediate_size)
    , kv_heads(plan.data().kv_heads)
    , head_dim(plan.data().head_dim)
    , workers(plan.data().workers)
    , waves(plan.data().wave_count)
    , max_context(plan.data().max_context)
    , scratch_count(plan.data().scratch_bytes / sizeof(float))
    , kv_count(plan.data().kv_bytes / sizeof(float))
    , numeric(plan.data().numeric)
    , stack(plan.data().stack)
    , weight_dtype(plan.data().weight_dtype)
{
    for (const auto id :
        {TensorId::u, TensorId::q, TensorId::k, TensorId::v, TensorId::q_rot, TensorId::attention,
            TensorId::r, TensorId::z, TensorId::gate, TensorId::up, TensorId::product}) {
        const std::size_t count =
            id == TensorId::k || id == TensorId::v ? kv_heads * head_dim
            : id == TensorId::gate || id == TensorId::up || id == TensorId::product
                ? intermediate_size
                : hidden_size;
        scratch_regions.emplace_back(plan.data().tensors[tensor_index(id)].offset, count);
    }
    cuda::copy_bytes(stream, plan.data().weights, weights);
    cuda::copy_bytes(stream, plan.data().tasks, tasks);
    cuda::copy_bytes(stream, plan.data().ranges, ranges);
    cuda::copy_bytes(stream, plan.data().tensors, tensors);
    cuda::copy_bytes(stream, plan.data().rotary_frequencies, rotary_frequencies);
    stream.sync();
}

WorkerProgramView UploadedPlan::view() const
{
    return {tasks.data(), ranges.data(), tensors.data(), weights.data(), rotary_frequencies.data(),
        waves, workers, hidden_size, intermediate_size, kv_heads, head_dim, numeric, stack,
        weight_dtype};
}

} // namespace tileinfer::detail

namespace tileinfer {

DecoderModel::DecoderModel(const LoadedPlan& plan)
{
    if (!plan.data().stack.full_stack)
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
