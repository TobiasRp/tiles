#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

namespace tileinfer::detail {

UploadedPlan::UploadedPlan(const LoadedPlan& plan)
    : weights(stream, pool, plan.weights.size(), cuda::no_init)
    , tasks(stream, pool, plan.tasks.size(), cuda::no_init)
    , ranges(stream, pool, plan.ranges.size(), cuda::no_init)
    , tensors(stream, pool, plan.tensors.size(), cuda::no_init)
    , hidden_size(plan.hidden_size)
    , intermediate_size(plan.intermediate_size)
    , kv_heads(plan.kv_heads)
    , head_dim(plan.head_dim)
    , workers(plan.workers)
    , waves(plan.wave_count)
    , max_context(plan.max_context)
    , scratch_count(plan.scratch_bytes / sizeof(float))
    , kv_count(plan.kv_bytes / sizeof(float))
    , numeric(plan.numeric)
    , stack(plan.stack)
    , weight_dtype(plan.weight_dtype)
{
    stream.sync();
    check(cudaMemcpyAsync(weights.data(), plan.weights.data(), plan.weights.size(),
              cudaMemcpyHostToDevice, stream.get()),
        "copy model weights");
    check(cudaMemcpyAsync(tasks.data(), plan.tasks.data(), plan.tasks.size() * sizeof(TaskRecord),
              cudaMemcpyHostToDevice, stream.get()),
        "copy model tasks");
    check(cudaMemcpyAsync(ranges.data(), plan.ranges.data(), plan.ranges.size() * sizeof(Range),
              cudaMemcpyHostToDevice, stream.get()),
        "copy model ranges");
    check(cudaMemcpyAsync(tensors.data(), plan.tensors.data(),
              plan.tensors.size() * sizeof(TensorRecord), cudaMemcpyHostToDevice, stream.get()),
        "copy model tensors");
    stream.sync();
}

WorkerProgramView UploadedPlan::view() const
{
    return {tasks.data(), ranges.data(), tensors.data(), weights.data(), waves, workers,
        hidden_size, intermediate_size, kv_heads, head_dim, numeric, stack, weight_dtype};
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
