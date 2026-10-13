#include "detail/cuda_check.hpp"
#include "detail/uploaded_plan.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>
#include <cuda/algorithm>
#include <cuda/std/span>

#include <memory>
#include <stdexcept>
#include <string>

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
    auto copy = [this](auto source, auto destination, const char* stage) {
        try {
            cuda::copy_bytes(stream, source, destination);
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(stage) + ": " + error.what());
        }
    };
    copy(cuda::std::span<const std::uint8_t>(plan.weights.data(), plan.weights.size()),
        cuda::std::span<std::uint8_t>(weights.data(), weights.size()), "copy model weights");
    copy(cuda::std::span<const TaskRecord>(plan.tasks.data(), plan.tasks.size()),
        cuda::std::span<TaskRecord>(tasks.data(), tasks.size()), "copy model tasks");
    copy(cuda::std::span<const WorkerTaskRange>(plan.ranges.data(), plan.ranges.size()),
        cuda::std::span<WorkerTaskRange>(ranges.data(), ranges.size()), "copy model ranges");
    copy(cuda::std::span<const TensorRecord>(plan.tensors.data(), plan.tensors.size()),
        cuda::std::span<TensorRecord>(tensors.data(), tensors.size()), "copy model tensors");
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
