#pragma once

#include "detail/launch.cuh"

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace tileinfer::detail {

// Owns a validated plan's immutable GPU data before any context can use it.
class UploadedPlan
{
  public:
    explicit UploadedPlan(const LoadedPlan& plan);

    [[nodiscard]] WorkerProgramView view() const;

    cuda::device_ref device{0};
    cuda::stream stream{device};
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(device);
    cuda::device_buffer<std::uint8_t> weights;
    cuda::device_buffer<TaskRecord> tasks;
    cuda::device_buffer<WorkerTaskRange> ranges;
    cuda::device_buffer<TensorRecord> tensors;
    cuda::device_buffer<float> rotary_frequencies;
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    int workers;
    int waves;
    int max_context;
    std::size_t scratch_count;
    std::size_t kv_count;
    std::size_t attention_count;
    bool matrix_mma;
    std::vector<std::pair<std::size_t, std::size_t>> scratch_regions;
    NumericConfig numeric;
    StackConfig stack;
    WeightDtype weight_dtype;
};

} // namespace tileinfer::detail
