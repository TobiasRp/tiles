#pragma once

#include "detail/launch.cuh"

#include <cuda_bf16.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>

#include <cstdint>
#include <vector>

namespace tileinfer::detail {

void enqueue_widen(cuda::stream& stream, const void* source, float* destination, std::size_t count,
    WeightDtype dtype);

// Byte allocation with checked model precision; host diagnostics always widen.
class ActivationBuffer
{
  public:
    ActivationBuffer(cuda::stream& stream, cuda::device_memory_pool_ref& pool, std::size_t count,
        WeightDtype dtype)
        : stream_(stream)
        , pool_(pool)
        , count_(count)
        , dtype_(dtype)
        , bytes_(stream, pool, count * element_bytes(dtype), cuda::no_init)
    {
    }

    void* data() { return bytes_.data(); }
    const void* data() const { return bytes_.data(); }
    std::size_t size() const { return count_; }

    template <typename Host>
    void copy_to(cuda::stream& stream, Host& host) const
    {
        if (dtype_ == WeightDtype::float32) {
            cuda::copy_bytes(
                stream, cuda::std::span(static_cast<const float*>(data()), count_), host);
        } else {
            cuda::device_buffer<float> widened(stream, pool_, count_, cuda::no_init);
            enqueue_widen(stream, data(), widened.data(), count_, dtype_);
            cuda::copy_bytes(stream, widened, host);
            // Destruction queues deallocation after the copy on this stream.
        }
    }

    // Used only by the blocking block-test adapter, never by generation.
    void upload(const std::vector<float>& host)
    {
        if (dtype_ == WeightDtype::float32) {
            cuda::copy_bytes(stream_, host, cuda::std::span(static_cast<float*>(data()), count_));
            stream_.sync();
        } else {
            std::vector<__nv_bfloat16> values(host.size());
            for (std::size_t i = 0; i < host.size(); ++i)
                values[i] = __float2bfloat16_rn(host[i]);
            cuda::copy_bytes(
                stream_, values, cuda::std::span(static_cast<__nv_bfloat16*>(data()), count_));
            stream_.sync();
        }
    }

  private:
    cuda::stream& stream_;
    cuda::device_memory_pool_ref& pool_;
    std::size_t count_;
    WeightDtype dtype_;
    cuda::device_buffer<std::uint8_t> bytes_;
};

} // namespace tileinfer::detail
