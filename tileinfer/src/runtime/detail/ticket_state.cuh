#pragma once

#include "detail/cuda_check.hpp"
#include "runtime.hpp"

#include <cuda/__event/timed_event.h>

#include <cstddef>
#include <memory>

namespace tileinfer::detail {

// Keeps host storage pinned until its stream-ordered transfer has finished.
template <typename T>
class PinnedBuffer
{
  public:
    PinnedBuffer() = default;
    ~PinnedBuffer()
    {
        if (data_ != nullptr)
            cudaFreeHost(data_);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(std::size_t count)
    {
        if (data_ != nullptr)
            throw std::runtime_error("pinned buffer already allocated");
        if (count == 0)
            return;
        void* storage = nullptr;
        check(cudaMallocHost(&storage, count * sizeof(T)), "allocate pinned result");
        data_ = static_cast<T*>(storage);
        size_ = count;
    }

    [[nodiscard]] T* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }
    T& operator[](std::size_t index) const { return data_[index]; }

  private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace tileinfer::detail

namespace tileinfer {

// Retains the event and all host buffers needed by one queued operation.
struct Ticket::State
{
    enum class Kind
    {
        prefill,
        decode,
        generate,
        reset,
    };

    explicit State(Kind value);
    ~State();

    cudaEvent_t event = nullptr;
    bool recorded = false;
    const void* owner = nullptr;
    Kind kind;
    int start_position = 0;
    int end_position = 0;
    int requested = 0;
    detail::PinnedBuffer<TokenId> input_host;
    detail::PinnedBuffer<TokenId> output_host;
    detail::PinnedBuffer<int> status_host;
    detail::PinnedBuffer<float> logits_host;
    detail::PinnedBuffer<float> kv_host;
    detail::PinnedBuffer<float> blocks_host;
    std::unique_ptr<cuda::timed_event> compute_start;
    std::unique_ptr<cuda::timed_event> compute_end;
};

} // namespace tileinfer
