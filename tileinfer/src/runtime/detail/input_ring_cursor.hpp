#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace tileinfer::detail {

// Tracks FIFO reservations; callers release them only after their copy's event completes.
class InputRingCursor
{
  public:
    explicit InputRingCursor(std::size_t capacity)
        : capacity_(capacity)
    {
        if (capacity == 0)
            throw std::runtime_error("input ring requires positive capacity");
    }

    [[nodiscard]] std::size_t reserve(std::size_t count)
    {
        if (count > capacity_ - (write_ - read_))
            throw std::runtime_error("pinned input ring exhausted");
        if (write_ > std::numeric_limits<std::size_t>::max() - count)
            throw std::runtime_error("input ring cursor overflow");
        const std::size_t begin = write_ % capacity_;
        write_ += count;
        return begin;
    }

    void release(std::size_t count)
    {
        if (count > write_ - read_)
            throw std::runtime_error("input ring released too many tokens");
        read_ += count;
        if (read_ == write_)
            read_ = write_ = 0;
    }

  private:
    std::size_t capacity_;
    std::size_t read_ = 0;
    std::size_t write_ = 0;
};

} // namespace tileinfer::detail
