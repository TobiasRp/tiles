#pragma once

#include "runtime.hpp"

#include <cuda/stream>

namespace tileinfer::detail {
// Internal test seam for holding the context stream while exercising its FIFO.
struct SequenceTestAccess
{
    static std::size_t input_capacity(const SequenceContext& context);
    static cuda::stream_ref stream(SequenceContext& context);
};
} // namespace tileinfer::detail
