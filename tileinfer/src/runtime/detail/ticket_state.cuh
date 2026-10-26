#pragma once

#include "detail/cuda_check.hpp"
#include "detail/launch.cuh"
#include "runtime.hpp"

#include <cuda/__event/timed_event.h>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <optional>

namespace tileinfer {

// Retains a request's completion and optional results independently of its handle.
struct Ticket::State
{
    enum class Kind
    {
        prefill,
        generate,
        reset,
    };

    struct Generation
    {
        int start_position;
        int requested;
        cuda::host_buffer<TokenId> output;
        cuda::host_buffer<int> status;
        cuda::timed_event compute_start;
        cuda::timed_event compute_end;

        Generation(cuda::stream_ref stream, int start, int count)
            : start_position(start)
            , requested(count)
            , output(stream, cuda::pinned_default_memory_pool(), count, cuda::no_init)
            , status(
                  stream, cuda::pinned_default_memory_pool(), detail::kStatusWords, cuda::no_init)
            , compute_start(stream)
            , compute_end(stream)
        {
        }
    };

    struct Diagnostic
    {
        std::optional<cuda::host_buffer<float>> logits;
        std::optional<cuda::host_buffer<float>> kv;
        std::optional<cuda::host_buffer<float>> blocks;
    };

    explicit State(Kind value);
    ~State();

    cudaEvent_t event = nullptr;
    bool recorded = false;
    const void* owner = nullptr;
    Kind kind;
    int end_position = 0;
    std::size_t input_begin = 0;
    std::size_t input_count = 0;
    std::optional<Generation> generation;
    std::optional<Diagnostic> diagnostic;
};

} // namespace tileinfer
