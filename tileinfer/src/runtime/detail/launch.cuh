#pragma once

#include "plan.hpp"

#include <cuda/stream>

namespace tileinfer::detail {

// Immutable device data for one worker launch
struct WorkerProgramView
{
    const TaskRecord* instructions;
    const WorkerTaskRange* streams; // one per worker
    LayerLayout layout;
    const void* weights;
    const float* rotary_frequencies;
    int workers;
    int hidden_size;
    int intermediate_size;
    int query_heads;
    int kv_heads;
    int head_dim;
    int max_context;
    int attention_chunks;
    int attention_chunk_steps;
    int attention_fused_steps;
    NumericConfig numeric;
    StackConfig stack;
    WeightDtype weight_dtype;
};

// Thread 0 of the executing worker records {start SM cycle, end SM cycle,
// start %globaltimer ns} for each instruction. Cycles time short tasks; the
// coarser global timer aligns workers, whose cycle counters are independent.
inline constexpr int kTimelineFields = 3;

// Borrows one sequence's mutable device storage for a worker launch.
struct WorkerBuffers
{
    void* input;
    void* output;
    void* scratch;
    void* kv;
    float* logits;
    float* block_trace;
    void* attention_scores;
    float* attention_partials; // FP32 partial P*V, then chunk statistics
    float* ffn_trace = nullptr;
    // Optional timeline of kTimelineFields words per encoded instruction.
    std::uint64_t* timeline = nullptr;
};

// Reads a device ID when ids is set, otherwise passes the immediate ID.
struct TokenInput
{
    TokenId immediate = -1;
    const TokenId* ids = nullptr;
    int index = 0;
};

void enqueue_worker(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token);

// Defines unused diagnostic cache entries without touching the valid prefix.
void enqueue_zero_unused_kv(cuda::stream& stream, void* kv, WeightDtype dtype, std::size_t count,
    int layers, int layer_stride, int width, int capacity, int prefix, const int* generated_status);

void enqueue_greedy_sampling(cuda::stream& stream, const float* logits, int vocab_size,
    TokenId* chosen, int* status, int step, TokenId eos_id);

} // namespace tileinfer::detail
