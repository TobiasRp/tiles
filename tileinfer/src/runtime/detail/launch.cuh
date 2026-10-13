#pragma once

#include "plan.hpp"

#include <cuda/stream>

namespace tileinfer::detail {

// Immutable device data for one worker launch
struct WorkerProgramView
{
    const TaskRecord* tasks;
    const WorkerTaskRange* ranges;
    const TensorRecord* tensors;
    const void* weights;
    int waves;
    int workers;
    int hidden_size;
    int intermediate_size;
    int kv_heads;
    int head_dim;
    NumericConfig numeric;
    StackConfig stack;
    WeightDtype weight_dtype;
};

// Borrows one sequence's mutable device storage for a worker launch.
struct WorkerBuffers
{
    float* input;
    float* output;
    float* scratch;
    float* kv;
    float* logits;
    float* block_trace;
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

void enqueue_greedy_sampling(cuda::stream& stream, const float* logits, int vocab_size,
    TokenId* chosen, int* status, int step, TokenId eos_id);

} // namespace tileinfer::detail
