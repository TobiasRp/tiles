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
    ModelShape shape;
    int workers;
    int selection_slots;
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
    float* attention_partials; // FP32 partial P*V, chunk statistics, then FFN down partials
    Candidate* selection;      // stacks: each logits task's greedy candidate
    float* ffn_trace = nullptr;
    // Optional timeline of kTimelineFields words per encoded instruction.
    std::uint64_t* timeline = nullptr;
};

// A launch decodes `steps` consecutive positions. Step s reads ids[s] when
// ids is set, otherwise the immediate ID; generation ignores both.
struct TokenInput
{
    TokenId immediate = -1;
    const TokenId* ids = nullptr;
    int steps = 1;
};

// Words of a generation request's device status. Workers read the stop
// words when a launch starts, so a request split over launches stops in all
// of them at once.
inline constexpr int kStatusCount = 0; // tokens selected so far
inline constexpr int kStatusEos = 1;   // nonzero once EOS was selected and decoded
inline constexpr int kStatusError = 2; // a DeviceError
inline constexpr int kStatusWords = 3;

enum DeviceError : int
{
    no_error = 0,
    non_finite_logits = 1,  // greedy selection saw a NaN or infinite logit
    token_out_of_range = 2, // a selected ID outside the vocabulary
};

// Most steps one launch decodes. The RTX 3050 runs a kernel watchdog (a
// display is attached), so long requests are split into launches of about
// 0.85 s each at 4K context.
inline constexpr int kMaxLaunchSteps = 64;

// Generation: every step selects its token from the previous step's
// candidates, which a previous launch may have written, and the first worker
// stores it in generated[first + s] and the status words.
struct Selection
{
    TokenId* generated = nullptr; // null: not generating
    int* status = nullptr;        // kStatusWords
    int first = 0;                // the request step of this launch's first step
    TokenId eos_id = -1;          // -1: none
};

void enqueue_worker(cuda::stream& stream, const WorkerProgramView& program,
    const WorkerBuffers& buffers, int position, TokenInput token, Selection selection = {});

// Defines unused diagnostic cache entries without touching the valid prefix.
void enqueue_zero_unused_kv(cuda::stream& stream, void* kv, WeightDtype dtype, std::size_t count,
    int layers, int layer_stride, int width, int capacity, int prefix, const int* generated_status);

} // namespace tileinfer::detail
