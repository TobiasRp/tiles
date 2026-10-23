#include "detail/launch.cuh"

#include <cuda_runtime.h>
#include <cuda/launch>
#include <cuda/std/limits>

#include <climits>

namespace tileinfer {
namespace {
constexpr int kSelectThreads = 1024;

// The better of two candidates: larger value, then lower ID. NaN never wins.
__device__ inline void keep_better(float& value, TokenId& id, float other, TokenId other_id)
{
    if (other > value || (other == value && other_id < id)) {
        value = other;
        id = other_id;
    }
}

// One block reduces the entire vocabulary with a stable lowest-ID tie break.
// Threads read four logits at a time, then warps and the block reduce.
__global__ void select_greedy(
    const float* logits, int vocab_size, TokenId* chosen, int* status, int step, TokenId eos_id)
{
    if (status[1] != 0) {
        if (threadIdx.x == 0)
            chosen[step] = -1;
        return;
    }
    float best = -cuda::std::numeric_limits<float>::infinity();
    TokenId best_id = INT_MAX;
    const int vectors = vocab_size / 4;
    const auto* packed = reinterpret_cast<const float4*>(logits);
    for (int i = threadIdx.x; i < vectors; i += blockDim.x) {
        const float4 four = packed[i];
        keep_better(best, best_id, four.x, 4 * i);
        keep_better(best, best_id, four.y, 4 * i + 1);
        keep_better(best, best_id, four.z, 4 * i + 2);
        keep_better(best, best_id, four.w, 4 * i + 3);
    }
    for (int id = 4 * vectors + threadIdx.x; id < vocab_size; id += blockDim.x)
        keep_better(best, best_id, logits[id], id);
    for (int offset = 16; offset > 0; offset /= 2)
        keep_better(best, best_id, __shfl_down_sync(0xffffffff, best, offset),
            __shfl_down_sync(0xffffffff, best_id, offset));
    __shared__ float values[kSelectThreads / 32];
    __shared__ TokenId ids[kSelectThreads / 32];
    if (threadIdx.x % 32 == 0) {
        values[threadIdx.x / 32] = best;
        ids[threadIdx.x / 32] = best_id;
    }
    __syncthreads();
    if (threadIdx.x != 0)
        return;
    for (int warp = 1; warp < kSelectThreads / 32; ++warp)
        keep_better(best, best_id, values[warp], ids[warp]);
    chosen[step] = best_id;
    ++status[0];
    if (best_id == eos_id)
        status[1] = 1;
}

} // namespace

namespace detail {

void enqueue_greedy_sampling(cuda::stream& stream, const float* logits, int vocab_size,
    TokenId* chosen, int* status, int step, TokenId eos_id)
{
    auto config = cuda::make_config(cuda::block_dims<kSelectThreads>(), cuda::grid_dims<1>());
    cuda::launch(stream, config, select_greedy, logits, vocab_size, chosen, status, step, eos_id);
}

} // namespace detail
} // namespace tileinfer
