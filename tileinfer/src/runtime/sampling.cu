#include "detail/launch.cuh"

#include <cuda_runtime.h>
#include <cuda/launch>
#include <cuda/std/limits>

#include <climits>

namespace tileinfer {
namespace {
// One block reduces the entire vocabulary with a stable lowest-ID tie break.
__global__ void select_greedy(
    const float* logits, int vocab_size, TokenId* chosen, int* status, int step, TokenId eos_id)
{
    if (status[1] != 0) {
        if (threadIdx.x == 0)
            chosen[step] = -1;
        return;
    }
    __shared__ float values[256];
    __shared__ TokenId ids[256];
    float best = -cuda::std::numeric_limits<float>::infinity();
    TokenId best_id = INT_MAX;
    for (int id = threadIdx.x; id < vocab_size; id += blockDim.x) {
        const float value = logits[id];
        if (value > best || (value == best && id < best_id)) {
            best = value;
            best_id = id;
        }
    }
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = values[threadIdx.x + stride];
            const TokenId other_id = ids[threadIdx.x + stride];
            if (other > values[threadIdx.x] ||
                (other == values[threadIdx.x] && other_id < ids[threadIdx.x])) {
                values[threadIdx.x] = other;
                ids[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        chosen[step] = ids[0];
        ++status[0];
        if (ids[0] == eos_id)
            status[1] = 1;
    }
}

} // namespace

namespace detail {

void enqueue_greedy_sampling(cuda::stream& stream, const float* logits, int vocab_size,
    TokenId* chosen, int* status, int step, TokenId eos_id)
{
    auto config = cuda::make_config(cuda::block_dims<256>(), cuda::grid_dims<1>());
    cuda::launch(stream, config, select_greedy, logits, vocab_size, chosen, status, step, eos_id);
}

} // namespace detail
} // namespace tileinfer
