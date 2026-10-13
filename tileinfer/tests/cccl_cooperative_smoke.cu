#include "plan.hpp"

#include <cooperative_groups.h>
#include <cuda/__event/timed_event.h>
#include <cuda_runtime.h>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ void gate(cuda::std::span<int> values, cuda::std::span<int> results)
{
    const int block = blockIdx.x;
    if (threadIdx.x == 0)
        values[block] = block + 1;
    cooperative_groups::this_grid().sync();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < gridDim.x; ++i)
            sum += values[i];
        results[block] = sum;
    }
}

int main()
{
    try {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, 0) != cudaSuccess)
            throw std::runtime_error("device properties");
        if (!props.cooperativeLaunch)
            throw std::runtime_error("no cooperative launch");
        cuda::device_ref device{0};
        cuda::stream stream{device};
        auto& pool = cuda::device_default_memory_pool(device);
        cuda::device_buffer<int> values{
            stream, pool, static_cast<size_t>(props.multiProcessorCount), cuda::no_init};
        cuda::device_buffer<int> results{
            stream, pool, static_cast<size_t>(props.multiProcessorCount), cuda::no_init};
        auto config = cuda::make_config(cuda::block_dims<tileinfer::kThreads>(),
            cuda::grid_dims(props.multiProcessorCount), cuda::cooperative_launch{});
        cuda::timed_event start{stream}, end{stream};
        start.record(stream);
        cuda::launch(stream, config, gate, cuda::std::span<int>{values.data(), values.size()},
            cuda::std::span<int>{results.data(), results.size()});
        end.record(stream);
        end.sync();
        std::vector<int> host(props.multiProcessorCount);
        if (cudaMemcpy(host.data(), results.data(), host.size() * sizeof(int),
                cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("copy");
        const int expected = props.multiProcessorCount * (props.multiProcessorCount + 1) / 2;
        for (int result : host)
            if (result != expected)
                throw std::runtime_error("grid sync mismatch");
        std::cout << "CCCL cooperative gate passed: " << props.multiProcessorCount << " blocks of "
                  << tileinfer::kThreads << " threads, "
                  << cuda::std::chrono::duration<double, cuda::std::milli>(end - start).count()
                  << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
