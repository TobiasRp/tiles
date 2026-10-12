#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void add(int lhs, int rhs, int* result) {
    *result = lhs + rhs;
}

}  // namespace

int main() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && device_count == 0)) {
        std::cout << "CUDA runtime check skipped: no usable NVIDIA GPU/driver.\n";
        return 77;
    }

    try {
        check(status, "cudaGetDeviceCount");

        int* allocation = nullptr;
        check(cudaMalloc(&allocation, sizeof(int)), "cudaMalloc");
        const std::unique_ptr<int, decltype(&cudaFree)> device_result(allocation, &cudaFree);

        add<<<1, 1>>>(20, 22, device_result.get());
        check(cudaGetLastError(), "kernel launch");
        check(cudaDeviceSynchronize(), "kernel execution");

        int result = 0;
        check(cudaMemcpy(&result, device_result.get(), sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy");
        if (result != 42) {
            throw std::runtime_error("CUDA result mismatch: expected 42, got " +
                                     std::to_string(result));
        }

        std::cout << "CUDA environment ready: 20 + 22 = " << result << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
