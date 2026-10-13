#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tileinfer::detail {

inline void check(cudaError_t status, const char* stage)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(status));
}

} // namespace tileinfer::detail
