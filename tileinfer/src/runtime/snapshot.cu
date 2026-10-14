#include "detail/activation_buffer.cuh"

#include <cuda/launch>

namespace tileinfer::detail {
namespace {
template <typename Scalar>
__global__ void zero_unused_kv(Scalar* kv, std::size_t count, int layer_stride, int width,
    int capacity, int prefix, const int* generated_status)
{
    const int valid = prefix + (generated_status ? generated_status[0] : 0);
    const int key_count = capacity * width;
    const int value_offset =
        (static_cast<std::size_t>(key_count) * sizeof(Scalar) + 15) / 16 * (16 / sizeof(Scalar));
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
        index += gridDim.x * blockDim.x) {
        const int local = index % layer_stride;
        const bool key = local < key_count && local / width < valid;
        const bool value = local >= value_offset && local < value_offset + key_count &&
                           (local - value_offset) / width < valid;
        if (!key && !value)
            kv[index] = 0.0f;
    }
}
template <typename Scalar>
__global__ void widen(const Scalar* source, float* destination, std::size_t count)
{
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
        index += gridDim.x * blockDim.x)
        destination[index] = static_cast<float>(source[index]);
}
} // namespace

void enqueue_widen(cuda::stream& stream, const void* source, float* destination, std::size_t count,
    WeightDtype dtype)
{
    if (count == 0)
        return;
    const auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(32));
    if (dtype == WeightDtype::bfloat16)
        cuda::launch(stream, config, widen<__nv_bfloat16>,
            static_cast<const __nv_bfloat16*>(source), destination, count);
    else
        cuda::launch(
            stream, config, widen<float>, static_cast<const float*>(source), destination, count);
}

void enqueue_zero_unused_kv(cuda::stream& stream, void* kv, WeightDtype dtype, std::size_t count,
    int layers, int layer_stride, int width, int capacity, int prefix, const int* generated_status)
{
    // Block packages use the same per-layer K/V layout with one layer.
    if (layer_stride == 0)
        layer_stride = static_cast<int>(count / layers);
    const auto config = cuda::make_config(cuda::block_dims<128>(), cuda::grid_dims(32));
    if (dtype == WeightDtype::bfloat16)
        cuda::launch(stream, config, zero_unused_kv<__nv_bfloat16>, static_cast<__nv_bfloat16*>(kv),
            count, layer_stride, width, capacity, prefix, generated_status);
    else
        cuda::launch(stream, config, zero_unused_kv<float>, static_cast<float*>(kv), count,
            layer_stride, width, capacity, prefix, generated_status);
}
} // namespace tileinfer::detail
