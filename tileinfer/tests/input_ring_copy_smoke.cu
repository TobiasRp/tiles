#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>

#include <iostream>
#include <stdexcept>

int main()
{
    try {
        cuda::device_ref device{0};
        cuda::stream stream{device};
        auto& device_pool = cuda::device_default_memory_pool(device);
        auto host_pool = cuda::pinned_default_memory_pool();
        cuda::host_buffer<int> ring(stream, host_pool, 8, cuda::no_init);
        cuda::host_buffer<int> readback(stream, host_pool, 8, cuda::no_init);
        cuda::device_buffer<int> tokens(stream, device_pool, 8, cuda::no_init);
        for (int i = 0; i < 8; ++i) {
            ring.data()[i] = 10 + i;
            readback.data()[i] = -1;
        }

        cuda::copy_bytes(stream, cuda::std::span<const int>(ring.data(), 8),
            cuda::std::span<int>(tokens.data(), 8));
        // A four-token prefill starts at ring index six and wraps to zero.
        cuda::copy_bytes(stream, cuda::std::span<const int>(ring.data() + 6, 2),
            cuda::std::span<int>(tokens.data(), 2));
        cuda::copy_bytes(stream, cuda::std::span<const int>(ring.data(), 2),
            cuda::std::span<int>(tokens.data() + 2, 2));
        // Read only the requested prefix of the larger device allocation.
        cuda::copy_bytes(stream, cuda::std::span<const int>(tokens.data(), 3),
            cuda::std::span<int>(readback.data(), 3));
        stream.sync();
        if (readback.data()[0] != 16 || readback.data()[1] != 17 || readback.data()[2] != 10)
            throw std::runtime_error("wrapped input was copied out of order");
        for (int i = 3; i < 8; ++i)
            if (readback.data()[i] != -1)
                throw std::runtime_error("partial result copy overwrote its destination suffix");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "input ring copy smoke: " << error.what() << '\n';
        return 2;
    }
}
