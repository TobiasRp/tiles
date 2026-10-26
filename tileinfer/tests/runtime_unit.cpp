#include "detail/cleanup.hpp"
#include "detail/input_ring_cursor.hpp"
#include "runtime.hpp"

#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::barrier) == 0);
static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::qkv) == 1);
static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::down) == 6);
static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::logits) == 9);
static_assert(static_cast<std::uint32_t>(tileinfer::StorageClass::weight) == 4);
static_assert(static_cast<std::uint32_t>(tileinfer::TensorId::Wdown) == 17);
static_assert(sizeof(tileinfer::TokenId) == 4);
static_assert(std::is_move_constructible_v<tileinfer::Ticket>);
static_assert(!std::is_copy_constructible_v<tileinfer::Ticket>);
static_assert(std::is_standard_layout_v<tileinfer::TaskRecord>);
static_assert(std::is_trivially_copyable_v<tileinfer::TaskRecord>);
static_assert(sizeof(tileinfer::TaskRecord) == 24);
static_assert(sizeof(tileinfer::WorkerTaskRange) == 8);
static_assert(sizeof(tileinfer::TensorRecord) == 8);
static_assert(std::is_trivially_copyable_v<tileinfer::TensorRecord>);
static_assert(std::is_trivially_copyable_v<tileinfer::StackConfig>);

int main()
{
    // A failed cleanup during unwinding must preserve the original failure.
    struct Owner
    {
        ~Owner()
        {
            tileinfer::detail::cleanup_noexcept([] { throw std::runtime_error("cleanup"); });
        }
    };
    try {
        Owner owner;
        throw std::runtime_error("original");
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()) != "original")
            return 8;
    }
    tileinfer::detail::InputRingCursor ring(8);
    if (ring.reserve(3) != 0 || ring.reserve(3) != 3)
        return 3;
    ring.release(3);
    if (ring.reserve(4) != 6) // two elements at the tail, then two at the head
        return 4;
    bool exhausted = false;
    try {
        (void)ring.reserve(2);
    } catch (const std::runtime_error&) {
        exhausted = true;
    }
    if (!exhausted)
        return 5;
    ring.release(3);
    if (ring.reserve(2) != 2)
        return 6;
    ring.release(4);
    ring.release(2);
    if (ring.reserve(8) != 0)
        return 7;
    ring.release(8);
    if (tileinfer::sha256({}) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        return 1;
    if (tileinfer::sha256({static_cast<std::uint8_t>('a'), static_cast<std::uint8_t>('b'),
            static_cast<std::uint8_t>('c')}) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        return 2;
    try {
        tileinfer::load_plan("/this/package/does/not/exist", {});
    } catch (const std::runtime_error&) {
        return 0;
    }
    return 1;
}
