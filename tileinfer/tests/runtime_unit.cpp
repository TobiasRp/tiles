#include "runtime.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::rms_norm_f32) == 1);
static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::gemv_f32) == 2);
static_assert(static_cast<std::uint32_t>(tileinfer::Opcode::silu_product_f32) == 6);
static_assert(static_cast<std::uint32_t>(tileinfer::StorageClass::weight) == 4);
static_assert(static_cast<std::uint32_t>(tileinfer::TensorId::Wdown) == 22);
static_assert(sizeof(tileinfer::TokenId) == 4);
static_assert(std::is_move_constructible_v<tileinfer::Ticket>);
static_assert(!std::is_copy_constructible_v<tileinfer::Ticket>);
static_assert(std::is_standard_layout_v<tileinfer::TaskRecord>);
static_assert(std::is_trivially_copyable_v<tileinfer::TaskRecord>);
static_assert(sizeof(tileinfer::TaskRecord) == 40);
static_assert(sizeof(tileinfer::Range) == 8);
static_assert(sizeof(tileinfer::TensorRecord) == 8);
static_assert(std::is_trivially_copyable_v<tileinfer::TensorRecord>);
static_assert(std::is_trivially_copyable_v<tileinfer::StackConfig>);

int main()
{
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
