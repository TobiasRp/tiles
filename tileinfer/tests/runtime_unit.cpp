#include "runtime.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(std::is_standard_layout_v<tileinfer::Task>);
static_assert(std::is_trivially_copyable_v<tileinfer::Task>);
static_assert(sizeof(tileinfer::Task) == 32);
static_assert(sizeof(tileinfer::Range) == 8);

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
