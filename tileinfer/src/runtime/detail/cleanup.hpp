#pragma once

#include <cstdio>
#include <exception>

namespace tileinfer::detail {

// Cleanup must preserve the active exception, including in noexcept destructors.
template <typename Action>
void cleanup_noexcept(Action&& action) noexcept
{
    try {
        action();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "tileinfer cleanup: %s\n", error.what());
    } catch (...) {
        std::fputs("tileinfer cleanup failed\n", stderr);
    }
}

} // namespace tileinfer::detail
