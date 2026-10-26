#pragma once

#include "plan.hpp"

#include <nlohmann/json.hpp>

namespace tileinfer::detail {

// Checks the supplied whole-token program without choosing tiles, waves, or
// workers, then encodes one instruction stream per worker with explicit barriers.
void validate_program(
    const nlohmann::json& document, const ModelShape& shape, int workers, LoadedPlan& result);

} // namespace tileinfer::detail
