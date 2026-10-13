#pragma once

#include "plan.hpp"

#include <nlohmann/json.hpp>

namespace tileinfer::detail {

// Dimensions checked by the package loader before task validation.
struct PlanShape
{
    int d;
    int f;
    int nq;
    int nkv;
    int dh;
    int context;
    int tile;
};

// Checks the supplied program without choosing tiles, waves, or workers.
void validate_program(
    const nlohmann::json& document, const PlanShape& shape, int workers, PlanData& result);

} // namespace tileinfer::detail
