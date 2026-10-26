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
    int chunks; // attention chunk tasks per K/V head
    int steps;  // most time steps one attention chunk task covers
    int fused;  // longest prefix one fused attention task per K/V head handles
    int ffn;    // FFN chunk tasks per layer, each with one FP32 down partial
};

// Checks the supplied whole-token program without choosing tiles, waves, or
// workers, then encodes one instruction stream per worker with explicit barriers.
void validate_program(
    const nlohmann::json& document, const PlanShape& shape, int workers, LoadedPlan& result);

} // namespace tileinfer::detail
