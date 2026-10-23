#include "plan.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string_view>
#include <type_traits>

static_assert(std::is_aggregate_v<tileinfer::LoadedPlan>);

int main(int argc, char** argv)
{
    using json = nlohmann::json;
    try {
        tileinfer::Target target{};
        target.major = 8;
        target.minor = 6;
        target.sm_count = 4;
        target.cooperative_launch = true;
        target.active_blocks_per_sm = 1;
        if (argc == 2 && std::string_view(argv[1]) == "--describe-target-json") {
            // Synthetic capabilities for CPU loader tests; never a GPU planning probe.
            std::cout << json{{"device", {{"compute_capability", {8, 6}}, {"sm_count", 4},
                                             {"cooperative_launch", true}}},
                             {"executor", {{"abi", tileinfer::kAbi},
                                              {"fingerprint", tileinfer::kFingerprint},
                                              {"threads_per_block", tileinfer::kThreads},
                                              {"active_blocks_per_sm", 1},
                                              {"opcodes", tileinfer::supported_opcodes()}}}}
                             .dump()
                      << '\n';
            return 0;
        }
        if (argc != 2)
            throw std::runtime_error("expected a package path");
        const auto plan = tileinfer::load_plan(argv[1], target);
        std::cout << json{{"tasks", plan.task_count}, {"waves", plan.wave_count},
                         {"instructions", plan.instructions.size()}, {"loads", plan.worker_loads},
                         {"rope_pairs", plan.rotary_frequencies.size()}}
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
