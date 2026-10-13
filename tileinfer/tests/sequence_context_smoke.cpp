#include "runtime.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "expected a stack package path");
        const auto plan = tileinfer::load_plan(argv[1], tileinfer::describe_target());
        require(plan.stack.full_stack && plan.max_context >= 5, "expected a small stack package");
        tileinfer::DecoderModel model(plan);
        const std::array<tileinfer::TokenId, 2> prompt{0, 1};
        const std::array<tileinfer::TokenId, 2> other_prompt{1, 0};

        auto isolated = model.create_context();
        isolated.enqueue_prefill(other_prompt);
        auto isolated_generation = isolated.enqueue_generate({2, std::nullopt});
        const auto other_expected = isolated.wait(isolated_generation).generated;

        auto first = model.create_context();
        auto second = model.create_context();
        first.enqueue_prefill(prompt);
        second.enqueue_prefill(other_prompt);
        auto first_generation = first.enqueue_generate({2, std::nullopt});
        auto second_generation = second.enqueue_generate({2, std::nullopt});
        const auto first_result = first.wait(first_generation);
        const auto second_result = second.wait(second_generation);
        require(first_result.generated.size() == 2 && first_result.position == 4,
            "first context produced an invalid result");
        require(second_result.generated == other_expected,
            "shared model contexts did not keep independent sequence state");
        require(first.ready(first_generation), "completed ticket is not ready");

        first.enqueue_reset();
        first.enqueue_prefill(prompt);
        auto one = first.enqueue_generate({1, std::nullopt});
        require(
            first.wait(one).generated == std::vector<tileinfer::TokenId>{first_result.generated[0]},
            "first generated token changed after reset");

        auto eos = first.enqueue_generate({2, first_result.generated[1]});
        first.wait(one);
        bool blocked = false;
        try {
            first.enqueue_decode(0);
        } catch (const std::runtime_error&) {
            blocked = true;
        }
        require(blocked, "waiting on an older ticket released the pending EOS position");
        const auto stopped = first.wait(eos);
        require(stopped.generated == std::vector<tileinfer::TokenId>{first_result.generated[1]} &&
                    stopped.position == 4 && stopped.stop_reason == tileinfer::StopReason::eos,
            "EOS result or position is incorrect");

        auto continued = first.enqueue_generate({1, std::nullopt});
        require(first.wait(continued).position == 5, "continuation started at the wrong position");
        first.enqueue_reset();
        auto final_prefill = first.enqueue_prefill(prompt);
        require(first.wait(final_prefill).position == 2, "reset did not restore the empty prefix");
        std::cout << "sequence context smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sequence context smoke: " << error.what() << '\n';
        return 2;
    }
}
