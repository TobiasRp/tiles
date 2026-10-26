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
        bool wrong_context = false;
        try {
            (void)second.ready(first_generation);
        } catch (const std::runtime_error&) {
            wrong_context = true;
        }
        require(wrong_context, "cross-context ticket was accepted");

        first.enqueue_reset();
        first.enqueue_prefill(prompt);
        auto one = first.enqueue_generate({1, std::nullopt});
        require(
            first.wait(one).generated == std::vector<tileinfer::TokenId>{first_result.generated[0]},
            "first generated token changed after reset");

        auto eos = first.enqueue_generate({2, first_result.generated[1]});
        first.wait(one);
        const auto stopped = first.wait(eos);
        require(stopped.generated == std::vector<tileinfer::TokenId>{first_result.generated[1]} &&
                    stopped.position == 4 && stopped.stop_reason == tileinfer::StopReason::eos,
            "EOS result or position is incorrect");

        auto continued = first.enqueue_generate({1, std::nullopt});
        require(first.wait(continued).position == 5, "continuation started at the wrong position");
        // A dropped EOS ticket: the next request waits for the generation and
        // continues where it stopped, and a reset discards it.
        first.enqueue_reset();
        first.enqueue_prefill(prompt);
        first.enqueue_generate({1, std::nullopt});
        (void)first.enqueue_generate({2, first_result.generated[1]});
        auto after_dropped = first.enqueue_generate({1, std::nullopt});
        require(first.wait(after_dropped).position == 5,
            "a dropped EOS ticket left the wrong continuation position");
        first.enqueue_reset();
        first.enqueue_prefill(prompt);
        (void)first.enqueue_generate({2, first_result.generated[1]});
        first.enqueue_reset();
        auto final_prefill = first.enqueue_prefill(prompt);
        require(first.wait(final_prefill).position == 2, "reset did not restore the empty prefix");

        auto queued = model.create_context();
        auto reference = model.create_context();
        const std::array<tileinfer::TokenId, 1> single{0};
        auto retained = queued.enqueue_prefill(single, {.logits = true});
        queued.enqueue_prefill(single);
        queued.enqueue_decode(1);
        auto queued_generation = queued.enqueue_generate({2, std::nullopt});
        reference.enqueue_prefill(single);
        reference.enqueue_prefill(single);
        reference.enqueue_decode(1);
        auto reference_generation = reference.enqueue_generate({2, std::nullopt});
        require(queued.wait(queued_generation).generated ==
                    reference.wait(reference_generation).generated,
            "queued prefills and decode changed generated IDs");
        const auto first_snapshot = queued.collect(retained);
        require(!first_snapshot.logits.empty() &&
                    queued.collect(retained).logits == first_snapshot.logits,
            "retained diagnostic result changed after ring reuse");
        require(queued.wait(retained).position == 1 && queued.ready(retained),
            "retained ticket expired after later requests");

        queued.enqueue_reset();
        auto after_reuse = queued.enqueue_prefill(prompt);
        require(queued.wait(after_reuse).position == 2 && queued.wait(retained).position == 1,
            "ring reuse changed a retained ticket");
        std::cout << "sequence context smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sequence context smoke: " << error.what() << '\n';
        return 2;
    }
}
