#include "runtime.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

// Requires `call` to throw an error whose message contains `expected`.
template <typename Call>
void require_error(Call call, std::string_view expected, const char* message)
{
    try {
        call();
    } catch (const std::runtime_error& error) {
        require(std::string_view(error.what()).find(expected) != std::string_view::npos, message);
        return;
    }
    throw std::runtime_error(message);
}

// A package whose logits are NaN for some token the prompt {0} avoids:
// generation stops with a device error, and the context needs a reset.
void check_non_finite(const tileinfer::DecoderModel& model)
{
    const std::array<tileinfer::TokenId, 1> prompt{0};
    auto context = model.create_context();
    context.enqueue_prefill(prompt);
    auto failed = context.enqueue_generate({2, std::nullopt});
    require_error([&] { context.wait(failed); }, "non-finite logits",
        "non-finite logits did not stop generation");
    require_error([&] { context.enqueue_decode(0); }, "reset required",
        "a stopped generation left the context valid");
    // A dropped ticket: the next request reports the error.
    context.enqueue_reset();
    context.enqueue_prefill(prompt);
    (void)context.enqueue_generate({2, std::nullopt});
    require_error([&] { context.enqueue_decode(0); }, "non-finite logits",
        "a dropped failed generation was not reported");
    // After a reset, waiting on the old ticket reports it again but leaves
    // the new sequence valid.
    context.enqueue_reset();
    auto prefill = context.enqueue_prefill(prompt);
    require_error([&] { context.wait(failed); }, "non-finite logits",
        "a completed failed ticket changed its result");
    require(context.wait(prefill).position == 1, "reset context did not accept a prefill");
    context.enqueue_decode(0);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const bool non_finite = argc == 3 && std::string_view(argv[2]) == "--non-finite";
        require(argc == 2 || non_finite, "expected a stack package path [--non-finite]");
        const auto plan = tileinfer::load_plan(argv[1], tileinfer::describe_target());
        require(plan.stack.full_stack && plan.max_context >= 5, "expected a small stack package");
        tileinfer::DecoderModel model(plan);
        if (non_finite) {
            check_non_finite(model);
            std::cout << "sequence context non-finite smoke passed\n";
            return 0;
        }
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
