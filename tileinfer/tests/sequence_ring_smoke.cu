#include "detail/cuda_check.hpp"
#include "detail/sequence_test_access.cuh"
#include "runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

// The callback uses only host synchronization; it never calls a CUDA API.
class StreamGate
{
    struct State
    {
        std::mutex mutex;
        std::condition_variable condition;
        bool entered = false;
        bool released = false;
    };

  public:
    explicit StreamGate(cuda::stream_ref stream)
        : state_(std::make_shared<State>())
    {
        auto argument = std::make_unique<std::shared_ptr<State>>(state_);
        tileinfer::detail::check(
            cudaLaunchHostFunc(stream.get(), callback, argument.get()), "queue test gate");
        argument.release(); // the callback retains its state even on exception unwinding
    }
    ~StreamGate() { release(); }
    StreamGate(const StreamGate&) = delete;
    StreamGate& operator=(const StreamGate&) = delete;
    void entered()
    {
        std::unique_lock lock(state_->mutex);
        require(state_->condition.wait_for(
                    lock, std::chrono::seconds(480), [&] { return state_->entered; }),
            "stream did not reach test gate");
    }
    void release()
    {
        std::lock_guard lock(state_->mutex);
        state_->released = true;
        state_->condition.notify_all();
    }

  private:
    static void callback(void* pointer)
    {
        std::unique_ptr<std::shared_ptr<State>> owner(
            static_cast<std::shared_ptr<State>*>(pointer));
        auto& state = **owner;
        std::unique_lock lock(state.mutex);
        state.entered = true;
        state.condition.notify_all();
        state.condition.wait(lock, [&] { return state.released; });
    }
    std::shared_ptr<State> state_;
};
} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 2, "expected stack package");
        const auto loaded = tileinfer::load_plan(argv[1], tileinfer::describe_target());
        const auto& plan = loaded;
        require(plan.max_context >= 5, "ring test requires five-token capacity");
        tileinfer::DecoderModel model(loaded);
        auto queued = model.create_context();
        auto reference = model.create_context();
        StreamGate first(tileinfer::detail::SequenceTestAccess::stream(queued));
        first.entered();
        const std::size_t capacity = tileinfer::detail::SequenceTestAccess::input_capacity(queued);
        std::size_t remaining = capacity - 2;
        tileinfer::Ticket prefix;
        while (remaining != 0) {
            const auto count = std::min(remaining, static_cast<std::size_t>(plan.max_context));
            queued.enqueue_reset();
            const std::vector<tileinfer::TokenId> prompt(count, 0);
            prefix = queued.enqueue_prefill(prompt);
            remaining -= count;
        }
        StreamGate second(tileinfer::detail::SequenceTestAccess::stream(queued));
        queued.enqueue_reset();
        const std::array<tileinfer::TokenId, 1> single{1};
        auto retained = queued.enqueue_prefill(single);
        bool exhausted = false;
        try {
            const std::array<tileinfer::TokenId, 2> excess{0, 1};
            queued.enqueue_prefill(excess);
        } catch (const std::runtime_error& error) {
            exhausted = std::string_view(error.what()) == "pinned input ring exhausted";
        }
        require(exhausted, "pending inputs did not exhaust the ring");
        first.release();
        // Wait until prefix work reaches the gate before asking the CUDA tool
        // to drain an event: instrumentation may also drain later callbacks.
        second.entered();
        // One reservation remains at capacity-2. The new span starts at
        // capacity-1 and must copy one tail element followed by three head elements.
        queued.enqueue_reset();
        std::array<tileinfer::TokenId, 4> prompt{0, 1, 1, 0};
        auto wrapped = queued.enqueue_prefill(prompt, {.logits = true});
        prompt.fill(1); // enqueue must retain its own copy of the caller's bytes
        second.release();
        queued.wait(prefix);
        require(queued.wait(wrapped).position == 4, "wrapped prefill position changed");
        const std::array<tileinfer::TokenId, 4> expected{0, 1, 1, 0};
        auto comparison = reference.enqueue_prefill(expected, {.logits = true});
        require(queued.collect(wrapped).logits == reference.collect(comparison).logits,
            "wrapped staging changed logits or retained caller storage");
        require(queued.wait(retained).position == 1 && queued.ready(retained),
            "ring reclamation expired a retained ticket");
        auto generated = queued.enqueue_generate({1, std::nullopt});
        auto reference_generated = reference.enqueue_generate({1, std::nullopt});
        require(queued.wait(generated).generated == reference.wait(reference_generated).generated,
            "ring exhaustion or reuse changed continuation");
        std::cout << "controlled context ring wraparound and exhaustion passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
