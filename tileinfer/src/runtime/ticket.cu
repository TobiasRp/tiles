#include "detail/ticket_state.cuh"

#include <utility>

namespace tileinfer {

Ticket::State::State(Kind value)
    : kind(value)
{
    detail::check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create ticket event");
}

Ticket::State::~State()
{
    if (event != nullptr) {
        if (recorded)
            cudaEventSynchronize(event);
        cudaEventDestroy(event);
    }
}

Ticket::Ticket(std::shared_ptr<State> state)
    : state_(std::move(state))
{
}

Ticket::~Ticket() = default;
Ticket::Ticket(Ticket&&) noexcept = default;
Ticket& Ticket::operator=(Ticket&&) noexcept = default;

} // namespace tileinfer
