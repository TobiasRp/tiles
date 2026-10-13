# TileInfer Refactoring

Status: Implemented on `tux`, 2026-10-13.

## Minor Tasks

- Rename `Range` in `plan.hpp` to `WorkerTaskRange`.

## Refactorings

### Reuse Pinned Input Storage Without Blocking Enqueues

`SequenceContext::Impl` should preallocate input staging so ordinary prefill and
decode requests do not allocate pinned input memory. Keep request results in
shared ticket states: one state and its normal completion storage may still be
allocated per request, not per generated token. Optional diagnostics may also
allocate when requested; they are outside the critical runtime path.

Normal enqueue, generation, and ring reclamation must not synchronize the
device or stream. `ready()` queries a completion event without blocking.
`wait()` explicitly waits for its ticket's event to return a `Completion`; it
must not synchronize unrelated later requests. Diagnostic paths may perform
explicit synchronization. Remove the current whole-stream synchronization
from context construction and successful enqueues. Context destruction and
partially enqueued failures still need safe resource lifetime handling; do not
remove their synchronization until cleanup can be ordered without freeing
buffers still in use.

#### Ticket states and handle lifetime

- Keep `Ticket` as a move-only handle with `std::shared_ptr<Ticket::State>`.
  The context retains unfinished states in `outstanding` even if their handles
  are dropped, and retains an unresolved EOS-position state separately. A
  caller holding a ticket can still call `ready()`, `wait()`, or `collect()`
  after other requests complete. Keep the existing context-ownership check.
- Remove a state from `outstanding` only after its completion event succeeds;
  a live handle or the EOS-position reference may continue to own the state.
  Releasing input-ring space must not depend on destruction of the ticket
  state. No fixed ticket-slot pool, slot index, retired-ID watermark, or
  expired-result behavior is needed.

`Ticket::State` currently combines completion tracking, request positions,
input staging, generation readback, diagnostics, and timing. Give those
responsibilities clear homes while keeping the shared state lifetime:

| Current member | Purpose and proposed home |
| --- | --- |
| `event` | Common completion event, recorded after any result or diagnostic copies; `ready()` queries it and `wait()` waits on it. |
| `recorded` | Common guard for the current destructor's event cleanup, including partially enqueued requests. Review that cleanup before removing any synchronization. |
| `owner` | Common context-ownership check; reject a ticket passed to another `SequenceContext`. |
| `kind` | Common request kind (`prefill`, `decode`, `generate`, or `reset`) used to interpret completion. |
| `end_position` | Common planned result position: position after prefill/decode, zero for reset, and the maximum possible position for generation before EOS count is known. |
| `start_position` | Keep with generation results to compute the actual EOS-adjusted position. Prefill/decode need it only while enqueuing, so use a local value there. |
| `requested` | Generation-only limit used for enqueue count, readback size, and returned-count validation. |
| `input_host` | Remove; token bytes live in the context's pinned ring. Keep only the reservation metadata needed to reclaim that ring space after completion. |
| `output_host`, `status_host` | Generation-only pinned readbacks for selected token IDs, actual count, and EOS flag. |
| `compute_start`, `compute_end` | Generation-only timed events used for `Completion::compute_us`. |
| `logits_host`, `kv_host`, `blocks_host` | Optional diagnostic snapshots, allocated only when requested and retained for repeated `collect()` calls. |

A small common state plus a generation-specific payload and optional diagnostic
payload is sufficient. Do not make every request carry generation buffers or
timed events. A ticket still retains its result for its lifetime; this
organization does not introduce expiration or change the public API.

#### Pinned input ring

- Replace per-ticket `input_host` allocations with one fixed-size pinned
  `cuda::host_buffer<TokenId>` ring owned by `SequenceContext::Impl` and backed
  by `cuda::pinned_default_memory_pool()`. Reserve one element for
  `enqueue_decode` and a contiguous logical span of `token_ids.size()` elements
  for `enqueue_prefill`. Copy the caller's input into the reservation before
  returning from enqueue. Write through `data()` or `get_unsynchronized()`;
  `cuda::host_buffer::operator[]` may synchronize.
- If a prefill reservation crosses the physical end of the ring, enqueue two
  host-to-device copies, from the tail and then the head, into consecutive
  positions of the existing device token buffer. Both copies and the decoder
  launches use the context stream. Lack of contiguous tail space alone is not
  an exhaustion condition.
- Keep each ring reservation unavailable until the asynchronous copy has
  completed. Initially, reclaim reservations in enqueue order when the
  request's existing completion event completes; this conservatively holds
  input space through the decoder work and needs no additional event. Reclaim
  that space even if the caller still holds the completed ticket. Never
  overwrite a reservation merely because its host-to-device copy was queued.
- Size the ring at context construction for the maximum supported queued
  input workload, including a maximum-length prefill and the intended number
  of queued requests. Check size arithmetic. If the ring is exhausted, fail
  enqueue explicitly; do not wait or allocate a spill buffer. The per-request
  state records its reservation so completion can release it.

#### Result and diagnostic buffers

- Replace `detail::PinnedBuffer` with pinned `cuda::host_buffer` storage.
  Normal completion buffers remain owned by their shared ticket states.
- Use `std::optional` for host buffers that exist only when requested.
  Allocate logits, K/V, and block-output snapshots when diagnostics are
  requested, and retain them with the state so repeated `collect()` calls
  work while the caller holds the ticket. Do not preallocate full K/V
  snapshots for ordinary requests.

#### CCCL copies

- Replace `cudaMemcpyAsync` calls involving `cuda::device_buffer` with
  `cuda::copy_bytes` on the same explicit stream. Start with the
  `SequenceContext` input and result transfers alongside the pinned input
  ring. Convert `UploadedPlan` setup transfers and `DiagnosticContext`
  readbacks as a separate cleanup; they are outside the critical runtime
  path. Do not assume the API change improves transfer performance.
- Pass the source first and destination second, unlike `cudaMemcpyAsync`.
  Use exactly sized `cuda::std::span` views for partial copies, especially
  each wrapped ring segment and the requested prefix of the larger generated
  ID device buffer. The source view determines the byte count; verify the
  destination has enough space. Keep the source and destination allocations
  alive until the stream-ordered copy completes.
- `cuda::copy_bytes` reports failures with C++ exceptions instead of a
  `cudaError_t`. Let them propagate to the existing invalid-context handling
  for failed enqueues. A pinned
  host source is necessary for the asynchronous input path; copies involving
  pageable `std::vector` storage in model setup or diagnostics may block.
  Do not use a source-access configuration that changes the ring's
  stream-ordered lifetime contract.

See the CCCL documentation for
[`copy_bytes`](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/algorithm.html),
[`cuda::host_buffer` and unsynchronized access](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/buffer.html),
and [runtime exception handling](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/cudart_interactions.html).

Keep the current queued-call, dropped-ticket, repeated-`wait()`, EOS-position,
reset, and diagnostic behavior. Add checks for ring wraparound, multiple
queued prefills and decodes, ring exhaustion, retained tickets after ring
reuse, cross-context handles, and EOS tickets. Verify copy sizes and ordering
on wrapped input and partial generated-ID transfers. Run the existing decoder
regressions and Compute Sanitizer on the changed lifetime paths; measure host
enqueue time before making a performance claim. This host-side refactor does
not change the plan format or worker kernel.

## Validation

The CUDA workstation build passed all nine CTest checks, including decoder and
stack regressions. Ruff, ty, and the 18 Python unit tests passed. Compute
Sanitizer memcheck, racecheck, and synccheck found no device errors in the
queued-ticket and wrapped-copy smoke tests. The installed CCCL pinned-pool
initializer emits a host `cuMemPoolSetAccess` API error under Compute Sanitizer;
the device checks were run with API-error reporting disabled after confirming
that the call originates in CCCL's pool initialization.
No host enqueue benchmark or performance claim is included here.
