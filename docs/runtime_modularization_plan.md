# TileInfer runtime modularization plan

Status: implemented on `tux` on 2026-10-13. The sections below record the split and its acceptance criteria.

## Starting point

Before the split, `tileinfer/src/runtime/worker.cu` was 912 lines. It contained the cooperative decoder kernel and target probe (lines 1–169), the blocking `DiagnosticContext` (170–451), ticket storage and cleanup (452–522), model upload (523–589), and `SequenceContext` (590–912). The greedy selection kernel sat beside the decoder kernel, while both contexts duplicated the immutable weights/task upload and the typed worker launch. The public `runtime.hpp` also exposed the test-only `DiagnosticContext` and its fixed-position benchmark result.

The aim is one owner for each resource and one host launch entry point per kernel. File boundaries should follow ownership and execution semantics, not a line-count target.

## Resulting layout

| File | Responsibility |
| --- | --- |
| `plan.hpp`, `plan.cpp` | Public package records, `LoadedPlan`, loader, hashes, and target declarations. Move the plan types currently at the top of `runtime.hpp` into this header. |
| `runtime.hpp` | Production `TokenId`, `DecoderModel`, `SequenceContext`, `Ticket`, generation options, completions, and optional snapshot requests/results. Include `plan.hpp`. |
| `diagnostic.hpp` | `DiagnosticContext` and `BenchmarkResult`, used by the test/profiling executable. |
| `detail/launch.cuh` | Small borrowed device-pointer views for the immutable program and mutable sequence buffers, plus declarations for `enqueue_worker` and `enqueue_greedy`. It owns no allocations or synchronization. |
| `worker_kernel.cu` | `decoder_worker<float>` and `<__nv_bfloat16>`, their single CCCL cooperative-launch wrapper, and `describe_target()` probing those exact compiled kernels. Keep `ops.cuh` here. |
| `sampling.cu` | Greedy reduction and its launch wrapper. Preserve lowest-ID tie breaking, device-resident token output, and the uniform skip after EOS. Keep selection state in the sequence context. |
| `detail/uploaded_plan.cuh`, `uploaded_plan.cu` | One internal `UploadedPlan` that owns immutable device weights, tasks, ranges, tensors, and launch metadata. Its upload stream completes before contexts use the data. `DecoderModel` shares it across contexts; the diagnostic harness can use it for block packages too. |
| `detail/ticket_state.cuh`, `ticket.cu` | Ticket event, request kind/positions, timed events, and pinned staging/readback buffers. Typed RAII pinned buffers replace the raw pointer/count pairs and manual six-buffer cleanup. |
| `sequence_context.cu` | Per-sequence stream, K/V, scratch, logits, generated IDs, queued position, outstanding tickets, enqueue methods, and completion handling. It calls the two launch wrappers and owns all sequence ordering decisions. |
| `diagnostic_context.cu` | Blocking hidden/token checks, snapshots, and fixed-position profiling. It uses `UploadedPlan` and `enqueue_worker`, without duplicating uploads or kernel dispatch. |

`ops.cuh` remains the device arithmetic library. `main.cpp` includes `diagnostic.hpp` only for the `TILE_TEST_HARNESS` build. The production `check` command continues to use `SequenceContext` snapshot requests; it does not depend on `DiagnosticContext`.

## Boundaries and invariants

1. `enqueue_worker` accepts a complete borrowed launch description: immutable program view, mutable buffer view, position, and either an immediate token or a device token pointer/index. Both contexts call this wrapper, so dtype dispatch and the long kernel argument list occur once. Its implementation and the kernel definition live in `worker_kernel.cu`; other translation units call a host function. `enqueue_greedy` follows the same pattern in `sampling.cu`. This avoids cross-translation-unit device calls or a new CUDA separable-compilation requirement.
2. `UploadedPlan` accepts a validated `LoadedPlan` for either a block or stack. `DecoderModel` enforces the full-stack requirement at its boundary. Its buffers and upload stream outlive every sharing context. Each `SequenceContext` owns only mutable buffers and its own ordered stream.
3. A ticket owns its completion event and pinned input/result memory. The context retains dropped, unfinished tickets until their event completes, and destruction synchronizes the stream before releasing context buffers. Snapshot copies are queued before the ticket event. Use a strong reference to the ticket whose EOS result leaves the final position unknown; waiting on an older ticket must not release that lock.
4. `SequenceContext` remains the sole authority for capacity, validity, reset, and the actual position after EOS. The sampling module chooses IDs and updates device count/stop flags, but does not modify host sequence state. The worker's EOS skip remains uniform across every block before any cooperative barrier.
5. Keep `DiagnosticRequest` and `Diagnostics` in the production API because `check` and external callers can request snapshots. The blocking `DiagnosticContext` and fixed-position profiler belong to the diagnostic target.

## Implementation sequence

1. **Record a baseline.** Capture a clean release build, all current CTest results, the target probe's register/shared-memory/occupancy fields, and `bench` samples for one synthetic and one Llama package. Add a direct API regression for two contexts sharing a model, multiple outstanding tickets, an older-ticket wait while EOS is pending, continuation, and reset.
2. **Create the launch boundary.** Add `detail/launch.cuh`; move the decoder and greedy kernels with their launch wrappers into `worker_kernel.cu` and `sampling.cu`. Switch both contexts to `enqueue_worker`. Build and run the GPU regressions before moving ownership.
3. **Unify immutable uploads.** Introduce `UploadedPlan`; make `DecoderModel` and `DiagnosticContext` use it. Keep the model ready only after its upload completes. Verify block and stack packages, float32/BF16, and two independent contexts.
4. **Separate async and diagnostic lifetimes.** Move ticket resources to `ticket.cu`, sequence logic to `sequence_context.cu`, and blocking methods to `diagnostic_context.cu`. Split `runtime.hpp` into the three public-facing headers above. Preserve the current CLI JSON and error behavior through this move.
5. **Update CMake and finish cleanup.** Build `tileinfer_runtime` from plan, worker, sampling, upload, ticket, and sequence sources. Build a `tileinfer_diagnostics` library from `diagnostic_context.cu` and link it only into `tile_runtime_test`; keep `tile_runtime` linked to the production library. Add all new sources/headers to clang-format checks and remove `worker.cu` once no definitions remain.

Each step should compile and pass its relevant tests before the next. Give new interface types short comments that explain ownership or caller obligations. Keep the existing C++23 host and CUDA20 settings, CCCL streams/buffers/launches/timing, and CMake presets.

## Acceptance checks

- Full CTest, Ruff, `ty`, and formatting checks pass. Synthetic block and stack references still match, including BF16, tail tiles, reset, capacity, `check` snapshots, greedy generation, and EOS.
- Compute Sanitizer memcheck, racecheck, and synccheck report no errors on generation with EOS, `check`, and the test-only block path.
- Reprobe the actual worker after the translation-unit move. Compare occupancy, registers, shared memory, and target fingerprint with the baseline; regenerate packages if the compiled worker contract changes.
- The direct API regression proves that ticket storage survives early handle destruction, an older ticket cannot unlock a pending EOS request, contexts have independent caches, and continuation starts at the reported position.
- Compare generated IDs and event timings before and after the split on the same packages and prompts. Investigate any sustained timing change with profiler data; do not use a single short run as a performance gate.

## Verification result

All eight CTest checks pass, including the direct sequence-context smoke test on float32 and BF16 stack packages. Ruff, `ty`, and all 18 Python unit tests pass. Compute Sanitizer memcheck, racecheck, and synccheck report zero errors on generation with EOS, `check`, the test-only block path, and the direct context smoke test. The compiled worker retained its fingerprint, seven active blocks per SM, 72 registers per thread, and four static shared-memory bytes. On the same Llama package and six-token prompt, generated IDs remained `[12366, 13]`; three-round mean CUDA event time was 136.95 ms before and 136.88 ms after the split.
