# TileInfer decoder runtime

TileInfer validates version-3 float32 or BF16-weight decoder packages, as well
as version-2 float32 packages. It executes either one block from a hidden
vector or a full stack from a token ID. The compiled cooperative worker uses
one logical block per SM. A full-stack call performs embedding lookup, repeats
the validated nine-wave block schedule, then computes
final RMSNorm and vocabulary logits in one launch. Block scratch is reused
across layers; each layer has its own K/V cache and packed weight stride.
The cooperative schedule and barriers are in `src/runtime/worker_kernel.cu`;
GPU greedy selection is in `src/runtime/sampling.cu`. The arithmetic
operators and tensor views are in `src/runtime/ops.cuh`;
named opcodes and storage classes are part of the fixed task ABI.
Weight-consuming operators and the worker are instantiated for each supported
weight type. Activations, K/V storage, reductions, and output logits remain
float32; BF16 weights are converted to float32 as they are read.

`DecoderModel` uploads a validated package once. Each `SequenceContext` owns an
ordered stream, hidden buffers, and K/V cache. `enqueue_prefill` queues the
one-token worker over prompt IDs without a host wait between tokens.
`enqueue_generate` alternates GPU greedy selection and the same worker; only
the selected IDs and compact status return to the host. A `Ticket` identifies
queued work. Call `ready` to poll or `wait` to synchronize once and obtain its
result. Prompt plus generated tokens must fit `max_context`. The public plan
records are in `src/runtime/plan.hpp`, and the async C++ interface is in
`src/runtime/runtime.hpp`; both link through `tileinfer_runtime`.

`src/runtime/uploaded_plan.cu` owns the shared device weights and task program.
`src/runtime/sequence_context.cu` owns per-sequence execution and completion;
`src/runtime/ticket.cu` owns ticket resources. The blocking
`DiagnosticContext` is declared in `src/runtime/diagnostic.hpp` and linked
only into the test executable through `tileinfer_diagnostics`.

`check` waits for per-token snapshots. Its input may set
`"trace_blocks": true`; each step then includes every layer's output hidden
vector. The normal generation path does not copy logits or the full cache to
the host after each token. `--run-block` lives only in the test harness for
isolated arithmetic regressions.

## Build and test on tux

```sh
cd ~/dev/tile-infer/tileinfer
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

CTest includes the CCCL cooperative-launch gate, package and corruption
checks, test-only one-block intermediate comparisons, full-stack logits/cache
comparisons, and greedy generation against PyTorch. The tests cover GQA,
tail tiles, reset, capacity, tied and untied output embeddings, for both
weight types. A direct C++ smoke test covers shared-model contexts, dropped
tickets, EOS position locking, continuation, and reset. Run Compute Sanitizer
`memcheck`, `racecheck`, and `synccheck`
on `generate`, `check`, and the test-only block harness after kernel changes.

## Commands

```sh
cd ~/dev/tile-infer
tileinfer/build/release/tile_runtime --describe-target-json
tileinfer/build/release/tile_runtime --inspect-plan PACKAGE
tileinfer/build/release/tile_runtime generate PACKAGE INPUTS.json --max-new-tokens 16
tileinfer/build/release/tile_runtime bench PACKAGE INPUTS.json --new-tokens 16 --warmup 1 --rounds 3
tileinfer/build/release/tile_runtime check PACKAGE INPUTS.json
```

`generate` and `bench` accept `{"tokens":[0,7,15]}`. `generate` also accepts
`--eos-id ID`; greedy selection runs on the GPU. It returns generated token
IDs, stop reason, final position, and GPU selection-plus-decode time. `bench`
times the same GPU generation loop with EOS disabled, excluding setup,
prefill, and readback. `check` accepts token IDs plus optional
`{"reset":true}` markers and returns full per-token logits and K/V values.
It accepts `--trace-blocks` for layer outputs. `--run-stack` remains a
compatibility alias for `check`.
The `tile_runtime_test` binary retains `--run-block`, `--state-smoke`, and
`--benchmark-stack` for regression and fixed-position profiling scripts.
The [TilePlan README](../tileplan/README.md) has a complete command sequence
that creates a package, writes token IDs, and runs generation.

`--inspect-plan` reports dimensions, wave count, worker loads, task count,
scratch and K/V bytes, weight dtype, and package hashes. It checks the
compiled-kernel fingerprint and cooperative residency before returning. Plans
contain byte offsets and strides, never addresses.
