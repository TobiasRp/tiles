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
cd ~/dev/tiles/tileinfer
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
cd ~/dev/tiles
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


## Structural validation and worker variant

The loader validates the supplied tasks and schedule independently of Python's
canonical planning policy. Legal worker reassignment and task-list reordering
are accepted; output coverage, typed operands, producer dependencies (including
transitive dependencies), wave order, and complete worker lists remain checked.
`LoadedPlan` can only be constructed by `load_plan`; callers inspect immutable
metadata with `plan.data()`.

GEMV and logits share a warp-per-row CUB reduction. Attention reduces each
query/key score once per head and uses online softmax with constant shared
storage, including for heads wider than a block. RoPE inverse frequencies are
computed and checked once during package loading. Block tracing allocates its
device buffer on first use. Normalization shares one helper; its reduction
order remains unchanged.

The current worker fingerprint differs from the earlier serial worker.
Regenerate version-3 packages using a fresh probe from this executable.
Version-2 float32 package compatibility remains available. No dependency or
CMake-preset changes are needed.

## Additional verification

The CPU-only preset now includes package rejection and alternative-schedule
tests under UBSan. To refresh a build cache from the old checkout path:

```sh
cmake --preset cpu-debug --fresh
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

The release suite also covers long/extreme-score prefixes, wide attention
heads, uneven GEMV/logits dimensions, and controlled context-ring wraparound
and exhaustion. CUDA host code receives `-Wall -Wextra` through NVCC; C++ code
also receives `-Wpedantic`. CMake warns if `uv` is missing and regressions
cannot be registered.

Run the four-tool device suite with:

```sh
cmake --build build/release --target sanitizers
```

It generates fresh float32/BF16 packages, keeps logs and a summary under the
build directory, and runs memcheck, racecheck, synccheck, and initcheck. CUDA
API-error reporting defaults to disabled for the installed CCCL pinned-pool
initialization issue; the script accepts `--report-api-errors all`. Initcheck's
API-copy memory inspection is disabled only for the two intentionally gated
ring cases because that inspection synchronizes the blocked copies; kernel
reads remain checked. Ordinary copies and diagnostics retain full initcheck
memory inspection. Optional K/V snapshots define unused cache entries as zero,
and scratch snapshots omit uninitialized alignment padding.

A fixed shape/tile measurement protocol separates pilot selection from final
rounds and rotates case order deterministically:

```sh
~/.local/bin/uv run --locked --project ../tileplan python tests/sweep_tiles.py \
  --executor "$PWD/build/release/tile_runtime_test" --out ../runs/tile-sweep
```

The sweep saves its protocol before measuring, verifies each top token/logit
against the CPU reference, records all tile candidates and raw batch-average
samples, then measures the selected tiles in independent final rounds. These
are fixed-context decoder timings, not autoregressive throughput or individual
latency percentiles.
