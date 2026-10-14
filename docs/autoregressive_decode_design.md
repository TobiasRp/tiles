# Autoregressive TileInfer: runtime and CLI design

Status: implemented and GPU-validated on `tux` on 2026-10-13. The validation
section below records the original version-3 implementation. Version 4 replaces
its BF16-weight/float32-activation arithmetic; see the
[BF16 implementation results](bf16_compute_results.md) for current precision and checks.

## Why change the current path

`--run-stack` is a correctness and inspection adapter. It accepts a fixed list of token IDs, calls `SequenceContext::run_token` for each, copies the entire vocabulary logits and the entire K/V allocation to the host after every token, and emits a large JSON trace. `run_token` itself synchronizes after the worker launch and again after copying logits. The next token therefore cannot be selected and fed back without a CPU round trip. A second CLI invocation constructs a new context and loses the sequence state.

`benchmark_token` avoids those copies during timing, but repeatedly overwrites one fixed cache position with one fixed token. It measures a useful isolated decoder launch, not an autoregressive generation loop. Benchmarking generation needs a path that chooses each next token from the preceding logits, keeps the token and cache on the GPU, and queues work without a synchronization per token.

The existing package contract remains useful: one validated block schedule is repeated for every layer, weights are immutable, the K/V cache belongs to a sequence, and the cooperative worker performs one token's embedding, blocks, final norm, and logits. The loader continues to reject incompatible plans before launch.

## Target behavior

1. A caller supplies prompt token IDs once. Prefill processes them in order on one stream, with no per-token host synchronization. For the first version, prefill uses the existing one-token decoder worker once per prompt token. The final prompt token leaves next-token logits on the GPU.
2. A GPU selection kernel reads those logits, writes one chosen token ID to device memory, and records it in a device output buffer. The existing decoder worker reads that device token ID, appends its K/V at the next position, and produces the following logits. Repeat for the requested number of new tokens. The host may enqueue the whole fixed maximum sequence before waiting.
3. Generation reads back only the selected IDs, count, stop reason, and optional summary values. Full logits, K/V, and block outputs are explicit diagnostic requests.
4. A request has a single completion point. No normal generation step calls `stream.sync()` or copies vocabulary logits to the CPU.

Initially, implement greedy selection on the GPU. Define ties as choosing the smallest token ID so tests and repeated runs agree. A later sampling policy can add temperature, top-k/top-p, and device RNG state without changing the sequencing model. Do not label a host argmax loop as the new fast path.

## Ownership and C++ interface

Separate immutable, uploaded model data from mutable sequence state:

```cpp
// A vocabulary index with the same representation on host and device.
using TokenId = int32_t;

// Identifies queued work and retains its completion and result storage.
class Ticket;
// Carries final counts, stop reason, and generated IDs after completion.
struct Completion;
// Chooses intermediate values to preserve before later tokens overwrite them.
struct DiagnosticRequest {
    bool logits = false;
    bool kv = false;
    bool block_outputs = false;
};
// Owns host copies of the intermediate values requested at enqueue time.
struct Diagnostics;
// Sets the token limit, selection policy, and optional stopping IDs.
struct GenerateOptions;

class SequenceContext;
// Uploads validated weights and the task program once for shared contexts.
class DecoderModel {
public:
    explicit DecoderModel(const LoadedPlan& plan); // upload a validated package once
    SequenceContext create_context() const;         // one stream and cache per sequence
};

// Owns one sequence's ordered work stream, cache, and pending decode state.
class SequenceContext {
public:
    Ticket enqueue_prefill(std::span<const TokenId> token_ids,
                           DiagnosticRequest diagnostics = {});
    Ticket enqueue_decode(TokenId token_id, DiagnosticRequest diagnostics = {});
    Ticket enqueue_generate(GenerateOptions options,
                            DiagnosticRequest diagnostics = {});
    Ticket enqueue_reset();

    bool ready(const Ticket& ticket) const;    // nonblocking event query
    Completion wait(const Ticket& ticket);     // explicit synchronization and error check
    Diagnostics collect(const Ticket& ticket); // optional explicit diagnostic readback
};
```

These names specify behavior, not a required final spelling. `DecoderModel` owns device weights, task records, ranges, and tensor descriptors. `SequenceContext` owns its stream, K/V cache, alternating hidden buffers, scratch, logits, current device token, generated-ID buffer, stop state, and queued/completed positions. The model allocation must outlive every context using it. Multiple contexts may share one model, but each context permits one ordered stream of work and is not concurrently writable by callers.

`TokenId` is the type for prompt IDs, EOS IDs, selected IDs, generated-ID buffers, and the worker's token input; cache positions and counts are separate types. JSON input must reject IDs outside the signed 32-bit range and then validate them against the model vocabulary.

`enqueue_prefill` must copy its host input into context-owned pinned staging before returning, then enqueue a single asynchronous host-to-device transfer and one decoder-worker launch per prompt token. Each launch reads its own offset in the device token array and uses the next position. The caller's span need not outlive the call. `enqueue_decode` similarly stages its scalar without synchronizing. The worker must accept a device token pointer (or an offset into a device token array) instead of the current by-value host `token_id` argument. CUDA stream order connects copies, selection kernels, worker launches, and optional readbacks. The caller may enqueue generation immediately after prefill on the same context; the stream preserves the dependency without a host wait. This first prefill path repeats a growing attention-prefix scan and has one launch per token; a chunked prefill kernel is a later performance change behind the same interface.

### Ticket and completion contract

A `Ticket` is an opaque, move-only handle returned as soon as work has been queued. It identifies a completion event on this context's stream and retains the result storage for that request. It does not contain a completed answer, start a CPU thread, or cancel work when dropped. A context must outlive its tickets; destroying the context waits for outstanding work before freeing buffers. Several tickets may be outstanding on one context, but their operations execute in enqueue order.

`ready(ticket)` checks the event without waiting. The event follows the small, stream-ordered result copy; `wait(ticket)` waits only through that event, checks asynchronous CUDA errors, and returns the compact `Completion`. For generation, that includes selected `TokenId` values, actual count, stop reason, and valid context length. A later queued operation may still be running. On a failed launch or execution, the context becomes invalid until reset.

Live logits and scratch may be overwritten by the next token. Therefore `DiagnosticRequest` is supplied **when enqueuing** work, so any requested intermediate snapshots get reserved storage and are captured at the right point in the stream. Their device-to-host copies are queued before the ticket event; after completion, `collect(ticket)` returns host-owned vectors. It cannot recover values that were not requested before they were overwritten. Normal generation requests no diagnostics. Reset is stream ordered; it changes the valid prefix and selection state without clearing old K/V bytes.

Remove blocking `run_token` and `run_hidden` from the public hot-path interface after callers migrate. Preserve the current one-block correctness coverage in a test-only block harness that can supply a hidden vector to the same worker and explicitly capture Q, K, V, attention, residual, FFN, and output. Synchronization is appropriate at that test boundary. The public CLI does not need a block-decoding command, but removing the command must not remove the isolated float32/BF16, tail-tile, reset, and capacity checks. Move `benchmark_token` to a separate profiling harness. This keeps synchronization and readback visible at their call sites.

## Generation semantics

For a prompt of length `P`, prefill consumes positions `0..P-1` and leaves logits predicting token `P`. Each generation step selects a token from the current logits, records it, and runs the worker at the next cache position to produce the next logits. Thus `N` requested new tokens entail `N` selection operations and `N` decoder calls after prefill. The final selected token is also in the cache, making a completed context ready for another generation request. Require `P > 0` (the caller supplies a BOS token if needed) and `P + N <= max_context`.

The GPU maintains an enabled flag and generated count. If EOS is selected, that token is recorded and decoded once; later queued steps skip cache writes and arithmetic. All blocks of a cooperative launch take the same skip decision so barriers remain legal. The completed result reports the actual generated count, stop reason (`eos` or `limit`), and actual valid context length. Because EOS makes the final position unknown to the host until completion, no further append may be queued on that context until the ticket is collected. For fixed-length benchmarking, disable EOS so every requested step executes.

The first implementation may enqueue one selection kernel and one cooperative worker launch per token. Both operate on resident buffers in the same stream, and the host queues the sequence without waiting between steps. A captured graph or a persistent multi-token device loop can be evaluated later against measured launch overhead; neither is needed to establish asynchronous autoregressive behavior.

## CLI: generation first, diagnostics explicit

Proposed commands:

```text
tile_runtime generate PACKAGE INPUTS.json --max-new-tokens N [--eos-id ID] [--selection greedy]
tile_runtime bench PACKAGE INPUTS.json --new-tokens N --warmup W --rounds R
tile_runtime check PACKAGE INPUTS.json [--trace-blocks]
```

`generate` accepts `{"tokens":[...]}` with `TokenId` values only. It returns generated IDs, count, stop reason, final position, and package identity as compact JSON. It does not tokenize text or print vocabulary logits by default; Python remains the tokenizer and text decoder. A future interactive or streaming mode can emit completed IDs in chunks, with chunk boundaries as explicit synchronization points. `--selection greedy` is the initial supported policy; stochastic flags appear only when their GPU implementation exists.

`check` preserves the current `--run-stack` trace schema, including reset markers and optional per-layer block outputs, for correctness regressions. Keep `--run-stack` as a compatibility alias until its test callers move; do not silently change its JSON. Migrate `decoder_regression.py` from `--run-block` to the test-only block harness, then remove `--run-block` and do not add a public block-check command. The test harness remains able to load a block package and inspect intermediate values. Diagnostic commands can be slow and blocking because their purpose is inspection. The top-level help and README should lead with `generate` and `bench`.

`bench` uses the exact generation path. It resets and prefills the same prompt before each timed round, starts a CUDA event after prefill, queues a fixed number of greedy selection plus decoder steps with EOS disabled, and ends the event after the final step. It synchronizes once per round, then reads generated IDs outside the timed interval for correctness checks. Report total milliseconds, microseconds per generated token, tokens per second, prompt length, generated count, selection policy, weight dtype, and package hashes. State clearly that setup, prompt prefill, and readback are excluded. A separate end-to-end wall-time mode can include them.

The fixed-position benchmark is still useful for diagnosing the worker: it holds context length constant, isolates decoder arithmetic from token selection and sequence growth, and lets Nsight Compute profile one launch at a chosen position. It can reveal whether a change in attention length or weight reads caused a kernel regression. It does not measure autoregressive throughput, so `--benchmark-stack` lives in the developer `tile_runtime_test` executable rather than the primary CLI. The existing profiling scripts use that executable. Label its results kernel-only and fixed-context, and never present them as the generation benchmark.

## Migration and checks

1. Introduce `DecoderModel`, device token input, async enqueue/completion, and explicit diagnostic capture while keeping old wrappers temporarily. Verify multiple queued prompt tokens produce the same logits and K/V as the existing blocking path.
2. Add GPU greedy selection and the generation loop. Compare generated IDs, logits at selected boundaries, and K/V with the existing PyTorch reference across multiple tokens, reset, capacity, EOS, and repeated continuation. Check deterministic tie behavior and asynchronous failure handling.
3. Preserve the `--run-stack` regression through `check` without changing its expected JSON. Move the `--run-block` regression to a test-only block harness before removing that CLI mode. Add `generate` and `bench`, then migrate README examples and the benchmark/profile scripts; move fixed-position profiling out of the primary CLI.
4. Measure generation with GPU events and also record wall time. Compare identical prompt and generation lengths against a PyTorch autoregressive path; do not compare generation throughput directly with the old fixed-position CUDA Graph number. Run CTest and Compute Sanitizer on the altered cooperative worker, especially its EOS skip path.
5. Give interface types short, useful comments wherever they clarify a contract or lifetime rule. Cover aliases, option/result structures, handles, and public classes; explain what callers need to know rather than restating each type's name.

No plan-format change is required for this interface proposal. The worker signature and device-side selection are runtime changes; any compiled-worker change still requires an occupancy reprobe and, if its fingerprint changes, regenerated packages.

## Historical validation of the version-3 implementation

- The current worker was reprobed on the RTX 3050, and all eight CTest checks passed, including CUDA smoke, the isolated block regression, and the full-stack regression with float32 and BF16 generation.
- Compute Sanitizer memcheck, racecheck, and synccheck reported zero errors on generation with EOS, `check`, and the test-only block path. A direct C++ smoke test also exercised queued tickets, EOS position locking, continuation, and reset under memcheck.
- A fresh BF16 Llama 3.2 1B package generated IDs `[12366, 13]` from a six-token prompt, matching a BF16 PyTorch eager reference with GPU argmax feedback. With two generated tokens and prompt processing excluded, mean CUDA event times were 137.0 ms for TileInfer and 29.3 ms for PyTorch on this RTX 3050. TileInfer reads BF16 weights into float32 arithmetic, so the paths still differ in computation. This is one short benchmark case, not a throughput characterization.
- Ruff, `ty`, and all 18 Python unit tests passed.
