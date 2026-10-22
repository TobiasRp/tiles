# Next steps: a tile-graph megakernel that beats CUDA Graph replay

Status: proposal, 2026-10-22. Written from the committed local tree
(`fa4f46f`) and the uncommitted result documents
([BF16 computation results](bf16_compute_results.md),
[quality implementation](tileinfer_quality_implementation.md),
[code review](tileinfer_code_review.md),
[refactoring plan](tileinfer_refactoring.md)). `tux` was unreachable while
this was written, so no number below was re-measured. Values marked
*estimate* are arithmetic from the model shape and the committed planner.
They are not measurements.

**Progress, 2026-10-22: the Section 3 cleanup and milestone S1 are done.**
See [Section 8](#8-cleanup-and-s1-results) for what changed and the
measurements. The next step is S0's instruments (roofline kernel, row-dot
microbenchmark, device timeline trace), then S2.

The goal is a prototype that **shows what a tile-based megakernel can do**.
Running correctly in one launch is already done. The prototype should show a
measured, explained advantage over PyTorch CUDA Graph replay on the same
RTX 3050, using the same checkpoint and numerical policy.

## 1. Where the project stands

The reported BF16 results (policy `hf-llama-eager-bf16-v1`, SIMT row-dot,
tile 64, 20 workers × 1024 threads, one cooperative launch per token):

| Measurement | TileInfer | PyTorch eager | PyTorch graph |
| --- | ---: | ---: | ---: |
| Fixed context 1, ms | 12.99 | 14.09 | 12.88 |
| Fixed context 64, ms | 13.18 | 14.42 | 13.05 |
| Fixed context 256, ms | 13.84 | 14.89 | 13.41 |
| Generation (6-token prompt, 2 new), ms/token | 13.158 | 14.247 | 12.978 |

TileInfer is already faster than eager. It is 1–3% behind graph replay, and
the gap grows with context (+0.85 ms from context 1 to 256, versus +0.53 ms
for graph replay).

### The roofline frames everything

Batch-one decode reads every weight once per token and does about one FMA per
weight byte pair. On this GPU it is limited by memory bandwidth, not by
arithmetic.

| Quantity | Value |
| --- | ---: |
| BF16 weight bytes read per token (16 layers + tied `lm_head`) | 2,471,628,800 |
| of which `lm_head` | 525,336,576 (21%) |
| per layer: QKV / Wo / gate+up / down | 12.6 / 8.4 / 67.1 / 33.6 MB |
| BF16 K/V bytes read per token at context T | 32,768 × T (134 MB at 4K) |
| Nominal DRAM bandwidth (128-bit, 14 Gbps GDDR6) | 224 GB/s |
| Bandwidth floor, weights only | 11.0 ms |
| Graph replay at 12.98 ms | ≈ 190 GB/s, 85% of nominal |
| TileInfer at 13.16 ms | ≈ 188 GB/s, 84% of nominal |

A streaming-read kernel usually reaches about 90–93% of nominal bandwidth. If
that holds here, the realistic floor is **about 12.0 ms/token** *(estimate)*.
That leaves graph replay roughly 1 ms (7–8%) of headroom at short context.

Three conclusions follow:

1. **Faster arithmetic will not help.** This is why native BF16 MMA did not
   help: the reported MMA path is 2.15× slower because of how it loads data,
   and the tensor cores have nothing to speed up at batch one.
2. **Scheduling only matters as far as it keeps DRAM busy.** The schedule's
   job is to make every microsecond of the token stream weight bytes at full
   bandwidth. The causes of idle DRAM are barriers, unbalanced waves, serial
   small phases, and attention.
3. **The claim we can show is "no bubbles", not "fewer FLOPs".** At short
   context, an honest target is about 5–8% faster than graph replay. At long
   context the target is larger (Section 2.5).

## 2. Where the remaining ~1.2 ms probably goes

These are hypotheses ranked by the estimated size of their effect. Milestone
S0 measures each of them before anything changes.

### 2.1 Wave quantization (largest suspect)

The planner splits each GEMV into fixed 64-row tiles and assigns them per
wave to 20 workers. Most projections do not divide evenly:

| Wave | Tasks (tile 64) | Rounds on 20 workers | Slot utilization |
| --- | ---: | ---: | ---: |
| Q+K+V | 32+8+8 = 48 | 3 | 80% |
| Wo | 32 | 2 | 80% |
| gate+up | 256 | 13 | 98% |
| down (64 × 8192 tiles) | 32 | 2 | 80% |
| attention heads | 32 | 2 | 80% |

Suppose each SM had a fixed share of bandwidth. Then this layout would make
each layer cost about 136 MB worth of time for 122 MB of bytes. That is up to
**12% (about 1.5 ms/token)** *(upper-bound estimate)*. The real cost is lower,
because busy SMs can pull more than 1/20 of DRAM bandwidth while others sit
idle. How much more depends on how many bytes each SM keeps in flight. The
device trace in S0 measures the actual cost.

### 2.2 Grid barriers

The committed planner produces 9 waves per layer: norm, QKV, RoPE/cache,
attention, Wo, norm, gate/up, SiLU, down. Each wave ends with a
`grid.sync()`. Adding embedding and final norm gives about **146 grid barriers
per token**. At 2–5 µs each, including the tail of the slowest worker, that is
0.3–0.7 ms *(estimate)*. During a barrier no worker issues weight loads.

### 2.3 Serial low-bandwidth phases

Norm, RoPE/cache, SiLU/product, embedding, final norm, and the argmax kernel
move almost no bytes. They still cost a wave each, plus a barrier. In the
committed source, `rms_norm` and `final_norm` sum squares serially on thread 0.
In `final_norm`, every block repeats that sum. The tux tree has a
PyTorch-order block reduction for BF16. Greedy selection is a separate kernel
launch per token.

### 2.4 Per-SM memory-level parallelism

How well wave tails are hidden depends on how many outstanding bytes one SM
can sustain. That depends on vector width, unrolling, and whether loads are
asynchronous. Little's law at about 224 GB/s and about 0.5–0.6 µs latency
calls for roughly 120–130 KB in flight across the GPU, or about 6 KB per SM
when all 20 are busy. When only 8 SMs are busy, each needs about 16 KB.
Measure the row-dot routine alone (Section S0). That separates kernel
efficiency from scheduling.

### 2.5 Attention at long context (not yet measured)

Measurements stop at context 256, where K/V traffic is 8 MB/token. At 4K it is
134 MB/token, 5% of the weight bytes. The current mapping is one task per
query head, which means:

- 32 tasks on 20 workers, so a second round with 12 idle workers;
- each KV head read up to 4 times (GQA group of 4), unless L2 (about 2 MB)
  catches the reuse;
- one SM streaming a whole head's prefix at single-SM bandwidth.

*Estimate:* about 1.5 ms/token at 4K, against an ideal of about 0.65 ms.
Hugging Face eager attention in the baseline also materializes `repeat_kv`.
That writes and then reads 4× expanded K/V, on the order of 1 GB of extra
traffic per token at 4K *(estimate; it depends on the static-cache
configuration)*. Graph replay therefore likely degrades faster than
TileInfer. If so, long context is where the megakernel
can show a clear win. Report it honestly as a win against *eager-attention*
PyTorch (Section 6).

## 3. Is a code/architecture cleanup needed?

**Yes, but a narrow one. Do not do a broad refactor.** The architecture is
sound for the goal. The Python offline planner, the C++ structural validator
and encoder, a single cooperative worker, and an immutable model plus mutable
`SequenceContext` are the right boundaries for a megakernel. The cleanup that
matters is the cleanup that unblocks scheduling work.

### Do first (these gate the scheduling work)

1. **Land the tux working tree.** *(Correction: this work was already
   committed and pushed as `60ff850`; the local checkout used for this
   proposal was seven commits behind. Only the paired-vector `warp_dot`
   experiment was uncommitted on tux; it is parked on the local tux branch
   `experiment/vector-warp-dot`.)* Every later ablation is measured against
   `60ff850`.
2. **Narrow the performance variants.**
   - Remove the native-MMA decode variant from the worker. Keep its
     description and probe results in the docs as a prefill candidate. It is
     2.15× slower, and it doubles the cases the probe and sanitizers must
     cover.
   - Keep the pure-float32 policy as a precision reference that is tested but
     never tuned. The probe already takes the minimum occupancy across
     variants, so a resource-heavy reference variant can limit the BF16
     worker. Consider building it only into `tile_runtime_test`.
   - When the version-5 ABI arrives (S1), stop reading packages in versions
     2 and 3. Regenerating a package is cheap, and every legacy reader adds
     validation surface.
3. **Make the device program the whole token.** Today the kernel hard-codes
   the phases: embedding, a layer loop over a repeated block template, final
   norm, logits. Argmax is a separate launch. Wave boundaries are implicit in
   the `Range[wave × workers + block]` table. Scheduling work needs three
   changes:
   - a **flat, per-worker instruction stream** covering all layers;
   - every phase as an opcode (`embed`, `norm_stats`, `logits_tile`,
     `argmax_partial`, ...);
   - explicit synchronization **instructions** (`barrier`, later
     `wait counter ≥ n` / `signal counter`).

   This is the cleanup and the scheduling enabler in one change. It is
   milestone S1.

### Do alongside, when touching the same code

- Review finding 2 (NaN/INT_MAX greedy selection). It goes away when argmax
  moves into the worker in S5. Design the device error status then.
- Review finding 3 (throwing destructors). This is a small fix. Do it before
  adding spin-wait code, because a hung or failed kernel must produce a
  clean error.
- Rename `Range` to `WorkerTaskRange`, or retire it with the v5 program.

### Defer (not on the per-token critical path)

- The pinned input ring and ticket-state split in
  [the refactoring plan](tileinfer_refactoring.md). `generate` enqueues a whole
  request with no per-token host work, so host enqueue cost cannot affect
  ms/token. Do it when serving or streaming becomes a goal.
- Review finding 4 (dropped EOS ticket). This is an API lifecycle issue,
  unrelated to the megakernel claim.

## 4. Scheduling direction

### The central idea: weight streaming has no data dependencies

In batch-one decode, nearly every dependency is dense. Every row of every
projection reads the whole input vector, and RMSNorm needs the whole
residual. A tile graph therefore cannot start much *computation* early:
`QKV ← x`, `Wo ← all heads`, `gate/up ← r`, `down ← all of SiLU·up` are
inherently all-to-all.

What *can* start early is loading the weights. A weight tile never depends on
the token. Split each GEMV tile into two graph nodes:

```text
load(W tile → shared memory)    predecessors: none (only buffer-slot reuse)
compute(tile)                   predecessors: input-vector producers, load(W tile)
```

With that split, the 9–10 µs that 20 × ~96 KB of shared memory holds at full
bandwidth covers a barrier, a norm, a RoPE step, a short attention, or a layer
boundary. DRAM keeps streaming while dependencies resolve.

This is a capability that graph replay **lacks on this GPU**. Programmatic
dependent launch, which lets kernel N+1 start loading while kernel N drains,
requires compute capability 9.0. The RTX 3050 is 8.6. It is the clearest
mechanism the prototype can demonstrate.

### Why static assignment beats a dynamic queue here

A worker can prefetch the weights for its next task only if it knows which
task comes next. A static per-worker instruction stream, produced by the
planner, makes prefetch trivially correct and deterministic. A global atomic
work queue would balance load automatically but destroy prefetch: a worker
does not know its next tile until it claims it. This supports the project's
"offline planner, static schedule" thesis. Keep a dynamic queue as an
*ablation* (S4), not as the design.

Use static *assignment* with dynamic *timing*. The order on each worker is
fixed. Waits on dependency counters replace global barriers, so a worker
proceeds as soon as its own inputs are ready.

### Partial dependencies worth exposing

These are where the tile graph does let computation start early:

```mermaid
flowchart LR
    X[x / residual] -->|all rows| QKV[QKV tiles + norm scale + RoPE + cache write]
    QKV -->|per KV head| ATT[attention chunks, per KV head × context chunk]
    ATT -->|per head| COMB[softmax combine per head]
    COMB -->|per head columns| WO[Wo split-K partials]
    WO -->|all partials| R[r = x + Wo·a, norm stats]
    R -->|all rows| GU[gate/up interleaved tiles + SiLU·up]
    GU -->|per column chunk| DN[down split-K partials]
    DN -->|all partials| X2[next layer x, norm stats]
```

- **QKV → attention, per KV head.** Attention for a KV head group can start
  as soon as that group's Q, K, and V rows exist. Today it waits for all QKV
  tiles and a separate RoPE wave.
- **Attention → Wo, split-K by head.** A Wo partial over head `h`'s columns
  needs only head `h`.
- **gate/up → down, split-K by column chunk.** A down partial over product
  chunk `k` needs only gate/up tile `k`.
- **Norm statistics produced by the producers.** Each Wo or down reducer can
  emit the partial sum of squares for its rows. Consumers combine the 20
  partials in a fixed order and apply the scale while loading `x`. This
  removes the norm tasks and their barriers entirely.

To keep results deterministic, split-K partials go to an FP32 partial buffer
and are reduced in a fixed order. Do not use float `atomicAdd`. The BF16
policy's "round the projection before the residual" boundary is applied after
that reduction. Summation order changes against the unsplit row-dot, so the
existing calibrated tolerances apply, not bit equality.

## 5. Milestones

Each milestone keeps the decoder reference and checkpoint regressions green,
reprobes the compiled worker, and runs clang-format, Ruff, ty, CTest, and
Compute Sanitizer before any timing is reported (AGENTS.md). Each one adds a
row to the ablation table in S6.

### S0 — Baseline and instruments (no kernel behavior change)

- Land the tux tree (Section 3) and tag the baseline.
- **Bandwidth roofline.** Add a test-only streaming kernel that reads the
  package's weight blob in tile order, with the same 20 × 1024 launch, and in
  the best configuration found. Report attainable GB/s. This replaces the
  "~12.0 ms" estimate with a measured floor.
- **Row-dot microbenchmark.** Time each GEMV shape (QKV, Wo, gate/up, down,
  `lm_head`) alone across all 20 workers and on k = 1…20 active workers. This
  measures per-SM bandwidth and settles how much of Section 2.1 is real.
- **Device timeline trace.** Optional, runtime-enabled, off when timing. For
  each instruction, thread 0 writes `{task, worker, start, end}` using
  `%globaltimer` (check its update resolution on this part, and fall back to
  `clock64` deltas plus one alignment point per worker). It also records time
  spent waiting at each barrier. Render it with the existing
  `tileplan visualize`/`tile_flow` code as a per-worker Gantt chart, with
  bytes per task. This chart is both the main diagnostic and the main showcase
  figure.
- **Extend the context axis.** Use contexts 1, 256, 1024, 2048, and 4096 with
  a 4K-capacity package, for TileInfer and both PyTorch paths.
- Report the **bubble budget**: measured time minus bytes ÷ attainable
  bandwidth, split into barrier wait, wave tails, small phases, attention, and
  `lm_head`/argmax.

*Done when* the per-token time is explained to within about 0.2 ms by named
components.

### S1 — Version-5 program ABI (cleanup, no change in behavior)

- The planner emits one flat program for the whole token: embedding, all
  layers, final norm, `lm_head` tiles. Each worker gets an ordered instruction
  list. Waves become explicit `barrier` instructions on every worker.
- The C++ validator checks the existing structure. It also checks that all
  workers have the same barrier count and that, for any later counter waits,
  every wait names producers with a strictly smaller global topological index
  (see S4).
- The kernel becomes `for (instr : my_stream) dispatch(instr)` with no
  hard-coded phases. Drop version-2/3 readers and the MMA decode variant.

*Acceptance:* logits, K/V, and block outputs are **bit-identical** to the
S0 baseline for float32 and BF16. The arithmetic and order do not change, so
any difference is a bug. Timing should be unchanged within noise.

### S2 — Fewer, better-balanced waves (still barriers)

- **Byte-balanced tiles.** For each GEMV, split rows (or K) into exactly
  `k × workers` near-equal ranges. Concatenate Q, K, and V into one
  3,072-row stage. Interleave gate and up rows, so SiLU·up happens in the
  epilogue of the same tile. The planner's cost becomes *bytes*, not
  `estimated_work = rows × width`.
- **Fuse the small phases into epilogues or prologues:**
  - RMSNorm: each consumer recomputes the 2,048-element statistic (4 KB, in
    the fixed PyTorch reduction order) or reads producer partials.
  - RoPE and cache write: in the QKV tile epilogue when the tile owns whole
    heads.
  - SiLU/product: in the gate/up epilogue.

  The target is 4 barriers per layer instead of 9. Every fusion must keep the
  rounding boundaries in the [BF16 plan](bf16_compute_plan.md). The operator
  unit tests already pin them.
- **GQA-grouped, context-split attention** (flash-decoding style). Each task
  handles one KV head × one context chunk for all 4 query heads of the group.
  A combine step per head follows. Choose the chunk count so the task count
  is a multiple of 20 at long context. Keep the explicit eager-softmax
  rounding contract. See [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).

*Expected:* most of 2.1–2.3 and 2.5 are recovered without any new
synchronization protocol. This alone may reach parity or better at short
context.

### S3 — Weight prefetch across synchronization

- Give each worker a shared-memory ring of weight slots (for example 2–3 ×
  32 KB, sized from the probe). Fill it with `cuda::memcpy_async` and
  `cuda::pipeline` from CCCL/libcu++, which lowers to `cp.async` on SM 8.x.
- The planner inserts `prefetch(tile)` instructions before each `barrier`, so
  each worker starts loading its next tiles before it waits. The GEMV consumes
  from shared memory while the next slot fills. Weight tiles larger than the
  ring stream through it.
- The occupancy probe must now use the worker's **dynamic** shared-memory
  size. `target.json` records it, and the validator rejects plans whose
  prefetch depth exceeds it.
- Read activations written in the same launch with ordinary loads, never
  through the read-only path (`__ldg`, or `const __restrict__` that compiles
  to `LDG.NC`). Weights may use it.

*Expected:* the remaining barrier and small-phase bubbles become invisible to
DRAM. The trace should show near-continuous streaming. Of all the
milestones, this is the most likely to beat graph replay *(hypothesis)*.

### S4 — Dataflow counters instead of grid barriers

- Replace `barrier` with `wait(counter, target)` / `signal(counter)`.
  Counters live in the `SequenceContext`, one per dependency group (for
  example per layer × stage × KV head).
- **Memory ordering.** The producer does: all threads store, then
  `__syncthreads()`, then thread 0 does `__threadfence()` followed by an
  atomic increment (CUTLASS's semaphore pattern). The consumer does: thread 0
  spins on an acquire load with `__nanosleep` backoff, then
  `__syncthreads()`. Use `cuda::atomic_ref` with device scope.
- **Reset without a reset pass.** Counters are monotonic. Each launch passes a
  generation number, and the targets are `generation × producers + k`.
  `enqueue_reset` clears them in stream order. Check headroom against 2³².
- **Deadlock freedom by construction.** The cooperative launch guarantees all
  20 workers are resident. Keep it for that reason even though `grid.sync()`
  goes away. If every worker's instruction order follows one global
  topological order, the lowest-numbered unfinished task always has its
  inputs and can run, so progress is guaranteed. The C++ validator enforces
  this order. Debug builds add a bounded spin that records the stuck counter
  and traps.
- **Testing.** Compute Sanitizer racecheck covers shared memory only, so
  global ordering bugs need their own test. Add a debug mode that inserts
  random `__nanosleep` delays before tasks, and compare against the S3
  barrier executor. Without split-K the outputs must stay bit-identical.
- Then add split-K Wo/down with deterministic reduction and producer-side
  norm statistics, so the partial dependencies in Section 4 are expressed in
  the graph.
- **Ablation.** Add a dynamic global work queue (atomic task claim, no
  prefetch). Comparing it with static assignment plus prefetch is a result in
  itself.

*Expected:* smaller gains than S3 at short context. The value is in the
showcase, and in long context where attention and Wo overlap.

### S5 — One launch per generation request

- Fold greedy selection into the program as instructions. Logits tiles emit
  per-worker `(max, lowest id)` partials, and one fixed worker combines them
  after a counter wait. Make non-finite logits a device error status (fixes
  review finding 2). Bounds-check the token before the embedding lookup.
- Loop over N tokens inside the kernel. Every worker reads the same EOS and
  status word, so all exit uniformly. Prefetch layer-0 weights during the
  `lm_head` tail.
- *Expected:* a small effect (launch gaps are microseconds against a 13 ms
  token), but it completes the "single megakernel" story. Do it after
  S2–S4.

### S6 — Showcase package

- One command that imports the model, probes, plans each ablation stage,
  checks correctness, runs the controlled sweep, and writes a report.
- **Ablation table**, in ms/token and % of attainable bandwidth: S0 baseline
  → S2 balanced/fused waves → S3 prefetch → S4 dataflow → S4 + split-K/
  split attention → S5 persistent, each at contexts 1/256/1K/2K/4K. Include
  PyTorch eager and graph replay.
- **Timeline figures** from the device trace: baseline (gaps at every
  barrier) next to the final schedule (continuous streaming), annotated with
  bytes per task.
- **Size sweep** with synthetic stacks of the same structure (hidden widths
  such as 512/1024/2048, 4–16 layers). Fewer bytes per kernel make launch and
  dependency overhead dominate, so the megakernel's advantage over graph
  replay should grow as the model shrinks. This shows where the approach pays
  off.

## 6. Measurement and honesty rules for the showcase claim

- Keep the existing protocol: frozen candidates, pilot rounds that select
  tiles, independent final rounds, interleaved order, raw samples, telemetry,
  and hashes. Report batch averages as batch averages.
- Report attainable bandwidth next to every latency. "x% of the measured
  streaming floor" is more convincing than "y% faster than PyTorch".
- The graph-replay baseline uses Hugging Face eager attention, including
  `repeat_kv`. A long-context win against it is real but partly reflects a
  weak baseline. Also report an SDPA-backed graph baseline as a separate
  numerical policy, or state the limitation in the headline.
- To attribute a speedup to *scheduling* rather than to a better operator,
  each ablation step changes one thing. The S0 row-dot microbenchmark and the
  bit-identical S1 → S3 checks provide matched-operator controls.
- Prior art to cite and compare against in the write-up: Hazy Research's
  [Llama-1B "no bubbles" megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
  (on-GPU instruction interpreter, weight prefetch across instructions,
  counter-based dependencies, on H100/B200) and
  [Mirage Persistent Kernel](https://github.com/mirage-project/mirage)
  (compiling a model into an event-driven task-graph megakernel). This
  project's differences: a consumer 8 GB Ampere GPU with no programmatic
  dependent launch, an explicit offline Python planner, and a validated
  static schedule.

## 7. Risks and open questions

- **Small headroom.** If S0 measures attainable bandwidth near 190 GB/s, graph
  replay is already at the floor for short context. The showcase then rests on
  long context, the size sweep, and timeline evidence. Find this out in S0,
  before investing in S3–S4.
- **Shared-memory prefetch and occupancy.** The design is already one block
  per SM. Using about 64–96 KB of dynamic shared memory does not reduce
  residency, but the probe and validator must account for it.
- **Spin-waiting at 1024 threads per block.** A single spinning thread with
  `__nanosleep` keeps the other warps idle at `__syncthreads()`. Confirm with
  the trace that waiting does not take issue slots from in-flight `cp.async`.
- **Numerical policy under fusion and split-K.** Every fusion must keep the
  BF16 rounding boundaries. Split-K changes the order of FP32 summation within
  the calibrated tolerances. Never tighten or refit tolerances to match
  executor output.
- **Program size.** A flat 16-layer program has a few thousand instruction
  records (about 100–200 KB). Instruction fetch stays out of the critical path
  if each worker prefetches its next records with its weights.
- **Open:** is one logical worker per SM still right once workers spin-wait?
  Two smaller workers per SM could overlap one worker's wait with the other's
  streaming. Evaluate only after S4, against the trace, per the AGENTS.md
  policy.

## 8. Cleanup and S1 results

Implemented and checked on `tux`, 2026-10-22, starting from `60ff850`.

### What changed

- **Variants narrowed.** The native BF16 MMA decode path (`warp_gemv_mma`,
  `UseMma` instantiations, `--matrix-kernel`, the `matrix_kernel` package
  field, and the SIMT/MMA benchmark script) is removed. The probe reports
  two worker variants: float32 and BF16 SIMT. Package versions 2–4 are no
  longer read; the loader, planner, and `generate.py` ask for regeneration.
- **Version-5 whole-token program.** TilePlan emits one flat task list:
  embedding, the block graph repeated for every layer (each task names its
  `layer`), final RMSNorm, and logits. Embedding, final norm, and logits are
  split into one contiguous row range per worker. The existing wave
  scheduler places the whole program; the stack `execution` stage list is
  gone. `layer_view()` recovers one layer's schedule for `visualize`.
- **Instruction streams.** The C++ loader validates the program and encodes
  one instruction stream per worker, inserting a `barrier` instruction between
  consecutive waves in every stream. The kernel is a single loop over its
  stream; device opcodes no longer carry a precision suffix, because the
  worker's storage type selects the precision. The per-layer block trace is
  written in the down-projection epilogue, so tracing adds no barriers.
- **Validator.** Besides operand and per-layer ownership checks, the loader
  runs the schedule symbolically over the physical buffers (two alternating
  hidden vectors, shared scratch, per-layer K/V rows, score workspace,
  logits). Every read must observe the value its producer wrote in an earlier
  wave, and no two tasks in one wave may write the same element or write an
  element another task reads. This catches cross-layer scratch reuse that
  the old single-layer check could not express. New loader cases cover a
  stale cross-layer scratch read, swapped layer indices, out-of-range layers,
  and stages tagged as layers.

The Llama 3.2 1B program has 9,180 tasks in 147 waves (1 + 16 × 9 + 2),
encoded as 12,100 instructions (533 KB). The compiled worker is unchanged at
64 registers/thread, 2,264 static shared bytes, and one resident block per SM.

### Verification

- CTest 13/13 (release) and 4/4 (`cpu-debug`, UBSan loader cases); Ruff,
  format, ty, and 26 Python unit tests.
- **Bit identity against `60ff850`.** Fixture stacks and blocks (float32 and
  BF16, tiles 4 and 7) produce identical `check` traces (logits, K/V,
  per-layer block outputs, including a reset), generation, and block runs.
  On the pinned checkpoint (BF16, tile 64, 512 context), a 6-token `check`
  trace with all block outputs and full logits, plus 48 generated tokens, are
  identical to the baseline binary and package.
- Compute Sanitizer (memcheck, racecheck, synccheck, initcheck): all 48
  checks passed both after the MMA/legacy removal and on the final version-5
  worker, with the suite's existing CCCL API-error-reporting workaround.

### A codegen lesson

The first version-5 kernel selected the hidden buffers through a local
pointer array indexed by `layer % 2`. It was 1.2% slower than the baseline
and read about 92 MB more from DRAM per token. Nsight Compute showed the
cause: global-load instructions fell from 82.1 M to 44.0 M because the
compiler lost the pointers' global address space and emitted generic loads
for activations. Selecting the buffer with a conditional restored `LDG`
loads and the baseline DRAM volume. Bisection ruled out the per-worker logits
partition and the task-record copy. Keep pointer selection in the worker
free of dynamically indexed local arrays.

### Measurements (exploratory, clocks unlocked)

Generation of 64 tokens after the 6-token `The capital of France is` prompt,
3 warmups and 7 rounds, three interleaved repetitions per binary:

| Build | µs/token (three runs) |
| --- | --- |
| `60ff850` baseline | 13,265 / 13,267 / 13,270 |
| MMA/legacy removal only | 13,267 / 13,270 |
| Version 5 | 13,196 / 13,195 / 13,194 |

Nsight Compute, one launch at context 64 with cold caches and base clocks:
2,483 MB DRAM reads and 319 M warp instructions for the baseline;
2,479 MB and 333 M for version 5. Version 5 is about 0.5% faster in this
exploratory measurement; this is not a controlled shape/tile sweep. A
plausible source is that each instruction is copied into registers once,
rather than re-read by reference inside the row loops, but that has not been
isolated.

