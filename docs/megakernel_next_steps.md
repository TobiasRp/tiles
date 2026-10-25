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

**Progress, 2026-10-23: the Section 3 cleanup, S1, S0 and S2 are done.**
[Section 8](#8-cleanup-and-s1-results) covers the cleanup and S1;
[Section 9](#9-s0-results-where-the-time-goes) has the S0 measurements, which
replace several estimates in Sections 1 and 2 (each marked **S0 measured**).
S2 is written up in [the S2 post](blog/s2-fused-tasks-2026-10.md):
TileInfer now matches CUDA Graph replay at context 1 and beats it by up to
15% at 4K.

**Re-plan, 2026-10-23: S3's probe came back negative.** Weights prefetched
before a grid barrier hide none of the time lost around it; refilled right
after the barrier, a 64 KB ring hides about half of the work that follows,
which projects to at most 0.1 ms per token
([Section 10](#10-s3-probe-results-and-the-re-plan),
[the S3 post](blog/s3-weight-prefetch-2026-10.md)). The ring is not built
into the decoder. The milestones in [Section 5](#5-milestones) are revised:
next is **S2b, short-context attention**, then **S4, fewer wave
boundaries**, and every milestone now opens with a probe gate.

**Re-plan, 2026-10-25: the S2b probe found a bigger problem than planned.**
The S2b gate passed, but its diagnosis was wrong. Attention at short context
is a chain of dependent global-memory round trips per task, and at long
context it is a one-row-in-flight K/V loop that reaches 25 GB/s of 213. A
`attention_probe` replacement with 16-byte loads and one task per block is
3.7 times faster at 4K and 2.8 to 3.7 times faster at context 1 to 256, which
projects to 0.3 ms per token at context 1 and 3.9 ms at 4K
([Section 11](#11-s2b-probe-results-attention-at-the-right-parallelism)).
S2b now covers both ends of the context range and stays first in line.

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

**S0 measured:** a streaming read reaches **212.9 GB/s** (95% of nominal), so
the weight-only floor is **11.61 ms/token**. Graph replay at context 1
(12.84 ms) is 1.23 ms above it; TileInfer (13.03 ms worker + 0.17 ms argmax)
is 1.58 ms above it.

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

**S0 measured:** small. One worker streams about 30 GB/s and 8 workers already
saturate DRAM, so a second or third round with 8–12 busy workers still runs
near full bandwidth. The 64-row tiles cost at most 3.5% against an even split
per GEMV. QKV and Wo waves achieve 180 GB/s and down 196 GB/s in the decoder.

### 2.2 Grid barriers

The committed planner produces 9 waves per layer: norm, QKV, RoPE/cache,
attention, Wo, norm, gate/up, SiLU, down. Each wave ends with a
`grid.sync()`. Adding embedding and final norm gives about **146 grid barriers
per token**. At 2–5 µs each, including the tail of the slowest worker, that is
0.3–0.7 ms *(estimate)*. During a barrier no worker issues weight loads.

**S0 measured:** an empty wave costs 0.79 µs. In the decoder the latency from
the last arrival to release is 0.13–0.16 ms per token, about 1 µs per barrier.

### 2.3 Serial low-bandwidth phases

Norm, RoPE/cache, SiLU/product, embedding, final norm, and the argmax kernel
move almost no bytes. They still cost a wave each, plus a barrier. In the
committed source, `rms_norm` and `final_norm` sum squares serially on thread 0.
In `final_norm`, every block repeats that sum. The tux tree has a
PyTorch-order block reduction for BF16. Greedy selection is a separate kernel
launch per token.

**S0 measured:** 0.40 ms per token inside the worker (SiLU 0.22 ms, the two
norms 0.15 ms, RoPE 0.08 ms) plus **0.165 ms for the greedy-selection kernel**,
which scans all 128,256 logits with one 256-thread block.

### 2.4 Per-SM memory-level parallelism

How well wave tails are hidden depends on how many outstanding bytes one SM
can sustain. That depends on vector width, unrolling, and whether loads are
asynchronous. Little's law at about 224 GB/s and about 0.5–0.6 µs latency
calls for roughly 120–130 KB in flight across the GPU, or about 6 KB per SM
when all 20 are busy. When only 8 SMs are busy, each needs about 16 KB.
Measure the row-dot routine alone (Section S0). That separates kernel
efficiency from scheduling.

**S0 measured:** the row-dot routine reaches 192–210 GB/s with 20 workers and
about 30 GB/s with one. Its 2-byte loads are enough in this access pattern; a
plain grid-stride stream of 2-byte loads reaches only 124 GB/s.

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

**S0 measured, and the prediction was wrong.** Attention costs 0.25 ms at
context 64, 2.4 ms at 1K, and **9.4 ms at 4K**, where it reaches only
14 GB/s against 0.63 ms of unique K/V bytes. Graph replay degrades less: from
context 1 to 4K it gains 7.5 ms per token, TileInfer 9.3 ms. TileInfer is
slower than graph replay at every measured context.

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

> **S3 probe:** the idea above fails on this GPU when the load is issued
> *before* the barrier. DRAM stays saturated until the slowest worker
> finishes, so an early worker's prefetch only takes bandwidth from the
> stragglers, and it has landed long before the barrier releases. Loads
> issued *after* the barrier, into the window where DRAM idles, do overlap
> the post-barrier work, but only up to the ring's capacity (about 7 µs of
> DRAM time for the whole GPU). See
> [Section 10](#10-s3-probe-results-and-the-re-plan).

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

**Every milestone opens with a probe gate** (added after S3). Before the
package format or the worker changes, a microbenchmark on the worker's own
launch shape (`bandwidth_probe`, the operator unit harness, or the timeline
trace) must show that the mechanism recovers its expected share of time.
The gate states its threshold up front. Below it, the milestone is
re-planned instead of built.

Order after the re-plan: S0, S1, S2 (done) → S3 (probed, not built) →
**S2b** attention at the right parallelism (probed, go) → **S4** fewer wave
boundaries → S5 one launch per request → S6 showcase.

### S0 — Baseline and instruments (no kernel behavior change) — done

Results are in [Section 9](#9-s0-results-where-the-time-goes).

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

### S2 — Fewer, better-balanced waves (still barriers) — done

Implemented as fused QKV/gate-up tasks, context-chunked attention, even row
splits instead of tiles, and a faster argmax; results and what did not go as
hoped are in [the S2 post](blog/s2-fused-tasks-2026-10.md).

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

### S3 — Weight prefetch across synchronization — probed, not built

The plan was a per-worker shared-memory ring of weight stages, filled with
`cuda::memcpy_async`, that keeps loading the next task's weights while the
worker waits at a barrier. Phase 1 built it in `bandwidth_probe ring` on
decoder GEMV shapes and the worker's launch (commits `16f412e`, `00a68b8`).

- The ring keeps `project_rows`' per-lane summation order: outputs are
  bit-identical in every mode and geometry.
- With 16–32 KB stages it streams as fast as direct loads. Smaller stages
  lose, because per-stage instruction overhead, not memory, sets the pace.
- **Filled before the barrier, it hides none of the post-barrier time**, nor
  does an L2 prefetch issued there.
- Refilled right after the barrier, 4 × 16 KB hides about 2.3 µs per barrier
  when 5–10 µs of work follows, and costs 0.4 µs per wave when none does.
- `lm_head`, one long wave, gains about 1% (2 × 32 KB stages).

Projected onto the decoder, that is at most 0.1 ms per token. The gate
failed, so the ring stays a probe. Its after-barrier form (4 × 16 KB, refill
right after each wait) is folded into S4's GEMV rewrite. Details are in
[Section 10](#10-s3-probe-results-and-the-re-plan) and
[the S3 post](blog/s3-weight-prefetch-2026-10.md).

### S2b — Attention at the right parallelism — probed, go

S2's attention costs 0.42 ms per token above ideal at contexts 1–64 and
4.1 ms at 4K, in 40 chunk tasks per wave on 20 workers. The probe in
[Section 11](#11-s2b-probe-results-attention-at-the-right-parallelism)
replaced the original diagnosis (block reductions per query head) with two
others:

- **Short context is a chain of dependent global round trips per task**, not
  reduction arithmetic. Each task loads `q` and K, stores scores, reloads
  them for the statistics, stores the statistics, and the values task reloads
  those. That is about 7 µs per task (inferred from 15 µs per wave of two
  tasks), and 40 tasks on 20 workers run two in a row. The warp reductions
  in the dot products cost only 1 µs per layer.
- **Long context is memory-level parallelism.** The K loop keeps one 128-byte
  row in flight per warp with two dependent 2-byte-strided loads each (about
  3 µs per row, derived from the K-only loop), so the 4K waves reach
  25 GB/s. The K/V layout is not the cause: a pure row-read kernel with
  16-byte loads reaches 169 GB/s on the same layout.

- **Probe gate: passed.** Fixed per-task overhead outside K/V reads and grid
  barriers is 22 µs per layer at context 1, 0.36 ms per token against the
  0.2 ms threshold (0.3 ms after discounting the instrumentation's 15%).
- **Wide K/V loads.** A lane loads 16 bytes, a warp loads four rows per
  instruction, and four loads are in flight per lane. `q` stays in registers
  as packed BF16 and one warp reduces each query head's statistics.
- **One task per worker.** Tasks per wave are `kv_heads × chunks`, and the
  planner's rule (the fewest chunks that balance 20 workers, 5) put two
  serial tasks on every worker. Use the fewest chunks that make a task fit
  in shared memory: 1 up to 2,048 steps, 2 up to 4,096. Eight workers
  already pull 170 GB/s with wide loads.
- **A fused single-wave task.** When the prefix fits one chunk, the task keeps
  scores and probabilities in shared memory, finishes the softmax and P·V
  itself, and the barrier between the two attention waves is skipped.
  Every worker evaluates the same `position`, so the skip is uniform. The
  instruction carries the threshold, and the validator checks both paths.
- **Numerics.** Scores and probabilities still round to BF16 at the eager
  points. The FP32 summation order changes (8-lane dot trees, per-lane P·V
  accumulation, the warp-level statistics), so outputs match S2 up to
  rounding, not bitwise. The probe's worst difference is 2.9e-6 in the FP32
  attention vector. The decoder regression and the checkpoint tolerances must
  pass unchanged before any timing counts.

*Expected (hypothesis, from the probe):* attention above ideal falls by 0.3 ms
per token at context 1, 0.7 ms at 256, 1.1 ms at 1K, 1.6 ms at 2K and 3.9 ms
at 4K. That puts TileInfer at about 12.6 ms at context 1, 12.7 ms at 1K and
13.3 ms at 4K.

### S4 — Fewer wave boundaries: fused FFN, split-K, dataflow counters

The S3 probe split S2's 0.72 ms of GEMV excess at context 1 into about
0.32 ms of steady-state streaming below the roofline and about 0.40 ms at
wave boundaries and in prologues. Each boundary costs about 2 µs beyond the
steady-state rate, plus about 1 µs of barrier latency; S2 has 99 of them.
Prefetch cannot hide a boundary, so S4 removes boundaries.

- **Probe gate.** Extend `bandwidth_probe`'s layer program with the
  candidate structure before touching the decoder: an FFN wave fused per
  worker, Wo split by K/V group, and counter waits in place of `grid.sync()`.
  Go if the probe saves at least 0.1 ms per token over S2's barriers.
- **Fused FFN per worker.** Worker *k* computes gate/up rows for product
  chunk *k*, then the down partial over that chunk's columns: `Wdown[:,
  chunk k] · product[chunk k]`, which it produced itself. The gate/up → down
  barrier disappears. The 20 FP32 partials are reduced in fixed order by the
  next consumer, which also adds the residual and rounds `y`.
- **Wo split-K by K/V group.** A Wo task for group *g* needs only that
  group's four query heads, so it can start as soon as group *g*'s
  attention is done.
- **Counters instead of barriers** where dependencies are partial. They
  replace `barrier` with `wait(counter, target)` / `signal(counter)`, and
  counters live in the `SequenceContext`, one per dependency group (for
  example per layer × stage × K/V head). A hard synchronization remains
  where a norm needs the whole vector: after Wo (`r`) and after the FFN
  (`y`).
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
- **The GEMV loop moves to the weight ring.** A worker about to wait issues
  its next task's first stages, then waits. This is the after-barrier form
  that the S3 probe measured, with 4 × 16 KB stages. The same rewrite tries
  `lm_head` with 2 × 32 KB stages (+1% in the probe) and 16-byte loads per
  lane against the 0.32 ms steady-state shortfall, and adopts each only if
  the probe shows a gain on every shape.
- **Testing.** Compute Sanitizer racecheck covers shared memory only, so
  global ordering bugs need their own test. Add a debug mode that inserts
  random `__nanosleep` delays before tasks, and compare against the barrier
  executor. The split-K reductions change the FP32 summation order, so
  checkpoint outputs are held to the calibrated tolerances, not bit
  equality; every split is introduced on its own, so a deviation points at
  one change.
- **Ablation.** Add a dynamic global work queue (atomic task claim, no
  prefetch). Comparing it with static assignment is a result in itself.

*Expected (hypothesis):* 0.15–0.3 ms at short context, from fewer
boundaries and from Wo streaming while later K/V groups still run
attention. More at long context, where attention and Wo overlap.

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
  → S2 balanced/fused waves → S2b short-context attention → S4 fused FFN /
  split-K / dataflow → S5 persistent, each at contexts 1/256/1K/2K/4K.
  Include PyTorch eager and graph replay, and the S3 probe's prefetch result
  as a negative control.
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
  bit-identical S1 and S2 checks provide matched-operator controls.
- Report negative results with the same care as positive ones. A mechanism
  that the probe rejects (S3) belongs in the write-up, with its numbers.
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
- **Small wins per mechanism.** After S2 the remaining 1.3 ms at context 1
  is spread over steady-state streaming (about 0.32 ms), wave boundaries and
  prologues (about 0.40 ms), attention (0.42 ms), and barrier latency
  (0.13 ms). No single mechanism is likely to recover more than 0.3 ms, so
  each must pass its probe gate before it costs a format change.
- **Shared-memory ring and occupancy.** The design is already one block per
  SM. A 64 KB ring plus the worker's 23.8 KB of static shared memory fits the
  99 KB limit and does not reduce residency, but it shrinks L1 to about
  28 KB. The probe and validator must account for both.
- **Spin-waiting at 1024 threads per block.** A single spinning thread with
  `__nanosleep` keeps the other warps idle at `__syncthreads()`. Confirm with
  the trace that waiting does not take issue slots from in-flight `cp.async`.
- **Issue budget.** At 1,024 threads per SM, each 16 KB of weights allows
  only about 350 instructions per thread at full bandwidth. The S3 probe's
  first ring lost up to half its bandwidth to per-stage address arithmetic. Any new
  per-stage or per-wait bookkeeping must stay a few adds.
- **Wide attention changes numerics at the margin.** The wide kernels sum in
  a different FP32 order than S2, and the effective chunk count now depends
  on `position`. A BF16 probability can flip by one unit in rare cases. The
  CPU reference, package regression and checkpoint comparison must run at
  contexts on both sides of each chunk threshold before S2b is accepted.
- **Shared memory.** The fused task keeps up to 4 heads × 2,048 probabilities
  (16 KB, the existing input-vector region) plus an 8 KB reduction buffer.
  The worker's static shared memory grows from 23.8 KB to about 32 KB and the
  L1 shrinks with it.
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

## 9. S0 results: where the time goes

Measured on `tux`, 2026-10-22, at commit `80e9c66` plus the S0 instruments.
GPU clocks were unlocked (SM about 1.95–1.97 GHz under load, memory 6.8 GHz).
All timings are exploratory, not a controlled shape/tile sweep. Artifacts are
in `runs/s0-2026-10-22/` on tux.

### Instruments added

- **Roofline and row-dot probe.** `tileinfer/tests/bandwidth_probe.cu` (test
  build only) has two modes. `stream BYTES` sweeps streaming reads over the
  package's byte count. `rowdot` runs the worker's own `ops::gemv` on every
  Llama projection shape. It runs inside a cooperative kernel with a grid
  barrier per GEMV, on 1–20 active workers, with an even split or the
  planner's 64-row round-robin tiles. Iterations rotate through matrix copies,
  so L2 reuse does not inflate bandwidth.
- **Worker timeline.** `tile_runtime_test --timeline-stack PACKAGE POSITION
  TOKEN WARMUP SAMPLES` records, per instruction, start and end SM cycles and
  the start `%globaltimer`. `%globaltimer` ticks only every 1.024 µs on this
  GPU, too coarse for 2–5 µs tasks. `tileplan.timeline` therefore fits each
  worker's cycles to its timer samples and puts all 20 workers on one
  nanosecond axis. Tracing adds 15–30 µs per token and is off unless
  requested. The first version stored start times in registers across the
  task and cost 10% even when disabled: at 64 registers per thread, those
  live values forced spills. Start times are now stored immediately.
- **Bubble budget and Gantt charts.** `tileinfer/tests/profile_timeline.py`
  traces several contexts, keeps the median launch, and writes
  `timeline-report.json` and two SVGs per context: the whole token and
  layer 8. The analysis splits the token into waves at barrier releases.
  Grouped by decoder phase, ideal streaming at 212.9 GB/s, excess work, and
  barrier latency sum exactly to the traced kernel time.
- The 4K-context package (BF16, tile 64) and the existing PyTorch
  eager/graph comparison now cover contexts 1/256/1K/2K/4K.

CTest gained a trace regression (task and barrier coverage, monotonic
records, budget sums) and the sanitizer suite gained trace cases. CTest
(13/13), the CPU/UBSan suite (4/4), clang-format, Ruff, ty, 27 Python tests,
and all 56 Compute Sanitizer checks pass. The compiled worker is unchanged at
64 registers/thread, 2,264 static shared bytes, and one block per SM.

### Bandwidth

| Measurement | Result |
| --- | ---: |
| Streaming read, best configuration (16-byte loads) | 212.9 GB/s |
| Streaming read, 20 blocks × 1024 threads, 16-byte loads | 212.6 GB/s |
| Streaming read, 20 × 1024, 2-byte loads (grid-stride) | 123.6 GB/s |
| Weight-only floor (2.47 GB at 212.9 GB/s) | 11.61 ms |
| Empty wave (one grid barrier) | 0.79 µs |
| L2 cache | 1.5 MB |

Row-dot routine (BF16, `ops::gemv`), GB/s by active workers with an even
split, and per-GEMV time with all 20 workers:

| Shape | 1 | 2 | 4 | 8 | 12 | 20 | 20 even, µs | 20 tile-64, µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QKV 3072×2048 | 27 | 55 | 114 | 191 | 200 | 200 | 63.0 | 65.2 |
| Wo 2048×2048 | 29 | 58 | 111 | 185 | 188 | 192 | 43.6 | 43.5 |
| gate+up 16384×2048 | 30 | 59 | 116 | 203 | 208 | 208 | 322.4 | 333.7 |
| down 2048×8192 | 31 | 61 | 120 | 203 | 202 | 204 | 164.5 | 162.4 |
| lm_head 128256×2048 | 30 | 59 | 117 | 207 | 210 | 210 | 2497.5 | — |

One worker streams about 30 GB/s. Eight saturate DRAM. **Twelve of the 20 SMs
are therefore not needed to keep DRAM busy.** That capacity is free for
compute-light work (attention, norms, epilogues) if a schedule lets it
overlap with streaming.

### Context sweep (fixed position, ms per decoder call)

| Context | PyTorch eager | PyTorch graph | TileInfer | TileInfer − graph |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.04 | 12.84 | 13.01 | +0.18 |
| 256 | 14.89 | 13.37 | 13.58 | +0.22 |
| 1024 | 15.87 | 14.63 | 15.33 | +0.70 |
| 2048 | 17.47 | 16.22 | 17.63 | +1.41 |
| 4096 | 21.62 | 20.30 | 22.29 | +1.99 |

Every compared top token and logit passed the script's check. TileInfer
excludes the separate greedy-selection kernel here; generation adds it.

### Bubble budget per token (µs)

| Context | Traced | Ideal stream | Barrier sync | Streaming excess | Attention excess | Small phases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13,026 | 11,607 | 159 | 774 | 101 | 384 |
| 64 | 13,156 | 11,617 | 131 | 788 | 224 | 396 |
| 256 | 13,593 | 11,647 | 138 | 782 | 631 | 395 |
| 1024 | 15,352 | 11,765 | 145 | 797 | 2,252 | 393 |
| 2048 | 17,665 | 11,922 | 151 | 790 | 4,409 | 392 |
| 4096 | 22,291 | 12,238 | 139 | 800 | 8,717 | 397 |

At context 64, by phase (16 waves each unless noted; sync is included in the
duration):

| Phase | Duration µs | Ideal µs | Achieved GB/s | Note |
| --- | ---: | ---: | ---: | --- |
| QKV | 1,117 | 946 | 180 | 48 tiles, 2–3 per worker |
| Wo | 742 | 630 | 181 | 32 tiles |
| gate+up | 5,358 | 5,043 | 200 | 256 tiles |
| down | 2,733 | 2,521 | 196 | 32 tiles of 1 MB |
| lm_head (1 wave) | 2,507 | 2,467 | 210 | one contiguous chunk per worker |
| attention | 247 | 10 | 8.5 | one head per task |
| SiLU·up | 219 | 0 | — | 128 tiny tasks, about 2 µs each |
| attention / FFN norm | 75 / 78 | 0 | — | one worker reduces, 19 wait |
| RoPE + cache | 75 | 0 | — | 40 tiny tasks |
| embedding, final norm (1 wave each) | 6 | 0 | — | |

**Explained to within the S0 criterion.** Traced kernel time matches the
untraced fixed-position time to within 15–30 µs, the tracing cost. In
generation (64 tokens after a 6-token prompt, 13,276 µs/token), Nsight
Systems shows the worker at 13,103 µs and greedy selection at 165 µs per
token, leaving 8 µs of launch gaps.

### What this changes in the plan

1. **The short-context gap to graph replay is about 0.35 ms (0.18 ms in the
   worker plus the separate argmax kernel), and three cheap items exceed it.** These are the 0.165 ms argmax kernel (a parallel or
   in-worker argmax should take a few µs), the 0.22 ms SiLU wave (fuse into
   the gate/up epilogue), and the 0.15 ms of single-worker norms (each
   consumer recomputes the statistic, or producers emit partials). Fusing
   RoPE into the QKV epilogue saves another 0.08 ms. Together, about 0.6 ms
   would put TileInfer below graph replay at short context. These go first
   in S2.
2. **Attention is the largest loss and grows linearly with context.** At 4K
   it is 9.4 ms for 0.63 ms of unique bytes. GQA-grouped, context-split
   attention (one task per KV head × chunk, reading each K/V row once, with
   enough tasks for all 20 workers) is the second S2 item and the main
   long-context lever.
3. **Wave quantization matters less than estimated.** 2.1 assumed a fixed
   per-SM share of bandwidth. The measured per-SM rate (30 GB/s, saturation
   at 8 workers) means uneven rounds mostly still saturate DRAM. The
   remaining streaming excess, 0.79 ms, is spread across QKV (172 µs), Wo
   (112 µs), gate/up (316 µs), down (212 µs), and lm_head (40 µs). Each
   streaming wave restarts its DRAM pipeline after a barrier, so S3 prefetch
   is a better lever for most of it than byte-balanced tiles. *(The S3 probe
   disproved this; see Section 10.)* Balanced tiles
   stay in S2 but at lower priority.
4. **Barrier latency is real but small:** about 1 µs per barrier, 0.14 ms per
   token. Fusion in S2 removes about half the barriers.
5. **Free SMs are the megakernel's opportunity.** Only 8 workers are needed
   to saturate DRAM, so a dataflow schedule (S4) can run attention, norms,
   and epilogues on the other SMs while weights stream. That is the
   overlap a sequence of kernels cannot express.

The measured ceiling also changes the showcase targets. At context 1 the
floor is 11.61 ms and graph replay 12.84 ms, so about 9.6% is available. At
4K, graph replay spends 20.3 ms against a 12.24 ms floor, so a near-ideal
attention schedule could beat it by a wide margin there.

## 10. S3 probe results and the re-plan

Measured on 2026-10-23 with `bandwidth_probe ring` (commits `16f412e`,
`00a68b8`); raw runs in `runs/s3-2026-10-23/`. The full write-up, with
figures, is [the S3 post](blog/s3-weight-prefetch-2026-10.md). One layer's
four GEMV waves on 20 workers, time above the 571.3 µs floor, 4 × 16 KB
rings:

| Busy work after each barrier | Direct | Ring, before barrier | Ring, after barrier | L2 prefetch, 32 KB |
| ---: | ---: | ---: | ---: | ---: |
| 0 µs | 23.5 µs | 21.8 µs | 25.0 µs | 27.2 µs |
| 2 µs | 29.4 µs | 29.0 µs | 25.8 µs | 33.7 µs |
| 5 µs | 40.7 µs | 40.5 µs | 31.0 µs | 45.0 µs |
| 10 µs | 59.1 µs | 58.7 µs | 49.8 µs | 63.3 µs |

### What the probe established

1. **Prefetch before a barrier is zero-sum on this GPU.** Eight workers
   saturate DRAM, so until the last stragglers finish, an early worker's
   prefetch only slows them down. When the barrier releases, nothing is in
   flight. This holds for the shared-memory ring and for L2 prefetch alike.
2. **Prefetch after a barrier works, within capacity.** Loads issued as the
   barrier releases overlap the post-barrier work. A 64 KB ring per SM hides
   about half of the 4.5 µs it holds in flight. Without post-barrier work it
   costs about 0.4 µs per wave.
3. **The ring is cheap only if each stage is.** Per-stage overhead is paid
   by all 32 warps, against a budget of about 350 instructions per thread
   per 16 KB. With a constant-stride cursor, 16–32 KB stages tie with direct
   loads, and 4–8 KB stages lose 4–28%.
4. **The ring keeps the numerics.** It is bit-identical to `project_rows` in
   every configuration.
5. **The GEMV excess splits into two parts.** One long wave (`lm_head`)
   streams at 207 of 213 GB/s, so about 0.32 ms per token is steady-state
   shortfall. The other 0.40 ms comes from about 2 µs per wave boundary
   plus prologues. The S2 budget lumped these together.

### What changed in the plan

- S3 is closed as a probe. The ring's after-barrier form moves into S4.
- S2b, short-context attention, is next: 0.42 ms at context 64 and no new
  protocol.
- S4 now aims to *remove* boundaries (fused FFN with split-K down, Wo split
  by K/V group, counters for partial dependencies) rather than hide them.
- Every milestone opens with a probe gate and a stated threshold.

*Targets (hypotheses):* S2b about −0.2 ms and S4 about −0.15 to −0.3 ms at
short context. Together that is about 12.4–12.5 ms at context 1, 3–4% below
graph replay (12.92 ms) and 7% above the 11.61 ms floor.

## 11. S2b probe results: attention at the right parallelism

`attention_probe` (new, `tileinfer/tests/attention_probe.cu`) runs the
Llama 3.2 1B attention waves of 16 layers (32 query heads, 8 K/V heads,
head dimension 64, BF16) in one cooperative launch with one block per SM:
scores wave, grid barrier, values wave, grid barrier. K and V rotate through
16 layers of 4,096-step caches (128 MB) so every layer reads from DRAM.
Variants run the production `ops::attention_*`, an instrumented copy with
`clock64` stamps between phases and two ablations, and the candidates below.
Timings are CUDA-event averages, exploratory (unlocked clocks, SM clock
about 1.85 GHz, random K/V). Repeated sweeps agree within 3%. Raw runs are
in `runs/s2b-2026-10-25/` on tux.

The production variant reproduces S2's trace: 30.6 µs per layer at context 1
(0.43 ms per token beyond the two barriers) and 316 µs at 4K, against 297 µs
of attention waves in the S2 timeline.

### Where S2's attention time goes

Average block, µs per layer (the instrumented copy runs about 15% slower than
production):

| Context | dot | max | sum | combine | prob | P·V | final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6 | 4.8 | 5.7 | 5.8 | 1.8 | 2.1 | 4.2 |
| 64 | 5.9 | 5.7 | 6.1 | 6.0 | 3.2 | 3.0 | 4.6 |
| 256 | 19.7 | 6.5 | 6.5 | 6.8 | 4.5 | 7.4 | 4.8 |
| 1024 | 46.5 | 7.5 | 6.9 | 7.4 | 5.2 | 21.3 | 5.6 |
| 4096 | 149.7 | 9.3 | 7.1 | 13.3 | 8.2 | 133.4 | 16.5 |

- At contexts 1–64, max, sum, combine, prob and final are 22 to 26 µs per
  layer (0.36 to 0.41 ms per token, stamping included) and do not depend on K/V
  bytes. Each of
  the two tasks per block repeats the sequence for four heads, with a block
  reduction or a thread-0 loop and a `__syncthreads` each.
- Removing the warp reductions from the dot products saves 1 µs per layer
  (context 64: 5.9 → 5.1 µs). Skipping the `q` loads as well saves 7 µs per
  layer at context 256 (18.5 → 11.1 µs): `q` is reloaded from global memory
  for every time step.
- At 4K the K-only loop (no `q`, no reductions) still takes 141 µs for 4.2 MB,
  30 GB/s. The P·V loop takes 133 µs for 4.2 MB, 31 GB/s.

### Candidates against a pure row-read floor

Layer time in µs, both waves with their barriers (the empty program costs
3.4 to 5 µs). `production` uses 5 chunks. The floor is a kernel that only
reads the rows, with the chunk count of the wide column.

| Context | production | wide, chunks | fused, 1 chunk | row-read floor |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30.7 | 15.2 (1) | 10.9 | 5.6 |
| 64 | 38.2 | 16.7 (1) | 12.3 | 6.2 |
| 256 | 61.3 | 21.1 (1) | 16.6 | 7.5 |
| 1024 | 101.5 | 38.0 (2) | 35.7 | 17.6 |
| 2048 | 154.0 | 55.9 (2) | 60.4 | 28.9 |
| 4096 | 331.0 | 89.1 (2) | — | 49.6 |

- **The row-read floor is 169 GB/s at 4K** (8.4 MB in 49.6 µs), 79% of the
  213 GB/s roofline, for the decoder's own `[time][kv_head][head_dim]`
  layout. A head-major layout gives the same time (50.0 µs), so no format
  change is needed.
- **Wide loads give 3.7 times the production speed at 4K** (89 µs against
  331 µs) and stay 1.8 times above the floor. The rest is dot arithmetic
  (32 FMAs and 12 shuffles per four rows) and four heads' P·V groups each
  re-reading the V rows from L1.
- **Fewer tasks win.** Chunks 1 and 2 give 8 and 16 tasks, one per worker.
  At 4K, 2 chunks cost 89 µs against 108 (5), 106 (4), 123 (3) and 134 (8);
  3 and 4 chunks (24 and 32 tasks) give some workers a second serial task.
  Eight busy
  workers pull 173 GB/s.
- **Two rows in flight beat four** (78 against 83 µs at 4K, chunks 2), and
  eight are slower (138 µs), probably from register spills (not checked).
- **The fused task** saves another 4 µs per layer at short context (10.9
  against 15.2 µs at context 1) by dropping the barrier and the second
  task's round trips. At 2K it is no longer better than 2 chunks.

### Projected change per token

Savings are 16 times the per-layer difference between production and the best
candidate, assuming the attention waves stay serial with the rest of the
token (they are separated by barriers):

| Context | Best candidate | Saves per layer µs | Saves per token ms | TileInfer today → projected, ms |
| ---: | --- | ---: | ---: | --- |
| 1 | fused, 1 chunk | 19.8 | 0.32 | 12.90 → 12.58 |
| 64 | fused, 1 chunk | 25.9 | 0.41 | |
| 256 | fused, 1 chunk | 44.7 | 0.72 | |
| 1024 | fused, 1 chunk | 65.8 | 1.05 | 13.79 → 12.74 |
| 2048 | wide, 2 chunks | 98.1 | 1.57 | |
| 4096 | wide, 2 chunks | 241.9 | 3.87 | 17.20 → 13.33 |

llama.cpp with flash attention and f16 K/V takes 13.67, 13.81 and 14.31 ms at
those three contexts
([runtime comparison](runtime_comparison_plan.md)). If the projection holds,
TileInfer would lead it at all three, by 8%, 8% and 7%.

### What was not shown

- The probe isolates the attention waves with random K/V, so near-uniform
  softmax. It does not include the Wo, QKV or FFN waves, and the decoder
  regression has not been run on the candidates.
- The wide kernels hard-code head dimension 64 and four heads per K/V head,
  and keep probabilities in a 16 KB shared buffer (2,048 steps per task).
  Other models need the generic path or a global-memory probability buffer.
- The remaining gap to the floor at 4K (89 against 50 µs) is not
  decomposed. The next candidates are a shared P·V pass for all four heads
  and `q` in registers across the K loop.
- The latency of one global round trip on this GPU was inferred (about 1 to
  1.5 µs per dependent step), not measured with a pointer-chase.
