# Development log

How Tiles went from a two-layer MLP to a Llama 3.2 1B decoder that runs at
96% of the RTX 3050's streaming floor, in October 2026. The current design is
in [the architecture document](architecture.md); this document keeps the
milestones, their measurements, and what did not work.

Unless a section says *controlled*, its timings are exploratory: unlocked
clocks, one session. Raw artifacts of every milestone are in `runs/` on the
development machine and are not part of the repository.

## Final results

S6, 2026-10-27, at `c95bc03`, controlled: SM clock locked at 1,950 MHz and
memory at 6,801 MHz, runtimes interleaved, both spawn orders, Compute
Sanitizer passing at the same commit. Fixed-context decode, logits ready,
ms per token; percentages are floor ÷ time, the floor being weight and K/V
bytes at 212.9 GB/s:

| Context | Floor | TileInfer | llama.cpp f16 K/V | llama.cpp BF16 K/V | PyTorch graph | PyTorch graph, SDPA | PyTorch eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.61 | **12.129** (96%) | 13.681 (85%) | 13.738 (84%) | 12.843 (90%) | 12.917 (90%) | 13.956 (83%) |
| 256 | 11.65 | **12.204** (95%) | 13.681 (85%) | 13.740 (85%) | 13.370 (87%) | 13.437 (87%) | 14.769 (79%) |
| 1,024 | 11.76 | **12.383** (95%) | 13.820 (85%) | 14.117 (83%) | 14.640 (80%) | 14.977 (79%) | 15.819 (74%) |
| 2,048 | 11.92 | **12.579** (95%) | 13.996 (85%) | 14.667 (81%) | 16.229 (73%) | 16.945 (70%) | 17.402 (69%) |
| 4,096 | 12.24 | **12.985** (94%) | 14.323 (85%) | 15.666 (78%) | 20.335 (60%) | 21.072 (58%) | 21.480 (57%) |

64-token greedy generation from the prompt `1, 2, 3,`, selection included:

| Runtime | ms per token | tokens/s |
| --- | ---: | ---: |
| TileInfer | **12.156** | **82.3** |
| PyTorch CUDA Graph | 13.089 | 76.4 |
| llama.cpp, flash attention, f16 K/V | 13.922 | 71.8 |
| llama.cpp, flash attention, BF16 K/V | 13.961 | 71.6 |
| PyTorch eager | 14.423 | 69.3 |

All runtimes generate the same tokens. Paired differences against TileInfer
have 95% half-widths of 3 to 12 µs.

- **Graph replay** leaves only 40 µs idle per token, so its loss (0.71 ms at
  context 1, 7.35 ms at 4K) is inside the kernels: GEMV shapes below the
  roofline and eager attention with `repeat_kv`.
- **llama.cpp** loses an almost constant 1.3 to 1.6 ms, most of it between
  tokens: one 513 KB device-to-host copy of the float32 logits takes
  1.19 ms. TileInfer and PyTorch keep logits on the GPU in the decode table,
  so that table is not like-for-like at the API boundary; the generation
  table is.
- **SDPA does not help** PyTorch here: Transformers 4.57 dispatches to
  CUTLASS memory-efficient attention after expanding K/V to 32 heads, which
  costs more than eager attention.

Where TileInfer's own remaining time goes, from the worker trace:

| Context | Traced | Ideal streaming | GEMV below roofline | Attention | Barrier latency | Small phases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.125 ms | 11.607 ms | 322 µs | 93 µs | 100 µs | 4 µs |
| 4,096 | 12.975 ms | 12.237 ms | 331 µs | 283 µs | 120 µs | 4 µs |

## Overview

| Stage | Date | What changed | Context 1 | Context 4K |
| --- | --- | --- | ---: | ---: |
| MLP prototype | 10-12 | cooperative wave executor for a two-layer MLP | | |
| Decoder milestones 1–6 | 10-12 | Llama block, full stack, checkpoint bridge (BF16 weights, float32 math) | 241 ms | |
| Warp parallelism, async runtime | 10-13 | GEMV rework; queued generation on the GPU | 15.4 ms | |
| BF16 compute (format 4) | 10-14 | BF16 activations and K/V with PyTorch rounding | 12.99 ms | |
| S1 (format 5) | 10-22 | whole token as one flat program | unchanged | |
| S0 | 10-22 | instruments, roofline, loss budget | 13.01 ms | 22.29 ms |
| S2 (format 6) | 10-23 | fused small phases, attention by chunk | 12.90 ms | 17.22 ms |
| S3 | 10-23 | weight prefetch ring: probed, not built | | |
| S2b (format 7) | 10-25 | wide attention, fused short-context task | 12.32 ms | 13.16 ms |
| S2c (format 8) | 10-26 | QKV rows split evenly, RoPE in attention | 12.19 ms | 13.05 ms |
| S6a | 10-26 | controlled measurement protocol | | |
| S4 (format 9) | 10-26 | fused FFN with split-K; counter waits probed, not built | 12.11 ms | 12.98 ms |
| S5 (format 10) | 10-26 | selection and token loop in the worker | 12.12 ms | 12.98 ms |
| S6 | 10-27 | one-command showcase, controlled comparison | 12.13 ms | 12.99 ms |

Times are one decoder call at a fixed context, from each stage's own
measurement; S2c onward are controlled. The early rows used shorter
context sweeps.

## Before the megakernel plan

### MLP prototype

The first test of the idea was `h = GELU(W1 x + b1)`, `y = W2 h + b2` in
float32: row tiles as tasks, two waves, one cooperative launch with a
`grid.sync()` between them, and comparisons with PyTorch eager and CUDA
Graph replay. It established the pieces that stayed: the probe of the
compiled worker's own occupancy, the strict loader, CCCL for launches,
streams, buffers, and timing (after a cooperative-launch gate passed on the
RTX 3050), and the one-command pipeline. It could not test early handoff,
because every `W2` row reads all of `h`. Its code was later removed.

### Decoder milestones 1–6

A Llama-style one-token decoder in six steps, each with a PyTorch
reference:

1. **Reference and semantics.** A small independent block with fixed
   weights; RoPE variant, epsilon, GQA head mapping, cache layout, and
   tolerances frozen in fixtures. Sequential cache checked against
   recomputation.
2. **Plan and state ABI.** Typed task operands, a general wave scheduler,
   and an immutable plan separate from a mutable `SequenceContext`
   (position, per-layer K/V, scratch). Reset only moves the position; tasks
   never read past the valid prefix.
3. **One CUDA block,** compared at every intermediate value, with GQA, tail
   tiles, reset, and the capacity boundary.
4. **Full stack:** embedding, 16 layers, final norm, logits, one call per
   token.
5. **BF16 weights** with float32 activations and math.
6. **Checkpoint bridge** for the pinned Llama 3.2 1B, using Hugging Face's
   Q/K layout and split-half rotary pairing (the Meta permutation must not
   be applied twice). All 27 greedy choices of a three-prompt suite matched
   Transformers.

The first full-stack run was correct and slow: 241 ms per token against
24 ms for float32 PyTorch graph replay, at 6% SM throughput. More warp
parallelism per GEMV and a reworked execution path brought it to 15.4 ms the
next day.

### Autoregressive runtime

The blocking `run_token` path copied all logits and K/V to the host after
every token. It was replaced by `DecoderModel` plus `SequenceContext` with
stream-ordered prefill, decode, and generate requests that return tickets,
GPU greedy selection with lowest-ID ties, EOS handled uniformly across
blocks, and diagnostics requested at enqueue time. A follow-up split the
monolithic `worker.cu` by ownership, moved the blocking diagnostics into a
test-only library, and replaced per-request pinned allocations with a
pinned input ring and CCCL copies.

### BF16 compute

Format 4 replaced BF16-weight/float32-activation execution with BF16
weights, activations, and K/V that reproduce the operator dtypes and
rounding points of eager Transformers. A dtype trace and rounding probes
showed that every boundary matters (see
[the numerical contract](architecture.md#numerical-contract)). An
identical-input probe found one real mismatch, in the RMSNorm reduction
order, which now follows PyTorch exactly. The checkpoint tolerances were
calibrated from independent PyTorch runs, after a first provisional limit
proved to reject normal BF16 variation between PyTorch runs as well.

Native BF16 tensor-core GEMV (`m16n8k16`) was built and measured against
SIMT: **2.15 times slower** (28.5 against 13.0 ms per token). At batch one
there is almost no arithmetic for tensor cores to speed up, and the MMA
path's loads were worse. It was removed in S1.

After this step TileInfer was at 12.99 ms per token, faster than PyTorch
eager (14.09) and 1% behind graph replay (12.88).

## The megakernel plan

The goal from here was a measured, explained advantage over PyTorch CUDA
Graph replay on the same GPU, checkpoint, and numerical policy. Three
conclusions framed it:

1. Decode is bandwidth-bound, so faster arithmetic does not help.
2. Scheduling matters only as far as it keeps DRAM busy: barriers,
   unbalanced waves, small serial phases, and attention leave it idle.
3. The claim to show is "no bubbles", reported as a percentage of the
   measured streaming floor.

After S3, every milestone opened with a **probe gate**: a microbenchmark on
the worker's own launch shape had to recover a threshold, stated in
advance, before the package format or worker changed. Every milestone also
kept the reference and checkpoint regressions green and required bitwise
equality whenever the arithmetic order did not change.

### S1: whole-token program

The kernel had hard-coded phases around one repeated block template. S1
made the planner emit one flat program (embedding, every layer, final norm,
logits) with explicit `barrier` instructions, made the kernel a single loop
over each worker's stream, and gave the validator a symbolic dataflow check
over the physical buffers, which catches cross-layer scratch reuse. It also
removed the MMA variant and the old package readers. Outputs were
bit-identical. The program then had 9,180 tasks in 147 waves.

### S0: where the time goes

Instruments: a streaming roofline and a row-dot probe (`bandwidth_probe`),
a per-instruction worker trace aligned across workers from SM cycles and
`%globaltimer` (which ticks only every 1.024 µs here), and a loss budget
whose parts sum exactly to the traced time. Findings, each replacing an
estimate:

- The streaming floor is **11.61 ms** (212.9 GB/s, 95% of nominal).
- One worker streams 30 GB/s and **eight saturate DRAM**, so wave
  quantization (48 QKV tiles on 20 workers) cost little. Twelve SMs are free
  for compute-light work.
- A grid barrier costs 0.79 µs empty, about 1 µs in the decoder.
- Small phases (norms, RoPE, SiLU) cost 0.40 ms per token, plus 0.165 ms for
  the separate argmax kernel.
- **Attention was the real problem at long context:** 9.4 ms at 4K at
  14 GB/s, one task per query head. TileInfer was slower than graph replay
  at every context (13.01 against 12.84 ms at 1; 22.29 against 20.30 at 4K).

### S2: fewer, fatter tasks

Fused RMSNorm into its consumers, RoPE and the cache write into QKV, and
SiLU·up into gate/up; split rows evenly instead of 64-row tiles; split
attention by K/V head and context chunk so each K/V row is read once;
replaced the argmax kernel. The program shrank to 3,068 tasks in 99 waves
and 812 lines of source went away. TileInfer **matched graph replay at
context 1 (12.90 against 12.92 ms) and beat it by 15% at 4K**. Write-up:
[S2 post](blog/s2-fused-tasks-2026-10.md).

### S3: weight prefetch across barriers (negative)

The plan was a `cp.async` shared-memory ring that keeps streaming the next
task's weights while a worker waits at a barrier. The probe kept outputs
bit-identical but showed:

- Prefetched **before** a barrier, it hides nothing: DRAM stays saturated
  until the slowest worker finishes, so an early worker's prefetch only takes
  bandwidth from the stragglers. An L2 prefetch does no better.
- Refilled **after** a barrier, 4 × 16 KB hides about 2.3 µs per barrier,
  at most 0.1 ms per token.

The gate failed and the ring was not built. Write-up:
[S3 post](blog/s3-weight-prefetch-2026-10.md).

### S2b: attention at the right parallelism

The probe overturned the planned diagnosis. Short-context attention was a
chain of dependent global-memory round trips per task, and long-context
attention kept one 128-byte row in flight per warp, reaching 25 GB/s. The
rewrite uses 16-byte K/V loads with several rows in flight, `q` in
registers, one task per worker, and, up to 384 positions, one fused task
per K/V head with the following barrier skipped. TileInfer became **faster
than graph replay at every context: 12.32 against 12.85 ms at context 1 and
13.16 against 20.31 ms at 4K (−35%)**. The new FP32 summation order flips
one checkpoint step's greedy choice on a one-unit BF16 near-tie; every
tolerance passes.

### S2c: balanced QKV

QKV tasks owned one head each so that RoPE fitted in the task, which put 48
tasks on 20 workers and repeated the norm per task. S2c splits the 3,072
`[Q; K; V]` rows evenly over the workers and moves RoPE and the K cache
write into the attention task, which already owns its K/V head. Outputs are
bit-identical to S2b; the token is about **0.11 ms faster** at every
context (controlled).

### S6a: controlled measurement

S2c and S4 each aimed at about 0.1 ms, inside the spread of exploratory
timings, so the measurement protocol came before them
([measurement](architecture.md#measurement)). It resolves about 10 µs per
token in one session and 5 µs with both spawn orders, and it found that the
second-spawned server is consistently 1 to 8 µs faster than the first. Under
it, TileInfer was 5.1% faster than graph replay at context 1 and 35.7% at
4K, and faster than llama.cpp at every context.

### S4: fewer wave boundaries

- **Fused FFN (built).** Each worker computes the gate/up rows of one
  product chunk and that chunk's FP32 partial of the down projection, with
  `Wdown` stored transposed; one reduction wave sums the partials in fixed
  order. The probe found the transposed `Wdown` worth as much as the fusion,
  and one reduction of each worker's own slice cheaper than every consumer
  repeating it. In the decoder it saved **70 to 75 µs per token**
  (controlled), a third of the probe's 0.22 ms. Outputs match within the
  calibrated tolerances, not bitwise, because the summation order changed.
- **Counter waits around attention (negative).** The decoder's own package
  ran with the barriers around attention replaced by dependency counters.
  Outputs stayed bit-identical even with random delays, but the token got at
  most 16 µs faster and up to 16 µs slower against a 60 µs gate. `Wo` needs
  every head, and the work on both sides of each barrier is already
  balanced, so a counter only replaces a barrier's latency with its own. The
  counter executor was not built.

Write-up: [S2b to S4 post](blog/s2b-to-s4-2026-10.md).

### S5: one launch per request

Greedy selection moved into the logits tasks, and the token loop into the
worker, 64 steps per launch because of the display watchdog. Non-finite
logits became a device error. Outputs are bit-identical to S4. Generation
got about 10 µs per token faster, as expected: the point was the
single-kernel design, not speed.

The first version was 0.2 to 0.45 ms per token *slower*. At 64 registers
per thread, values that had been launch constants became expensive once
they varied per step: a per-step copy of the shape struct landed in local
memory, a 64-bit division by a run-time position compiled to a software
routine that slowed every task, and the step's token passed by value cost
80 µs. The fixes were a constant shape, 32-bit chunk arithmetic bounded by
the loader, and the step state in volatile shared memory. Every loop change
is now checked against the decoder's own controlled timing.

### S6: showcase

`experiments/showcase/showcase.py run` goes from the checkpoint to a report
with figures (33 minutes for the final run), reusing any step whose input
hashes match.
It builds, probes, imports, verifies against the checkpoint tolerances, runs
CTest, converts the GGUF and checks llama.cpp parity, measures the roofline,
runs the controlled sessions and generation, traces one token, records
Nsight kernel timelines of every runtime, and draws the figures in
`docs/blog/figures/s6-*.svg`. Compute Sanitizer runs separately
(`showcase.py sanitize`), and the report states whether it covers the
measured worker. The results are at the top of this document.

## Lessons

- **Measure the floor first.** Every claim is a percentage of a measured
  roofline, and the S0 loss budget decided the order of the later
  milestones.
- **Probe before changing the format.** Two of the most plausible
  mechanisms, weight prefetch (S3) and dataflow counters (S4), failed their
  gates on this GPU; neither reached the decoder.
- **Diagnoses change.** S0 expected wave quantization to dominate; it was
  attention. S2b expected reduction arithmetic; it was round trips and
  memory-level parallelism.
- **Static assignment helps here.** With about one task per worker per wave
  and DRAM saturated by eight SMs, a dynamic queue has little to balance.
- **Codegen at the register limit.** A worker at 64 registers loses 0.1 ms
  or more from a value moving into a register, local memory, or a 64-bit
  division. It happened in S0 (trace timestamps), S2b, and S5.
- **Consumer GPUs constrain the design.** No programmatic dependent launch,
  no TMA, a small L2, and a display watchdog all shaped the result.

## Open questions

- A matrix prefill kernel for long prompts.
- Closing the last 0.32 ms of GEMV streaming in QKV, Wo, and the FFN, for
  example with 16-byte weight loads.
- Two smaller workers per SM instead of one.
- Stronger PyTorch baselines (`torch.compile`, an SDPA dispatch that keeps
  grouped K/V) and a pinned logits buffer for llama.cpp.
- Porting the planner to another GPU, which needs a new probe and possibly
  different choices.

## Prior art

Hazy Research's [Llama-1B "no bubbles" megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
runs an on-GPU instruction interpreter with weight prefetch across
instructions and counter-based dependencies on H100 and B200.
[Mirage Persistent Kernel](https://github.com/mirage-project/mirage) compiles
a model into an event-driven task-graph megakernel. Tiles differs in its
target (an 8 GB consumer Ampere GPU without programmatic dependent launch),
in its explicit offline Python planner, and in a static schedule that the
runtime validates before running. On this GPU, prefetch across barriers and
counter dependencies did not pay.
