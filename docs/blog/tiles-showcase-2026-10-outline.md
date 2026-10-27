# Tiles on a small GPU: a static megakernel against PyTorch and llama.cpp: outline

Final post of the series. Its job: show the measured result of the final
TileInfer (S5) against PyTorch eager, CUDA Graph replay and llama.cpp on the
RTX 3050, explain how tile-based static scheduling produces it, and say which
choices are specific to this GPU. It does not retell the milestones; the
earlier records ([baseline](baseline-2026-10.md), [S2](s2-fused-tasks-2026-10.md),
[S3](s3-weight-prefetch-2026-10.md), [S2b to S4](s2b-to-s4-2026-10.md)) hold
those. Every number comes from one showcase run, `runs/s6-2026-10-27` on
tux (`experiments/showcase/showcase.py run`, commit `c95bc03`, locked
clocks); its `report.md` holds the tables, and
[Section 20 of the plan](../megakernel_next_steps.md#20-s6-results) the
discussion.
Items marked **[verify]** need a source or a measurement before they go in
the text.

Figures (all in `figures/`, drawn by [`make_s6_figures.py`](make_s6_figures.py)
from that run):

| Fig. | File | Carries |
| --- | --- | --- |
| 1 | `s6-comparison.svg` | the headline: ms per token against context, with the floor |
| 2 | `s6-kernels-ctx1.svg` | what each runtime's token looks like on the GPU |
| 3 | `s6-pipeline.svg` | offline planner, online executor |
| 4 | `s6-layer-tiles.svg` | how one layer becomes tiles |
| 5 | `s6-waves.svg` | waves and barriers, the conditional barrier |
| 6 | `s6-assignment-ctx4096.svg` | the static assignment and its balance |
| 7 | `s6-timeline-ctx1.svg`, `s6-timeline-ctx4096.svg` | the full token on 20 workers, the loss budget |
| 8 | `s6-token-loop.svg` | 64 steps per launch, selection on the GPU |
| 9 | `s6-rtx3050.svg` | the hardware constraints behind the design |
| 10 | `s6-prior-art.svg` | Hazy Research and Mirage against this design |
| – | `s6-kernels-ctx4096.svg`, `s6-assignment-ctx1.svg` | backups for questions |

## 1. The result (Fig. 1, decode table)

- One sentence of setup: Llama 3.2 1B, BF16, batch 1, RTX 3050 8 GB, the
  same weights in every runtime, logits ready, no sampling.
- Headline: TileInfer 12.13 ms per token at context 1 and 12.99 ms at 4K,
  96% and 94% of the streaming floor (11.61 and 12.24 ms). Graph replay
  12.84 and 20.34 ms (TileInfer 5.6% and 36% faster); llama.cpp f16 K/V
  13.68 and 14.32 ms (11% and 9% faster); eager 13.96 and 21.48 ms.
- The floor: weights plus K/V bytes at the measured streaming rate. Report
  "x% of the floor" next to every latency (rule from Section 6 of the plan).
- How measured, in two sentences: interleaved resident processes, paired
  differences with bootstrap intervals, both spawn orders, the clocks.
- Generation, 64 tokens, selection included: TileInfer 82.3 tokens/s,
  graph replay 76.4, llama.cpp 71.8 (f16 K/V), eager 69.3. Same 64 tokens
  everywhere.
- Parity in one paragraph: every runtime picks the same next token at
  every measured context; TileInfer matches Hugging Face eager attention in
  BF16 within the calibrated tolerances, with one step that differs at a
  0.125 near-tie of the reference; llama.cpp's teacher-forced parity has no
  real mismatches.

## 2. Why the baselines lose time (Fig. 2, Nsight table)

- PyTorch graph replay: 787 kernels per token and only 40 µs idle between
  them. Graphs remove the launch gaps; what is left is inside the
  kernels (ramp-up, tails, cuBLAS GEMV shapes) and in eager attention with
  `repeat_kv`, which grows with context.
- PyTorch eager: the same kernels with host launch gaps.
- llama.cpp: 213 kernels per token, a few µs idle between them, then
  1.8 ms idle before the next token. In that gap one 513 KB device-to-host
  copy of the float32 logits takes 1.19 ms. Its attention scales well (flash
  attention), so the loss is nearly constant per token. Be fair: in the decode
  table TileInfer and PyTorch keep the logits on the GPU; generation is the
  like-for-like row. **[verify]** why the copy runs at 0.43 GB/s (pageable
  buffer?).
- TileInfer: one kernel; its losses are inside it (Fig. 7).
- SDPA graph replay is slower than eager attention (21.07 against 20.34 ms
  at 4K): inside the model PyTorch picks CUTLASS memory-efficient attention
  (284 µs per layer at 4K) after expanding K/V to 32 heads; a stand-alone
  SDPA call with grouped K/V takes 84 µs. A separate numerical policy, not
  the reference; say that a better PyTorch path exists.

## 3. Tile-based static scheduling (Figs. 3 to 6)

- Fig. 3: the planner is Python and offline; the executor is C++/CUDA and
  makes no scheduling decision. The probe queries the compiled worker, the
  plan records its fingerprint.
- Fig. 4: rows split evenly over workers (QKV, Wo, FFN, FFN reduce); RMSNorm
  recomputed in each projection task instead of a norm wave; the FFN fused
  per worker with FP32 split-K partials, reduced in a fixed order
  (deterministic, no atomics); attention per K/V head and context chunk.
- Fig. 5: six waves per layer, one grid barrier after each; the barrier
  between the attention waves is skipped while context ≤ 384.
- Fig. 6: bytes per worker per wave are equal by construction; idle workers
  are visible (attention uses 8 or 16 of 20).
- What static buys: validation before launch, bit-reproducible results,
  every byte and every barrier known in advance. What it costs: no
  overlap across barriers (S3's probe said this would gain at most 0.1 ms).

## 4. One token, end to end (Figs. 7 and 8)

- Fig. 7: the whole token at context 1 and 4K. Above ideal streaming:
  518 µs at context 1 (GEMV below the roofline 322, barriers 100,
  attention 93) and 738 µs at 4K (attention 283). QKV and Wo stream at
  190 GB/s, the FFN and `lm_head` at 209.
- Fig. 8: the token loop inside the launch: candidates from the logits
  tasks, every worker combines them, EOS and error words; 64 steps per
  launch because of the display watchdog.

## 5. What is specific to the RTX 3050 (Fig. 9)

- About 8 of 20 SMs saturate DRAM, so the other SMs are slack for norms,
  attention and epilogues; one block of 1,024 threads per SM.
- Grid barriers cost under a microsecond; 99 per token cost about 0.1 ms.
- 1.5 MB L2 against 122 MB of weights per layer: nothing stays cached.
- No TMA and no programmatic dependent launch; plain loads; the prefetch
  ring was probed and not built.
- The 64-register budget: run-time values must not reach hot code as
  registers or divisions (S5's 0.2 ms lesson).
- The display watchdog limits a launch to 64 tokens.

## 6. Prior art (Fig. 10)

- Hazy Research's "No Bubbles" megakernel
  (<https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>) and
  Mirage Persistent Kernel (<https://github.com/mirage-project/mirage>).
- Same idea, a different regime: on H100/B200 the floor is under a
  millisecond and microsecond overheads matter; on the 3050 it is 11.6 ms and
  the job is staying at DRAM rate through every barrier.
- Keep it neutral: no claim against their numbers.

## 7. Limits and what is next

- One GPU, one model, batch 1, greedy, BF16; prefill still runs one token
  per step.
- The PyTorch baseline is Hugging Face eager attention (the numerical policy
  TileInfer reproduces); SDPA reported separately.
- llama.cpp numbers are host wall time around a synchronous API; TileInfer
  and PyTorch numbers are CUDA-event batches.
- Next: a matrix prefill; the remaining loss is GEMV below the roofline.

## Facts to check before writing

- Mirage: GPUs it targets and its dependency mechanism, from its paper or
  README, before filling the empty cells of Fig. 10.
- Hazy: the hardware-feature column of Fig. 10 against their post.
- The speed of llama.cpp's logits copy (Section 2).
