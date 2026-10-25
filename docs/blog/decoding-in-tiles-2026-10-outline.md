# Decoding in Tiles: A Review of Llama 3.2 1B: outline

Post 1 of the series. Job of this post: state the goal, the why, and introduce
the model through the lens of decoding. No measured results (no baseline
numbers); those start in the next post. Items marked **[verify]** need a
source or measurement before they go in the text.

Figures: `figures/architecture-model.svg` (Fig. 1) and
`figures/architecture-layer.svg` (Fig. 2), made by
[`make_architecture_figures.py`](make_architecture_figures.py).

## 1. The goal

- Batch size 1, lowest latency per token, on an 8 GB RTX 3050 (Ampere).
- Keywords: single-stream decode, time per token, interactive use.
- Why batch 1: achieve lowest possible latency.
- Question the series asks: how close to the bandwidth floor can a
  tile-graph, persistent-kernel design get on a small GPU?

## 2. Prior art: Hazy Research, "No Bubbles" (May 2025)

Link: <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>

**What they did**

- Same model (Llama 3.2 1B, ~1.24B parameters, BF16), batch 1, on H100 and
  B200.
- Problem: stock engines (vLLM, SGLang) leave the GPU idle between kernels.
  About 7 kernels per layer × 16 layers. Three sources of "bubbles":
  1. kernel launch and teardown (~2.1 µs per launch, ~1.3 µs with CUDA
     graphs, on H100),
  2. tail effects: a kernel waits for its slowest block (their example:
     512 blocks on 148 SMs leave 80 SMs idle at the end),
  3. memory-pipeline bubbles: weights cannot be loaded until the previous
     kernel is done, even when they do not depend on it.
- Answer: one **megakernel** for the whole forward pass, with an on-GPU
  **interpreter**. Each SM runs a pre-built instruction sequence (7
  instruction types, fused, e.g. RMSNorm + QKV + RoPE).
- Shared memory split into 13 × 16 KiB **pages**. Instructions request and
  release pages, so the next instruction's weight loads start while the
  previous one is still finishing.
- Fine-grained **counters in global memory** for dependencies; the MLP is
  split into 4 chunks so the down projection starts before the whole hidden
  state is ready.
- Results: ~78% of memory bandwidth on H100 (≈2.5× vLLM, >1.5× SGLang),
  under 680 µs per forward pass on B200 (>3.5× vLLM, >1.5× SGLang), against
  a theoretical limit of ~3,000 passes per second (~330 µs) on B200.
- Their B200 breakdown: ~250 µs activation store/load and consistency waits,
  ~200 µs matvec, ~30 µs waiting on weights, ~40 µs warp sync, ~80 µs setup.

**What stays the same on a 3050**

- The framing: decode is a dependent chain of small ops, and the gaps between
  them are lost time.
- Persistent kernel, ahead-of-time schedule, per-SM instruction streams.

**What is different, and easy to miss**

| | Hazy (H100 / B200) | Here (RTX 3050) |
| --- | --- | --- |
| Time floor per token | ~0.7 ms (H100: 2.48 GB ÷ 3.35 TB/s), ~0.3 ms (B200) | **11.6 ms** (2.47 GB ÷ 212.9 GB/s measured) |
| Size of a bubble relative to the floor | A few µs per kernel is a large fraction: ~5 µs × ~112 kernels is on the order of the floor itself | The same µs are a few percent; one grid barrier measured at 0.79 µs |
| What limits bandwidth | Needs many SMs and deep async copy pipelines to reach TB/s, hence the page scheme | ~8 of 20 SMs already saturate DRAM (~30 GB/s each); the rest are spare capacity for norms, attention, epilogues |
| Dependency tracking | Counters in global memory, chunked | Static dependency waves with grid barriers (cooperative launch) **[tease only]** |
| Granularity | 7 big fused instruction types | ~9,200 small tile tasks assigned to 20 workers **[tease only]** |
| Hardware features | TMA, clusters, larger shared memory | Ampere: `cp.async`, L2 prefetch, no TMA **[verify per-SM shared memory]** |
| Cache | Large L2 **[verify H100 / B200 size]** | 1.5 MB L2: weights never stay resident |
| Success metric | Fraction of peak bandwidth, vs. other engines | Distance to the measured floor |

- Takeaway: the same idea, but the bottleneck moves. On H100/B200 the fight
  is against microsecond overheads that rival the floor. On the 3050 the
  floor is ~16× longer, so the question becomes *sustaining* DRAM bandwidth
  through every barrier and every small op.
- Keep neutral: this is a different regime, not a claim to beat their numbers.

## 3. The RTX 3050 in one box

- Compute capability 8.6, 20 SMs, 1.5 MB L2, 8 GB.
- DRAM: 128-bit GDDR6, **224 GB/s nominal, 212.9 GB/s measured** (streaming
  read probe, ~95% of nominal). The measured number is the yardstick for the
  series.
- Contrast with H100 (3.35 TB/s HBM) and B200 (**[verify]** ~8 TB/s): ~16× and
  ~35× the bandwidth.
- Keywords: small GPU, GDDR6, bandwidth-bound, spare SMs.

## 4. Why autoregressive decode is memory-bound

- Token n+1 needs token n, so there is no parallelism across the sequence.
- Every token reads every weight once: 2,471,628,800 bytes in BF16.
- Arithmetic intensity at batch 1: one multiply-add per two weight bytes
  (~1 FLOP/byte) vs. a GPU needing tens of FLOP/byte to be compute-bound.
  GEMV, not GEMM.
- The bandwidth floor: bytes ÷ bandwidth = 11.6 ms per token. The floor grows
  with the KV cache: 32 KiB per context position (134 MB at 4K tokens).
- Side note: why batching and speculative decoding help real servers, and why
  they are out of scope here.

## 5. Llama 3.2 1B through a decode lens (Fig. 1)

- Walk the overview: embedding → 16 layers → final RMSNorm → `lm_head`.
- Bytes, not architecture: 1.24B parameters, layers 79%, embedding table 21%.
- Tied embedding: input side reads one 4 KB row; output side streams the whole
  525 MB matrix every token (21% of all traffic).
- Strict chain: 16 layers × (attention, MLP). The order cannot change; only the
  work inside a step can be spread across SMs.
- KV cache: the only state carried between tokens; 8 KV heads (GQA) keep it
  small. One sentence on GQA, no more.

## 6. One decode layer (Fig. 2)

- Walk the layer figure, annotated by cost at batch 1:
  - 7 weight matmuls: 121.6 MB per layer, nearly all the time; MLP = 83% of
    the bytes.
  - Small ops (RMSNorm ×2, RoPE, SiLU·×, residual adds): few bytes, but each is
    a dependency, and norms are global reductions.
  - Attention: the only cost that grows with position p.
- Each arrow with a dependency is a place to lose time to a barrier; this
  sets up the "bubbles" discussion of the next posts.
- Side note (2–3 sentences): RMSNorm, RoPE, SwiGLU and GQA are quality
  choices. For decode they matter only through the bytes and the dependencies
  they create.

## 7. What comes next

- Roadmap in one paragraph: tile graph and executor, then fused tasks, then
  weight prefetch across barriers.
- Closing line: the number to watch is time per token against the 11.6 ms
  floor.

## Facts to check before writing

- Hazy: H100 SM count and per-SM shared memory are not in their post, so do
  not quote them from there; get them from NVIDIA datasheets.
- RTX 3050 per-SM shared memory limit (opt-in) and H100/B200 L2 and HBM sizes.
- B200 bandwidth figure (their ~3,000 passes/s implies ~7–8 TB/s).
- Wording of the "5 µs × kernel count" comparison: it is their estimate
  (~770 passes/s vs ~1,350), not a measurement.
