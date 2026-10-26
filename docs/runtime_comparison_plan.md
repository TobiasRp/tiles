# Runtime comparison plan: llama.cpp next to TileInfer and PyTorch

Status: Phases 0 to 2 are done. The harness in `experiments/runtimes/` was
validated on an Apple M1 Pro (CPU, Phase 1) and then used for the measurements
on the RTX 3050 in `tux` (Phase 2, results below).

Phase 1 results (pinned checkpoint revision `4e20de36…`, llama.cpp b11871,
CPU, 6 threads; harness validation only, never compared with CUDA numbers):

- GGUF: 147 tensors, 113 BF16 matrices + 34 F32 vectors (norms and
  `rope_freqs`), no quantized tensor. sha256
  `40241de51bb5634628bf29a583c34bcefe8d31849c677bb26a9ad6b47e272760`; compare
  with the tux conversion in Phase 2 (identical, see below).
- Parity vs Transformers BF16 eager (CPU), teacher-forced, 3 prompts x 32
  steps: 0 real argmax mismatches, 3 mismatches at exact BF16 top-2 ties,
  max top-logit error 0.10 (tolerance 0.15 + 3%). Free-running comparison
  turned out to be fragile: exact ties exist (prompt 1 has " It" and " The" at
  21.625), the tie-break differs, and the sequences then diverge.
- Timing: `rt_probe` and `llama-bench` agree to within about 5% at the same
  thread count (about 207-218 ms/token, flat over contexts 1 to 256). Their
  default thread counts differed (4 vs 6), which first showed up as a 30%
  gap; the harness now passes an explicit count and records it.
- Greedy sampling on the host adds under 1 ms on CPU (within noise).
- Not exercised on the Mac: Metal. The TileInfer/PyTorch report adapters were
  first run on real reports in Phase 2.

## Phase 2 rerun after S2c (2026-10-26, locked clocks)

The same harness with TileInfer S2c (`7a38e10`), SM clock locked at
1,950 MHz and memory at 6,801 MHz, flash attention with BF16 or f16 K/V
only, two passes in opposite order (`run_phase2.sh OUT PACKAGE fa1-bf16
fa1-f16`). Raw records and `report-both.md` are in `runs/s6a-2026-10-26/rt/`
on tux. ms per token, decode only, both passes averaged:

| Context | TileInfer | llama.cpp FA, f16 K/V | llama.cpp FA, BF16 K/V | PyTorch graph | PyTorch eager | Floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.19 | 13.68 | 13.73 | 12.85 | 13.99 | 11.61 |
| 256 | 12.27 | 13.68 | 13.73 | 13.37 | 14.76 | 11.65 |
| 1,024 | 12.45 | 13.83 | 14.14 | 14.63 | 15.78 | 11.77 |
| 2,048 | 12.64 | 13.99 | 14.68 | 16.23 | 17.40 | 11.92 |
| 4,096 | 13.07 | 14.32 | 15.69 | 20.34 | 21.49 | 12.24 |

- TileInfer is now the fastest at every context: 11% ahead of llama.cpp's
  fastest configuration at context 1 and 9% at 4,096 (17% against BF16 K/V,
  the matched precision).
- Generation (the `1, 2, 3,` prompt, 64 tokens, greedy): TileInfer 12.24 ms
  per token, PyTorch graph 13.09, llama.cpp 13.91 (f16 K/V) and 14.01 (BF16
  K/V).
- llama.cpp's numbers equal the unlocked ones below within 0.01 ms; every
  runtime picks the same token at every swept context, with the logit offsets
  described below.
- The controlled comparison behind TileInfer's numbers is in
  [the S6a results](megakernel_next_steps.md#15-s6a-results).

## Phase 2 results: RTX 3050, CUDA

Measured on `tux` on 2026-10-25: RTX 3050 (20 SMs, driver 595.99.02, CUDA
13.4, unlocked clocks, nothing else on the GPU), llama.cpp b11871 built for
sm_86 with CUDA graphs on, PyTorch 2.14.1+cu130, TileInfer at the commit in
`env.txt` (after S2 fused tasks). The same BF16 weights everywhere: the tux
GGUF hash equals the Mac one (`40241de5…2760`). Everything ran in one session
as two passes in opposite order (`scripts/run_phase2.sh`); the two passes
differ by at most 0.5% in any cell (median 0.0%). Raw records, per-pass
reports and Nsight reports are in `runs/rt-2026-10-25/` on `tux`.

### Decode latency versus context (ms per token, logits ready, no sampling)

Fixed-position timing as in the S0 baseline: the prefix is filled first and
each timed call overwrites the same slot. llama.cpp numbers are from
`rt_probe`; CUDA graphs on.

| Context | PyTorch eager | PyTorch graph | TileInfer | llama.cpp FA, f16 K/V | llama.cpp FA, BF16 K/V | llama.cpp no FA, BF16 K/V | Floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.96 | 12.85 | 12.90 | 13.67 | 13.74 | 14.10 | 11.61 |
| 256 | 14.75 | 13.37 | 13.14 | 13.67 | 13.73 | 14.08 | 11.65 |
| 1,024 | 15.81 | 14.64 | 13.79 | 13.81 | 14.11 | 14.20 | 11.77 |
| 2,048 | 17.39 | 16.23 | 14.63 | 13.98 | 14.67 | 14.42 | 11.92 |
| 4,096 | 21.48 | 20.33 | 17.20 | 14.31 | 15.67 | 15.06 | 12.24 |

- **Short context:** TileInfer and PyTorch's CUDA Graph are tied (12.9 ms, 11%
  above the floor). llama.cpp is 6% slower at context 1 (13.7 ms, about
  181 GB/s of the 212.9 GB/s measured).
- **Crossover near 1,000 tokens:** TileInfer and llama.cpp with flash
  attention are equal at 1,024. Beyond that llama.cpp's attention scales much
  better: at 4,096 it is 17% faster than TileInfer (14.3 vs 17.2 ms, 17% above
  the floor against TileInfer's 41%) and 30% faster than PyTorch's graph.
  This is the attention weakness S2b targets.
- **Matched precision costs llama.cpp:** with BF16 K/V (what the others use)
  flash attention is slower than with f16 K/V, 9.5% at 4,096 (15.67 vs
  14.31 ms) and 0.5% at context 1. It still beats TileInfer by 9% at 4,096.
- **Flash attention helps in both:** without it llama.cpp is 3% slower at
  context 1 and 5% slower at 4,096 (f16 K/V). K/V type makes no difference
  without flash attention, which points to the flash-attention path as the
  source of the BF16 penalty.

The BF16 penalty is explained by the kernels. Nsight Systems on the probe
(a 4,095-token prefix filled one token at a time, then decode) shows that with BF16 K/V flash attention runs
`flash_attn_ext_f16` (the MMA kernel the source chooser picks on sm_86), fed
by 32 `convert_unary_cont_vec4<bf16, half>` launches per token (K and V of 16
layers): 261 kernels per token against 213 with f16 K/V, and 5.1% of GPU time
(averaged over contexts 0 to 4,095). About 90% of kernel time is the 97
`mul_mat_vec_f<bf16>` launches (weights), which makes llama.cpp's own BF16
mat-vec kernel, not cuBLAS, the weight path. TileInfer runs one launch per
token.

### Other llama.cpp settings

| Setting (flash attention on) | Context 1 | Context 4,096 |
| --- | ---: | ---: |
| f16 K/V, CUDA graphs on | 13.67 | 14.31 |
| f16 K/V, CUDA graphs off | 13.95 | 14.59 |
| BF16 K/V, CUDA graphs on | 13.74 | 15.67 |
| BF16 K/V, CUDA graphs off | 14.11 | 15.98 |

CUDA graphs are worth 0.3 ms (2%) in llama.cpp. `llama-bench` agrees with the
probe within 0.25 ms (1.6%) at every context; it is slightly higher because its
cache grows by 16 tokens during a test.

### Sampling

llama.cpp's greedy sampling on the host adds **0.21 ms per token** (1.5%),
the same at every context and configuration (13.89 vs 13.67 ms at context 1).
TileInfer's separate argmax kernel measured 0.165 ms in S0. The TileInfer
fixed-context numbers above exclude it, so TileInfer plus its argmax is about
13.1 ms at context 1, against llama.cpp plus sampling at 13.9 ms. Once
TileInfer moves sampling into the token, the "decode only" TileInfer rows need
a runtime switch or a trace subtraction.

### Generation (9-token prompt `1, 2, 3,`, 64 new tokens, greedy included)

| Runtime | ms per token | Tokens per second |
| --- | ---: | ---: |
| PyTorch eager | 14.51 | 69 |
| PyTorch graph | 13.09 | 76 |
| TileInfer | 12.94 | 77 |
| llama.cpp FA, f16 K/V | 13.92 | 72 |
| llama.cpp FA, BF16 K/V | 13.98 | 72 |

All three PyTorch/TileInfer runs produce the same 64 ids. The prompt had to
change from "The capital of France is": that prompt reaches an exact BF16
top-2 tie, after which TileInfer and PyTorch eager break it differently, and
the PyTorch ids even depend on the static-cache length. That is a real
property of BF16 greedy decoding, not a bug in either runtime.

### Correctness

- GGUF is BF16-only (113 matrices) and its hash matches the Mac conversion.
- Teacher-forced parity against Transformers BF16 eager on CUDA, 3 prompts x 32
  steps, for flash attention on/BF16, on/f16 and off/BF16: 0 real argmax
  mismatches, 1 tie mismatch each, max top-logit error 0.12 to 0.13 (within the
  0.15 + 3% rule).
- Cross-check at the swept contexts (next token and top logit versus PyTorch
  eager, same filled context): every runtime picks the same token. TileInfer
  stays within 0.06 of PyTorch. llama.cpp differs by 0.27 to 0.35 in every
  configuration, a constant offset across contexts and K/V types. That points
  to llama.cpp's numerics (float32 activations instead of BF16 rounding at
  operator boundaries), not to K/V precision, and is inside the BF16
  tolerance, but it is the largest margin of the three. The sweep context is
  one token repeated, so it checks consistency, not quality.

### Caveats

- The decode sweep fills the context with one repeated token. Attention cost
  does not depend on the values, but the cross-check logit is not a realistic
  one.
- TileInfer and PyTorch use CUDA events around batches of calls; the probe
  uses wall clock with a synchronize per call. The 1.6% bench/probe gap and the
  PyTorch graph match with TileInfer's S0 numbers suggest this is not a large
  effect, but it is not identical instrumentation.
- Clocks are unlocked; the 0.5% pass spread covers the thermal state of this
  session only.
- No investigation yet of why llama.cpp's weight streaming reaches about 85% of
  peak bandwidth at context 1 while TileInfer reaches about 90%.

## Goal

Put llama.cpp on the same latency-vs-context chart as TileInfer, PyTorch eager
and PyTorch CUDA Graph for **Llama 3.2 1B, BF16**, at batch 1.

Yardstick stays the measured 212.9 GB/s floor (11.61 ms/token at context 1).

## Design decisions

### 1. Isolation: a separate project, subprocess boundary

New top-level directory, with its own Python project and lockfile. Nothing
under `tileplan/` or `tileinfer/` changes (`uv.lock` and CMake presets stay as
they are).

```
experiments/runtimes/
  README.md
  pyproject.toml, uv.lock        # own env: gguf, numpy, transformers (conversion/parity only)
  scripts/
    fetch_llamacpp.sh            # clone ggml-org/llama.cpp at a pinned SHA into third_party/
    build_llamacpp.sh            # cmake, CUDA arch 86 on tux; CPU (+Metal) on the Mac
    build_probe.sh               # build probe/rt_probe against a llama.cpp build
  src/rtbench/
    schema.py                    # common result record (below)
    llamacpp.py                  # runs llama-bench and rt_probe, parses -> records
    gguf_tools.py                # HF -> BF16 GGUF conversion, tensor-type check, sha256
    parity.py                    # token/logit parity vs the HF reference
    tileinfer.py                 # reads benchmark_decode/generate reports -> records
    report.py                    # merges records -> tables (SVG figures later)
    cli.py                       # rtbench convert | bench | parity | report
  probe/                         # rt_probe.cpp: greedy + fixed-context decode on libllama
  tests/                         # unit tests for the harness (no model needed)
  third_party/                   # Git-ignored: llama.cpp checkout + builds
  models/, results/              # Git-ignored: GGUF files, result records
```

- llama.cpp is **built from source and driven as a subprocess**
  (`llama-bench` and a small probe). No `llama-cpp-python`, so no CUDA
  wheel/toolchain coupling with the torch 2.14/cu130 env.
- Pin the llama.cpp commit in `scripts/` (recorded in every result file) and
  the CUDA toolkit used. Rebuild when either changes.
- The existing PyTorch and TileInfer benchmarks are **not re-implemented**.
  The adapter ingests their JSON; the only shared piece is the record schema
  and the report.
- `.gitignore` gets `/experiments/runtimes/third_party/` and
  `/experiments/runtimes/models/`. `AGENTS.md` gets one line pointing here.
- Why a new directory rather than a new `tileplan` group: llama.cpp needs a
  different toolchain lifecycle (C++ build at a pinned SHA) and different
  Python deps (`gguf`). Keeping them out of `tileplan/uv.lock` means
  experiments cannot disturb the planner or `generate.py`.

### 2. Common result record

One JSON line per (runtime, model, test point):

`runtime`, `runtime_version`/commit, `model`, `weight_dtype`, `kv_dtype`,
`device`, `driver`/`cuda`, `test` (`decode_fixed_ctx` | `generate`), `sampling` (`none` = logits ready, `greedy` = next token id
known), `context_length`, `samples_us[]`, `mean_us`, `notes` (what else is
included/excluded: logits copy, launch gaps), `flags`.

`sampling` is part of the record key, so the report can show both variants
side by side. TileInfer rows today are `none` for the fixed-context decode
benchmark and `greedy` for generation; once the runtime folds sampling into the
token program, its fixed-context rows become `greedy` and the `none` variant
needs a runtime switch or a subtraction from a trace (record which).

This makes "what each number includes" explicit, which is the main honesty
problem when comparing runtimes.

### 3. Fairness rules (to write into the README and the post)

| Item | TileInfer / PyTorch today | llama.cpp setting | Caveat to record |
| --- | --- | --- | --- |
| Weights | HF BF16 safetensors | `convert_hf_to_gguf.py --outtype bf16`; verify no tensor was quantized (all `bf16` except norms, which GGUF keeps F32) | Norm weights are F32 in GGUF, tiny |
| Activations | BF16 storage, FP32 accumulation | ggml computes in F32 activations with BF16 weights | Not the same rounding; parity is by tolerance, not bits |
| KV cache | BF16 | `-ctk bf16 -ctv bf16`; also report the llama.cpp default f16 | **Verify** the CUDA kernels accept BF16 K/V without a CPU fallback or a slow path; watch GPU use and logs |
| Attention | eager (torch), per-head tasks (TileInfer) | `-fa 1` (primary) and `-fa 0` | Report both; flash attention is what a user would run |
| Offload | all on GPU | `-ngl 99` | Check the load log says 17/17 layers |
| Context sweep | positions 0, 255, 1023, 2047, 4095 (fixed position, same slot overwritten) | `llama-bench -d <pos> -n 32 -p 0` for `tg @ d<pos>` | llama-bench *appends* tokens, so depth grows by n during the test: use small n (e.g. 16) and report depth as context. It also returns logits to the host each token (sync); note it. |
| Sampling | TileInfer's decode benchmark excludes its separate 0.165 ms argmax kernel; its generation run includes it. The runtime will move logits sampling into the measured token in the future | llama-bench times `llama_decode` only (logits are read back, no sampler). The probe adds a greedy sampler step | Every runtime gets two numbers per point: **decode** (logits ready, no sampling) and **decode + sampling** (next token id known). Never mix them in one column |
| Clocks | unlocked, ~0.6% drift | same | Run all runtimes in the same session, interleaved |
| Prompt | 6-token prompt + 64 new tokens (generation test) | same token ids, fed with `llama-bench -pg` or the probe | Tokenizer differences: feed **token ids**, not text, for timing |

Report as "exploratory" or "controlled" per AGENTS.md: the context sweep and
generation row are controlled; anything with different flags is exploratory.

### 4. Correctness before numbers

Timing alone is not comparable if llama.cpp computes something different.
`llamacpp_parity.py` runs a tiny probe (either `llama-completion` with
logits/probs output, or a ~100-line C++ program against `libllama` that does
greedy decode on token ids and prints the top token and logit):

- the frozen three-prompt suite from `tileinfer/tests/llama32_checkpoint.py`
  (27 greedy decisions there), extended to 32 new tokens per prompt, run
  **teacher-forced**: the probe is fed the reference's tokens and reports its
  own argmax and top logit at every step, so each step has the same context;
- compare top-1 logit at the benchmark positions with the BF16 tolerance
  already used in `benchmark_decode.py` (atol 0.15, rtol 0.03);
- split argmax mismatches into real ones and exact near-ties (reference
  top-2 gap within the logit tolerance); ties are expected in BF16 and are
  not failures, real mismatches are.

The tokenizer is checked separately (llama.cpp's GGUF tokenizer vs HF
`tokenizers` on the suite) because a mismatch would look like a model error.

## Phases

### Phase 0: scaffold (Mac, no GPU)
- Create `experiments/runtimes` skeleton, pyproject, schema, ignores, README.
- `fetch_llamacpp.sh` / `build_llamacpp.sh` with a `--backend cpu|metal|cuda`
  switch (CUDA only for `tux`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`).
- Needs the gated Llama 3.2 1B checkpoint at the pinned revision (not present
  on this Mac: `checkpoints/` does not exist). Run `hf auth login` and
  download it, as in `tileplan/README.md`.
- Done when: `llama-bench` runs on this Mac's CPU against a BF16 GGUF.

### Phase 1: llama.cpp text track on the Mac CPU (harness validation)
- `convert_gguf.sh`, GGUF tensor-type check, sha256 of the GGUF recorded.
- `llamacpp_bench.py` + record schema + `report.py` with fake/CPU data.
- Parity probe: tokens and logits vs HF **on CPU** (torch CPU BF16) for the
  prompt suite. This validates the tooling and finds tokenizer problems.
- CPU numbers are **not** published or compared; they only prove the harness.
- Done when: the whole pipeline produces a report from the Mac with correct
  parity.

### Phase 2: CUDA on tux (done)
- Rebuild llama.cpp with CUDA arch 86; regenerate the GGUF from the pinned
  checkpoint on tux and compare its sha256 to the Mac one (don't copy files
  from the Mac).
- Probe BF16 KV / flash-attention behaviour first (see fairness table); settle
  the flag set before measuring.
- Run, in one session and interleaved: TileInfer, PyTorch eager, PyTorch
  Graph (existing scripts), llama.cpp `-fa 1` / `-fa 0`, KV bf16 / f16.
- Add: effective bandwidth (bytes read / time), Nsight Systems kernel count
  per token for llama.cpp (to show the launch-count story the blog already
  tells for PyTorch), and CUDA-graph use (`GGML_CUDA_GRAPHS` default on;
  record whether it was active).
- Done when: the context sweep and generation row for four runtimes sit in one
  table/SVG, with parity results next to them. Done as tables above; SVG
  figures are left for the blog post.

## Risks and open points

### Resolved by reading the pinned source (b11871)

- **BF16 weights on CUDA:** batch-1 BF16 matrix products take llama.cpp's own
  mat-vec kernel (`mmvf.cu`, accepted for BF16 when the batch is 1 on Ampere),
  not cuBLAS. That is the path being measured.
- **BF16 K/V with flash attention:** a `bf16-bf16` vector kernel is compiled
  by default (`GGML_CUDA_FA_QUANTS`). Mixed pairs such as `f16`/`bf16` would
  fall back to an f16 conversion with a warning; use the same type for K and V.
  On compute capability 8.6 the kernel chooser sends Llama's decode shape
  (head size 64, GQA 4, padded KV) to the **MMA kernel, not the vector
  kernel**, which per the source comment converts K/V to f16 first. That
  would add a K/V conversion pass per token and f16 range limits to BF16
  values. **Confirmed on tux:** 32 conversion kernels per token and a 9.5%
  penalty at context 4,096, so both K/V types are reported, with BF16 K/V
  as the matched-precision row.
- **CUDA graphs:** on by default in CUDA builds (`GGML_CUDA_GRAPHS`);
  `GGML_CUDA_DISABLE_GRAPHS=1` turns them off at run time. Measure both; the
  graph is rebuilt per token, so "graphs on" is not the same as PyTorch graph
  replay.
- **Sampling:** llama-bench does not sample; sampling needs the probe (see
  fairness table). The probe samples on the host (`llama_sampler_sample`).
  The pinned `llama.h` also has a backend-sampling path (samplers attached to
  the context), which would put llama.cpp's greedy selection on the GPU like
  TileInfer's. Add it as a third sampling variant if the host path turns out
  to dominate on tux.
- **Tokenizer:** feeding token ids avoids it for timing; the converter picks
  the pre-tokenizer from a hash of the HF tokenizer and fails loudly if it is
  unknown, so a successful conversion is already a partial check. Parity still
  compares ids.

### Closed on tux

- The BF16 K/V MMA/conversion hypothesis holds (see Phase 2 results).
- `llama-bench` and the probe agree within 1.6% with graphs on.
- The tux GGUF hash equals the Mac one.

### Still open

- Why llama.cpp reaches a lower share of DRAM bandwidth than TileInfer at
  context 1 (its mat-vec kernel versus TileInfer's GEMV).
- Backend (GPU) sampling in llama.cpp as a third sampling variant. The host
  path costs 0.21 ms, so the effect is bounded by that.
- Whether the constant 0.27 to 0.35 top-logit offset is float32 activations
  or something else; a per-layer comparison would tell.

### Other risks

- Numerics differ by design; "faster" claims need the parity table beside them.
- The GGUF tokenizer path can differ from HF; compare on ids to isolate it.
- Sustained-clock drift: interleave runtimes and keep the 3-round averages.
- The Mac cannot validate any performance number; Metal/CPU runs are only for
  harness correctness.
- `AGENTS.md` says tux is the working checkout. Phases 0 and 1 were built
  and committed on the Mac while tux was down; Phase 2 and later changes were
  made on tux.

## Deliverables

- `experiments/runtimes/` as above, with a README of the one-command flow
  (fetch, build, convert, bench, report).
- A `docs/blog/` follow-up record, in the style of `baseline-2026-10.md`, for
  the four-runtime chart.
- A line in `AGENTS.md` and the top-level README linking the experiment.
