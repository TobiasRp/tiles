# Runtime comparison plan: llama.cpp next to TileInfer and PyTorch

Status: Phases 0 and 1 are implemented in `experiments/runtimes/` and were run
on an Apple M1 Pro (CPU). Phase 2 (CUDA on the RTX 3050) waits for `tux`.

Phase 1 results (pinned checkpoint revision `4e20de36…`, llama.cpp b11871,
CPU, 6 threads; harness validation only, never compared with CUDA numbers):

- GGUF: 147 tensors, 113 BF16 matrices + 34 F32 vectors (norms and
  `rope_freqs`), no quantized tensor. sha256
  `40241de51bb5634628bf29a583c34bcefe8d31849c677bb26a9ad6b47e272760`; compare
  with the tux conversion in Phase 2.
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
- Not yet exercised on real output: the TileInfer/PyTorch report adapters
  (unit-tested with synthetic reports only), `--fa on`, `--kv f16`, Metal.

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

### Phase 2: CUDA on tux (when available)
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
  table/SVG, with parity results next to them.

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
  values. Expect this to show up in Nsight Systems as extra conversion
  kernels; treat as a hypothesis until measured on tux, and report f16 K/V as
  the main llama.cpp configuration if it holds.
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

### Still open (need tux)

- Whether the BF16 K/V MMA/conversion hypothesis above holds, and its cost.
- Whether `llama-bench`'s graph behaviour matches the probe's.
- Whether the `tux` GGUF hash matches the Mac one.

### Other risks

- Numerics differ by design; "faster" claims need the parity table beside them.
- The GGUF tokenizer path can differ from HF; compare on ids to isolate it.
- Sustained-clock drift: interleave runtimes and keep the 3-round averages.
- The Mac cannot validate any performance number; Metal/CPU runs are only for
  harness correctness.
- `AGENTS.md` says tux is the working checkout and local is read-only. Phases
  0 and 1 need a place to commit while tux is down.

## Deliverables

- `experiments/runtimes/` as above, with a README of the one-command flow
  (fetch, build, convert, bench, report).
- A `docs/blog/` follow-up record, in the style of `baseline-2026-10.md`, for
  the four-runtime chart.
- A line in `AGENTS.md` and the top-level README linking the experiment.
