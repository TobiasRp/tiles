# BF16 computation implementation and validation

Implemented in `tux:~/dev/tiles` from revision
`ed925a4c3a0b955c7016267e6f71ae911dc1f8d9`, completed October 14, 2026.
The [precision plan](bf16_compute_plan.md) remains the detailed operator contract.

## Implemented behavior

Version 4 has exactly two storage/activation policies: pure float32 and
`hf-llama-eager-bf16-v1`. The BF16 policy replaces the old BF16-weight/
float32-activation path. Version-2/3 pure-float32 packages remain readable;
retired BF16-weight packages fail with regeneration guidance.

Weights, hidden states, projection results, residuals, K/V, attention scores,
and probabilities are BF16 under the replacement policy. Dot-product
accumulators, norm statistics, softmax reductions, SiLU internals, and RoPE
frequencies/trigonometry are FP32. Norm, rotary, SiLU/product, projection/
residual, and logits preserve eager PyTorch's intermediate BF16 rounding.
Public logits and requested diagnostic snapshots widen the computed BF16
values to float32. Generation does not widen the persistent activation/cache
buffers or introduce per-token host synchronization.

The planner serializes explicit precision fields, BF16 opcodes, dtype-correct
byte extents, and a per-context BF16 attention workspace reused across layers.
The loader rejects incompatible dtype combinations and opcode/layout mismatches.
The Python reference and checkpoint importer implement the same two policies.
`generate.py` defaults to BF16/SIMT for new packages; `--dtype float32` widens
the checkpoint weights offline and selects the pure-float32 path. Reusing a
package inherits its precision and matrix kernel; explicit options must match.

`--matrix-kernel simt` widens BF16 operands for FP32 SIMT arithmetic.
`--matrix-kernel mma` uses native BF16 `m16n8k16` MMA with FP32 accumulation
for projections and the language-model head. It pads row/K tails and replicates
the vector across eight columns. Two independent FP32 K accumulators are
combined before the BF16 cast. An isolated 512-output probe matched PyTorch
exactly with this reduction; the original serial MMA reduction differed on two
outputs. Attention QK/PV remain SIMT FP32 reductions
over BF16 operands in both variants. The variants share the same rounding
contract. No dependency, lockfile, or CMake preset changed.

The actual compiled worker probe reports 64 registers/thread, 2,264 static
shared bytes, 1,024 threads/block, and one resident block/SM for each of
float32 SIMT, BF16 SIMT, and BF16 MMA. RTX 3050 has 20 SMs, so the initial
cooperative grid remains 20 logical workers. SASS contains BF16 HMMA
instructions in the matrix routines.

## Numerical oracle and calibration

The oracle is the pinned Llama 3.2 1B checkpoint, Transformers 4.57.6 eager
attention, and PyTorch 2.14.1+cu130 on RTX 3050. Correctness runs use cuBLASLt,
TF32 disabled, no autocast/compilation, inference mode,
`TORCH_DISABLE_NATIVE_JIT=1`, and
`allow_bf16_reduced_precision_reduction = (False, False)` to disable both
reduced reductions and split-K. Prefixes are built one token at a time.

Initial provisional whole-model limits of `atol=.05, rtol=.02` failed.
Independent PyTorch comparisons subsequently showed that these limits also
reject normal BF16 shape/backend variation. This calibration happened after
the first executor comparison; the final limits were derived from separate
PyTorch runs rather than fitted to executor output. Strict sequential versus
strict full-prefix inference had maximum logit error .1875 and RMS .02775;
strict sequential versus default cuBLAS sequential had .1875 and RMS .03225.
An initial `.15` absolute floor still rejected a block-output difference
that also occurs between the independent PyTorch runs. The final calibration
therefore uses an explicit reference-only rule: keep relative terms fixed at
`.03` (`.02` for V), compute the maximum positive
`abs(error) - rtol*abs(reference)` across those independent runs, multiply by
1.25, round up to .01, and enforce a minimum floor of .05. RMS ceilings are
twice the largest reference RMS, rounded up to .01. No executor output enters
this calculation. This corrects the provisional calibration after inspecting
the failed checks; it does not claim the intended calibration-first order was
followed. The saved arrays and `reference-calibration.json` make it reproducible.
The checked-in `tileinfer/tests/bf16_checkpoint_tolerances.json` freezes the
result and its provenance.

The final checkpoint limits are:

| Stage | Absolute floor | Relative term | RMS ceiling |
| --- | ---: | ---: | ---: |
| BF16 block outputs | .34 | .03 | .04 |
| BF16 K | .15 | .03 | .03 |
| BF16 V | .05 | .02 | .01 |
| BF16 logits | .18 | .03 | .07 |
| Float32, every stage | .001 | .0001 | .001 |

Every element must satisfy `abs(error) <= atol + rtol * abs(reference)` and
the stage RMS must satisfy its ceiling. Required smoke prompts must also
preserve every greedy decision. Reports include BF16 ULP percentiles,
finiteness, mismatch counts, top-five overlap, and reference top-two margins.
These are tolerance-based model checks, not a claim of bitwise model parity.

An identical-input checkpoint probe exposed a real RMSNorm reduction mismatch.
The BF16 norm now follows the pinned PyTorch FP32 mean reduction order, with
four accumulators, a block tree, and warp shuffles, followed by separate FP32
mean/epsilon operations and reciprocal square root. Sixteen independent
2,048-wide norm fixtures require exact BF16 results. The existing float32
norm/references retain their original behavior.

## Tests and artifacts

`tileinfer/tests/ops_unit.cu` invokes the actual device functions from
`ops.cuh`; `ops_unit.py` supplies matching CUDA PyTorch inputs and expected
results. Its 167 cases cover float32, BF16 SIMT/MMA GEMV, row/K tails,
projection-before-residual rounding, ties-to-even, RMSNorm/final norm,
RoPE through position 8192, SiLU/product, embedding, logits, and attention
with random/flat/peaked scores and prefixes through 1025. Exact bit checks
cover signed zero, BF16 subnormals, finite extremes, and positive/negative
rounding ties. Sentinels check inactive output ownership; poisoned unused
K/V capacity must not be read.
BF16 outputs must be exactly representable after widening.

Additional package/reference tests cover explicit dtype rules, malformed
workspace/opcodes, deterministic serialization, legacy float32 compatibility,
and retirement of mixed-mode packages. Generation-wrapper tests cover new
BF16/SIMT defaults, explicit overrides, inherited existing-package precision
and matrix kernels, legacy float32 metadata, and explicit-option conflicts.
Existing block/stack/context/ticket,
reset, EOS, and capacity regressions cover both policies and MMA. A zero-logit
generation case requires smallest-token-ID selection for all three variants.

The sanitizer suite runs all four tools over 17 cases (68 invocations),
including isolated operators and the three ring variants. After changing only
MMA accumulation, the final rerun selects MMA and includes all isolated
operators and ring-copy checks: 7 cases × 4 tools, or 28 invocations. The controlled ring
test waits until its second gate is reached before draining ticket events and
allows extra time for instrumented cooperative work. The pre-existing CCCL
API-error-reporting workaround remains; initcheck API-copy inspection is
disabled only for deliberately gated ring cases, while kernel reads remain
checked.

Artifacts are under `runs/bf16-compute-2026-10-13/` on tux. The checkpoint
BF16 weight hash is
`34cb4260034f41a7b629325a4f29e8fef074f978321b9b70d95487aa6bf3d7ce`;
the pure-float32 import hash is
`3d096cfdcbed32f967074bb9c9cf64f796d342fed7ee926640e85707a15d18bd`.
Each BF16 package stores 2,471,628,800 weight bytes; float32 stores
4,943,257,600 bytes. Capacity is 512 for the checkpoint validation packages.

## Final verification

Ruff, ty, clang-format, all 25 Python tests, and all 13 CTest checks passed.
The full 68-invocation sanitizer suite passed before the MMA reduction change;
the final 28-invocation MMA rerun also passed, including all 167 operator cases
under every sanitizer tool. The pre-existing narrow API checks exclusions are
disclosed above.
CTest includes the CCCL cooperative gate, 167 isolated operator cases, full
block/stack numerical checks, context/ticket/ring behavior, and greedy ties.

The final checkpoint comparison uses 27 teacher-forced steps over three frozen
prompts, including two generated tokens per prompt. Float32, BF16 SIMT, and
BF16 MMA preserve all 27 greedy decisions under the same sequential oracle.
Model comparisons remain limited to this checkpoint and prompt suite.

| Implementation | Block max / RMS | K max / RMS | V max / RMS | Logits max / RMS |
| --- | --- | --- | --- | --- |
| Float32 | .000320 / .00000448 | .0000324 / .00000320 | .0000108 / .000000894 | .0000381 / .00000598 |
| BF16 SIMT | .125 / .002413 | .1875 / .011171 | .0390625 / .003525 | .1875 / .029070 |
| BF16 MMA | .5 / .003081 | .1875 / .011494 | .0390625 / .003612 | .1875 / .027141 |

Current packages and comparisons have the `-mma2` suffix under the artifact
root; earlier attempts are retained for diagnosis. `target.json`,
`resource-usage.txt`, and `sass-bf16.txt` describe the final worker.
## Controlled fixed-context measurements

The frozen candidates were SIMT/MMA × 64/256 output rows, at contexts 1/64/256
with token ID 1. Pilots used one warmup, two calls per round, and two rounds.
Each implementation selected the tile minimizing the geometric mean over the
three pilot contexts. Independent final runs used three warmups and five rounds
of five calls. Seeded random order interleaved the candidates; CUDA events
excluded loading, sequential prefix setup, and readback. PyTorch uses the same
BF16 eager oracle, with an additional CUDA Graph baseline. GPU clocks were
unlocked; telemetry, raw samples, hashes, compiler/driver versions, and the
full protocol are in `matrix-comparison-mma2-final/report.json`.

| Context tokens | PyTorch eager ms | PyTorch graph ms | BF16 SIMT, tile 64 ms | BF16 MMA, tile 256 ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.09 | 12.88 | 12.99 | 28.50 |
| 64 | 14.42 | 13.05 | 13.18 | 28.62 |
| 256 | 14.89 | 13.41 | 13.84 | 29.05 |

SIMT remains the default. Native MMA is about 2.15 times slower across this
suite's geometric mean, despite executing BF16 HMMA. This is a controlled
comparison of these frozen shapes and tiles, not a general model/shape sweep.
No hardware traffic or universal speedup is claimed; the runtime's memory
report describes validated payload extents and excludes allocator overhead.



## Short autoregressive generation

A separately frozen six-token prompt (`The capital of France is`, including
BOS) generated two tokens with EOS disabled, three warmups, and five measured
rounds. It used the tiles selected by the independent fixed-context pilots.
CUDA events included GPU argmax feedback and both decoder calls, excluding
loading, prefill, and readback. All implementations generated IDs
`[12366, 13]` (` Paris.`); PyTorch eager and graph replay agreed.

| Matrix implementation | TileInfer ms/token | Matching PyTorch eager ms/token | Matching PyTorch graph ms/token |
| --- | ---: | ---: | ---: |
| BF16 SIMT, tile 64 | 13.158 | 14.247 | 12.978 |
| BF16 MMA, tile 256 | 28.752 | 14.255 | 12.972 |

The protocol and every sample are in `generation-protocol.json`,
`generation-simt-final.json`, and `generation-mma-final.json`. This short smoke
case does not characterize long-generation throughput or agreement.

The public command was also checked with an existing v4 BF16 SIMT package:

```sh
tileplan/.venv/bin/python generate.py "The capital of France is" \
  --package runs/bf16-compute-2026-10-13/matrix-comparison-mma2-final/package-simt-64 \
  --dtype bfloat16 --matrix-kernel simt --max-new-tokens 2
```

It printed ` Paris.`. `verification.json` records the final binary hashes and
gate counts; those hashes match the checkpoint and benchmark reports.
