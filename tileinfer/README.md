# TileInfer

TileInfer executes TilePlan decoder packages in C++23/CUDA20. One cooperative
worker launch executes a whole token: embedding lookup, every decoder layer,
final normalization, and logits. The loader validates the package's flat task
program and encodes one instruction stream per worker; grid barriers are
explicit `barrier` instructions at the same stream positions in every worker.
GPU greedy selection feeds back the next token. The current RTX 3050 worker
uses 1024 threads per block and one logical worker per SM; worker IDs have no
physical SM affinity.

Version-6 packages use either float32 or BF16 weights, activations, and K/V.
BF16 matrix operands accumulate in float32; RMSNorm statistics, softmax, RoPE
frequency/trigonometric work, and SiLU internals retain float32 math. BF16
rounding follows the pinned PyTorch eager operator boundaries. Public logits
and diagnostics widen the modeled values to float32. Projections, attention,
and the language-model head use SIMT warp reductions. A native BF16 MMA
variant was measured 2.15 times slower for batch-one decode and has been
removed; see [the BF16 results](../docs/bf16_compute_results.md).

The runtime reads only packages of the current format; regenerate older
packages with TilePlan. For a text demo, use [`generate.py`](../generate.py),
which creates BF16 packages by default. The runtime follows the precision
stored in each package.

## Build

```sh
cd tileinfer
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

## Use the runtime

From `tileinfer/`, with a package and an input JSON such as `{"tokens":[0,7,15]}`:

```sh
build/release/tile_runtime generate PACKAGE INPUTS.json --max-new-tokens 16
build/release/tile_runtime bench PACKAGE INPUTS.json --new-tokens 16 --warmup 3 --rounds 7
```

`generate` accepts `--eos-id`; `bench` disables EOS for constant work.
`check` returns per-token logits and K/V snapshots, optionally `--trace-blocks`.
`--inspect-plan` summarizes a package; `--describe-target-json` probes the compiled worker.

The C++ API is in [runtime.hpp](src/runtime/runtime.hpp): `DecoderModel` shares
uploaded weights; each `SequenceContext` owns its stream and K/V state.
Prefill and generation return tickets; `wait` retrieves the result.
[ops.cuh](src/runtime/ops.cuh) contains arithmetic and
[worker_kernel.cu](src/runtime/worker_kernel.cu) the instruction-stream loop;
[plan_validation.cpp](src/runtime/plan_validation.cpp) checks operands, output
ownership, and every read against its producer across layers and shared scratch.
The blocking diagnostics API is linked only into the test harness.

## Historical measurements on RTX 3050

These tables describe the superseded version-3 BF16-weight/float32-activation
implementation. They do not measure the BF16-compute path. Current
validation and controlled arithmetic-variant comparisons are described in
[the BF16 implementation results](../docs/bf16_compute_results.md).

Pinned Llama 3.2 1B checkpoint, batch one, TF32 disabled; measured October 13, 2026.
The TileInfer package uses 64-row tiles. No GPU clocks were locked.

### Autoregressive generation

Six prompt tokens, 16 generated tokens, three warmup runs and seven measured
runs per mode, plus graph-capture setup. CUDA events include GPU argmax feedback
and decoding, excluding loading, prefill, and readback. PyTorch uses eager
attention and `StaticCache` in both modes. The graph captures the entire 16-token loop;
each replay starts from a freshly prefilled cache.

| Implementation | Weights / activations | Average ms/token |
| --- | --- | ---: |
| TileInfer v3 (retired) | BF16 / float32 | 16.03 |
| PyTorch eager | float32 / float32 | 25.55 |
| PyTorch graph replay | float32 / float32 | 24.43 |
| PyTorch eager | BF16 / BF16 | 14.42 |
| PyTorch graph replay | BF16 / BF16 | 13.06 |

Float32 PyTorch and TileInfer generated identical IDs. BF16 PyTorch generated
a different continuation; its eager and graph paths agreed. The BF16 rows use
BF16 activations and matrix operations, while the float32 reference stores twice as many
weight bytes as TileInfer.

### Fixed-context decoder step

Repeated token ID 1, three warmups, seven rounds of five calls per context.
These timings exclude selection and hold the context length fixed. PyTorch
uses float32 weights/arithmetic, eager attention, and `StaticCache`; graph
replay captures one decoder step. Eager/graph logits agree, and TileInfer's
top token/logit matches the reference at every context.

| Context tokens | PyTorch eager (ms) | PyTorch graph (ms) | TileInfer (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 25.56 | 24.33 | 15.42 |
| 64 | 26.04 | 24.49 | 17.48 |
| 256 | 26.54 | 25.07 | 23.75 |

Raw samples and environment details are in `runs/tileinfer-baselines-2026-10-13/`
on tux. These are workload measurements, not a general model/shape sweep.

To repeat generation measurements from `tileinfer/`, set `PACKAGE` and `INPUTS`
to your package and prompt JSON:

```sh
uv run --locked --project ../tileplan --group checkpoint python tests/benchmark_generate.py \
  --checkpoint ../checkpoints/Llama-3.2-1B --package "$PACKAGE" \
  --executor build/release/tile_runtime --inputs "$INPUTS" \
  --new-tokens 16 --warmup 3 --rounds 7 --out ../runs/generation-comparison.json
```

The reference precision now follows the package; an explicit
`--reference-dtype` must match it.
[benchmark_decode.py](tests/benchmark_decode.py) measures fixed contexts;
[profile_decode.py](tests/profile_decode.py) collects Nsight counters.
Both expose their arguments through `--help`.

To see where a token's time goes, measure the bandwidth roofline once, then
trace the worker at chosen contexts. The trace records every instruction's
start and end on every worker; the report splits the token into ideal
streaming, per-phase excess, and barrier latency, and draws Gantt charts.

```sh
build/release/bandwidth_probe stream 2471628800 > ../runs/stream.json
build/release/bandwidth_probe rowdot > ../runs/rowdot.json
uv run --locked --project ../tileplan python tests/profile_timeline.py \
  --executor build/release/tile_runtime_test --package "$PACKAGE" \
  --stream-report ../runs/stream.json --positions 0 1023 --out ../runs/timeline
```

`bandwidth_probe rowdot` times the worker's GEMV routine on the Llama 3.2 1B
projection shapes with 1–20 active workers. See
[the S0 results](../docs/megakernel_next_steps.md#9-s0-results-where-the-time-goes).
`bandwidth_probe ring` runs the same shapes as wave programs separated by
grid barriers. It compares direct loads with a `cp.async` shared-memory
weight ring, refilled before or after each barrier, and with an L2 prefetch;
see [the S3 results](../docs/blog/s3-weight-prefetch-2026-10.md).
`attention_probe [POSITIONS [CHUNKS]]` runs the attention waves of 16 layers
with one block per SM: the production kernels, an instrumented copy with
per-phase cycle counts, and wide-load candidates, for each context and chunk
count. See [the S2b results](../docs/megakernel_next_steps.md#11-s2b-probe-results-attention-at-the-right-parallelism).

## Checks

CTest covers the CCCL cooperative-launch gate, loader/UBSan validation,
float32/BF16 numerical references, generation, and context/ring behavior.
[ops_unit.py](tests/ops_unit.py) tests the actual device GEMV, attention, norm,
rotary, SiLU/product, embedding, and logits routines in isolation, including
tails, ownership, ties-to-even, and exact BF16 norm cases.
The `cpu-debug` preset runs host-only checks. Run the four Compute Sanitizer
tools after kernel changes:

```sh
cmake --build build/release --target sanitizers
```

The suite retains its documented CCCL API-error-reporting workaround;
initcheck disables API-copy inspection only for the deliberately gated
ring tests. Kernel memory reads remain checked.
[llama32_checkpoint.py](tests/llama32_checkpoint.py) compares layer outputs,
K/V, logits, and greedy choices with matching-precision Transformers.
See the [decoder design](../docs/tile_graph_decoder.md)
for the full numerical and execution contract.
