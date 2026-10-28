# TileInfer

TileInfer executes TilePlan decoder packages in C++23/CUDA20. The cooperative
worker executes a whole token per pass over its program: embedding lookup,
every decoder layer, final normalization, and logits. The loader validates the
package's flat task program and encodes one instruction stream per worker;
grid barriers are explicit `barrier` instructions at the same stream positions
in every worker. One launch decodes up to 64 consecutive tokens with a grid
barrier between them. During generation the logits tasks also store each
row range's greedy candidate, and every worker combines them into the next
token at the start of the following step; a non-finite logit stops the
request with a device error. The current RTX 3050 worker
uses 1024 threads per block and one logical worker per SM; worker IDs have no
physical SM affinity.

Version-10 packages use either float32 or BF16 weights, activations, and K/V.
BF16 matrix operands accumulate in float32; RMSNorm statistics, softmax, RoPE
frequency/trigonometric work, and SiLU internals retain float32 math. BF16
rounding follows the pinned PyTorch eager operator boundaries. Public logits
and diagnostics widen the modeled values to float32. Projections, attention,
and the language-model head use SIMT warp reductions. A native BF16 MMA
variant was measured 2.15 times slower for batch-one decode and has been
removed; see [BF16 compute](../docs/development.md#bf16-compute).

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

## Measure

Current results against PyTorch and llama.cpp are in
[the development log](../docs/development.md#final-results);
[the showcase command](../experiments/showcase/README.md) reproduces them.

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

For a comparison between binaries or packages, use the controlled protocol in
[controlled_decode.py](tests/controlled_decode.py). Each package stays
resident in a `tile_runtime_test --benchmark-serve` process, blocks of the
arms alternate in time, a background `nvidia-smi` log records clocks,
temperature, power and clock-event reasons for every block, and the report
gives paired differences with bootstrap intervals. Lock the GPU clocks first
when possible (`sudo nvidia-smi -lgc 1950,1950` and `-lmc 6801,6801` on the
RTX 3050) and record that in `--note`:

```sh
uv run --locked --project ../tileplan python tests/controlled_decode.py \
  --arm old=OLD/tile_runtime_test,OLD_PACKAGE \
  --arm new=build/release/tile_runtime_test,"$PACKAGE" \
  --positions 0 255 1023 4095 --synthetic-prefix --note "locked clocks" \
  --out ../runs/compare
```

`--synthetic-prefix` fills K/V with deterministic values instead of decoding
the prefix; timing is unchanged within 0.03%, but the next token is no longer
compared with PyTorch. `--pytorch CHECKPOINT` adds eager and CUDA Graph arms.
Measurement modes trust the package's declared weight checksum (hashing
2.5 GB took 19 s per process), so run `tileplan verify` once per package. Run
the same package as two arms (A/A) to see the protocol's resolution; see
[S6a](../docs/development.md#s6a-controlled-measurement).

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

All positions are traced in one process, so the prefix is decoded once;
`--synthetic-prefix` skips it.

`bandwidth_probe rowdot` times the worker's GEMV routine on the Llama 3.2 1B
projection shapes with 1–20 active workers. See
[S0](../docs/development.md#s0-where-the-time-goes).
`bandwidth_probe ring` runs the same shapes as wave programs separated by
grid barriers. It compares direct loads with a `cp.async` shared-memory
weight ring, refilled before or after each barrier, and with an L2 prefetch;
see [the S3 results](../docs/blog/s3-weight-prefetch-2026-10.md).
`bandwidth_probe qkv` runs the QKV waves of 16 layers as S2b's one-head tasks
and as the worker's even row split; see
[S2c](../docs/development.md#s2c-balanced-qkv).
`bandwidth_probe ffn` runs the FFN of 16 layers as S2c's split waves and as
fused per-worker tasks with split-K down partials; see
[S4](../docs/development.md#s4-fewer-wave-boundaries).
`counter_probe PACKAGE [POSITION...]` runs a BF16 stack package with the worker's
operators and replaces the barriers around attention with counter waits; see
[S4](../docs/development.md#s4-fewer-wave-boundaries).
`attention_probe [POSITIONS [CHUNKS]]` runs the attention waves of 16 layers
with one block per SM: S2's kernels as the reference, an instrumented copy
with per-phase cycle counts, the S2b wide-load candidates, and the worker's
own attention tasks with the planner's default chunk rule and barrier skip.
See [S2b](../docs/development.md#s2b-attention-at-the-right-parallelism).

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
See the [architecture document](../docs/architecture.md)
for the full numerical and execution contract.
