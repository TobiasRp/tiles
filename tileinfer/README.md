# TileInfer

TileInfer executes TilePlan decoder packages in C++23/CUDA20. One cooperative
worker launch performs embedding lookup, the repeated decoder schedule, final
normalization, and logits. GPU greedy selection feeds back the next token.
The current RTX 3050 worker uses 1024 threads per block and one logical worker
per SM; worker IDs have no physical SM affinity.

Weights may be BF16 or float32. Arithmetic, activations, logits, and K/V state
are float32. The executor supports version-3 packages and legacy version-2
float32 packages. Regenerate version-3 packages after a worker fingerprint change.
For a text demo, use [`generate.py`](../generate.py).

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
[worker_kernel.cu](src/runtime/worker_kernel.cu) the cooperative wave loop.
The blocking diagnostics API is linked only into the test harness.

## Measurements on RTX 3050

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
| TileInfer | BF16 / float32 | 16.03 |
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

Add `--reference-dtype bfloat16` for the BF16 baseline.
[benchmark_decode.py](tests/benchmark_decode.py) measures fixed contexts;
[profile_decode.py](tests/profile_decode.py) collects Nsight counters.
Both expose their arguments through `--help`.

## Checks

CTest covers the CCCL cooperative-launch gate, loader/UBSan validation,
float32/BF16 numerical references, generation, and context/ring behavior.
The `cpu-debug` preset runs host-only checks. Run the four Compute Sanitizer
tools after kernel changes:

```sh
cmake --build build/release --target sanitizers
```

The suite retains its documented CCCL API-error-reporting workaround;
initcheck disables API-copy inspection only for the two deliberately gated
ring tests. Kernel memory reads remain checked.
[llama32_checkpoint.py](tests/llama32_checkpoint.py) compares layer outputs,
K/V, logits, and greedy choices with float32 Transformers.
[sweep_tiles.py](tests/sweep_tiles.py) separates synthetic tile-selection pilots
from independent final rounds. See the [decoder design](../docs/tile_graph_decoder.md)
for the full numerical and execution contract.
