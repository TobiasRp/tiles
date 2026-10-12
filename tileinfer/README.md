# TileInfer decoder runtime

TileInfer validates version-3 float32 or BF16-weight decoder packages, as well
as version-2 float32 packages. It executes either one block from a hidden
vector or a full stack from a token ID. The compiled cooperative worker uses
one logical block per SM. A full-stack call performs embedding lookup, repeats
the validated nine-wave block schedule, then computes
final RMSNorm and vocabulary logits in one launch. Block scratch is reused
across layers; each layer has its own K/V cache and packed weight stride.
The cooperative schedule and barriers are in `src/runtime/worker.cu`.
The arithmetic operators and tensor views are in `src/runtime/ops.cuh`;
named opcodes and storage classes are part of the fixed task ABI.
Weight-consuming operators and the worker are instantiated for each supported
weight type. Activations, K/V storage, reductions, and output logits remain
float32; BF16 weights are converted to float32 as they are read.

`SequenceContext` owns the stream, device buffers, valid-prefix position, and
failure state. A call appends exactly one token. It rejects token IDs outside
the vocabulary and calls beyond `max_context`; a failed launch invalidates
the context until reset. Reset changes the valid prefix without clearing old
cache bytes.

For diagnostics, a `--run-stack` request may set `"trace_blocks": true`.
Each token step then includes the output hidden vector from every decoder
layer. The normal inference path leaves this trace disabled.

## Build and test on tux

```sh
cd ~/dev/tile-infer/tileinfer
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

CTest includes the CCCL cooperative-launch gate, package and corruption
checks, one-block intermediate comparisons, and full-stack logits/cache
comparisons against PyTorch. The tests cover GQA, tail tiles, reset, capacity,
tied and untied output embeddings, for both weight types. Compute Sanitizer
is a separate check for kernel changes: run `memcheck`, `racecheck`, and
`synccheck` on both `--run-block` and `--run-stack`.

## Commands

```sh
cd ~/dev/tile-infer
tileinfer/build/release/tile_runtime --describe-target-json
tileinfer/build/release/tile_runtime --inspect-plan PACKAGE
tileinfer/build/release/tile_runtime --state-smoke PACKAGE
tileinfer/build/release/tile_runtime --run-block PACKAGE INPUTS.json
tileinfer/build/release/tile_runtime --run-stack PACKAGE INPUTS.json
```

`--run-block` accepts `{"tokens":[{"hidden":[...]}]}`.
`--run-stack` accepts `{"tokens":[0,7,15]}`. Either token list may
contain `{"reset":true}`. These diagnostic commands return one trace per
token; the public C++ API uses `SequenceContext::run_hidden` or
`SequenceContext::run_token`.
The [TilePlan README](../tileplan/README.md) has a complete command sequence
that creates a package, writes token IDs, and runs `--run-stack`.

`--inspect-plan` reports dimensions, wave count, worker loads, task count,
scratch and K/V bytes, weight dtype, and package hashes. It checks the
compiled-kernel fingerprint and cooperative residency before returning. Plans
contain byte offsets and strides, never addresses.
