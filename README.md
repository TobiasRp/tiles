# Tiles

An experimental inference engine for [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B),
built on an 8 GB RTX 3050.

The idea is to execute a decoder as a program of small tensor tiles. Python
builds the task graph and schedule offline; a cooperative C++/CUDA worker
executes that schedule once per token and loops over tokens inside the
kernel, up to 64 per launch. Weights and K/V state stay on the GPU, and the
worker itself selects and feeds back the next token.

```mermaid
flowchart LR
    A[Checkpoint] --> B[Offline tile plan] --> C[Cooperative CUDA worker]
    C --> D[Next token] --> C
```

The current prototype uses fixed waves with grid barriers. Finer-grained tile
scheduling is still work in progress.

## Try it

With CUDA, CMake, `uv`, and the [pinned checkpoint](tileplan/README.md#checkpoint-setup)
in `checkpoints/Llama-3.2-1B`, run from the repository root:

```sh
./generate.py "The capital of France is"
./generate.py "Once upon a time" --max-new-tokens 32
./generate.py "The capital of France is" --dtype float32
```

`generate.py` builds the runtime, probes the compiled worker, imports a package,
and prints the continuation and timing. New packages default to BF16 weights,
activations, and K/V; `--dtype float32` selects the pure-float32 path.

Reuse a package with `--package runs/llama32-…`; an omitted `--dtype` follows
the package, and an explicit one must match it. Each run prints its
output directory. The checkpoint is the base model, so prompts are text
continuations rather than chat messages.

## Explore

- [TilePlan](tileplan/README.md): checkpoint setup, packages, and graph visualization.
- [TileInfer](tileinfer/README.md): runtime API, tests, and PyTorch eager/graph measurements.
- [Architecture](docs/architecture.md): the package, the worker, the numerical contract, and the repository layout.
- [Development log](docs/development.md): milestones, measurements, and negative results.
- [Showcase](experiments/showcase/README.md): one command that measures TileInfer
  against PyTorch and llama.cpp and draws the figures.
