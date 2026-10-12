# tile-infer

Prototype for statically scheduled tile-graph inference on the RTX 3050.
Python creates a fixed MLP reference, probes the compiled CUDA worker, and
writes a tile plan. A C++/CUDA runtime will execute that plan; Python will
compare it with PyTorch eager and CUDA Graph replay.

**Status:** project scaffold. The CLI and runtime executable exist so the
build and package boundaries are reviewable. Inference, plan generation,
benchmarks, and the CCCL cooperative-launch gate are still pending.

## Layout

```text
docs/                  Hypothesis, Python architecture, CUDA runtime design
tileplan/
  src/tileplan/         Installable planner and pipeline package
  tests/                Python checks
  pyproject.toml        uv-managed dependencies and CLI entry point
tileinfer/
  src/runtime/          C++/CUDA executor
  tests/                Compiler and CUDA environment checks
  CMakePresets.json     Debug, release, and CPU-only builds
```

See the [core design](docs/tile_graph_inference.md),
[Python design](docs/tile_graph_python_architecture.md), and
[runtime design](docs/tile_graph_cuda_runtime.md).

## Existing environment

The target workstation is `tux`, with an RTX 3050 (compute capability 8.6),
CUDA Toolkit 13.4, driver 595.91.07, CMake, Ninja, uv, and a Python 3.14
environment with PyTorch 2.14.1. The CMake CUDA presets use
`/usr/local/cuda/bin/nvcc` and architecture 86. Override these preset values
for another host.

## Build checks

From the repository root:

```sh
cd tileinfer
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cd ../tileplan
~/.local/bin/uv sync --locked
~/.local/bin/uv run --locked python -m tileplan --help
~/.local/bin/uv run --locked python -m unittest discover -s tests
```

A CPU-only CMake preset is available as `cpu-debug`. The CUDA smoke check
proves that the basic toolchain and device work; it is not the CCCL go/no-go
test.

## Intended entry point

Once implemented, the complete experiment will run from `tileplan/`:

```sh
uv run --locked python -m tileplan demo --out ../runs/demo
```

That command will build or locate `tile_runtime`, generate the PyTorch
fixture, probe the actual compiled worker, build a plan, execute it, check
outputs, and run the PyTorch baselines. The command currently exits with a
clear "implementation pending" message.
