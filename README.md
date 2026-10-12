# tile-infer

Prototype for statically scheduled tile-graph inference on the RTX 3050.
Python creates a fixed MLP reference, probes the compiled CUDA worker, and
writes a tile plan. A C++/CUDA runtime will execute that plan; Python will
compare it with PyTorch eager and CUDA Graph replay.

**Status:** the Python TilePlan package, its offline planner, file bridge, and
PyTorch baselines are implemented and tested. The C++/CUDA executor remains a
scaffold. The CCCL cooperative-launch gate and real GPU execution are the next
steps, after review of TilePlan.

## Layout

```text
docs/                  Hypothesis, Python architecture, CUDA runtime design
tileplan/
  src/tileplan/         Installable planner and pipeline package
  tests/                Python checks, including a mock executor bridge
  pyproject.toml        uv-managed dependencies and CLI entry point
tileinfer/
  src/runtime/          C++/CUDA executor scaffold
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

## Checks

From the repository root:

```sh
cd tileplan
~/.local/bin/uv sync --locked
~/.local/bin/uv run --locked python -m unittest discover -s tests -v

cd ../tileinfer
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

See [TilePlan usage](tileplan/README.md) for its offline commands and the
eventual `demo` entry point. The current C++ executable does not yet report
worker capabilities, load a plan, or run inference.
