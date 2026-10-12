# tile-infer

Prototype for statically scheduled tile-graph inference on an RTX 3050.
[TilePlan](tileplan/README.md) generates a fixed-shape float32 MLP fixture,
probes the compiled CUDA worker, and writes a versioned plan and weights.
[TileInfer](tileinfer/README.md) validates that package and executes its tile
tasks in one cooperative kernel per inference. Python checks the intermediate
and final values against a saved PyTorch reference and runs PyTorch eager and
CUDA Graph baselines.

**Status:** The two-wave MLP path works end to end on `tux`. The runtime uses
one logical worker block per reported SM, CUB warp reduction for GEMV, and a
grid barrier between MLP stages. A decoder, new task opcodes, partial tile
handoff, and a tuned shape/tile sweep are future work. The current benchmark
is an exploratory workload comparison, not evidence that static scheduling
alone improves latency.

## Layout

```text
docs/                  Hypothesis and design notes
tileplan/
  src/tileplan/         Planner, fixture, probe, visualization, and demo
  tests/                Python unit and file-bridge tests
  pyproject.toml        uv-managed dependencies and CLI entry point
tileinfer/
  src/runtime/          C++23 loader/CLI and CUDA20 cooperative worker
  tests/                Unit, PyTorch regression, CCCL, and sanitizer checks
  CMakePresets.json     Debug, release, and CPU-only builds
```

See the [core design](docs/tile_graph_inference.md),
[Python design](docs/tile_graph_python_architecture.md), and
[runtime design](docs/tile_graph_cuda_runtime.md) for the experiment and its
planned extensions.

## Target environment

The target workstation has an RTX 3050 (compute capability 8.6), CUDA Toolkit
13.4, driver 595.91.07, CMake, Ninja, uv, and a Python 3.14 environment with
PyTorch 2.14.1. The CMake CUDA presets use `/usr/local/cuda/bin/nvcc` and
architecture 86. Adjust the presets before building on another GPU.

## Run the complete demo on `tux`

```sh
cd ~/dev/tile-infer/tileplan
~/.local/bin/uv sync --locked
~/.local/bin/uv run --locked python -m tileplan demo --out ../runs/demo
```

`demo` builds the release executor, probes its actual worker occupancy,
generates the reference and package, checks CUDA `h` and `y` against PyTorch,
and records the persistent, eager, and CUDA Graph measurements under
`runs/demo/`. Choose a fresh output directory for each run. See the TilePlan
README for individual commands and custom dimensions or tile heights.

## Checks

```sh
cd ~/dev/tile-infer/tileplan
~/.local/bin/uv run --locked ruff check src tests ../tileinfer/tests
~/.local/bin/uv run --locked ruff format --check src tests ../tileinfer/tests
~/.local/bin/uv run --locked ty check src tests ../tileinfer/tests
~/.local/bin/uv run --locked python -m unittest discover -s tests -v

cd ../tileinfer
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The release tests on `tux` include clang-format, C++ unit tests, the CCCL
cooperative-launch gate, generated-plan regression against PyTorch, and
Compute Sanitizer memcheck, racecheck, and synccheck. CMake also provides a
`cpu-debug` preset for host-only tests. Optional tools are added to CTest when
found during configuration.
