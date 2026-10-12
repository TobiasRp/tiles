# TilePlan

TilePlan is the offline Python half of the tile-graph experiment. It creates a
deterministic float32 MLP fixture, probes the **final compiled** CUDA worker,
lowers the MLP into row-tile tasks, schedules those tasks in two waves, and
writes a versioned `plan.json` plus little-endian `weights.bin`. `verify` checks
the package without CUDA. The PyTorch eager and CUDA Graph baselines run on the
GPU and compare their outputs with the saved CPU reference.

The [TileInfer executor](../tileinfer/README.md) implements the target probe,
file bridge, and cooperative MLP worker. `demo` now runs the complete pipeline
on `tux` and checks the executor's hidden and final outputs against the saved
PyTorch reference. The Python file bridge also has mock-executor unit tests.

## Install and test on `tux`

```sh
cd ~/dev/tile-infer/tileplan
~/.local/bin/uv sync --locked
~/.local/bin/uv run --locked ruff check src tests
~/.local/bin/uv run --locked ruff format --check src tests
~/.local/bin/uv run --locked ty check src tests
~/.local/bin/uv run --locked python -m unittest discover -s tests -v
```

## Offline workflow

```sh
uv run --locked python -m tileplan reference --out ../runs/fixture
uv run --locked python -m tileplan probe --executor ../tileinfer/build/release/tile_runtime --out ../runs/target.json
uv run --locked python -m tileplan build --model ../runs/fixture/model.json --weights ../runs/fixture/weights.npz --target ../runs/target.json --tile-h1 16 --tile-h2 16 --out ../runs/package
uv run --locked python -m tileplan inspect ../runs/package
uv run --locked python -m tileplan verify ../runs/package
uv run --locked python -m tileplan visualize ../runs/package --out ../runs/schedule.svg
uv run --locked python -m tileplan benchmark --package ../runs/package --cases ../runs/fixture/reference_cases.npz --out ../runs/baselines.json
```

`probe` uses the final compiled worker, not the CCCL smoke-test kernel. Unit
tests also use synthetic target descriptions to exercise the offline planner.
`benchmark` reports PyTorch baselines only and labels them as such. It does not
claim a persistent-kernel speedup.

## Complete demo

```sh
uv run --locked python -m tileplan demo --out ../runs/demo
```

`demo` builds the release executor and drives the complete pipeline. Use
`--executor PATH` to reuse a prebuilt binary. The executor's `--run-plan`
bridge reads raw float32 inputs and writes final outputs and optional hidden
values via `--outputs` and `--hidden`. Its `--benchmark` mode reads the same
saved input used by PyTorch and reports its SHA-256. The demo checks that hash
before combining results. Choose a fresh `--out` directory for each run.

Packages are reproducible for the same fixture, target, and tile heights. The
builder writes to a fresh directory via a temporary directory and rename;
`planning_report.json` records the plan and weight hashes, per-wave worker
loads, and peak buffer bytes. The schedule uses logical worker IDs and a full
barrier between the two MLP stages.

`visualize` writes a self-contained SVG with the row ranges and execution
order of each worker's tasks in both waves. It also shows estimated work and
the full barrier; vertical spacing is illustrative, not GPU elapsed time.
Pass `--note TEXT` to label a synthetic or experimental target in the image.
