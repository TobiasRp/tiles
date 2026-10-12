# Python side

`src/tileplan/` is the installable Python package. Its CLI names the planned
stages, but each execution command currently reports that implementation is
pending. The existing uv environment and lockfile are retained.

Planned responsibilities:

- `reference`: seeded PyTorch model, weights, and expected values.
- `probe`: normalize facts from the final compiled CUDA worker into `target.json`.
- `build`: tile DAG, wave schedule, memory plan, validation, and package output.
- `benchmark`: PyTorch eager and CUDA Graph replay baselines plus comparison.
- `demo`: single command orchestrating the full pipeline.

The package format and pass design live in
[`docs/tile_graph_python_architecture.md`](../docs/tile_graph_python_architecture.md).
