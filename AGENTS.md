# Repository guidance

This repository tests the tile-graph inference hypothesis on the RTX 3050.
Read the four design documents in `docs/` before changing the plan format or
executor. The active workload is the Llama-style one-token decoder in
`docs/tile_graph_decoder.md`. Its CPU reference, version-5 whole-token plan,
SequenceContext, CUDA arithmetic, and
synthetic full stack with float32 or BF16 weights/activations/KV, and the pinned Llama 3.2 1B
checkpoint bridge are implemented.
The MLP prototype is historical and its source/tests were removed.

- Keep Python as the offline planner, fixture generator, baseline runner, and
  one-command orchestrator. Keep per-inference scheduling in the C++/CUDA
  executor.
- Use the actual compiled worker kernel to query occupancy before generating
  `target.json`. One logical block per SM is the initial policy; logical
  workers have no physical SM affinity.
- The CCCL cooperative-launch gate passed on the RTX 3050. Keep CCCL for
  launches, streams, buffers, transfers, and timing; rerun the gate when
  changing the worker or CUDA toolchain.
- Preserve the existing `uv.lock` and CMake presets when changing
  dependencies or build inputs. Host code uses C++23; CUDA code uses C++20.
- Keep the decoder reference and package-regression checks when changing
  plans or kernels. Reprobe the compiled decoder worker after changing arithmetic
  opcodes or the execution loop. Run clang-format, Ruff, ty, CTest, and Compute Sanitizer before
  performance claims; distinguish exploratory timings from a controlled
  shape/tile sweep.
- Work in `tux:~/dev/tiles` over SSH, if available: edit, build, run, test,
  and commit there. Treat any other local checkout as read-only; do not edit
  or commit in it, and do not mirror files from it to `tux`.
