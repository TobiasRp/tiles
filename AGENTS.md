# Repository guidance

This repository tests the tile-graph inference hypothesis on the RTX 3050.
Read the three design documents in `docs/` before changing the plan format or
executor. The MLP is the first workload; its two stages require a full barrier.

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
- The float32 MLP executor works end to end. Keep its `h` and `y` regression
  checks against the PyTorch fixture when changing plans or kernels. Run
  clang-format, Ruff, ty, CTest, and Compute Sanitizer before performance
  claims; distinguish exploratory timings from a controlled shape/tile sweep.
