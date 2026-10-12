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
- The CCCL cooperative-launch smoke test is the next implementation gate.
  Use CCCL consistently only if it passes on this GPU. If it fails, revise
  the runtime design to use the standard CUDA Runtime API throughout.
- Preserve the existing `uv.lock` and CMake presets when changing
  dependencies or build inputs. Host code uses C++23; CUDA code uses C++20.
- Do not describe scaffold commands as functional inference. Add correctness
  checks against the PyTorch fixture before reporting performance.
