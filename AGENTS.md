# Repository guidance

This repository tests the tile-graph inference hypothesis on the RTX 3050.
Read `docs/architecture.md` (and `docs/development.md` for the measured
history) before changing the plan format or executor. The active workload is
the Llama 3.2 1B one-token decoder. Implemented: its CPU reference, the
version-10 whole-token plan, `SequenceContext`, the CUDA arithmetic, the
synthetic full stack with float32 or BF16 weights, activations, and K/V, and
the pinned Llama 3.2 1B checkpoint bridge. The earlier MLP prototype was
removed. The S6 showcase command (`experiments/showcase/showcase.py`)
reproduces the final comparison with PyTorch and llama.cpp and the blog
figures.

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
