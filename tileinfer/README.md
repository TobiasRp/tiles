# TileInfer C++/CUDA executor

`tile_runtime` loads TilePlan's version 1 float32 MLP package, validates its
target, storage, checksum, tasks and wave schedule, then runs one cooperative
worker block per SM. Workers execute compact task records; a grid barrier
separates the hidden and output waves. The first two opcodes are warp GEMV
with bias, with or without tanh GELU. The device program loops over arbitrary
wave and worker ranges; adding attention will require new operator records,
dependency validation, and device routines.

`src/runtime/plan.cpp` owns the version 1 MLP schema and schedule checks;
`src/runtime/worker.cu` owns the compact task dispatch and CCCL execution
context. This separation keeps future model-specific validation out of the
wave loop. The loader rejects stale kernel fingerprints and weight checksums
before allocating GPU buffers.

Build and check on tux:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The test suite includes a CCCL cooperative launch gate, malformed-package
checks, and execution of TilePlan packages against PyTorch reference cases
with both sparse and many-task schedules. When installed, `clang-format` checks
the C++/CUDA sources and Compute Sanitizer runs memcheck, racecheck, and
synccheck on another generated plan. The gate passed on the RTX 3050 with 20
blocks of 128 threads.

```sh
build/debug/tile_runtime --describe-target-json
build/debug/tile_runtime --run-plan PACKAGE --inputs INPUTS.bin \
    --outputs OUTPUTS.bin --hidden HIDDEN.bin --report RUN.json
build/debug/tile_runtime --benchmark PACKAGE --input INPUT.bin \
    --iterations 100 --rounds 5 --report BENCH.json
```

Inputs and outputs are contiguous little-endian float32 vectors. `--run-plan`
accepts multiple inputs concatenated in one file and reuses the loaded plan
and device buffers across them. `--benchmark` uploads one input once and times
repeated launches with CCCL events. Host code uses C++23; CUDA code uses C++20.
The reported GPU timing is a batch average per forward pass, not a latency
percentile. The current implementation supports one execution context and one
in-flight invocation, the two MLP opcodes, and a full barrier between stages.
It does not implement decoder state, partial tile handoff, or a performance
shape/tile search. Execution errors return nonzero and print a diagnostic to
stderr.

The JSON and SHA-256 headers under `third_party/` are pinned, header-only
dependencies with their licenses in the source files.
