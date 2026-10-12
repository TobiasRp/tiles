# TileInfer C++/CUDA executor

`tile_runtime` loads TilePlan's version 1 float32 MLP package, validates its
target, storage, checksum, tasks and wave schedule, then runs one cooperative
worker block per SM. Workers execute compact task records; a grid barrier
separates the hidden and output waves. The first two opcodes are warp GEMV
with bias, with or without tanh GELU. The device program loops over arbitrary
wave and worker ranges; adding attention will require new operator records,
dependency validation, and device routines.

Build and check on tux:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

The test suite includes a CCCL cooperative launch gate, malformed-package
checks, and execution of TilePlan packages against PyTorch reference cases
with both sparse and many-task schedules. The gate passed on the RTX 3050
with 20 blocks of 128 threads.

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
