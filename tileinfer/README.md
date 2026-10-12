# C++ / CUDA side

`src/runtime/` is reserved for the C++/CUDA executor. `tests/` retains the
existing compiler and CUDA environment checks. The `tile_runtime` executable
currently exposes only `--help`; plan loading and inference are pending.

The first implementation gate is a CCCL 3.2 cooperative launch check on the
RTX 3050, using one block per SM and grid synchronization. It must pass before
the executor adopts CCCL for launches, streams, buffers, spans, and timing. A
failure changes the entire runtime API choice to the standard CUDA Runtime API,
as specified in [the runtime design](../docs/tile_graph_cuda_runtime.md).

Host-only sources use C++23. CUDA sources use C++20.
