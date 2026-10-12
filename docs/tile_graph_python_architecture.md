# Python architecture for the tile-graph inference prototype

Status: design draft, 2026-10-12. This document specifies the offline Python side of [the RTX 3050 tile-graph prototype](tile_graph_inference.md). The CUDA executor and its kernels have a separate [runtime design](tile_graph_cuda_runtime.md).

## 1. Boundary and deliverable

Python creates the test fixture, probes the target, and then acts as a small **offline compiler**. Given a fixed MLP description, float32 weights, tile sizes, and a target configuration, TilePlan produces a validated execution package. The initial target is the RTX 3050 on Linux with CUDA 13.4 and driver 595.91.07, as recorded in the parent design; device properties still come from the actual machine. Python drives the demo and its file bridge but is absent from the executor's per-inference device path and does not choose tasks at runtime. It also runs the separate PyTorch comparison paths during benchmarking. Rebuilding the package is required when the model, weights, dimensions, tile configuration, worker count, or executor ABI changes; changing only the input vector does not rebuild it.

```text
PyTorch reference ──► model.json + weights.npz + reference_cases.npz
CUDA executable ────► Python probe_target.py ──► target.json
model.json + weights.npz + target.json
                    │
                    ▼
          TilePlan graph builder and planner
                    │
                    ▼
              plan.json + weights.bin
                    │
                    ▼
      C++/CUDA execute ──► compare with reference_cases.npz
PyTorch eager + CUDA Graph replay ──► compare and report alongside tile plan
```

`probe_target.py` calls the **same final C++/CUDA executable that will load the plan** to obtain raw device and compiled-kernel facts. It validates and normalizes the installed RTX 3050's SM count, compute capability, cooperative-launch support, memory limits, supported task opcodes, executor ABI, threads per block, and maximum active blocks per SM. It then computes the initial policy's worker count and cooperative grid size, checks residency, and writes a versioned `target.json` with both the raw facts and derived choices. The smoke-test kernel or a worker stub cannot supply a planning target: occupancy must come from the compiled worker with its GEMV routine. The runtime repeats the compatibility and launch checks before execution. One worker block per reported SM is the first policy. Worker IDs are logical CUDA block IDs, never physical SM IDs.

`target.json` has three groups: `device` (GPU identity, compute capability, SM count, memory limits, cooperative-launch support, driver/runtime versions), `executor` (ABI, opcodes, block size, resource use and occupancy for the compiled kernel), and `launch` (derived worker count, grid size, and the residency limit used to validate it). The probe calculates the legal cooperative-grid bound from the reported SM count and active blocks per SM, then requires the chosen one-block-per-SM grid to fit. Kernel register use and occupancy come from the compiled executable's CUDA query; the Python script does not estimate them from model dimensions.

The PyTorch fixture generator runs on CPU, builds a small deterministic MLP, and writes `model.json` (kind, widths `D/H/O`, schema version, `GELU(approximate="tanh")`), `weights.npz` (the four named arrays), and `reference_cases.npz` (test inputs plus expected `h` and `y`). This step can run without the RTX 3050; the baselines and executor measurements run on the target GPU. No Python control flow is traced into the plan; the accepted model and operation vocabulary are explicit.

## 2. Python components

| Component | Responsibility | Main result |
| --- | --- | --- |
| PyTorch fixture generator | Create a seeded small MLP, export its weights, and evaluate fixed test inputs. | `model.json`, `weights.npz`, `reference_cases.npz`. |
| PyTorch baseline runner | Reconstruct the packaged MLP on the target GPU; time eager execution and CUDA Graph replay on fixed-shape inputs. | Baseline outputs, timings, and environment metadata. |
| Target probe | Invoke the compiled CUDA executable's capability query; derive and validate the one-worker-per-SM launch configuration. | `target.json`. |
| Model importer | Read `model.json` and named arrays `W1`, `b1`, `W2`, `b2` from `weights.npz`; reject missing, extra, incorrectly shaped, or non-finite weights. | Immutable MLP specification and float32 arrays. |
| Tensor graph builder | Create tensor descriptors and whole-operation nodes for `h = GELU(W1 × x + b1)` and `y = W2 × h + b2`. | Logical DAG with explicit dimensions and semantics. |
| Tile lowerer | Divide output rows into fixed-size tiles and replace each operation with executable tile tasks. | Tile DAG with read/write regions and dependencies. |
| Wave scheduler | Topologically group independent tasks, then assign each wave's tasks to logical workers. | Ordered task list per worker, per wave. |
| Memory planner | Calculate lifetimes and byte ranges for input, output, weights, and scratch. | Buffer descriptors and a peak-memory estimate. |
| Plan validator | Check shapes, bounds, dependencies, schedule, capabilities, and deterministic serialization. | Errors or a complete, internally consistent plan. |
| Package writer | Write the versioned JSON plan and packed float32 weights atomically, with a checksum. | Directory the C++ runtime can load. |
| Checker and inspector | Compare CUDA outputs with saved PyTorch values; print graph, waves, work balance, and memory usage. | Correctness and planning diagnostics. |

This can start as one Python package with modules such as `reference`, `baselines`, `probe_target`, `model`, `ir`, `lower`, `schedule`, `memory`, `validate`, `emit`, and `check`. These are responsibility boundaries, not a requirement for eleven files on day one. The planner core should depend on plain Python and NumPy; PyTorch generates the independent model fixture, expected values, and comparison timings but does not define the plan format.

## 3. Intermediate representation

Use typed, immutable records between passes so each pass has one input and one output. Stable integer IDs and a canonical ordering make a plan reproducible from the same inputs.

| Record | Required fields | Invariant |
| --- | --- | --- |
| `TensorDesc` | ID, name, shape, float32 dtype, row-major strides, storage class (`input`, `weight`, `scratch`, `output`) | Byte size and bounds follow from shape/strides. |
| `OpNode` | ID, opcode, input/output tensor IDs, fixed attributes | Operator semantics and shapes are fully specified. |
| `TileTask` | ID, tile opcode, output row interval `[begin, end)`, read regions, write region, predecessor IDs | One task owns each output row of a tensor; no two tasks write the same region. |
| `Wave` | Wave ID and ordered task IDs for every logical worker, including empty lists | Every predecessor is in an earlier wave; all workers reach the same barrier. |
| `BufferAllocation` | Tensor ID, storage class, byte offset, byte length, alignment | Ranges fit their allocation and do not overlap while both are live. |

The first task vocabulary is deliberately narrow: `gemv_bias_gelu_f32` for `W1` tiles and `gemv_bias_f32` for `W2` tiles. A task computes complete output rows across the full input dimension; **no split-K reduction** is needed in version 1. Version 1 uses the tanh GELU approximation, fixed in the model and executor ABI; a mismatch is a load error, not an implicit numerical change. A general-purpose graph tracing or arbitrary PyTorch operator importer is outside this prototype.

For an MLP with input width `D`, hidden width `H`, and output width `O`, `W1` is `[H, D]`, `b1` is `[H]`, `W2` is `[O, H]`, and `b2` is `[O]`. An `H`-row tile reads all of `x`, its rows of `W1` and `b1`, and writes its unique slice of `h`. An `O`-row tile reads all of `h`, its rows of `W2` and `b2`, and writes its unique slice of `y`. Thus every second-layer task depends on every first-layer task. The tile graph expresses that dependency and the first executor enforces it with a wave barrier. This MLP cannot demonstrate early handoff from one `W1` tile to one `W2` tile.

## 4. Planning passes and initial policies

1. **Parse and normalize.** Validate schema versions, dimensions, tile sizes, float32 dtype, row-major weight layout, named inputs, and `target.json` capabilities. Require cooperative launch and at least one resident block per SM for the compiled executor. Convert arrays to contiguous little-endian float32 in a defined order. Preserve the exact activation choice in the model specification.
2. **Build the tensor DAG.** Resolve dimensions and ensure each logical tensor has a single producer, except immutable inputs and weights. Reject cycles and unsupported operations before tiling.
3. **Lower to row tiles.** Generate half-open row intervals using configured tile heights for `W1` and `W2`; the last tile may be shorter. Give tasks stable IDs in operation order, then row order. Derive read/write regions and predecessor edges from those intervals.
4. **Form waves.** The first MLP has wave 0 for all `W1`/GELU tasks and wave 1 for all `W2` tasks. More generally, put a task in the first wave after all predecessors. A wave contains no producer–consumer pair. Every worker executes its assigned tasks in order and joins the barrier before the next wave.
5. **Assign workers.** Set `worker_count = device.SM_count`. Within each wave, sort tasks by descending estimated work (`output_rows × input_width`), then assign each to the currently least-loaded worker, breaking ties by worker ID and task ID. This deterministic estimate balances row tails without pretending to predict GPU time. Record estimated work per worker for inspection.
6. **Plan memory.** Version 1 gives `h` a dedicated contiguous scratch buffer; `x` and `y` are runtime-bound buffers, and weights are immutable. Compute lifetimes and peak bytes, but defer scratch reuse until the simple layout is correct. No pointer values appear in the plan.
7. **Validate and emit.** Verify the graph and schedule independently of the implementation that produced them. Serialize only after all checks pass; write a complete temporary package directory and rename it to a fresh destination after checksums are complete, so readers cannot see a new plan with old weights. The plan stores the execution order; a verifier can reconstruct data hazards from task regions without storing a redundant predecessor list.

Tile heights are explicit planner inputs, not values silently inferred from the model. Freeze a small candidate set before measuring, generating a separate plan for each. Report every candidate's pilot result, choose a tile height from those pilot rounds, then measure the chosen plan again in independent final rounds. The Python cost estimate balances a wave; it does not declare the fastest tile size.

## 5. Package and C++/CUDA contract

`plan.json` is the control artifact. `weights.bin` holds contiguous, little-endian float32 arrays in manifest order; it contains no Python pickle or runtime pointers. The runtime resolves tensor IDs and byte offsets to device addresses after allocation. The plan is inspectable text for version 1; task records can be converted into the executor's compact device representation when loaded. Serialize `plan.json` canonically and record its SHA-256 alongside the weight SHA-256 in the run report, so a measured result identifies the exact package.

| `plan.json` section | Contents |
| --- | --- |
| Header | Plan format version, executor ABI, model kind, activation variant, and generator version. |
| Target | Compute capability, SM count, cooperative launch required, worker count, threads per block, kernel capability fingerprint. |
| Model | `D`, `H`, `O`, float32 dtype, tile heights, tensor shapes and strides. |
| Storage | Input/output binding sizes, scratch size and tensor offsets, weight entries with byte offsets and lengths. |
| Tasks | Stable task ID, supported opcode, output row interval, referenced tensor IDs, wave and worker. |
| Waves | For each worker, its ordered task IDs; all workers share the same wave count. |
| Integrity | Size and SHA-256 of `weights.bin`; format and generator versions for diagnostics. |

A task record might say `{"id": 3, "opcode": "gemv_bias_gelu_f32", "wave": 0, "worker": 3, "rows": [96, 128], "input": "x", "weight": "W1", "bias": "b1", "output": "h"}`. `[96, 128]` is a half-open row interval. The task describes *what* to execute and *which slice* to write; it neither embeds a host address nor asks CUDA to place worker 3 on physical SM 3.

The C++ loader is authoritative for device compatibility. It rejects an unknown format/ABI/opcode, a changed SM count or block size, insufficient buffer sizes, a failed weight checksum, or a cooperative grid that cannot be resident with the compiled kernel. Python should catch as much as its input capability report allows, while treating runtime rejection as a valid outcome if the device or binary changed after planning. This follows the [CUDA cooperative-launch and occupancy requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html).

## 6. Python-facing workflow and checks

Keep the installable `tileplan` package and the `tile_runtime` CMake target in one project. The driver builds the executable for the host GPU before requesting kernel occupancy; with CMake 3.24 or newer, [`CUDA_ARCHITECTURES=native`](https://cmake.org/cmake/help/v3.30/prop_tgt/CUDA_ARCHITECTURES.html) is one way to select that architecture. Record the resolved architecture, compiler flags, executable identity, Python/PyTorch versions, and dependency versions in the run directory. The one-command entry point starts after environment installation; it does not silently switch away from CCCL if the hardware go/no-go test fails.

The proposed single entry point builds or locates the CUDA executable, then runs reference generation, target probing, planning, persistent execution, PyTorch baselines, correctness checks, and a short benchmark, keeping all artifacts in one run directory. It runs on the Linux GPU host after installing the Python project's declared dependencies and CUDA build prerequisites. The command checks these prerequisites and the actual GPU before doing work; a supplied `--executor` can reuse a prebuilt binary:

```text
python -m tileplan demo --out RUN_DIR
```

Individual stages remain callable for debugging or reusing a plan:

```text
python -m tileplan reference --seed S --out RUN_DIR
python -m tileplan probe --executor ./tile_runtime --out RUN_DIR/target.json
python -m tileplan build --model RUN_DIR/model.json --weights RUN_DIR/weights.npz \
                         --target RUN_DIR/target.json --tile-h1 N --tile-h2 M \
                         --out PACKAGE_DIR
python -m tileplan inspect PACKAGE_DIR
python -m tileplan verify PACKAGE_DIR
python -m tileplan benchmark --package PACKAGE_DIR --cases RUN_DIR/reference_cases.npz \
                             --out RUN_DIR/benchmarks.json
```

`build` also writes a concise planning report: task counts by opcode, wave count, worker loads and idle workers, scratch/weight bytes, package hashes, and target properties. `inspect` reads the package without CUDA. `verify` rechecks schema, hashes, bounds, unique writers, topological wave order, and complete worker coverage; it does not claim GPU correctness.

The PyTorch fixture includes fixed test vectors, including zeros and seeded random inputs. The checker compares both `h` and `y` with the C++/CUDA output; `h` can be exposed through diagnostic readback rather than the normal inference interface. Run it against packages whose dimensions produce partial tail tiles as well as full tiles. Apply one stated `abs(actual - reference) <= atol + rtol * abs(reference)` rule to each output element, with tolerances fixed before benchmarking; report maximum absolute error and maximum error divided by that element's allowed tolerance, as well as pass/fail. `reference_cases.npz` records seeds, dimensions, activation mode, inputs, and expected values separately from the runtime package.

The baseline runner reconstructs the MLP from `plan.json` model metadata and `weights.bin`, so the package alone identifies the weights it times. It evaluates `h = GELU(W1 @ x + b1)` and `y = W2 @ h + b2` under `torch.inference_mode()` on the target GPU. It pins full float32 matmul precision with TF32 disabled for the first comparison and records the effective policy, following [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) rather than relying on a default that may change. It checks eager and replay outputs against the saved reference before accepting timings. Following [PyTorch's CUDA Graph guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs), warm up on a side stream, keep graph input/output storage at stable device addresses, and capture the fixed-shape operation sequence without host synchronization. Before a replay for a new case, copy that case into the static input buffer. Record PyTorch/CUDA versions, device identity, math policy, warmup count, replay count, and the measured distributions. The eager and replay paths use the same PyTorch operations, while PyTorch may choose a different number and type of kernels than the tile executor.

The one-command report places persistent, PyTorch eager, and PyTorch CUDA Graph timings next to each other for the same shape and saved benchmark input. Python writes that input once as contiguous float32 for the C++ benchmark mode and requires its hash to match the hash in the runtime report before combining results. Each path warms up, keeps weights and the input resident, synchronizes before its timed rounds, and uses CUDA events on the executing stream. Report event-bracketed latency per forward and round-to-round spread. Measure host-to-host latency separately within an already loaded process, from an available host input vector through synchronized host output; exclude process startup, plan loading, and graph capture from that per-inference number and report setup costs separately. A batch average is labeled as such; a p95 requires individual-invocation samples. Run the paths serially and repeat them in rotated order to reduce clock and temperature drift. The report identifies the PyTorch results as workload-level baselines: different kernels, fusion, and tiling can affect their latency, so these numbers alone do not isolate the benefit of a static schedule.

The first implementation milestone is a single tiny MLP package that can be inspected, loaded, and checked against the reference. Next, freeze a shape and tile-height suite before recording comparative results, including a shape with fewer tasks than SMs and a shape with several tasks per worker. A one-token decoder should add new operation/task types and state contracts to this IR only after the MLP plan and loader interface are stable. Its dependency graph, or a focused companion micrograph, must contain partial tile dependencies before it can test early handoff.
