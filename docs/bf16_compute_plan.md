# BF16 computation with PyTorch-compatible precision

Implementation plan, 2026-10-13. Target workspace: `tux:~/dev/tiles`.
Inspected revision: `ed925a4c3a0b955c7016267e6f71ae911dc1f8d9`.
Status: implemented on 2026-10-14 in `tux:~/dev/tiles`. See
[implementation and validation results](bf16_compute_results.md) for the final
kernels, reference calibration, tests, and measured arithmetic default.
The baseline evidence below describes the inspected pre-change revision.

## Implemented result

The implementation replaces the BF16-weight/float32-activation path with a
BF16 inference policy that uses BF16 weights, activations, intermediate operator results, and K/V,
while retaining FP32 accumulation and the FP32 operations used by the installed
PyTorch/Transformers implementation. The existing float32 path remains available. The completed executor supports two policies:
float32 weights/activations/KV and BF16 weights/activations/KV. The former
BF16-weight/float32-activation policy is removed, including its runtime dispatch,
configuration options, and positive execution tests.

The compatibility target is **Transformers 4.57.6 eager Llama inference with
PyTorch 2.14.1 on the RTX 3050**, loading the pinned checkpoint directly with
`dtype=torch.bfloat16`, without autocast, compilation, or an alternate attention
backend. Match observable operator dtypes and rounding boundaries; use explicit
tolerances for reductions. Bitwise equality of complete model executions is
not the contract.

The implementation uses two stages: first establish the precision semantics with a
simple BF16-input/FP32-accumulating worker; then add and evaluate native BF16
Tensor Core matrix arithmetic inside that worker. The first stage is a valid
mixed-precision execution path, but must be reported as SIMT FP32 arithmetic
over BF16 operands, not as Tensor Core acceleration.

## Evidence and pre-change implementation

Read the four tile-graph design documents, the active decoder design, and the
autoregressive runtime design. The historical MLP notes describe old work;
the current source and `tileinfer/README.md` define the implementation baseline.

Verified on tux:

| Item | Observed value |
| --- | --- |
| GPU | RTX 3050, compute capability 8.6, 8 GiB; PyTorch reports BF16 support |
| Driver | 595.99.02 |
| PyTorch | `2.14.1+cu130`, CUDA build 13.0 |
| PyTorch source revision | `5c4886908584029761b579af026dcfb627c84070` |
| Transformers | `4.57.6` |
| Checkpoint | Llama 3.2 1B base, revision `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Default BF16 reduced-precision GEMM reduction | Enabled |
| Default SDPA math reduced-precision setting | Disabled |
| Float32 matmul setting | `highest`, TF32 disabled |
| Current worker | 1024 threads per block; one logical worker per SM |

At the inspected baseline, packages were version 3, with version-2 float32 compatibility.
`WeightDtype` selects only weight storage. `ops::TensorViews`, worker arguments,
sequence allocations, and diagnostics use float pointers for every activation
and K/V buffer. `row_dot` widens each BF16 weight to float. All arithmetic is
float32, including embeddings after lookup, residuals, final normalization,
and logits. Opcode names explicitly end in `_f32`.

Several baseline fusions change the result if a BF16 cast is added only at the
end: norm includes the scale multiplication, GEMV includes the residual,
RoPE combines two products and an addition, and SwiGLU combines SiLU and its
product. Attention uses an online float32 softmax recurrence. Its probability
and output rounding differ from eager BF16 attention.

A one-off GPU investigation is saved under
`runs/bf16-compute-research-2026-10-13/` on tux. `probe.py` and `probe.json`
record the environment, installed Llama source hash, a first-layer operator
dtype trace, and seeded rounding examples. A real checkpoint forward returned
BF16 logits and BF16 K/V; its RoPE inverse-frequency buffer was FP32. In the
synthetic probes, omitting an intermediate rounding boundary changed:

| Probe | Different output elements | Maximum absolute difference |
| --- | ---: | ---: |
| RMSNorm cast before scale | 883 / 4096 | 0.03125 |
| SiLU cast before product | 1109 / 4096 | 0.03125 |
| RoPE casts between products and sum | 693 / 2048 | 0.015625 |
| Projection cast before residual | 1185 / 4096 | 0.03125 |
| Eager attention versus FP32 attention rounded only at output | 1005 / 2048 | 0.015625 |

These examples establish that the boundaries matter. They are not estimates
of checkpoint accuracy, a validation of future kernels, or timing measurements.

## Online research and its implications

1. The version-pinned [Transformers Llama implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py)
   explicitly promotes RMSNorm and RoPE frequency/trigonometric work to FP32.
   It casts normalized values before applying the norm weight, returns RoPE
   coefficients in the activation dtype, and requests FP32 attention softmax
   followed by a cast to the query dtype. In inference its language-model head
   leaves logits in the model dtype. The installed-source dtype trace confirms
   these boundaries for this checkpoint.
2. PyTorch's pinned [operator math type definitions](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/OpMathType.h)
   use float internally for BF16 elementwise math. Its [CUDA SiLU kernel](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)
   computes in that math type and returns the tensor scalar type. BF16 tensors
   therefore do not imply BF16 exponential evaluation or BF16 accumulation.
3. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
   describes FP32 accumulation for reduced-precision GEMMs and warns that
   backend reduction order can change answers. The [CUDA precision controls](https://docs.pytorch.org/docs/2.14/notes/cuda.html#reduced-precision-reduction-in-bf16-gemms)
   allow disabling BF16 reduced-precision reductions and split-K separately.
   Pin these controls for the correctness oracle and report the ordinary
   default setting as a separate comparison.
4. [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
   says the math backend keeps intermediates in float for BF16 inputs and that
   backend fusion changes numerical results. Eager attention and SDPA require
   separate policies; the existing benchmarks already select eager attention.
5. NVIDIA's [Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations)
   specifies BF16 matrix inputs with FP32 accumulators. The [PTX MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
   provides BF16 `m16n8k16`/`m16n8k8` operations on SM80 and later, including
   SM86. [CUDA BF16 conversions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)
   provide explicit round-to-nearest-even conversion. Use those conversions
   at numerical boundaries and verify generated instructions before claiming
   native BF16 matrix execution.

## Precision contract

Let `B(v)` mean round a FP32 value to BF16 with round-to-nearest, ties-to-even.
Let `F(v)` mean its exact widening to FP32. A BF16 temporary can remain in a
register, but the cast must actually occur. The following table is the implemented
contract derived from the installed dtype trace and the rounding probes.

| Stage | Stored operands | Internal math / reduction | Required result and boundary |
| --- | --- | --- | --- |
| Token IDs / positions | int32 IDs; integer positions | Integer indexing; FP32 position conversion for RoPE | IDs remain integers |
| Embedding lookup | BF16 weight | Copy the stored BF16 value | BF16 hidden vector |
| RMSNorm statistics | BF16 hidden | Widen, square, mean, epsilon addition, reciprocal square root in FP32 | FP32 normalization factor |
| RMSNorm application | BF16 hidden and scale | FP32 normalize; cast; FP32 scale multiplication | `B(F(B(F(x) * factor)) * F(weight))` |
| Q/K/V and FFN projections | BF16 weights and inputs | BF16 matrix operands, FP32 accumulation | Cast each projection result to BF16 |
| RoPE frequencies / angle / sin / cos | FP32 frequency table; integer position | FP32 frequency scaling, angle, trigonometry | Cast sin/cos to BF16 before application |
| RoPE application | BF16 Q/K and coefficients | Each multiply and add uses widened operands | BF16 product temporaries, then BF16 sum |
| K/V append / history | BF16 rotated K and projected V | Copy | BF16 K/V throughout all layers |
| QK dot product | BF16 Q/K | FP32 accumulation | BF16 raw score |
| Attention scaling | BF16 raw score; FP32 scalar | Widen score and multiply by FP32 `head_dim**-0.5` | BF16 scaled score |
| Attention mask, when needed | BF16 score and additive mask | FP32 elementwise addition | BF16 masked score; no future keys are read |
| Attention softmax | BF16 scaled/masked scores | FP32 maximum, subtraction, exponentials, sum, normalization | Round normalized probabilities to BF16 |
| Attention probability × V | BF16 probabilities and V | FP32 accumulation | One BF16 cast after the complete dot product |
| Output projection + residual | BF16 attention, weights, residual | FP32 projection accumulation; BF16 projection result; widened add | BF16 residual result; two casts, not one |
| SiLU | BF16 gate projection | FP32 exponential, denominator, division | BF16 SiLU result |
| SwiGLU product | BF16 SiLU result and up projection | FP32 multiplication | BF16 product |
| Down projection + residual | BF16 product, weights, residual | Same projection/add boundaries as above | BF16 next-layer hidden |
| Final RMSNorm | BF16 hidden and final scale | Same norm contract | BF16 normalized hidden |
| Language-model head | BF16 final hidden and tied embedding | FP32 accumulation | BF16 mathematical logits |
| Greedy selection / public logits | BF16 mathematical logits widened to FP32 | FP32 comparisons, integer tie breaking | Keep existing float32 API; choose smallest token ID on ties |
| Diagnostic snapshots | BF16 tensors | Widen only after the modeled computation | Float32 host arrays containing exact widened BF16 values |

Both supported policies retain FP32 scalar configuration constants and FP32
reduction state. Do not quantize epsilon, attention scale, or inverse frequencies
just because activations are BF16. Keep FP32 math for SiLU, softmax, and
normalization. Do not accumulate dot products or the attention output in BF16.

### Boundaries that fused routines must preserve

For every projection followed by a residual, retain
`B(F(B(accumulated_projection)) + F(residual))`. Adding the residual to the
FP32 accumulator and casting once gives a different function. Keep the existing
fused scheduling if useful; insert the conversion inside the epilogue.

For SwiGLU, retain `B(F(B(silu(F(gate)))) * F(up))`. The SiLU output is rounded
even when it is never written to a separate global tensor.

For a split-half rotary pair `(a,b)`, first round the FP32 sine/cosine. Compute
`B(F(B(F(a)*F(cos))) - F(B(F(b)*F(sin))))` and the corresponding second
coordinate with addition. Negation is exact. Prevent compiler contraction or
reassociation across these BF16 temporaries; a single fused multiply-add does
not implement this contract.

For RMSNorm, keep the FP32 normalized value, its BF16 cast, and the later
weight multiplication distinct. Improve the current serial norm reduction
only after isolating precision changes; reduction order remains tolerance-based.
PyTorch explicitly uses reciprocal square root; compare a suitable FP32 rsqrt
implementation with the current `1/sqrt` path using isolated operator inputs.

For logits, cast the accumulator to BF16 and then widen it for the existing
float32 logits buffer. Sampling unrounded FP32 accumulators could select a
different token when BF16 makes two logits equal. Widening BF16 for comparison
preserves its ordering and lets the existing sampling kernel remain in place.

## Attention implementation

Retain the current online FP32 attention for the pure-float32 policy. Implement a
separate eager-compatible BF16 attention routine; replacing its pointer type
alone would round its running output after every cache entry and be incorrect.

Initially keep one task per complete query head, the existing GQA mapping,
and the visible prefix `0..position`. Reserve a BF16 score/probability workspace
of `[query_heads, max_context]`, reused across layers. Each head owns a disjoint
slice. A simple implementation has these steps within that task:

1. Compute every QK dot in FP32. Apply `B` to the raw dot, then separately
   apply the scale and `B` again. Store the scaled BF16 score. Compute the FP32
   maximum over the valid prefix.
2. Widen those scores and compute the stable FP32 exponential sum. Use FP32
   block/warp reductions; synchronize before reusing shared reduction storage.
3. Compute each normalized probability in FP32, round it to BF16, and overwrite
   the corresponding score slot after the denominator is available.
4. Read BF16 probabilities and V, accumulate each output coordinate in FP32
   registers/reduction storage across the full prefix, and write BF16 once.

Use explicit barriers for each handoff. Never read unused cache capacity.
For one-token decode over an exact prefix, every represented key is visible,
so adding a zero causal mask can be omitted. Static-cache reference runs must
mask unused capacity explicitly. Do not use non-square `is_causal=True` as a
substitute for the visible-prefix rule.

The workspace needs 256 KiB at 32 heads × 4096 positions, or 512 KiB at 8192
positions; it is shared across the 16 sequential layers. Avoid a full
FP32 attention matrix or one workspace allocation per layer. Register/shared
FP32 reductions and optional diagnostic buffers are additional allocations.

This routine is deliberately explicit about eager rounding. A later online or
fused attention optimization must preserve those casts, or declare a distinct
numerical policy and validate against its corresponding PyTorch backend.

## Package, runtime, and compatibility design

### Explicit modes and versioning

Introduce version 4 and `tileinfer-decoder-v4`; do not reinterpret version-3
`_f32` opcodes as BF16 arithmetic. Represent two supported combinations:

| Policy | Weight dtype | Activation / K/V dtype | Reduction dtype |
| --- | --- | --- | --- |
| Existing float32 | float32 | float32 | float32 |
| Replacement eager BF16 | bfloat16 | bfloat16 | float32 |

Reject mismatched weight/activation/cache dtypes, including the retired BF16
weights with float32 activations. Keep a small explicit contract instead of
implicitly accepting unsupported combinations. FP32 reductions and selected
FP32 internal math remain part of the BF16 policy, not a separate activation mode.

Version 4 should declare `numeric_policy`, `weight_dtype`, `activation_dtype`,
`kv_dtype`, `accumulation_dtype`, `logits_compute_dtype`, and `api_output_dtype`.
Use `numeric_policy: "hf-llama-eager-bf16-v1"` for the new policy. Tensor
descriptors remain authoritative for storage dtype and byte extent. Include
the policy and implementation variant in the executor capability/fingerprint.
Public JSON reports must distinguish mathematical logits from their API dtype.

Use precision-neutral v4 model kinds (`llama_decoder_block` and
`llama_decoder_stack`). Replace the old ambiguous `model.dtype` field with
the explicit dtype fields above. Continue accepting the `_f32` model kinds
and fixed float32 dtype under the v2/v3 readers only for pure-float32 packages,
so no header contradicts the new activation policy.

Add BF16-specific opcode meanings for norm, GEMV, rotary query/cache, eager
attention, and SiLU/product, plus stack embedding/final-norm/logit stages.
The reduction dtype belongs to their documented semantics, not to their names.
Existing float32 opcode meanings remain intact.

Keep v2 and v3 pure-float32 package compatibility, including their existing
target-validation rules and known fingerprints. Reject v3 BF16-weight packages
with an actionable message to regenerate them as v4 BF16 packages: their
float32 activation/cache layout and arithmetic are superseded. Do not retain
a legacy mixed-mode execution specialization or silently reinterpret old
packages. Unknown or incompatible worker targets still require regeneration.
New formats and fingerprints must fail clearly in old readers. Regenerate
packages from the source model/checkpoint instead of editing JSON in place.

### Typed storage and offset rules

Use one compile-time storage scalar type for weights, activations, scratch,
and K/V, dispatching only to float32 or BF16. Keep reduction types explicitly
float. Parameterize the tensor views and the worker's input, output, scratch,
and K/V arguments. Remove the BF16-weight/float32-activation specialization.
Keep logits and requested block traces as FP32 views with explicit BF16
widening at their write boundaries; these API views are not a mixed
activation policy.

Store allocation extents and serialized offsets in bytes. Validate alignment,
dtype divisibility, overflow, shape, tensor ownership, and aliases before
turning them into typed element offsets. The initial activation/scratch/KV
storage classes can each have one uniform dtype; the new attention workspace
shares the BF16 scratch dtype. A fully generic mixed-type tensor allocator is
not needed for this milestone.

Both `SequenceContext` and the test-only `DiagnosticContext` need matching
typed allocations, launches, and snapshot conversion. Use a shared helper for
buffer ownership/conversion so their behavior cannot diverge. Preserve CCCL
streams, buffers, transfers, launch/timing, ticket lifetimes, and stream ordering.

Update every hardcoded four-byte activation calculation: scratch counts and
regions, per-layer K/V stride, K/V base pointers, input/output bindings,
snapshots, and zero-unused-cache logic. Weight strides still use weight dtype.
The full-stack input remains an int32 token ID. Block-test hidden inputs can
remain float32 externally and convert once before entering the BF16 worker.

Diagnostic snapshot conversion must be queued before a ticket's completion
event and before later launches overwrite the source. Return widened float32
vectors to callers. Convert only on an explicit diagnostic request; ordinary
generation must not introduce full-cache copies or host synchronization.

The block harness currently reports `ffn` by subtracting the residual from the
final output. That cannot recover the actual down-projection result after a
rounded BF16 residual addition. Capture the projection before addition in the
test-only trace, and distinguish it from a reconstructed residual delta. Likewise,
index widened scratch/KV snapshots in tensor elements, using dtype-correct
offset conversion rather than dividing every serialized offset by four.

Preserve the current cache packing and its alignment when deriving new strides.
The checkpoint's K/V payload falls from 64 KiB/token to 32 KiB/token: 128 MiB
at 4K capacity and 256 MiB at 8K. Alignment, allocator overhead, scratch, and
other buffers must also appear in the memory report. Weights stay unchanged.

### Capability probing

Query attributes and occupancy for each actual compiled worker specialization,
including a later MMA specialization. Report per-variant registers, static and
dynamic shared memory, block size, and resident-block limit. Either retain a
conservative minimum for a combined target or select and validate the exact
variant in the package; never borrow occupancy from the old worker.

Build before generating `target.json`, and reject missing BF16 capabilities.
Keep one logical block per SM initially. Rerun the CCCL cooperative gate after
worker changes. Physical SM affinity, device queues, and new scheduling
protocols are outside this precision change.

## Planned implementation sequence and affected files

### 1. Freeze the oracle and precision tests

Extend `tileplan/src/tileplan/decoder_reference.py` and `decoder_stack.py` with
an explicit policy argument. Preserve the present CPU float32 reference and
fixtures. Add a reference that holds true BF16 tensors and executes separate
PyTorch operations at the boundaries above. Do not simulate the entire policy
by quantizing weights and returning them to float32.

Use CPU tests for conversion, schema, and logical dtype rules, but make CUDA
PyTorch on tux the numerical authority for matrix/reduction comparisons.
Capture inputs and outputs for norm, Q/K/V, rotary, scaled scores, probabilities,
attention, both residuals, SiLU/product, final norm, and logits. Diagnostic
traces should record original dtype before widening to NumPy-compatible FP32.

For controlled correctness, pin `torch.inference_mode()`, eager attention,
no autocast, `highest` FP32 matmul, TF32 disabled, and
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)`.
Record both reduction and split-K flags. The research probe initially used
the Boolean `False`, which disables reduced reductions but leaves split-K
enabled; its rounding evidence is not a full strict-oracle regression.
Also run the ordinary default-reduction setting as a separately labeled
compatibility check. Do not claim to reproduce every default cuBLAS algorithm.

Acceptance: isolated examples distinguish the required boundaries from a
single final cast; reference caches and outputs carry the intended dtypes.

### 2. Add the package policy and independent validation

Change `decoder_plan.py`, `decoder_stack.py`, `probe.py`, and `cli.py`, then
`tileinfer/src/runtime/plan.hpp`, `plan.cpp`, and `plan_validation.cpp`.
Add v4 descriptors, opcodes, activation/cache sizes, attention workspace,
stack stages, reports, and policy checks in both Python and C++.

Select the complete policy with `--dtype {float32,bfloat16}`; the choice sets
weight, activation, and K/V storage together. Existing weight-dtype builder
arguments may remain as aliases for this complete policy, but must not expose
an independent compute-dtype override. BF16 selects the mixed internal-math
contract above, including FP32 reductions. Thread the policy through the
checkpoint importer and `generate.py`, which default to BF16 for new packages.
Preserve the synthetic builders' float32 default; new BF16 packages use the
replacement policy. Gate its release on checkpoint
validation rather than keeping the retired path as an opt-in/default fallback.
When `generate.py --package` reuses an existing package, read its declared
policy and matrix kernel, inherit omitted options, and reject conflicting
explicit options; changing a launch flag must not reinterpret the package's storage or arithmetic.

Acceptance: reproducible packages for both supported policies; independent
Python/C++ rejection of inconsistent dtypes, wrong strides, overlapping head
workspaces, incorrect byte counts, unsupported precision/opcode combinations,
bad tied-weight aliases, and stale targets. Retain v2/v3 pure-float32 coverage
and add explicit rejection/regeneration tests for v3 BF16-weight packages.

### 3. Implement typed buffers and the simple numerical worker

Change `detail/launch.cuh`, `detail/uploaded_plan.cuh`, `uploaded_plan.cu`,
`worker_kernel.cu`, `ops.cuh`, `sequence_context.cu`, `diagnostic_context.cu`,
`snapshot.cu`, and ticket diagnostic storage/conversion as needed.

First retain the current GEMV ownership/reduction structure: widen both BF16
operands, accumulate in FP32, apply the required BF16 result/epilogue casts.
Implement the norm, rotary, residual, and SiLU boundaries without changing the
wave DAG. Add the eager-compatible attention routine and workspace. Embedding
copies BF16; final norm and logits use the same policy as their block counterparts.

Keep FP32 running attention sums in registers/shared memory until final output,
even though persistent activations are BF16. Use explicit RN conversion helpers;
do not rely on an overloaded BF16 expression's incidental intermediate types.
No `--use_fast_math` or cross-boundary contraction changes in this stage.

Acceptance: operator/block/stack checks pass in both supported policies;
conversion does not break reset, tickets, context sharing, EOS, capacity, or
device token feedback.
Remove retired mixed-mode dispatch, buffer handling, and execution tests.
Reprobe this compiled worker before building its packages.

### 4. Integrate the real checkpoint and comparisons

Extend `llama32_import.py`, `tileinfer/tests/llama32_checkpoint.py`,
`benchmark_generate.py`, `benchmark_decode.py`, and runtime JSON reporting in
`main.cpp`. BF16 imports retain the exact checkpoint BF16 bytes and tied
embedding/LM-head aliasing. A float32 import widens checkpoint values once
offline and stores float32 weights for the pure-float32 policy. Reuse checkpoint
provenance and source hashes; record each generated package's own weight hash.
Regenerate old BF16-weight/float32-activation packages as v4 BF16 packages.

Compare repeated one-token calls on the same token sequence first, including
layer outputs and the valid K/V prefix. Run free greedy generation separately
so token divergence cannot conceal the first numeric mismatch. Match the
reference's prefix/cache/mask shape. Keep sequential prefill as the primary
correctness control; full-prompt PyTorch prefill uses different matrix shapes
and may produce different reductions even with the same dtype.

Remove hardcoded claims that runtime arithmetic always stays float32.
`benchmark_generate.py` currently enforces generated-ID agreement only against
the float32 reference; update it to enforce the selected compatible policy.
Compare the pure-float32 policy against the float32 reference and the replacement
BF16 policy against the BF16 reference. Convert former mixed-mode positive
regressions to BF16-policy regressions, preserving their semantic coverage.

Acceptance: checkpoint tensor/logit validation and short greedy regressions
pass under fixed acceptance rules; reports state every precision/backend flag.
Only then release the replacement as the BF16 path for newly imported packages.
An explicit float32 choice selects float32 weights, activations, and K/V;
there is no BF16-weight/float32-activation option. Old pure-float32 packages
retain their semantics; old BF16-weight packages fail with regeneration guidance.

### 5. Add native BF16 matrix arithmetic

Add a device-callable matrix tile routine, preferably a small CUDA/PTX wrapper
using BF16 MMA with FP32 accumulators on SM86. Keep the validated SIMT routine
as the numerical control and tail fallback. Begin with Q/K/V, FFN, output
projection, and LM head; attention QK/PV can adopt MMA later under the same
rounding contract.

For batch-one GEMV, map output rows to MMA's M dimension and reduction columns
to K. A candidate `m16n8k16` tile has only one useful vector column in N;
explicitly initialize padded lanes and discard unused output columns. Stage
BF16 operands, retain FP32 accumulators across all K tiles, then run the same
BF16 epilogue. Do not round the accumulator between K tiles. Specify fragment
ownership, lane mapping, alignment, and K/row tails before writing the kernel.

This mapping wastes columns and may lose on a bandwidth-sensitive one-token
workload. Evaluate it; do not assume Tensor Cores improve latency. Record the
implementation variant in the fingerprint, inspect SASS/Nsight for BF16 HMMA,
and compare resource costs and output error with the SIMT implementation.
If a library tile component proves preferable, preserve `uv.lock` and CMake
presets and introduce the dependency deliberately. Host-side cuBLAS calls
cannot serve as per-task routines inside the cooperative worker.

Acceptance: real native BF16 instructions confirmed, compatible precision
boundaries retained, all correctness/sanitizer gates pass, and fresh occupancy
permits the cooperative launch. Select the default arithmetic variant from
measured results; report SIMT versus MMA honestly if the simpler path wins.

## Validation and completion criteria

Use three levels of numerical checks:

- **Representation and local boundary checks:** bitwise BF16 conversion/copy
  tests, ties-to-even cases, signed zero, finite extremes, and isolated
  multiplication/addition with identical operands. Inspect BF16 bits rather
  than rounded JSON decimal text. Norm/SiLU/trigonometric/reduction checks
  remain tolerance-based where the underlying FP32 algorithm differs.
- **Operator, synthetic block, and stack checks:** report max absolute error,
  RMS error, `abs(error)/(atol + rtol*abs(reference))`, BF16 ULP distribution,
  mismatched elements, and finiteness. Compare each operator with the same
  recorded input to separate local arithmetic error from accumulated drift.
- **Checkpoint checks:** teacher-forced token-by-token logits, layer outputs,
  K/V prefixes, top-k agreement, top-two margins, and short greedy sequences.
  Widen BF16 outputs for measurement only. Keep precision-specific tolerance
  tables in fixtures/reports.

Set provisional isolated CUDA BF16 output tolerances to `atol=1e-2,
rtol=1e-2` for initial diagnosis, not as an automatic whole-model acceptance
standard. During step 1, measure independent reference variation for the
actual shapes, freeze per-stage/model thresholds before judging the executor,
and include absolute floors for near-zero outputs. Do not reuse tight FP32
tolerances for BF16 or loosen tolerances after observing a failing kernel.
Require all frozen numerical gates and existing FP32 gates to pass.

For frozen greedy smoke prompts, require token agreement. For broader prompt
sets, classify a disagreement using the first divergent step and its logit
margin. If each candidate's maximum logit error is bounded by `e`, a reference
top-two gap greater than `2e` should preserve its winner. A near tie is useful
diagnostic evidence, not permission to silently pass a required smoke test.
BF16 creates more ties; cover the smallest-token-ID tie rule explicitly.

Preserve existing semantic coverage while replacing retired mixed-mode cases:

- `tileplan/tests/test_decoder_reference.py`, `test_decoder_stack.py`, and
  `test_decoder_planning.py`: all policies, dtype/size/alias validation,
  deterministic serialization, reference cache semantics, legacy pure-float32
  packages, and rejection of retired mixed-mode packages.
- `tileinfer/tests/loader_regression.py`, `decoder_regression.py`,
  `stack_regression.py`, and `kernel_regression.py`: new opcodes, both supported
  policies, malformed packages, GQA, odd row/K tails, reset, capacity, and
  stable softmax.
- Ticket/context/ring tests: typed allocations, diagnostic lifetime, shared
  immutable model across independent caches, EOS skip behavior, and no new
  per-token host synchronization.
- `sanitizers.py`: include the new policy and later MMA variant in memcheck,
  racecheck, synccheck, and initcheck, including attention workspace tails and
  uninitialized padded MMA lanes. Preserve and disclose the existing narrow
  CCCL pinned-pool/API-reporting exceptions.

Cover positions 0, 1, 63/64, 255/256, and the last valid cache slot on small
fixtures; separately probe RoPE near 4095/4096, 8191/8192 and its frequency
scaling boundaries. Include zero inputs, cancellation, high-magnitude scores,
strongly peaked/flat attention, long prefixes, and all-invalid-package cases.
No unsupported long-context allocation is needed to test rotary arithmetic.

Run the repository's existing commands, with new modes integrated into their
coverage. From the repository root on tux:

```sh
uv run --locked --project tileplan ruff check --config tileplan/pyproject.toml tileplan tileinfer/tests generate.py
uv run --locked --project tileplan ruff format --check --config tileplan/pyproject.toml tileplan tileinfer/tests generate.py
uv run --locked --project tileplan ty check tileplan/src tileplan/tests tileinfer/tests generate.py
uv run --locked --project tileplan python -m unittest discover -s tileplan/tests
```

From `tileinfer/`:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
cmake --build build/release --target sanitizers
```

CTest includes clang-format and the cooperative gate. Run the checkpoint
comparison separately with the newly added precision selector, then benchmark
only validated packages generated from a fresh compiled-worker probe. Preserve
host C++23, CUDA C++20, the lockfile, and build presets.

## Measurement and final deliverables

Keep the numeric-policy change separate from schedule, block-size, and tile-size
tuning. Compare replacement BF16 SIMT, BF16 MMA, and PyTorch BF16 eager/graph
using identical checkpoint bytes, prompts, visible contexts, precision flags,
EOS settings, and generation work. Compare pure-float32 execution separately
with its float32 PyTorch reference. Archived pre-change mixed-mode reports
may provide historical context; their path is not kept in the completed executor
or in the supported-mode benchmark matrix.

Measure fixed contexts and autoregressive generation separately. Start with
the existing 1/64/256 contexts, then extend to 1024/4096 if feasible. Freeze
shape/tile candidates, separate pilot selection from independent final rounds,
rotate implementation order, and record clocks/temperature, all samples,
package/binary hashes, compiler versions, policy, worker resources, and traffic.
Do not infer a speedup from halved activation/cache storage: weights are already
BF16, and extra attention passes or MMA padding can offset the savings.

Completion produces a documented v4 precision contract for the two supported
policies, preserved pure-float32 regressions, replacement BF16 operator/block/
checkpoint validation reports, retired-package rejection tests, all required
format/static/CTest/sanitizer gates, freshly probed worker variants, and a
controlled comparison selecting the BF16 arithmetic implementation. The old
BF16-weight/float32-activation path and its independent configuration options
are removed. The implementation and its accuracy/performance evidence are
recorded in the linked results document; archived mixed-mode reports remain historical.
