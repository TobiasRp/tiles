# Llama-style one-token decoder design

Status: milestones 1–6 implemented; BF16 computation supersedes BF16-only
weight storage in version 4, 2026-10-13. The current precision and migration
contract is in [the BF16 implementation plan](bf16_compute_plan.md). Version-4
packages use either float32 or BF16 weights, activations, and K/V; reductions
and selected internal operations remain float32. Version 5 (2026-10-22) keeps
that arithmetic and stores the whole token as one flat task program:
embedding, every layer, final normalization, and logits, scheduled into global
waves. Older package formats are no longer read and must be regenerated; see
[the megakernel plan](megakernel_next_steps.md). The historical milestone
results below describe version 3.

Historical status, 2026-10-12: The CPU reference,
version-3 decoder package, cooperative CUDA arithmetic, and compact repeated-block
stack are available. Synthetic float32 block and full-stack fixtures are under
`tileplan/tests/fixtures/decoder_m1` and `decoder_m4`; the planner can pack
their weights as BF16. The pinned Llama 3.2 1B base checkpoint also imports
and passes a real-model regression.

This document specifies the path to one-token inference with the
[`meta-llama/Llama-3.2-1B` base model](https://huggingface.co/meta-llama/Llama-3.2-1B)
for [TilePlan](tile_graph_python_architecture.md) and
[TileInfer](tile_graph_cuda_runtime.md). BF16 weight storage and Hugging Face
checkpoint loading are implemented.

## Goal and first supported model

Build a batch-one, **one-token decode** path, first for one deterministic
Llama-style block, then for the complete Llama 3.2 1B stack ending in
vocabulary logits. The end-to-end goal is to import a pinned revision of the
official base checkpoint, keep its weights in **BF16** on the GPU, process
prompt tokens through repeated decode calls, and generate tokens whose logits
match a reference using the same stored weight values.

Each call accepts one token's hidden vector (or, for the full model, one token
ID) and one mutable sequence context. It consumes positions `0..p` and advances the
context from length `p` to `p + 1`. The plan and weights are fixed; the input,
position, cache contents, and output vary across calls. Only one invocation
may use a context at a time. A fixed maximum context length is part of the
plan; reaching it returns an error instead of silently overwriting cache.

The first synthetic block uses float32 weights, activations, KV cache, and
accumulation, batch size one, standard full causal attention, no dropout, and
the existing cooperative wave barrier. This isolates decoder semantics before
adding BF16 weight storage. The historical version-3 checkpoint path stores **all model weights**
(including embeddings and norm scales) as BF16, converts them to float32 when
consumed by the initial kernels, and keeps activations, KV cache, reductions,
and softmax in float32. Version 4 replaces this path with BF16 activations/KV and PyTorch-compatible
rounding; its required float32 internal math is specified in the BF16 plan.
Start with small synthetic dimensions and an independent PyTorch reference.
The first importer supports the pinned Llama 3.2 1B base checkpoint rather
than every model with a `Llama` label.

## Concrete Llama 3.2 1B target

The model has hidden width 2,048, 16 layers, 32 query heads, 8 KV heads,
head width 64, feed-forward width 8,192, vocabulary size 128,256, and
`rms_norm_eps = 1e-5`. Its Llama 3 RoPE uses `rope_theta = 500000`, frequency
scaling factor 32, low/high frequency factors 1/4, and original context 8,192.
The output projection shares the token-embedding weight. The checkpoint
supports a 128K-token context, but the first GPU package should allocate a
smaller fixed capacity, such as 4K or 8K tokens. Record the checkpoint
repository and immutable revision in the package and regression report.
These values come from [Meta's model specification](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py),
the [model card](https://huggingface.co/meta-llama/Llama-3.2-1B), and
[Hugging Face's Llama 3.2 conversion](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).

When consuming the Hugging Face safetensors, use its Q/K row layout and
split-half rotary pairing. Hugging Face's converter permutes Meta-format Q/K
weights for that convention. Do not apply the Meta-format permutation a
second time; compare rotated Q/K and logits against the same Hugging Face
revision before accepting the importer. See the
[conversion mapping](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)
and [rotary implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

## Block computation

For hidden width `d`, feed-forward width `f`, `n_q` query heads, `n_kv` KV
heads, and head width `d_h`, require `n_q % n_kv == 0`, even `d_h`, and
`d = n_q * d_h` for the initial block. Projection matrices use PyTorch-style
`[out, in]` layout. The base block has bias-free projections and RMSNorm
scales, with `rms_norm_eps` supplied by the model specification.

```text
u = RMSNorm(x, attn_norm)
q = Wq @ u;  k = Wk @ u;  v = Wv @ u
q = RoPE(q, p);  k = RoPE(k, p)
K[layer, p] = k;  V[layer, p] = v
for query head h:
    kv_head = h // (n_q // n_kv)
    scores[t] = dot(q[h], K[layer, t, kv_head]) / sqrt(d_h), t = 0..p
    a[h] = softmax(scores) @ V[layer, 0:p+1, kv_head]
r = x + Wo @ concat(a)
z = RMSNorm(r, ffn_norm)
g = Wgate @ z;  up = Wup @ z
y = r + Wdown @ (SiLU(g) * up)
```

Use a numerically stable softmax. RoPE pairing, frequency calculation,
`rope_theta`, and any scaling rule must be explicit metadata, with one
supported variant in the first decoder ABI. Do not infer them from tensor
shapes. At a single-token call, all visible keys are at positions `0..p`;
there are no future keys to mask. The reference should compute attention
over that exact prefix (or use an explicit mask) so a non-square causal-mask
convention cannot shift the answer. The first block can take a hidden vector
directly. The full model adds token embedding, repeated blocks, final
RMSNorm, and `lm_head` to produce one vocabulary-logit vector. Weight tying,
if present, is an explicit model property.

## State and execution contract

Separate immutable `LoadedPlan` (weights, encoded tasks, shape/config, target
fingerprint) from mutable `SequenceContext` (position, per-layer K/V storage,
hidden scratch, stream, and validity). The initial cache is contiguous
`K` and `V` storage with logical shape
`[layers, max_context, n_kv, d_h]`; the serialized plan describes byte sizes
and strides, not addresses. `position` is the count of valid cache entries
per layer. On a call at position `p`, each layer writes its own `K/V[p]`, then
its attention task reads entries `0..p`. On successful completion of every
layer and the output stage, set `position = p + 1`. A failed asynchronous
call may have partially written cache, so mark the context invalid until an
explicit reset. Reset sets position to zero; old bytes need no clearing
because tasks may read only the valid prefix. Context capacity and position
must be checked before launch.

Retain the existing logical-worker, cooperative-kernel model: no physical SM
affinity, no per-call graph construction, and every worker reaches each wave
barrier. The first decoder schedule has waves for norm, Q/K/V projection,
RoPE/cache write, attention, output projection/residual, second norm,
gate/up projection, activation/product, down projection/residual. Adjacent
stages may be fused only when their read/write contract remains clear.
Attention may begin after this layer's new K/V entry is published; the next
block begins after the prior block's residual output. An attention task can
initially own a complete query head and scan the full visible prefix. This
is intentionally a simple, correct mapping; tiling the context dimension
requires a later partial reduction.

The MLP version-1 task record hardcodes GEMV buffers and two waves. Add a
new plan format/executor ABI for decoder work. Plan tasks should reference typed tensor/storage IDs,
an opcode with a small validated operand payload, output region, predecessor
IDs, wave, and worker. Extend the scheduler to form topological waves from
the task DAG. The loader validates opcode-specific dimensions, ownership,
buffer bounds, cache reads/writes, and every dependency before encoding a
device program. Keep the first opcode set narrow: RMSNorm, GEMV, RoPE/cache
write, attention over a prefix, vector add, SiLU/product, embedding, and
logits. A planner tensor DAG and generic memory reuse can follow when the
first concrete block exposes a need; dedicated scratch buffers are adequate
at the start. Reprobe occupancy for the **compiled decoder worker** because
new opcodes change its resource use.

The version-2 planner chooses the earliest legal wave for each task and balances
estimated work across logical workers. Its automatic tile-height heuristic
compares candidate schedules and records every score in the planning report.
The compiled worker executes the whole token in one cooperative launch.
Through version 4, the stack package stored one block task template that the
kernel repeated across layers. Since version 5, the planner emits every
layer's tasks, plus row-split embedding, final RMSNorm, and logits tasks, as one
program. Layers reuse block scratch, and each layer has its own K/V region and
weight stride. The output embedding may alias
the input embedding by an explicit package flag. The target report declares
`execution_ready: true` and records occupancy from this compiled kernel.

## Milestones and acceptance checks

1. **Reference and semantics (complete).** Implement a small independent PyTorch block
   with fixed weights and a token sequence. Freeze shapes, RoPE variant,
   epsilon, head mapping, cache layout, and tolerances in test fixtures.
   Verify its sequential cache against recomputation on the same prefix.
2. **Plan and state ABI (complete).** Add a versioned decoder model specification,
   typed task operands, general wave scheduling, decoder package validation,
   and `SequenceContext`. Existing MLP packages and tests can be deleted.
   Inspect output shows waves, per-worker load, scratch, and KV bytes.
3. **One CUDA block (complete).** Execute one block for one token at `p=0`, then a
   sequence of tokens through the same context. Compare Q/K/V after RoPE,
   cache entries, attention output, both residuals, FFN output, and final
   hidden vector with PyTorch. Cover GQA, tail tiles, reset, and capacity
   boundary. Run CTest and Compute Sanitizer memcheck/racecheck/synccheck.
4. **Full decoder stack (complete).** Add embedding lookup, repeated blocks with one
   cache per layer, final norm, and logits. Compare token-by-token logits
   and cache with a PyTorch reference. Preserve a one-token API so the same
   plan can process a sequence by repeated calls.
5. **BF16 weight path (complete).** Extend the decoder package with per-tensor dtypes
   and packed BF16 weight storage; validate sizes, offsets, hashes, and tied
   weight aliases. Add BF16-loading GEMV, embedding, and norm paths that
   accumulate/compute in float32 while preserving float32 activations and
   cache. Test synthetic BF16-quantized weights against a PyTorch reference
   using the same BF16 values converted to float32 for computation. Keep
   float32 decoder packages working. The version-3 package selects one weight
   dtype (`float32` or `bfloat16`) and records it on every weight tensor;
   activations and K/V remain float32. The CUDA worker and weight-consuming
   operators are instantiated for both storage types, with float32 arithmetic
   after each load. Version-2 float32 packages remain readable. The compiled
   worker was reprobed at seven resident blocks per SM, and BF16 block and
   stack runs passed Compute Sanitizer memcheck, racecheck, and synccheck.
6. **Llama 3.2 1B checkpoint bridge (implemented).** Pin an accessible revision of
   `meta-llama/Llama-3.2-1B`. Read its `config.json` and safetensors, reject
   unsupported variants, map all named tensors to a BF16 plan, and alias the
   tied embedding/output weight without a second GPU allocation. Record
   revision and hashes. Compare intermediate block values, KV cache, and
   token-by-token logits for several short prompts against
   `AutoModelForCausalLM` with its stored BF16 weights converted to float32
   for the same computation policy. Run a short greedy-generation regression.
   The imported revision is `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
   A 64-token validation package from the base checkpoint has 2,471,628,800
   bytes of BF16 weights and aliases `lm_head` to the embedding. Across 27
   tokens from three prompts, the maximum absolute errors against a float32
   Transformers run were 0.000504 for block outputs, 0.0000477 for K,
   0.0000146 for V, and 0.0000801 for logits. All 27 next-token choices
   matched. The clean one-command report is at
   `runs/llama32-import-2026-10-12/comparison-final/report.json` on `tux`.
   A separate 4K-context package also completed a real-token GPU smoke run.
7. **Performance experiments.** Measure decode against PyTorch eager and
   CUDA Graph replay at fixed context lengths. Profile the decoder worker
   and memory traffic. Add chunked prefill, lower-precision activations/cache,
   cache paging/sliding-window variants, or per-tile readiness only when a
   measured workload justifies each change. A companion micrograph can
   isolate partial tile handoff if the decoder schedule does not.

   First measurements on `tux` (2026-10-12) use the pinned BF16 1B package
   with a 4K capacity and token ID 1 repeated in each prefix. GPU-resident
   one-token latency, in milliseconds, at context lengths 1/64/256 was
   241.0/247.8/270.9 for TileInfer, 25.7/26.0/26.5 for float32 PyTorch
   eager, and 24.4/24.6/25.1 for PyTorch CUDA Graph replay. Every compared
   top token and top logit passed the fixed tolerance. Nsight Compute at
   length 64 measured 2,492,031,104 DRAM read bytes, 4,821,632 DRAM write
   bytes, 2,689,235,136 L2 bytes, and 5.96% SM throughput for one worker
   launch. The package contains 2,471,628,800 BF16 weight bytes. These
   measurements expose a large worker throughput gap; no performance
   variants in this milestone have been implemented. Commands and report
   paths are in the root README.

Each milestone has deterministic unit tests for the planner/loader contract
and GPU regression tests against PyTorch for the paths it adds. Keep the existing formatting/static-check gates.

## Path from synthetic weights to a real checkpoint

The full-model milestone supplies the operations required by the Llama 3.2
1B base model. Its official Hugging Face repository is gated, so checkpoint
access is required for the importer and end-to-end regression. Match the
pinned config exactly: layer/head/FFN/vocabulary sizes; epsilon; RoPE base,
pairing and scaling; bias flags; and tied output embedding. The importer maps
checkpoint tensor names and layouts into a checked, versioned BF16
`weights.bin` package and rejects unsupported variants. Tokenization can run
in Python initially; the executor receives token IDs and returns logits.
Generation then selects a token (greedy selection is enough for the first
end-to-end test) and feeds it back with the same context. Chat-template
formatting is needed only if an instruct checkpoint is chosen later.

A dedicated prefill kernel is **not required for functional support**. Feed
prompt token IDs to the one-token API in order from an empty context; each
call appends its K/V state, and the last call yields next-token logits. This
is the first prefill path. It performs one launch per prompt token and
repeatedly scans growing prefixes, so prompt latency will be poor for long
prompts. Chunked/matrix-matrix prefill is a later performance milestone.
The fixed `max_context` still limits prompt plus generated tokens.

Llama 3.2 1B has about 1.23 billion parameters: approximately 2.3 GiB of
BF16 weights, compared with 4.6 GiB in float32. With 16 layers, 8 KV heads,
and head width 64, a float32 K/V cache uses 64 KiB per token: about 512 MiB
at 8K tokens and 8 GiB at 128K. The RTX 3050 has 8 GiB total, so the first
package should cap the cache at 4K or 8K and account for scratch, program,
CUDA runtime, and loading overhead. Keep the PyTorch reference and TileInfer
weights off the GPU at the same time during checkpoint comparison. A later
128K experiment requires a different cache/storage strategy and much faster
prefill. BF16 weight storage reduces traffic and residency; its latency
effect must still be measured on the worker kernel.

The [Meta Llama 3 reference implementation](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py)
illustrates the block ingredients. The
[Hugging Face Llama configuration](https://huggingface.co/docs/transformers/model_doc/llama)
lists checkpoint-level variants such as GQA and RoPE scaling; its
[KV-cache guide](https://huggingface.co/docs/transformers/kv_cache)
describes incremental generation. The
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
specifies its causal-mask behavior for non-square attention shapes.
