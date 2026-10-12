# Llama-style one-token decoder design

Status: design and milestones agreed, 2026-10-12. The implemented software
still executes only the float32 MLP described in the [core design](tile_graph_inference.md).
This document specifies the next workload for [TilePlan](tile_graph_python_architecture.md)
and [TileInfer](tile_graph_cuda_runtime.md); it is not a claim that decoder
support or Hugging Face checkpoint loading already works.

## Goal and first supported model

Build a batch-one, **one-token decode** path, first for one deterministic
Llama-style block, then for a stack ending in vocabulary logits. Each call
accepts one token's hidden vector (or, for the full model, one token ID) and
one mutable sequence context. It consumes positions `0..p` and advances the
context from length `p` to `p + 1`. The plan and weights are fixed; the input,
position, cache contents, and output vary across calls. Only one invocation
may use a context at a time. A fixed maximum context length is part of the
plan; reaching it returns an error instead of silently overwriting cache.

The first implementation uses float32 weights, activations, KV cache, and
accumulation, batch size one, standard full causal attention, no dropout, and
the existing cooperative wave barrier. Start with small synthetic dimensions
that fit the RTX 3050 and an independent PyTorch reference. An actual
checkpoint defines exact parameter values and some semantic variants; the
first checkpoint importer will support one explicitly named architecture and
revision rather than every model with a `Llama` label.

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
    kv_head = h // (n_q / n_kv)
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
new plan format/executor ABI for decoder work while preserving acceptance of
existing MLP packages. Plan tasks should reference typed tensor/storage IDs,
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

## Milestones and acceptance checks

1. **Reference and semantics.** Implement a small independent PyTorch block
   with fixed weights and a token sequence. Freeze shapes, RoPE variant,
   epsilon, head mapping, cache layout, and tolerances in test fixtures.
   Verify its sequential cache against recomputation on the same prefix.
2. **Plan and state ABI.** Add a versioned decoder model specification,
   typed task operands, general wave scheduling, decoder package validation,
   and `SequenceContext`. Existing MLP packages and tests continue to run.
   Inspect output shows waves, per-worker load, scratch, and KV bytes.
3. **One CUDA block.** Execute one block for one token at `p=0`, then a
   sequence of tokens through the same context. Compare Q/K/V after RoPE,
   cache entries, attention output, both residuals, FFN output, and final
   hidden vector with PyTorch. Cover GQA, tail tiles, reset, and capacity
   boundary. Run CTest and Compute Sanitizer memcheck/racecheck/synccheck.
4. **Full decoder stack.** Add embedding lookup, repeated blocks with one
   cache per layer, final norm, and logits. Compare token-by-token logits
   and cache with a PyTorch reference. Preserve a one-token API so the same
   plan can process a sequence by repeated calls.
5. **Checkpoint bridge.** Choose one small, permitted Hugging Face checkpoint
   and pin its revision. Read `config.json` and safetensors shards, check the
   model variant, convert named tensors to the plan's weight layout, and
   record hashes. Compare logits and cache against `AutoModelForCausalLM`
   under a recorded precision policy for several short prompts.
6. **Performance experiments.** Measure decode against PyTorch eager and
   CUDA Graph replay at fixed context lengths. Profile the decoder worker
   and memory traffic. Add chunked prefill, reduced precision, cache
   paging/sliding-window variants, or per-tile readiness only when a
   measured workload justifies each change. A companion micrograph can
   isolate partial tile handoff if the decoder schedule does not.

Each milestone has deterministic unit tests for the planner/loader contract
and GPU regression tests against PyTorch for the paths it adds. Keep the MLP
regression and the existing formatting/static-check gates.

## Path from synthetic weights to a real checkpoint

The full-model milestone supplies the operations required by a basic
Llama-style causal model. To run actual weights, select an exact checkpoint
and match its config: layer/head/FFN/vocabulary sizes; epsilon; RoPE base,
pairing and scaling; bias flags; and tied or separate output embeddings.
The importer must map checkpoint tensor names and layouts into a checked,
versioned `weights.bin` package, including sharded safetensors. Unsupported
variants should fail with a clear error. Tokenization and chat-template
formatting can run in Python initially; the executor receives token IDs and
returns logits. Generation then selects a token (greedy selection is enough
for the first end-to-end test) and feeds it back with the same context.

A dedicated prefill kernel is **not required for functional support**. Feed
prompt token IDs to the one-token API in order from an empty context; each
call appends its K/V state, and the last call yields next-token logits. This
is the first prefill path. It performs one launch per prompt token and
repeatedly scans growing prefixes, so prompt latency will be poor for long
prompts. Chunked/matrix-matrix prefill is a later performance milestone.
The fixed `max_context` still limits prompt plus generated tokens.

Memory constrains checkpoint choice on this RTX 3050. Float32 weights alone
need roughly four bytes per parameter (a 7-billion-parameter model is about
28 GB), plus KV cache, scratch, and runtime overhead. Calculate the actual
package and cache budget before choosing the first checkpoint; use a small
model or add supported FP16/BF16/quantized kernels and packing. Model-family
names alone do not guarantee compatible math, layout, or memory footprint.

The [Meta Llama 3 reference implementation](https://github.com/meta-llama/llama-models/blob/main/models/llama3/model.py)
illustrates the block ingredients. The
[Hugging Face Llama configuration](https://huggingface.co/docs/transformers/model_doc/llama)
lists checkpoint-level variants such as GQA and RoPE scaling; its
[KV-cache guide](https://huggingface.co/docs/transformers/kv_cache)
describes incremental generation. The
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
specifies its causal-mask behavior for non-square attention shapes.
