# TilePlan decoder planner

TilePlan builds version-3 decoder packages with float32 or BF16 weights from
deterministic PyTorch fixtures and a probe of the compiled CUDA worker. It
supports one block and a full decoder stack. A stack package stores one block
task DAG and nine-wave schedule, then repeats that schedule across layers. It
reuses block scratch, allocates K/V storage per layer, and records whether the
output projection shares the token embedding. Python validates the package before
writing it; TileInfer validates it again before encoding device records.
Weight storage is selected at build time; all activations and the KV cache
remain float32. Existing version-2 float32 packages remain readable. The
`import-llama32` command reads the pinned Hugging Face Llama 3.2 1B base
checkpoint directly from safetensors, validates its config and tensor set, and
writes BF16 bytes one tensor at a time. The tied output projection aliases the
token embedding. Its plan records the checkpoint revision and source-file hashes.

## Install and check

```sh
cd ~/dev/tile-infer/tileplan
~/.local/bin/uv sync --locked
~/.local/bin/uv run --locked ruff check src tests
~/.local/bin/uv run --locked ruff format --check src tests
~/.local/bin/uv run --locked ty check src tests
~/.local/bin/uv run --locked python -m unittest discover -s tests -v
```

## Build and run on tux

Build the [TileInfer runtime](../tileinfer/README.md) first. Then:

```sh
cd ~/dev/tile-infer/tileplan
RUN=../runs/decoder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
~/.local/bin/uv run --locked python -m tileplan probe \
  --executor ../tileinfer/build/release/tile_runtime \
  --out "$RUN/target.json"
~/.local/bin/uv run --locked python -m tileplan build \
  --model tests/fixtures/decoder_m1/model.json \
  --weights tests/fixtures/decoder_m1/weights.npz \
  --target "$RUN/target.json" \
  --tile-rows auto --weight-dtype bfloat16 --out "$RUN/block"
~/.local/bin/uv run --locked python -m tileplan build-stack \
  --model tests/fixtures/decoder_m4/model.json \
  --weights tests/fixtures/decoder_m4/weights.npz \
  --target "$RUN/target.json" \
  --tile-rows auto --weight-dtype bfloat16 --out "$RUN/stack"
printf '%s\n' '{"tokens":[0,7,15,3,47]}' > "$RUN/inputs.json"
../tileinfer/build/release/tile_runtime \
  generate "$RUN/stack" "$RUN/inputs.json" --max-new-tokens 1 > "$RUN/generation.json"
~/.local/bin/uv run --locked python -m tileplan inspect "$RUN/stack"
~/.local/bin/uv run --locked python -m tileplan verify "$RUN/stack"
```

`generation.json` contains the selected next token ID for the synthetic
three-layer stack. These IDs have no text tokenizer yet. Use
`tile_runtime check "$RUN/stack" "$RUN/inputs.json"` for full per-token
logits and K/V diagnostics.

`reference` and `stack-reference` generate new deterministic fixtures.
Both `build` and `build-stack` accept `--weight-dtype float32|bfloat16`,
defaulting to float32. For BF16, the planner rounds the float32 fixture
weights to BF16 once and packs their raw 16-bit values. Tensor descriptors
record each weight's dtype; byte offsets, hashes, and tied aliases are checked
by both the planner and runtime.

## Visualize the schedule and tensor operations

After building a block or stack package, run `visualize` from the `tileplan`
directory. For the stack package created above, keep `RUN` set to that run
directory and write three self-contained SVGs:

```sh
~/.local/bin/uv run --locked python -m tileplan visualize "$RUN/stack" \
  --out "$RUN/stack-schedule.svg" \
  --tensor-graph-out "$RUN/stack-tensor-graph.svg" \
  --tile-flow-out "$RUN/stack-tile-flow.svg"
```

- `stack-schedule.svg` shows each tile task's opcode, output region, and logical
  worker, grouped by wave.
- `stack-tensor-graph.svg` shows the model's tensor operations, tensor and weight
  shapes, data flow, residual paths, embedding lookup, and final logits.
- `stack-tile-flow.svg` follows a small set of actual plan tasks through all
  nine decoder waves. It shows output slices, computations, workers, and direct
  dependencies. Gray cards count prerequisite tasks omitted from the example;
  wave boundaries mark full cooperative-grid barriers. The diagram shows one
  block template, repeated for each layer in a stack. It illustrates where
  tile dependencies are narrower than whole operators, but the current
  barrier-based runtime cannot hand a tile to the next wave early.

Replace `$RUN/stack` with `$RUN/block` to draw a single-block package.

The automatic tile-height search evaluates 2, 4, 8, and 16 rows using
per-wave maximum estimated worker load plus a per-task dispatch penalty.
The selected height and all candidate scores are recorded in
`planning_report.json`. This is an offline schedule heuristic, not a
latency prediction. A target probe must be refreshed after the worker changes.

## Import and check Llama 3.2 1B

Access to the [official base checkpoint](https://huggingface.co/meta-llama/Llama-3.2-1B)
is gated. Sign in on `tux` with your own Hugging Face account and download the
pinned revision. The token stays on `tux`.

```sh
~/.local/bin/uv tool install huggingface_hub
~/.local/bin/hf auth login
~/.local/bin/hf download meta-llama/Llama-3.2-1B \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --local-dir ~/dev/tile-infer/checkpoints/Llama-3.2-1B

cd ~/dev/tile-infer/tileplan
RUN=../runs/llama32-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
~/.local/bin/uv run --locked python -m tileplan probe \
  --executor ../tileinfer/build/release/tile_runtime --out "$RUN/target.json"
~/.local/bin/uv run --locked python -m tileplan import-llama32 \
  --checkpoint ../checkpoints/Llama-3.2-1B \
  --target "$RUN/target.json" --max-context 64 \
  --tile-rows auto --out "$RUN/package"
~/.local/bin/uv run --locked --group checkpoint python \
  ../tileinfer/tests/llama32_checkpoint.py \
  --checkpoint ../checkpoints/Llama-3.2-1B \
  --package "$RUN/package" \
  --executor ../tileinfer/build/release/tile_runtime \
  --out "$RUN/comparison"
```

The comparison converts the stored BF16 weights to float32 for the Transformers
reference, then releases that model before launching TileInfer on the 8 GB GPU.
It checks every block output, each layer's current K/V slice, and the full
vocabulary logits at every token of three short prompts. It also feeds back
two greedy tokens per prompt and checks the next-token choices. The report and
raw outputs are saved under `comparison/`. For a larger usable context, import
with `--max-context 4096`; the 64-token package above keeps validation fast.
