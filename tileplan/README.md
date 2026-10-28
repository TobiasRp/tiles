# TilePlan

TilePlan is the offline Python half of Tiles. It converts a checkpoint or
synthetic fixture into `plan.json` and `weights.bin`: a tiled task DAG for a
whole token (embedding, every layer, final norm, and logits), its wave
schedule over logical workers, tensor layout, and packed float32 or BF16
weights. Each layer has six task waves: the [Q; K; V] projection (with its
RMSNorm fused in; Q and K stay unrotated, V goes to the cache); attention
scores and attention values per K/V head and context chunk, where the scores
task applies RoPE and chunk 0 writes the rotated K cache row; the output
projection; one FFN task per product chunk, which computes its gate/up rows
(with SiLU·up) and that chunk's FP32 down partial; and the FFN reduction,
which sums the partials in chunk order into rows of the layer output. Every
projection splits its rows into one range per worker, and `Wdown` is stored
transposed so a chunk's rows are contiguous. Each logits task also owns
one slot of the greedy-selection workspace. `visualize` draws one layer of
that program. Version-10 packages
use the same storage dtype for weights, activations, and K/V, with float32
accumulation and selected float32 internal math. Python validates graph
dependencies and planning metadata; [TileInfer](../tileinfer/README.md) checks
the executable layout and schedule.

For text generation, use [`generate.py`](../generate.py). It handles the full
build/probe/import/tokenize/run sequence and creates BF16 packages by
default. Reusing a package inherits its precision unless an explicit matching
option is supplied.

## Checkpoint setup

From the repository root, download the pinned Llama 3.2 1B **base** checkpoint.
Access is gated; authenticate with your own Hugging Face account:

```sh
uv tool install huggingface_hub
hf auth login
hf download meta-llama/Llama-3.2-1B \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --local-dir checkpoints/Llama-3.2-1B
```

Model files and generated runs are Git-ignored. `generate.py` installs its
locked Python dependencies automatically. For planner development, use
`uv sync --locked --project tileplan --group checkpoint`.

## Work with packages

To control package creation directly, build TileInfer first and run from the
repository root. Choose a fresh output directory:

```sh
uv run --locked --project tileplan python -m tileplan probe \
  --executor tileinfer/build/release/tile_runtime --out runs/demo/target.json
uv run --locked --project tileplan python -m tileplan import-llama32 \
  --checkpoint checkpoints/Llama-3.2-1B --target runs/demo/target.json \
  --max-context 4096 --out runs/demo/package
uv run --locked --project tileplan python -m tileplan verify runs/demo/package
```

Probe the actual compiled worker after any kernel change. Packages record its
fingerprint and launch parameters. Older package formats and unknown worker
fingerprints are rejected; regenerate those packages.

`--dtype bfloat16` (the checkpoint default) selects BF16 weights, activations,
and K/V together; `--dtype float32` selects the pure-float32 path. Synthetic
builders default to float32 and also accept `--weight-dtype` as an alias.
Matrix arithmetic uses FP32 SIMT multiplication and accumulation over BF16
operands; [the measured comparison](../docs/development.md#bf16-compute) explains
why the native Tensor Core variant was removed.

There is no tile-size parameter. Projections split their rows evenly over the
target's workers. Attention chunk tasks follow the position: a visible prefix
of at most `--attention-fused-steps` (default 384) runs as one fused task per
K/V head and skips the barrier before the values wave; a longer prefix splits
into the fewest chunks of at most `--attention-chunk-steps`, but at least two.
The chunk steps default to the most the worker's shared attention region
holds (`attention_shared_bytes` in `target.json`; 2,048 steps for Llama 3.2
1B in BF16), so a 4K context needs two chunks, 16 tasks on 20 SMs.

Other commands are available through `python -m tileplan --help`:

| Command | Purpose |
| --- | --- |
| `build` / `build-stack` | Package a synthetic block or decoder stack. |
| `reference` / `stack-reference` | Generate deterministic CPU reference fixtures. |
| `inspect` / `verify` | Summarize or validate a package. |
| `visualize` | Draw one layer's wave schedule. |

```sh
uv run --locked --project tileplan python -m tileplan visualize runs/demo/package \
  --out runs/demo/schedule.svg
```

## Checks

```sh
uv run --locked --project tileplan python -m unittest discover -s tileplan/tests
```

The [block](tests/fixtures/decoder_m1/README.md) and
[stack](tests/fixtures/decoder_m4/README.md) fixtures cover incremental decoding
against full-prefix recomputation. Package tests cover storage, dependencies,
BF16 rounding and packing, and reproducibility. See the [runtime checks](../tileinfer/README.md#checks)
for GPU and real-checkpoint validation, and [pyproject.toml](pyproject.toml) for
developer dependencies and Ruff configuration.
