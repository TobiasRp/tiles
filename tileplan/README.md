# TilePlan

TilePlan is the offline Python half of Tiles. It converts a checkpoint or
synthetic fixture into `plan.json` and `weights.bin`: a tiled task DAG, wave
schedule, tensor layout, and packed float32 or BF16 weights. Activations and
K/V storage remain float32. Python validates graph dependencies and planning
metadata; [TileInfer](../tileinfer/README.md) checks the executable layout and schedule.

For text generation, use [`generate.py`](../generate.py). It handles the full
build/probe/import/tokenize/run sequence.

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
  --max-context 4096 --tile-rows 64 --out runs/demo/package
uv run --locked --project tileplan python -m tileplan verify runs/demo/package
```

Probe the actual compiled worker after any kernel change. Packages record its
fingerprint and launch parameters; an old version-3 package must be regenerated
when they change. Version-2 float32 compatibility remains available.

`--tile-rows auto` uses an offline load-balancing heuristic over 2/4/8/16 rows;
it does not benchmark the GPU. Explicit 64-row tiles gave the best measured
Llama checkpoint result with the current 1024-thread worker.

Other commands are available through `python -m tileplan --help`:

| Command | Purpose |
| --- | --- |
| `build` / `build-stack` | Package a synthetic block or decoder stack. |
| `reference` / `stack-reference` | Generate deterministic CPU reference fixtures. |
| `inspect` / `verify` | Summarize or validate a package. |
| `visualize` | Draw the schedule, tensor graph, and selected tile dependencies. |

```sh
uv run --locked --project tileplan python -m tileplan visualize runs/demo/package \
  --out runs/demo/schedule.svg --tensor-graph-out runs/demo/tensors.svg \
  --tile-flow-out runs/demo/tiles.svg
```

## Checks

```sh
uv run --locked --project tileplan python -m unittest discover -s tileplan/tests
```

The [block](tests/fixtures/decoder_m1/README.md) and
[stack](tests/fixtures/decoder_m4/README.md) fixtures cover incremental decoding
against full-prefix recomputation. Package tests cover storage, dependencies,
quantization, and reproducibility. See the [runtime checks](../tileinfer/README.md#checks)
for GPU and real-checkpoint validation, and [pyproject.toml](pyproject.toml) for
developer dependencies and Ruff configuration.
