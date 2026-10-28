# Runtime comparison

Puts llama.cpp next to TileInfer and PyTorch for Llama 3.2 1B in BF16 at
batch 1. The fairness rules are in
[the architecture document](../../docs/architecture.md#measurement) and the
results in [the development log](../../docs/development.md#final-results).

This directory has its own Python project and lockfile, and builds llama.cpp
out of tree at a pinned release (`scripts/llamacpp.pin`). It does not touch
`tileplan/` or `tileinfer/`.

## Setup

```sh
cd experiments/runtimes
scripts/fetch_llamacpp.sh                 # pinned llama.cpp into third_party/
scripts/build_llamacpp.sh cpu             # or metal (Mac) / cuda (RTX 3050, sm_86)
scripts/build_probe.sh cpu
UV_PYTHON=3.12 uv sync --group dev
```

The checkpoint is the pinned base model from
[tileplan/README.md](../../tileplan/README.md#checkpoint-setup), expected in
`checkpoints/Llama-3.2-1B` at the repository root.

## Run

```sh
uv run rtbench convert --checkpoint ../../checkpoints/Llama-3.2-1B
uv run rtbench parity --backend cpu --gguf models/Llama-3.2-1B-bf16.gguf \
  --checkpoint ../../checkpoints/Llama-3.2-1B
echo '{"tokens": [128000, 9906, 11, 856, 836, 374]}' > results/prompt.json
uv run rtbench bench --backend cpu --gguf models/Llama-3.2-1B-bf16.gguf \
  --inputs results/prompt.json --positions 0 63 --llama-bench
uv run rtbench report results/llamacpp.jsonl \
  --tileinfer-decode path/to/decode.json --tileinfer-generate path/to/generate.json
```

On a CUDA machine, `scripts/run_phase2.sh OUT_DIR PACKAGE_DIR` runs the whole
sweep (TileInfer, PyTorch eager and graph, six llama.cpp configurations, two
passes in opposite order). `rtbench report` accepts several records files and
TileInfer reports and averages cells that repeat, so the passes can be merged.

`rt_probe serve` keeps the model resident and answers fixed-position timing
requests on standard input, one context per K/V type; the S6 showcase uses it
to interleave llama.cpp with TileInfer under the controlled protocol
([experiments/showcase](../showcase/README.md)).

Every row records whether it includes sampling (`none`: logits ready;
`greedy`: next token id known). The report keeps the two apart.

CPU and Metal numbers only prove that the harness works; they are never
compared with CUDA numbers.

## Checks

```sh
uv run ruff check . && uv run ruff format --check .
uv run ty check src tests
uv run pytest tests
```
