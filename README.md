# tile-infer

Tile inference prototype targeting the
[Llama 3.2 1B **base** model](https://huggingface.co/meta-llama/Llama-3.2-1B)
on an 8 GB RTX 3050. [TilePlan](tileplan/README.md) turns the pinned Hugging
Face checkpoint into a validated, tiled execution package. [TileInfer](tileinfer/README.md)
executes one token at a time with a persistent CUDA worker and a per-layer KV
cache.

The checkpoint stays in BF16 storage; loads convert weights to float32 for
arithmetic. Activations, accumulation, logits, and the KV cache are float32.
The output projection aliases the token embedding. Tokenization runs in Python;
the CUDA executor accepts token IDs. The base model has no chat template in
this workflow.

## Run Llama 3.2 1B

You need access to the gated model on Hugging Face, then download the pinned
revision. The model files are kept under `checkpoints/`, which Git ignores.

```sh
~/.local/bin/uv tool install huggingface_hub
~/.local/bin/hf auth login
~/.local/bin/hf download meta-llama/Llama-3.2-1B \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --local-dir ~/dev/tile-infer/checkpoints/Llama-3.2-1B
```

Build the CUDA runtime, probe that exact worker, and import a package with a
4K-token context. Use a new `RUN` directory each time; TilePlan refuses to
overwrite an existing package. This package uses about 2.47 GB for weights
and 256 MiB for the float32 KV cache.

```sh
cd ~/dev/tile-infer/tileinfer
cmake --preset release
cmake --build --preset release

cd ../tileplan
~/.local/bin/uv sync --locked
RUN=../runs/llama32-$(date +%Y%m%d-%H%M%S)
MODEL=../checkpoints/Llama-3.2-1B
mkdir -p "$RUN"
~/.local/bin/uv run --locked python -m tileplan probe \
  --executor ../tileinfer/build/release/tile_runtime \
  --out "$RUN/target.json"
~/.local/bin/uv run --locked python -m tileplan import-llama32 \
  --checkpoint "$MODEL" --target "$RUN/target.json" \
  --max-context 4096 --tile-rows auto --out "$RUN/package"
~/.local/bin/uv run --locked python -m tileplan verify "$RUN/package"
```

To generate tokens, tokenize a prompt, then let the executor select and feed
back each next ID on the GPU:

```sh
PROMPT='The capital of France is'
~/.local/bin/uv run --locked --group checkpoint python - "$MODEL" "$PROMPT" "$RUN/inputs.json" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
tokens = tokenizer(sys.argv[2], add_special_tokens=True)["input_ids"]
Path(sys.argv[3]).write_text(json.dumps({"tokens": tokens}))
PY

../tileinfer/build/release/tile_runtime \
  generate "$RUN/package" "$RUN/inputs.json" --max-new-tokens 16 \
  > "$RUN/generation.json"

~/.local/bin/uv run --locked --group checkpoint python - "$MODEL" "$RUN/generation.json" <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
generated = json.loads(Path(sys.argv[2]).read_text())["generated"]
print(generated, repr(tokenizer.decode(generated)))
PY
```

The CLI starts a fresh context for each invocation. `generate` uses the
one-token worker for prefill, then performs greedy selection and feedback on
the GPU. `check` returns logits and each layer's current K/V values for every
input token when diagnostics are needed. Tokenization remains in Python;
fast chunked prefill is future work.

## Check correctness

The real-checkpoint regression compares every block output, per-layer K/V
slice, and full-vocabulary logits with Hugging Face Transformers using the
same BF16 checkpoint values converted to float32. It runs three short prompts
and checks their greedy next-token choices. The reference process exits before
TileInfer loads the weights on the 8 GB GPU.

```sh
~/.local/bin/uv run --locked --group checkpoint python \
  ../tileinfer/tests/llama32_checkpoint.py \
  --checkpoint "$MODEL" --package "$RUN/package" \
  --executor ../tileinfer/build/release/tile_runtime \
  --out "$RUN/comparison"

~/.local/bin/uv run --locked ruff check src tests ../tileinfer/tests
~/.local/bin/uv run --locked ruff format --check src tests ../tileinfer/tests
~/.local/bin/uv run --locked ty check src tests ../tileinfer/tests
~/.local/bin/uv run --locked python -m unittest discover -s tests -v
cd ../tileinfer
ctest --preset release --output-on-failure
```

The validated run on `tux` compared 27 tokens across three prompts: maximum
absolute errors were 0.000504 for block outputs and 0.0000801 for logits, and
all 27 next-token choices matched. The full report is in
`runs/llama32-import-2026-10-12/comparison-final/report.json` on `tux`.

## Measure autoregressive generation

`bench` times GPU greedy selection plus decode for a fixed number of new
tokens. It excludes model setup, prompt prefill, and result readback. Each
round resets and prefills the same prompt; EOS is disabled to keep work
constant.

```sh
cd ~/dev/tile-infer/tileplan
../tileinfer/build/release/tile_runtime \
  bench "$RUN/package" "$RUN/inputs.json" \
  --new-tokens 16 --warmup 1 --rounds 3 > "$RUN/generation-benchmark.json"
```

To compare the same autoregressive sequence with float32 Transformers eager,
run the reference in a separate process so its model leaves the GPU before
TileInfer loads its BF16 package:

```sh
~/.local/bin/uv run --locked --group checkpoint python \
  ../tileinfer/tests/benchmark_generate.py \
  --checkpoint "$MODEL" --package "$RUN/package" \
  --executor ../tileinfer/build/release/tile_runtime \
  --inputs "$RUN/inputs.json" --new-tokens 16 \
  --warmup 1 --rounds 3 --out "$RUN/generation-comparison.json"
```

## Profile an isolated decoder step

The historical fixed-position benchmark keeps weights, input token, and KV
cache on the GPU. It times
one-token decode at fixed positions after filling each prefix with token ID 1.
Every timed call overwrites the same new cache slot, so context length stays
fixed. This is useful for profiling the worker but does not measure
autoregressive generation. The PyTorch eager and CUDA Graph paths use the same float32
Transformers model, fixed `StaticCache`, and attention mask. TileInfer uses
the imported BF16 weight storage with float32 arithmetic. Setup, prefix fill,
and output readback are outside CUDA-event timings. The script checks the
top token and top logit against PyTorch at every length.

```sh
cd ~/dev/tile-infer/tileplan
~/.local/bin/uv run --locked --group checkpoint python \
  ../tileinfer/tests/benchmark_decode.py \
  --checkpoint "$MODEL" \
  --package "$RUN/package" \
  --executor ../tileinfer/build/release/tile_runtime_test \
  --positions 0 63 255 --warmup 3 --iterations 5 --rounds 3 \
  --out "$RUN/decode-benchmark.json"
```

Positions `0 63 255` measure context lengths 1, 64, and 256. To profile one
worker launch at a chosen position, use Nsight Compute; the script skips the
prefix launches and writes exact DRAM read/write and L2 byte counters. This
requires access to GPU performance counters. Profiler replay perturbs latency,
so use the benchmark report for timing.

```sh
~/.local/bin/uv run --locked python ../tileinfer/tests/profile_decode.py \
  --package "$RUN/package" \
  --executor ../tileinfer/build/release/tile_runtime_test \
  --position 63 \
  --out "$RUN/decode-profile-p63.json"
```

For a small deterministic example that needs no checkpoint, see the
[TilePlan synthetic stack instructions](tileplan/README.md). The
[decoder design](docs/tile_graph_decoder.md) describes the plan and state ABI,
current scope, and performance experiments still to do.
