# S6 showcase

One command from the pinned Llama 3.2 1B checkpoint to a report that
compares the final TileInfer with llama.cpp, PyTorch eager and PyTorch CUDA
Graph replay on the RTX 3050. The milestone is planned in
[S6 in the development log](../../docs/development.md#s6-showcase).

## Run

On `tux`, from the repository root, with the checkpoint in
`checkpoints/Llama-3.2-1B` and the `experiments/runtimes` environment synced
(see [its README](../runtimes/README.md)):

```sh
# Optional: lock the clocks (one setting per nvidia-smi call).
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1950,1950
sudo nvidia-smi -lmc 6801,6801
python3 experiments/showcase/showcase.py sanitize --out runs/s6-DATE
python3 experiments/showcase/showcase.py run --out runs/s6-DATE
sudo nvidia-smi -rgc   # unlock afterwards
sudo nvidia-smi -rmc
```

`run` takes about an hour on a fresh directory. Each step stores the hashes
of its inputs in `OUT/steps/` and is skipped when they are unchanged, so a
rerun after a failure continues where it stopped, and a changed worker,
package or protocol reruns what depends on it. `--force STEP...` reruns steps
anyway; `--report-only` redraws the figures and rewrites the report.

`sanitize` runs all Compute Sanitizer checks (several minutes) and records
the commit and worker hash; the report says whether that pass covers the
measured worker. It is kept out of `run` because it is slow.

## Steps

| Step | What it does | Output |
| --- | --- | --- |
| build | incremental CMake builds of TileInfer and `rt_probe` | |
| probe | `tileplan probe` on the compiled worker | `target.json` |
| import | `tileplan import-llama32`, 4K context | `package-4k/` |
| verify, checkpoint | package validation; calibrated tolerances on the checkpoint | `checkpoint/` |
| ctest | the CTest suite | `logs/ctest.log` |
| gguf, parity | GGUF conversion if missing; teacher-forced llama.cpp parity | `parity-*.json` |
| stream | `bandwidth_probe stream` and `rowdot` | `stream.json`, `rowdot.json` |
| decode-* | four controlled sessions (below) | `decode-*/report.json` |
| generate | 64-token greedy generation, two passes in opposite order | `generate/` |
| timeline | `profile_timeline.py` at contexts 1 and 4,096 | `timeline/` |
| kernels | one token of every runtime under Nsight Systems | `kernels/` |
| figures, report | `docs/blog/make_s6_figures.py`, `report.md` | `figures/`, `report.md` |

The decode sessions use `tileinfer/tests/controlled_decode.py` (the S6a
protocol: resident processes, interleaved blocks, telemetry, paired bootstrap
intervals) at contexts 1, 256, 1K, 2K and 4K. Three models do not fit the
8 GB GPU together, so TileInfer is paired with PyTorch (eager, graph replay,
graph replay with SDPA attention) in one session and with llama.cpp (flash
attention, f16 and BF16 K/V in one `rt_probe serve` process) in another.
Each runs in both spawn orders; the report averages them.

## Reading the numbers

- TileInfer and PyTorch rounds are CUDA-event batches of 20 back-to-back
  calls. llama.cpp's API returns when the logits are ready, so its rounds are
  host wall time around each synchronous `llama_decode`, as in the runtime
  comparison.
- PyTorch with SDPA is a separate numerical policy: the reference policy of
  this repository is Hugging Face eager attention, which the TileInfer
  package reproduces.
- Kernel traces run under Nsight Systems and are for the figures only.
