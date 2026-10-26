#!/usr/bin/env bash
# Phase 2 sweep on the RTX 3050: TileInfer, PyTorch eager/graph and llama.cpp
# in one session. Two passes in opposite order show session drift.
# Usage: run_phase2.sh OUT_DIR PACKAGE_DIR [CONFIG...]
# CONFIG names a subset of the llama.cpp configurations below (default: all).
# Needs: llama.cpp CUDA build, rt_probe CUDA build, the GGUF, a 4K BF16 package.
set -euo pipefail
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$(realpath -m "$1")"
package="$(realpath "$2")"
shift 2
selected=("$@")
gguf="$here/models/Llama-3.2-1B-bf16.gguf"
positions=(0 255 1023 2047 4095)
new_tokens=64
mkdir -p "$out"

# "1, 2, 3," has no BF16 near-tie in 64 tokens, so TileInfer and PyTorch generate
# the same ids. Prompts such as "The capital of France is" hit a tie and the
# PyTorch reference then depends on the static-cache length.
echo '{"tokens": [128000, 16, 11, 220, 17, 11, 220, 18, 11]}' > "$out/prompt.json"
{
  date -u +%FT%TZ
  git -C "$repo" rev-parse HEAD
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  # Current clocks show whether they were locked (idle clocks are low when not).
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,persistence_mode --format=csv
  nvidia-smi --query-compute-apps=pid,name --format=csv
  sha256sum "$gguf"
} > "$out/env.txt"

# name:fa:kv:graphs
configs=(
  "fa1-bf16:on:bf16:on"
  "fa1-f16:on:f16:on"
  "fa0-bf16:off:bf16:on"
  "fa0-f16:off:f16:on"
  "fa1-bf16-nograph:on:bf16:off"
  "fa1-f16-nograph:on:f16:off"
)
if ((${#selected[@]})); then
  chosen=()
  for c in "${configs[@]}"; do
    for name in "${selected[@]}"; do [[ ${c%%:*} == "$name" ]] && chosen+=("$c"); done
  done
  ((${#chosen[@]} == ${#selected[@]})) || { echo "unknown configuration in: ${selected[*]}" >&2; exit 1; }
  configs=("${chosen[@]}")
fi

tileinfer_pass() {
  local pass=$1
  (cd "$repo/tileinfer" && uv run --locked --project ../tileplan --group checkpoint \
    python tests/benchmark_decode.py --checkpoint ../checkpoints/Llama-3.2-1B \
    --package "$package" --executor build/release/tile_runtime_test \
    --positions "${positions[@]}" --warmup 5 --iterations 20 --rounds 5 \
    --out "$out/tileinfer-decode-p$pass.json")
  (cd "$repo/tileinfer" && uv run --locked --project ../tileplan --group checkpoint \
    python tests/benchmark_generate.py --checkpoint ../checkpoints/Llama-3.2-1B \
    --package "$package" --executor build/release/tile_runtime \
    --inputs "$out/prompt.json" --new-tokens "$new_tokens" --warmup 3 --rounds 7 \
    --out "$out/tileinfer-generate-p$pass.json")
}

llamacpp_config() {
  local pass=$1 name fa kv graphs flags=()
  IFS=: read -r name fa kv graphs <<< "$2"
  [[ $graphs == off ]] && flags+=(--no-graphs)
  (cd "$here" && uv run --no-sync rtbench bench --backend cuda --gguf "$gguf" \
    --fa "$fa" --kv "$kv" "${flags[@]}" --positions "${positions[@]}" \
    --warmup 5 --iterations 20 --rounds 5 --inputs "$out/prompt.json" \
    --new-tokens "$new_tokens" --llama-bench --out "$out/llamacpp-$name-p$pass.jsonl")
}

for pass in 1 2; do
  echo "=== pass $pass $(date -u +%T)"
  if [[ $pass == 1 ]]; then
    tileinfer_pass 1
    for c in "${configs[@]}"; do echo "--- $c"; llamacpp_config 1 "$c"; done
  else
    for ((i = ${#configs[@]} - 1; i >= 0; i--)); do echo "--- ${configs[i]}"; llamacpp_config 2 "${configs[i]}"; done
    tileinfer_pass 2
  fi
done
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv >> "$out/env.txt"
echo "=== done $(date -u +%T)"
