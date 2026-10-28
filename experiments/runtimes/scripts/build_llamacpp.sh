#!/usr/bin/env bash
# Build llama.cpp tools. Usage: build_llamacpp.sh [cpu|metal|cuda]
# cpu   - portable CPU build (harness validation)
# metal - Apple GPU (Mac harness checks only; never compared with CUDA numbers)
# cuda  - RTX 3050 (compute capability 8.6)
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
backend="${1:-cpu}"
src="$here/third_party/llama.cpp"
build="$here/third_party/build-$backend"
[[ -d "$src/.git" ]] || { echo "run fetch_llamacpp.sh first" >&2; exit 1; }

flags=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
       -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON)
case "$backend" in
  cpu)   flags+=(-DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON) ;;
  metal) flags+=(-DGGML_METAL=ON) ;;
  cuda)  flags+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86) ;;
  *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac

cmake -S "$src" -B "$build" "${flags[@]}"
cmake --build "$build" --target llama-bench llama-completion llama-tokenize -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
echo "built $backend in $build/bin"
