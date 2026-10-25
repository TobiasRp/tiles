#!/usr/bin/env bash
# Build probe/rt_probe against the llama.cpp build for one backend.
# Usage: build_probe.sh [cpu|metal|cuda]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
backend="${1:-cpu}"
llama_build="$here/third_party/build-$backend"
[[ -d "$llama_build/bin" ]] || { echo "run build_llamacpp.sh $backend first" >&2; exit 1; }
cmake -S "$here/probe" -B "$here/probe/build-$backend" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMACPP_SRC="$here/third_party/llama.cpp" -DLLAMACPP_BUILD="$llama_build"
cmake --build "$here/probe/build-$backend" -j
