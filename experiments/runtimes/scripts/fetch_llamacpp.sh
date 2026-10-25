#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit into third_party/llama.cpp.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=llamacpp.pin
source "$here/scripts/llamacpp.pin"
dest="$here/third_party/llama.cpp"

if [[ ! -d "$dest/.git" ]]; then
  mkdir -p "$here/third_party"
  git init -q "$dest"
  git -C "$dest" remote add origin https://github.com/ggml-org/llama.cpp
fi
git -C "$dest" fetch -q --depth 1 origin "$LLAMACPP_COMMIT"
git -C "$dest" checkout -q --detach "$LLAMACPP_COMMIT"
echo "llama.cpp $LLAMACPP_TAG ($(git -C "$dest" rev-parse HEAD)) in $dest"
