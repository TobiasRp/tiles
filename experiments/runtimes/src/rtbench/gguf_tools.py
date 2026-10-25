"""Convert a Hugging Face checkpoint to a BF16 GGUF and check what came out."""

import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .paths import LLAMACPP_SRC


def convert(checkpoint: Path, out: Path) -> None:
    """Run llama.cpp's converter (it bundles its own gguf-py)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(LLAMACPP_SRC / "convert_hf_to_gguf.py"),
            str(checkpoint),
            "--outtype",
            "bf16",
            "--outfile",
            str(out),
        ],
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_types(path: Path) -> Counter[tuple[str, int]]:
    """(type name, tensor rank) -> count."""
    sys.path.insert(0, str(LLAMACPP_SRC / "gguf-py"))
    from gguf import GGUFReader  # ty: ignore[unresolved-import]

    reader = GGUFReader(path)
    return Counter((t.tensor_type.name, len(t.shape)) for t in reader.tensors)


def check_bf16(path: Path) -> Counter[tuple[str, int]]:
    """Matrices must be BF16; 1D tensors (norm weights) stay F32 in GGUF."""
    types = tensor_types(path)
    for (name, rank), count in types.items():
        allowed = {"BF16"} if rank >= 2 else {"F32", "BF16"}
        if name not in allowed:
            raise ValueError(f"{count} tensor(s) of type {name} with rank {rank}")
    return types
