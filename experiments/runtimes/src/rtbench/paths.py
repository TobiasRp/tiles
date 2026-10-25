"""Locations inside experiments/runtimes."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"
LLAMACPP_SRC = THIRD_PARTY / "llama.cpp"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PIN = ROOT / "scripts" / "llamacpp.pin"


def llamacpp_build(backend: str) -> Path:
    return THIRD_PARTY / f"build-{backend}"


def llamacpp_bin(backend: str, name: str) -> Path:
    path = llamacpp_build(backend) / "bin" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not built; run scripts/build_llamacpp.sh {backend}"
        )
    return path


def probe_bin(backend: str) -> Path:
    path = ROOT / "probe" / f"build-{backend}" / "bin" / "rt_probe"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not built; run scripts/build_probe.sh {backend}"
        )
    return path


def llamacpp_pin() -> dict[str, str]:
    """Pinned tag and commit from scripts/llamacpp.pin."""
    pin = {}
    for line in PIN.read_text().splitlines():
        match = re.fullmatch(r"(\w+)=(\S+)", line.strip())
        if match:
            pin[match.group(1)] = match.group(2)
    return pin
