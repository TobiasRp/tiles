"""Read, validate, and summarize a package of any model kind."""

import json
from hashlib import sha256
from pathlib import Path

from . import decoder_block, decoder_stack

_KINDS = {
    decoder_block.MODEL_KIND: (decoder_block.validate, decoder_block.block_summary),
    decoder_stack.MODEL_KIND: (
        decoder_stack.validate_stack,
        decoder_stack.stack_summary,
    ),
}


def _handlers(plan: dict):
    kind = plan.get("header", {}).get("model_kind")
    if kind not in _KINDS:
        raise ValueError(
            f"unsupported model kind {kind!r}; regenerate the package with the "
            "current planner"
        )
    return _KINDS[kind]


def read_package(package: Path) -> tuple[dict, bytes, str]:
    data = (package / "plan.json").read_bytes()
    plan = json.loads(data)
    packed = (package / "weights.bin").read_bytes()
    _handlers(plan)[0](plan, packed)
    return plan, packed, sha256(data).hexdigest()


def summary(plan: dict, digest: str) -> dict:
    return _handlers(plan)[1](plan, digest)
