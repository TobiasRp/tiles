"""Single-block decoder packages: one layer, no embedding or logits."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from .decoder_reference import DecoderSpec
from .plan_core import (
    WEIGHTS,
    assemble_plan,
    block_spec,
    check_weight,
    model_section,
    numeric_policy,
    pack_weights,
    plan_target,
    require,
    scheduled_program,
    storage_section,
    tensor_shape,
    validate_header,
    validate_plan,
    validate_target,
    weight_size,
    write_package,
)
from .probe import read_target

MODEL_KIND = "llama_decoder_block"


def read_model(path: Path) -> DecoderSpec:
    document = json.loads(path.read_text())
    require(
        document.get("kind") == "llama_decoder_block_reference"
        and document.get("version") == 1,
        "unsupported decoder model fixture",
    )
    return block_spec(document)


def read_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def block_plan(
    spec: DecoderSpec, target: dict, dtype: str, weights_sha256: str
) -> dict:
    """Everything in a block package except the weight bytes themselves."""
    workers = target["worker_count"]
    io_bytes = spec.hidden_size * weight_size(dtype)
    return assemble_plan(
        MODEL_KIND,
        target,
        model_section({**spec.as_dict(), "layers": 1}, spec, dtype, workers),
        storage_section(spec, dtype, workers, 1, io_bytes, io_bytes),
        scheduled_program(spec, dtype, workers),
        weights_sha256,
    )


def make_plan(
    spec: DecoderSpec,
    weights: dict[str, np.ndarray],
    target: dict,
    weight_dtype: str = "float32",
) -> tuple[dict, bytes]:
    require(set(weights) == set(WEIGHTS), "weight names mismatch")
    for name, value in weights.items():
        check_weight(value, tensor_shape(spec, name), name)
    packed = pack_weights(spec, weights, weight_dtype)
    plan = block_plan(
        spec, plan_target(target), weight_dtype, sha256(packed).hexdigest()
    )
    return plan, packed


def validate(plan: dict, packed: bytes) -> None:
    """Validate every declaration before a package is handed to the runtime."""
    validate_header(plan, MODEL_KIND)
    model = plan["model"]
    spec = block_spec(model)
    dtype = model.get("weight_dtype", "float32")
    validate_target(plan["target"])
    validate_plan(
        plan, packed, lambda digest: block_plan(spec, plan["target"], dtype, digest)
    )


def block_summary(plan: dict, digest: str) -> dict:
    storage = plan["storage"]
    waves = plan["waves"]
    return {
        **numeric_policy(plan["model"].get("weight_dtype", "float32")),
        "attention_bytes": storage["attention_bytes"],
        "plan_sha256": digest,
        "weights_sha256": plan["integrity"]["weights_sha256"],
        "model": {
            name: plan["model"][name]
            for name in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_context",
                "attention_chunks",
            )
        },
        "task_counts": {
            opcode: sum(task["opcode"] == opcode for task in plan["tasks"])
            for opcode in sorted({task["opcode"] for task in plan["tasks"]})
        },
        "wave_count": len(waves),
        "worker_loads": [wave["estimated_work"] for wave in waves],
        "idle_workers": [
            sum(load == 0 for load in wave["estimated_work"]) for wave in waves
        ],
        "scratch_bytes": storage["scratch_bytes"],
        "kv_bytes": storage["kv_bytes"],
        "weight_bytes": storage["weight_bytes"],
        "peak_bytes": storage["attention_bytes"]
        + storage["attention_partial_bytes"]
        + sum(
            storage[name]
            for name in (
                "input_bytes",
                "output_bytes",
                "scratch_bytes",
                "kv_bytes",
                "weight_bytes",
            )
        ),
        "target": plan["target"],
    }


def build(
    model_path: Path,
    weights_path: Path,
    target_path: Path,
    out: Path,
    weight_dtype: str = "float32",
) -> dict:
    spec = read_model(model_path)
    plan, packed = make_plan(
        spec, read_weights(weights_path), read_target(target_path), weight_dtype
    )
    return write_package(out, plan, packed, block_summary)
