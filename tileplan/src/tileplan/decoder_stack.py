"""Synthetic decoder stack and compact repeated-block package."""

import json
import math
import mmap
import os
import shutil
import tempfile
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .decoder_plan import (
    FORMAT,
    GENERATOR,
    SCRATCH,
    WEIGHTS,
    _attention_bytes,
    _descriptors,
    _graph,
    _json_bytes,
    _model_fields,
    _pack_array,
    _pack_weights,
    _regions,
    _require,
    _schedule,
    _shape,
    _validate_policy,
    _weight_size,
    choose_tile_rows,
    numeric_policy,
)
from .decoder_reference import (
    DecoderSpec,
    ReferenceCache,
    _rms_norm,
    decode_one,
    fixed_weights,
    recompute_prefix,
)
from .probe import ABI, read_target


@dataclass(frozen=True)
class StackSpec:
    block: DecoderSpec = DecoderSpec()
    layers: int = 3
    vocab_size: int = 48
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.vocab_size <= 0:
            raise ValueError("layers and vocabulary must be positive")

    def as_dict(self) -> dict:
        return {
            **self.block.as_dict(),
            "layers": self.layers,
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def fixed_stack_weights(spec: StackSpec, seed: int = 37) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {
        "embedding": torch.randn(
            (spec.vocab_size, spec.block.hidden_size), generator=generator
        )
        / math.sqrt(spec.block.hidden_size),
        "final_norm": 1
        + 0.05 * torch.randn((spec.block.hidden_size,), generator=generator),
    }
    for layer in range(spec.layers):
        result.update(
            {
                f"L{layer}.{name}": value
                for name, value in fixed_weights(spec.block, seed + layer + 1).items()
            }
        )
    if not spec.tie_word_embeddings:
        result["lm_head"] = torch.randn(
            (spec.vocab_size, spec.block.hidden_size), generator=generator
        ) / math.sqrt(spec.block.hidden_size)
    return result


@dataclass
class StackCache:
    layers: list[ReferenceCache]

    @classmethod
    def empty(
        cls,
        spec: StackSpec,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> StackCache:
        return cls(
            [
                ReferenceCache.empty(spec.block, dtype, device)
                for _ in range(spec.layers)
            ]
        )

    @property
    def position(self) -> int:
        positions = {layer.position for layer in self.layers}
        if len(positions) != 1:
            raise ValueError("layer cache positions diverged")
        return positions.pop()

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


def decode_token(
    token_id: int,
    weights: dict[str, torch.Tensor],
    spec: StackSpec,
    cache: StackCache,
) -> dict:
    if not 0 <= token_id < spec.vocab_size:
        raise ValueError("token ID outside vocabulary")
    hidden = weights["embedding"][token_id]
    blocks = []
    for layer in range(spec.layers):
        layer_weights = {name: weights[f"L{layer}.{name}"] for name in WEIGHTS}
        trace = decode_one(hidden, layer_weights, spec.block, cache.layers[layer])
        hidden = trace["y"]
        blocks.append(trace)
    if hidden.dtype == torch.float32:
        factor = torch.rsqrt(hidden.square().mean() + spec.block.rms_norm_eps)
        final_hidden = hidden * factor * weights["final_norm"]
    else:
        final_hidden = _rms_norm(hidden, weights["final_norm"], spec.block.rms_norm_eps)
    projection = (
        weights["embedding"] if spec.tie_word_embeddings else weights["lm_head"]
    )
    logits = projection @ final_hidden
    return {"blocks": blocks, "final_hidden": final_hidden, "logits": logits}


def recompute_stack_prefix(
    token_ids: list[int], weights: dict[str, torch.Tensor], spec: StackSpec
) -> torch.Tensor:
    """Recompute every visible position using full causal matrices."""
    if not token_ids or len(token_ids) > spec.block.max_context:
        raise ValueError("invalid token prefix")
    if any(token < 0 or token >= spec.vocab_size for token in token_ids):
        raise ValueError("token ID outside vocabulary")
    hidden = weights["embedding"][token_ids]
    for layer in range(spec.layers):
        layer_weights = {name: weights[f"L{layer}.{name}"] for name in WEIGHTS}
        hidden = recompute_prefix(hidden, layer_weights, spec.block)["y"]
    if hidden.dtype == torch.float32:
        factor = torch.rsqrt(
            hidden.square().mean(dim=-1, keepdim=True) + spec.block.rms_norm_eps
        )
        final_hidden = hidden * factor * weights["final_norm"]
    else:
        final_hidden = _rms_norm(hidden, weights["final_norm"], spec.block.rms_norm_eps)
    projection = (
        weights["embedding"] if spec.tie_word_embeddings else weights["lm_head"]
    )
    return final_hidden @ projection.T


def _align(value: int, alignment: int = 64) -> int:
    return value + (-value) % alignment


def read_stack_model(path: Path) -> StackSpec:
    document = json.loads(path.read_text())
    _require(
        document.get("kind") == "llama_decoder_stack_reference"
        and document.get("version") == 1,
        "unsupported stack fixture",
    )
    names = {field.name for field in fields(DecoderSpec)}
    block = DecoderSpec(**{name: document[name] for name in names})
    _require(document["head_to_kv"] == block.head_to_kv, "head mapping mismatch")
    return StackSpec(
        block,
        document["layers"],
        document["vocab_size"],
        document["tie_word_embeddings"],
    )


def _expected_names(spec: StackSpec) -> set[str]:
    names = {"embedding", "final_norm"}
    names.update(f"L{layer}.{name}" for layer in range(spec.layers) for name in WEIGHTS)
    if not spec.tie_word_embeddings:
        names.add("lm_head")
    return names


def read_stack_weights(path: Path, spec: StackSpec) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        _require(
            set(source.files) == _expected_names(spec), "stack weight names mismatch"
        )
        weights = {name: source[name].copy() for name in source.files}
    _validate_weights(spec, weights)
    return weights


def _validate_weights(spec: StackSpec, weights: dict[str, np.ndarray]) -> None:
    _require(set(weights) == _expected_names(spec), "stack weight names mismatch")
    d, vocab = spec.block.hidden_size, spec.vocab_size
    for name, value in weights.items():
        if name in ("embedding", "lm_head"):
            shape = (vocab, d)
        elif name == "final_norm":
            shape = (d,)
        else:
            shape = _shape(spec.block, name.split(".", 1)[1])
        _require(
            value.shape == shape
            and value.dtype == np.float32
            and bool(np.isfinite(value).all()),
            f"invalid stack weight {name}",
        )


def make_stack_plan(
    spec: StackSpec,
    weights: dict[str, np.ndarray],
    target: dict,
    tile_rows: int,
    weight_dtype: str = "float32",
    matrix_kernel: str = "simt",
) -> tuple[dict, bytes]:
    _require(type(tile_rows) is int and tile_rows > 0, "tile rows must be positive")
    _validate_weights(spec, weights)
    block = spec.block
    weight_bytes = _weight_size(weight_dtype)
    embedding = _pack_array(weights["embedding"], weight_dtype)
    layer_packed = []
    for layer in range(spec.layers):
        packed, _ = _pack_weights(
            block,
            {name: weights[f"L{layer}.{name}"] for name in WEIGHTS},
            weight_dtype,
        )
        layer_packed.append(packed)
    layer_stride = _align(len(layer_packed[0]))
    layer_offset = _align(len(embedding))
    norm_offset = _align(layer_offset + spec.layers * layer_stride)
    cursor = norm_offset + block.hidden_size * weight_bytes
    lm_offset = 0
    if not spec.tie_word_embeddings:
        lm_offset = _align(cursor)
        cursor = lm_offset + len(embedding)
    packed = bytearray(cursor)
    packed[: len(embedding)] = embedding
    for layer, blob in enumerate(layer_packed):
        start = layer_offset + layer * layer_stride
        packed[start : start + len(blob)] = blob
    packed[norm_offset : norm_offset + block.hidden_size * weight_bytes] = _pack_array(
        weights["final_norm"], weight_dtype
    )
    if not spec.tie_word_embeddings:
        packed[lm_offset : lm_offset + len(embedding)] = _pack_array(
            weights["lm_head"], weight_dtype
        )
    plan = make_stack_plan_metadata(
        spec, target, tile_rows, weight_dtype, sha256(packed).hexdigest(), matrix_kernel
    )
    _require(len(packed) == plan["storage"]["weight_bytes"], "invalid packed size")
    validate_stack(plan, bytes(packed))
    return plan, bytes(packed)


def make_stack_plan_metadata(
    spec: StackSpec,
    target: dict,
    tile_rows: int,
    weight_dtype: str,
    weights_sha256: str,
    matrix_kernel: str = "simt",
) -> dict:
    """Build the stack graph and layout without holding checkpoint weights."""
    _require(type(tile_rows) is int and tile_rows > 0, "tile rows must be positive")
    _require(len(weights_sha256) == 64, "invalid weight hash")
    block = spec.block
    weight_bytes = _weight_size(weight_dtype)
    embedding_bytes = spec.vocab_size * block.hidden_size * weight_bytes
    weight_regions, layer_weight_bytes = _regions(block, WEIGHTS, 64, weight_bytes)
    layer_stride = _align(layer_weight_bytes)
    layer_offset = _align(embedding_bytes)
    norm_offset = _align(layer_offset + spec.layers * layer_stride)
    cursor = norm_offset + block.hidden_size * weight_bytes
    lm_offset = 0
    if not spec.tie_word_embeddings:
        lm_offset = _align(cursor)
        cursor = lm_offset + embedding_bytes
    scratch_regions, scratch_bytes = _regions(block, SCRATCH, 16, weight_bytes)
    kv_regions, layer_kv_bytes = _regions(block, ("K", "V"), 16, weight_bytes)
    tasks = _graph(block, tile_rows, weight_dtype)
    waves = _schedule(tasks, target["launch"]["worker_count"])
    stack = {
        "embedding_offset": 0,
        "embedding_bytes": embedding_bytes,
        "layer_weights_offset": layer_offset,
        "layer_weight_stride": layer_stride,
        "layer_weight_bytes": layer_weight_bytes,
        "final_norm_offset": norm_offset,
        "final_norm_bytes": block.hidden_size * weight_bytes,
        "lm_head_offset": lm_offset,
        "lm_head_alias": spec.tie_word_embeddings,
        "layer_kv_stride": layer_kv_bytes,
    }
    plan = {
        "header": {
            "format_version": FORMAT,
            "executor_abi": ABI,
            "generator": GENERATOR,
            "model_kind": "llama_decoder_stack",
        },
        "target": {
            "compute_capability": target["device"]["compute_capability"],
            "sm_count": target["device"]["sm_count"],
            "cooperative_launch": True,
            "worker_count": target["launch"]["worker_count"],
            "threads_per_block": target["launch"]["threads_per_block"],
            "kernel_fingerprint": target["executor"]["fingerprint"],
        },
        "model": {
            **_model_fields(spec.as_dict(), weight_dtype),
            "matrix_kernel": matrix_kernel,
            "tile_rows": tile_rows,
            "tensors": _descriptors(block, weight_dtype),
        },
        "storage": {
            "input_bytes": 4,
            "output_bytes": spec.vocab_size * 4,
            "scratch_bytes": scratch_bytes,
            "attention_bytes": _attention_bytes(block, weight_dtype),
            "scratch": scratch_regions,
            "kv_bytes": layer_kv_bytes * spec.layers,
            "kv": kv_regions,
            "weight_bytes": cursor,
            "weights": weight_regions,
            "stack": stack,
        },
        "execution": {
            "prelude": "embedding_lookup_bf16"
            if weight_dtype == "bfloat16"
            else "embedding_lookup_f32",
            "repeat_block_count": spec.layers,
            "postlude": ["rms_norm_bf16", "logits_bf16"]
            if weight_dtype == "bfloat16"
            else ["rms_norm_f32", "logits_f32"],
        },
        "tasks": tasks,
        "waves": waves,
        "integrity": {
            "weights_bytes": cursor,
            "weights_sha256": weights_sha256,
        },
    }
    return plan


def validate_stack(plan: dict, packed: bytes | mmap.mmap) -> None:
    header, model, storage = (plan[name] for name in ("header", "model", "storage"))
    legacy = header["format_version"] in (2, 3)
    _require(
        (
            (
                legacy
                and header["executor_abi"]
                == f"tileinfer-decoder-v{header['format_version']}"
            )
            or (header["format_version"] == FORMAT and header["executor_abi"] == ABI)
        )
        and header["model_kind"]
        == ("llama_decoder_stack_f32" if legacy else "llama_decoder_stack"),
        "unsupported stack header",
    )
    weight_dtype = _validate_policy(header, model)
    weight_bytes = _weight_size(weight_dtype)
    names = {field.name for field in fields(DecoderSpec)}
    block = DecoderSpec(
        **{name: model[name] for name in names if name != "dtype" or legacy}
    )
    spec = StackSpec(
        block, model["layers"], model["vocab_size"], model["tie_word_embeddings"]
    )
    _require(
        {name: model[name] for name in spec.as_dict() if name != "dtype" or legacy}
        == {
            name: value
            for name, value in spec.as_dict().items()
            if name != "dtype" or legacy
        }
        and model["tensors"] == _descriptors(block, weight_dtype),
        "invalid stack model",
    )
    tile_rows = model["tile_rows"]
    _require(type(tile_rows) is int and tile_rows > 0, "invalid tile rows")
    workers = plan["target"]["worker_count"]
    _require(
        type(workers) is int
        and workers > 0
        and workers == plan["target"]["sm_count"]
        and plan["target"]["cooperative_launch"] is True
        and type(plan["target"]["threads_per_block"]) is int
        and plan["target"]["threads_per_block"] > 0,
        "invalid stack target",
    )
    base_regions, layer_weights_bytes = _regions(block, WEIGHTS, 64, weight_bytes)
    scratch_regions, scratch_bytes = _regions(block, SCRATCH, 16, weight_bytes)
    kv_regions, layer_kv_bytes = _regions(block, ("K", "V"), 16, weight_bytes)
    embedding_bytes = spec.vocab_size * block.hidden_size * weight_bytes
    layer_offset = _align(embedding_bytes)
    layer_stride = _align(layer_weights_bytes)
    norm_offset = _align(layer_offset + spec.layers * layer_stride)
    lm_offset = (
        0
        if spec.tie_word_embeddings
        else _align(norm_offset + block.hidden_size * weight_bytes)
    )
    total = (
        norm_offset + block.hidden_size * weight_bytes
        if spec.tie_word_embeddings
        else lm_offset + embedding_bytes
    )
    expected_stack = {
        "embedding_offset": 0,
        "embedding_bytes": embedding_bytes,
        "layer_weights_offset": layer_offset,
        "layer_weight_stride": layer_stride,
        "layer_weight_bytes": layer_weights_bytes,
        "final_norm_offset": norm_offset,
        "final_norm_bytes": block.hidden_size * weight_bytes,
        "lm_head_offset": lm_offset,
        "lm_head_alias": spec.tie_word_embeddings,
        "layer_kv_stride": layer_kv_bytes,
    }
    _require(
        storage
        == {
            "input_bytes": 4,
            "output_bytes": spec.vocab_size * 4,
            "scratch_bytes": scratch_bytes,
            **(
                {"attention_bytes": _attention_bytes(block, weight_dtype)}
                if not legacy
                else {}
            ),
            "scratch": scratch_regions,
            "kv_bytes": layer_kv_bytes * spec.layers,
            "kv": kv_regions,
            "weight_bytes": total,
            "weights": base_regions,
            "stack": expected_stack,
        },
        "invalid stack storage",
    )
    _require(
        plan["execution"]
        == {
            "prelude": "embedding_lookup_bf16"
            if weight_dtype == "bfloat16"
            else "embedding_lookup_f32",
            "repeat_block_count": spec.layers,
            "postlude": ["rms_norm_bf16", "logits_bf16"]
            if weight_dtype == "bfloat16"
            else ["rms_norm_f32", "logits_f32"],
        },
        "invalid stack execution stages",
    )
    _require(
        plan["integrity"]
        == {"weights_bytes": len(packed), "weights_sha256": sha256(packed).hexdigest()}
        and len(packed) == total,
        "stack weight checksum or size mismatch",
    )
    tasks = _graph(block, tile_rows, weight_dtype)
    waves = _schedule(tasks, workers)
    _require(
        plan["tasks"] == tasks and plan["waves"] == waves,
        "invalid stack block graph or schedule",
    )


def build_stack(
    model_path: Path,
    weights_path: Path,
    target_path: Path,
    tile_rows: int | str,
    out: Path,
    weight_dtype: str = "float32",
    matrix_kernel: str = "simt",
) -> dict:
    spec = read_stack_model(model_path)
    weights = read_stack_weights(weights_path, spec)
    target = read_target(target_path)
    candidates = None
    if tile_rows == "auto":
        tile_rows, candidates = choose_tile_rows(
            spec.block, target["launch"]["worker_count"]
        )
    elif isinstance(tile_rows, str):
        tile_rows = int(tile_rows)
    plan, packed = make_stack_plan(
        spec, weights, target, tile_rows, weight_dtype, matrix_kernel
    )
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (temporary / "weights.bin").write_bytes(packed)
        data = _json_bytes(plan)
        (temporary / "plan.json").write_bytes(data)
        report = {
            **numeric_policy(weight_dtype),
            "matrix_kernel": matrix_kernel,
            "attention_bytes": plan["storage"]["attention_bytes"],
            "plan_sha256": sha256(data).hexdigest(),
            "weights_sha256": plan["integrity"]["weights_sha256"],
            "layers": spec.layers,
            "vocab_size": spec.vocab_size,
            "tile_rows": tile_rows,
            "block_task_count": len(plan["tasks"]),
            "block_wave_count": len(plan["waves"]),
            "scratch_bytes": plan["storage"]["scratch_bytes"],
            "kv_bytes": plan["storage"]["kv_bytes"],
            "weight_bytes": len(packed),
            "worker_loads": [wave["estimated_work"] for wave in plan["waves"]],
        }
        if candidates is not None:
            report["tile_search"] = candidates
        (temporary / "planning_report.json").write_bytes(_json_bytes(report))
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def generate_stack_fixture(
    out: Path, spec: StackSpec = StackSpec(), seed: int = 37
) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    weights = fixed_stack_weights(spec, seed)
    token_ids = [0, 7, 15, 3, 47]
    cache = StackCache.empty(spec)
    with torch.inference_mode():
        traces = [decode_token(token, weights, spec, cache) for token in token_ids]
    (out / "model.json").write_text(
        json.dumps(
            {
                "kind": "llama_decoder_stack_reference",
                "version": 1,
                "seed": seed,
                **spec.as_dict(),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    np.savez(
        out / "weights.npz",
        **{name: value.numpy() for name, value in weights.items()},
    )
    np.savez(
        out / "reference_cases.npz",
        token_ids=np.asarray(token_ids, dtype=np.int32),
        logits=np.stack([trace["logits"].numpy() for trace in traces]),
        k_cache=np.stack([layer.k.numpy() for layer in cache.layers]),
        v_cache=np.stack([layer.v.numpy() for layer in cache.layers]),
    )
