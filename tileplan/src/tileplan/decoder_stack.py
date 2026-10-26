"""Synthetic decoder stack and whole-token decoder program package."""

import json
import math
import mmap
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .decoder_reference import (
    DecoderSpec,
    ReferenceCache,
    _rms_norm,
    decode_one,
    fixed_weights,
    recompute_prefix,
)
from .plan_core import (
    CANDIDATE_BYTES,
    WEIGHTS,
    align,
    assemble_plan,
    attention_chunking,
    block_spec,
    check_weight,
    layer_view,
    model_section,
    numeric_policy,
    pack_array,
    pack_weights,
    plan_target,
    regions,
    require,
    scheduled_program,
    selection_slots,
    source_shape,
    storage_section,
    validate_header,
    validate_plan,
    validate_target,
    weight_size,
    write_package,
)
from .probe import read_target

MODEL_KIND = "llama_decoder_stack"


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


def stack_spec(source: dict) -> StackSpec:
    """Rebuild a stack spec from a reference fixture or a plan's model section."""
    return StackSpec(
        block_spec(source),
        source["layers"],
        source["vocab_size"],
        source["tie_word_embeddings"],
    )


def read_stack_model(path: Path) -> StackSpec:
    document = json.loads(path.read_text())
    require(
        document.get("kind") == "llama_decoder_stack_reference"
        and document.get("version") == 1,
        "unsupported stack fixture",
    )
    return stack_spec(document)


def read_stack_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def _expected_names(spec: StackSpec) -> set[str]:
    names = {"embedding", "final_norm"}
    names.update(f"L{layer}.{name}" for layer in range(spec.layers) for name in WEIGHTS)
    if not spec.tie_word_embeddings:
        names.add("lm_head")
    return names


def _check_weights(spec: StackSpec, weights: dict[str, np.ndarray]) -> None:
    require(set(weights) == _expected_names(spec), "stack weight names mismatch")
    d, vocab = spec.block.hidden_size, spec.vocab_size
    for name, value in weights.items():
        if name in ("embedding", "lm_head"):
            shape = (vocab, d)
        elif name == "final_norm":
            shape = (d,)
        else:
            shape = source_shape(spec.block, name.split(".", 1)[1])
        check_weight(value, shape, name)


def stack_layout(spec: StackSpec, dtype: str) -> tuple[dict, int]:
    """Offsets of the stack's regions in weights.bin, and its total size."""
    block, size = spec.block, weight_size(dtype)
    embedding_bytes = spec.vocab_size * block.hidden_size * size
    _, layer_weight_bytes = regions(block, WEIGHTS, 64, size)
    _, layer_kv_bytes = regions(block, ("K", "V"), 16, size)
    layer_stride = align(layer_weight_bytes)
    layer_offset = align(embedding_bytes)
    norm_offset = align(layer_offset + spec.layers * layer_stride)
    total = norm_offset + block.hidden_size * size
    lm_offset = 0
    if not spec.tie_word_embeddings:
        lm_offset = align(total)
        total = lm_offset + embedding_bytes
    return {
        "embedding_offset": 0,
        "embedding_bytes": embedding_bytes,
        "layer_weights_offset": layer_offset,
        "layer_weight_stride": layer_stride,
        "layer_weight_bytes": layer_weight_bytes,
        "final_norm_offset": norm_offset,
        "final_norm_bytes": block.hidden_size * size,
        "lm_head_offset": lm_offset,
        "lm_head_alias": spec.tie_word_embeddings,
        "layer_kv_stride": layer_kv_bytes,
    }, total


def stack_plan(
    spec: StackSpec,
    target: dict,
    dtype: str,
    weights_sha256: str,
    chunk_steps: int | None = None,
    fused_steps: int | None = None,
) -> dict:
    """Everything in a stack package except the weight bytes themselves.

    Checkpoint importers build this before streaming weights, so it never
    needs them; ``weights_sha256`` may be a placeholder until they are written.
    """
    require(len(weights_sha256) == 64, "invalid weight hash")
    block, workers = spec.block, target["worker_count"]
    stack, total = stack_layout(spec, dtype)
    chunking = attention_chunking(block, dtype, target, chunk_steps, fused_steps)
    storage = storage_section(
        block,
        dtype,
        chunking["attention_chunks"],
        workers,
        spec.layers,
        4,
        spec.vocab_size * 4,
        total,
    )
    return assemble_plan(
        MODEL_KIND,
        target,
        model_section(spec.as_dict(), block, dtype, chunking, workers),
        {
            **storage,
            "stack": stack,
            # One greedy candidate per logits task.
            "selection_bytes": CANDIDATE_BYTES
            * selection_slots(spec.vocab_size, workers),
        },
        scheduled_program(
            block, dtype, workers, chunking, spec.layers, spec.vocab_size
        ),
        weights_sha256,
    )


def make_stack_plan_metadata(
    spec: StackSpec,
    target: dict,
    weight_dtype: str,
    weights_sha256: str,
    chunk_steps: int | None = None,
    fused_steps: int | None = None,
) -> dict:
    return stack_plan(
        spec,
        plan_target(target),
        weight_dtype,
        weights_sha256,
        chunk_steps,
        fused_steps,
    )


def make_stack_plan(
    spec: StackSpec,
    weights: dict[str, np.ndarray],
    target: dict,
    weight_dtype: str = "float32",
    chunk_steps: int | None = None,
    fused_steps: int | None = None,
) -> tuple[dict, bytes]:
    _check_weights(spec, weights)
    stack, total = stack_layout(spec, weight_dtype)
    image = bytearray(total)

    def put(offset: int, blob: bytes) -> None:
        image[offset : offset + len(blob)] = blob

    put(0, pack_array(weights["embedding"], weight_dtype))
    for layer in range(spec.layers):
        layer_weights = {name: weights[f"L{layer}.{name}"] for name in WEIGHTS}
        put(
            stack["layer_weights_offset"] + layer * stack["layer_weight_stride"],
            pack_weights(spec.block, layer_weights, weight_dtype),
        )
    put(stack["final_norm_offset"], pack_array(weights["final_norm"], weight_dtype))
    if not spec.tie_word_embeddings:
        put(stack["lm_head_offset"], pack_array(weights["lm_head"], weight_dtype))
    packed = bytes(image)
    plan = make_stack_plan_metadata(
        spec,
        target,
        weight_dtype,
        sha256(packed).hexdigest(),
        chunk_steps,
        fused_steps,
    )
    return plan, packed


def validate_stack(plan: dict, packed: bytes | mmap.mmap) -> None:
    validate_header(plan, MODEL_KIND)
    model = plan["model"]
    spec = stack_spec(model)
    dtype = model.get("weight_dtype", "float32")
    validate_target(plan["target"])
    steps, fused = model["attention_chunk_steps"], model["attention_fused_steps"]
    validate_plan(
        plan,
        packed,
        lambda digest: stack_plan(spec, plan["target"], dtype, digest, steps, fused),
    )


def stack_summary(plan: dict, digest: str) -> dict:
    storage = plan["storage"]
    waves = plan["waves"]
    return {
        **numeric_policy(plan["model"].get("weight_dtype", "float32")),
        "attention_bytes": storage["attention_bytes"],
        "plan_sha256": digest,
        "weights_sha256": plan["integrity"]["weights_sha256"],
        "layers": plan["model"]["layers"],
        "vocab_size": plan["model"]["vocab_size"],
        "attention_chunks": plan["model"]["attention_chunks"],
        "attention_chunk_steps": plan["model"]["attention_chunk_steps"],
        "attention_fused_steps": plan["model"]["attention_fused_steps"],
        "task_count": len(plan["tasks"]),
        "wave_count": len(waves),
        "layer_wave_count": len(layer_view(plan)["waves"]),
        "scratch_bytes": storage["scratch_bytes"],
        "kv_bytes": storage["kv_bytes"],
        "weight_bytes": storage["weight_bytes"],
        "worker_loads": [wave["estimated_work"] for wave in waves],
    }


def build_stack(
    model_path: Path,
    weights_path: Path,
    target_path: Path,
    out: Path,
    weight_dtype: str = "float32",
    chunk_steps: int | None = None,
    fused_steps: int | None = None,
) -> dict:
    plan, packed = make_stack_plan(
        read_stack_model(model_path),
        read_stack_weights(weights_path),
        read_target(target_path),
        weight_dtype,
        chunk_steps,
        fused_steps,
    )
    return write_package(out, plan, packed, stack_summary)


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
