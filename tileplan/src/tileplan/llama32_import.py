"""Import the pinned Hugging Face Llama 3.2 1B base checkpoint."""

import json
import mmap
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .decoder_plan import WEIGHTS, _json_bytes, _require, _shape, choose_tile_rows
from .decoder_reference import DecoderSpec
from .decoder_stack import StackSpec, make_stack_plan_metadata, validate_stack
from .probe import read_target

MODEL_ID = "meta-llama/Llama-3.2-1B"
PINNED_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_SUFFIXES = {
    "attn_norm": "input_layernorm.weight",
    "Wq": "self_attn.q_proj.weight",
    "Wk": "self_attn.k_proj.weight",
    "Wv": "self_attn.v_proj.weight",
    "Wo": "self_attn.o_proj.weight",
    "ffn_norm": "post_attention_layernorm.weight",
    "Wgate": "mlp.gate_proj.weight",
    "Wup": "mlp.up_proj.weight",
    "Wdown": "mlp.down_proj.weight",
}


def read_llama32_config(path: Path, max_context: int) -> StackSpec:
    """Accept only the base 1B architecture supported by this executor."""
    config = json.loads(path.read_text())
    expected = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    for name, value in expected.items():
        _require(config.get(name) == value, f"unsupported Llama config {name}")
    rope = config.get("rope_scaling")
    _require(isinstance(rope, dict), "missing Llama 3 RoPE scaling")
    for name, value in {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }.items():
        _require(rope.get(name) == value, f"unsupported RoPE {name}")
    _require(
        type(max_context) is int
        and 1 <= max_context <= 8192
        and max_context <= config.get("max_position_embeddings", 0),
        "max_context must be between 1 and 8192 and fit the checkpoint",
    )
    block = DecoderSpec(
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_context=max_context,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_factor=32.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )
    return StackSpec(block, 16, 128256, True)


def _name_map(spec: StackSpec) -> dict[str, str]:
    names = {
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
    }
    for layer in range(spec.layers):
        for name in WEIGHTS:
            names[f"L{layer}.{name}"] = f"model.layers.{layer}.{_SUFFIXES[name]}"
    if not spec.tie_word_embeddings:
        names["lm_head"] = "lm_head.weight"
    return names


def _manifest(checkpoint: Path, expected: set[str], tied: bool) -> dict[str, Path]:
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        document = json.loads(index.read_text())
        weight_map = document.get("weight_map")
        _require(isinstance(weight_map, dict), "invalid safetensors index")
        _require(
            all(
                isinstance(name, str) and isinstance(file, str)
                for name, file in weight_map.items()
            ),
            "invalid safetensors index entries",
        )
        names = weight_map
    else:
        file = checkpoint / "model.safetensors"
        _require(file.is_file(), "missing model.safetensors or shard index")
        with safe_open(file, framework="pt", device="cpu") as source:
            names = dict.fromkeys(source.keys(), file.name)
    actual = set(names)
    optional = {"lm_head.weight"} if tied else set()
    _require(
        expected <= actual <= expected | optional, "checkpoint tensor names mismatch"
    )
    manifest = {}
    for name, filename in names.items():
        _require(
            filename == Path(filename).name and filename.endswith(".safetensors"),
            "invalid safetensors shard name",
        )
        path = checkpoint / filename
        _require(path.is_file(), f"missing safetensors shard {filename}")
        manifest[name] = path
    for path in set(manifest.values()):
        with safe_open(path, framework="pt", device="cpu") as source:
            _require(
                set(source.keys())
                == {name for name, shard in manifest.items() if shard == path},
                f"safetensors index disagrees with {path.name}",
            )
    return manifest


def _revision(checkpoint: Path, requested: str | None) -> tuple[str, str]:
    metadata = checkpoint / ".cache/huggingface/download/config.json.metadata"
    recorded = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    snapshot = checkpoint.parent.name == "snapshots" and checkpoint.name or None
    observed = recorded or snapshot
    if observed is not None:
        _require(
            bool(_HEX40.fullmatch(observed)), "invalid checkpoint revision metadata"
        )
    revision = requested or observed
    if revision is None:
        raise ValueError("checkpoint revision unavailable; pass --revision")
    _require(
        revision == PINNED_REVISION,
        "checkpoint revision is not the pinned 1B base revision",
    )
    _require(observed is None or observed == revision, "checkpoint revision mismatch")
    return (
        revision,
        "download_metadata"
        if recorded
        else "snapshot_path"
        if snapshot
        else "explicit",
    )


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tensor(
    source: Any,
    name: str,
    shape: tuple[int, ...],
    mapping: mmap.mmap,
    offset: int,
    dtype: str = "bfloat16",
) -> None:
    tensor_slice = source.get_slice(name)
    _require(
        tuple(tensor_slice.get_shape()) == shape and tensor_slice.get_dtype() == "BF16",
        f"invalid BF16 tensor {name}",
    )
    tensor = source.get_tensor(name)
    _require(bool(torch.isfinite(tensor).all()), f"non-finite tensor {name}")
    raw = (
        tensor.contiguous().view(torch.uint16).numpy().tobytes()
        if dtype == "bfloat16"
        else tensor.float().contiguous().numpy().tobytes()
    )
    mapping[offset : offset + len(raw)] = raw


def pack_safetensors(
    checkpoint: Path,
    spec: StackSpec,
    target: dict,
    tile_rows: int,
    out: Path,
    revision: str | None = None,
    dtype: str = "bfloat16",
) -> dict:
    """Stream HF BF16 tensors into the existing stack ABI."""
    resolved_revision, revision_evidence = _revision(checkpoint, revision)
    names = _name_map(spec)
    manifest = _manifest(checkpoint, set(names.values()), spec.tie_word_embeddings)
    for path in set(manifest.values()):
        metadata = checkpoint / ".cache/huggingface/download" / f"{path.name}.metadata"
        if metadata.is_file():
            _require(
                metadata.read_text().splitlines()[0] == resolved_revision,
                f"checkpoint shard {path.name} has a different revision",
            )
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        plan = make_stack_plan_metadata(spec, target, tile_rows, dtype, "0" * 64)
        storage = plan["storage"]
        stack = storage["stack"]
        weight_regions = {entry["tensor"]: entry for entry in storage["weights"]}
        # WEIGHTS is in descriptor order, matching the tensor IDs in the plan.
        descriptors = {item["name"]: item["id"] for item in plan["model"]["tensors"]}
        weights_path = temporary / "weights.bin"
        with weights_path.open("w+b") as stream:
            stream.truncate(storage["weight_bytes"])
            with mmap.mmap(stream.fileno(), 0) as mapping, ExitStack() as opened:
                sources = {
                    path: opened.enter_context(
                        safe_open(path, framework="pt", device="cpu")
                    )
                    for path in set(manifest.values())
                }

                def copy(internal: str, shape: tuple[int, ...], offset: int) -> None:
                    hf_name = names[internal]
                    _copy_tensor(
                        sources[manifest[hf_name]],
                        hf_name,
                        shape,
                        mapping,
                        offset,
                        dtype,
                    )

                d = spec.block.hidden_size
                copy("embedding", (spec.vocab_size, d), 0)
                for layer in range(spec.layers):
                    base = (
                        stack["layer_weights_offset"]
                        + layer * stack["layer_weight_stride"]
                    )
                    for internal in WEIGHTS:
                        offset = base + weight_regions[descriptors[internal]]["offset"]
                        copy(
                            f"L{layer}.{internal}", _shape(spec.block, internal), offset
                        )
                copy("final_norm", (d,), stack["final_norm_offset"])
                if spec.tie_word_embeddings and "lm_head.weight" in manifest:
                    alias = sources[manifest["lm_head.weight"]].get_tensor(
                        "lm_head.weight"
                    )
                    embedding = sources[manifest[names["embedding"]]].get_tensor(
                        names["embedding"]
                    )
                    _require(
                        torch.equal(alias, embedding),
                        "tied lm_head differs from embedding",
                    )
                if not spec.tie_word_embeddings:
                    copy("lm_head", (spec.vocab_size, d), stack["lm_head_offset"])
                plan["integrity"]["weights_sha256"] = sha256(mapping).hexdigest()
                validate_stack(plan, mapping)
                mapping.flush()
        plan["source"] = {
            "repository": MODEL_ID,
            "revision": resolved_revision,
            "revision_evidence": revision_evidence,
            "config_sha256": _hash_file(checkpoint / "config.json"),
            "safetensors_sha256": {
                path.name: _hash_file(path) for path in sorted(set(manifest.values()))
            },
        }
        data = _json_bytes(plan)
        (temporary / "plan.json").write_bytes(data)
        report = {
            "numeric_policy": plan["model"]["numeric_policy"],
            "activation_dtype": dtype,
            "accumulation_dtype": "float32",
            "attention_bytes": storage["attention_bytes"],
            "plan_sha256": sha256(data).hexdigest(),
            "weights_sha256": plan["integrity"]["weights_sha256"],
            "revision": resolved_revision,
            "layers": spec.layers,
            "vocab_size": spec.vocab_size,
            "max_context": spec.block.max_context,
            "tile_rows": tile_rows,
            "weight_bytes": storage["weight_bytes"],
            "kv_bytes": storage["kv_bytes"],
        }
        (temporary / "planning_report.json").write_bytes(_json_bytes(report))
        os.rename(temporary, out)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def import_llama32(
    checkpoint: Path,
    target_path: Path,
    out: Path,
    max_context: int = 4096,
    tile_rows: int | str = "auto",
    revision: str | None = None,
    dtype: str = "bfloat16",
) -> dict:
    spec = read_llama32_config(checkpoint / "config.json", max_context)
    target = read_target(target_path)
    candidates = None
    if tile_rows == "auto":
        tile_rows, candidates = choose_tile_rows(
            spec.block, target["launch"]["worker_count"]
        )
    elif isinstance(tile_rows, str):
        tile_rows = int(tile_rows)
    report = pack_safetensors(checkpoint, spec, target, tile_rows, out, revision, dtype)
    if candidates is not None:
        report["tile_search"] = candidates
        (out / "planning_report.json").write_bytes(_json_bytes(report))
    return report
