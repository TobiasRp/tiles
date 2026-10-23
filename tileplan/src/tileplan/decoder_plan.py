"""Version-6 decoder package builder and independent package checks."""

import copy
import json
import os
import shutil
import tempfile
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .decoder_reference import WEIGHT_SHAPES, DecoderSpec
from .probe import ABI, read_target

FORMAT = 6
GENERATOR = "tileplan-decoder-0.6"
WEIGHT_DTYPES = {"float32": 4, "bfloat16": 2}
# Weights keep this order inside each layer's block of weights.bin.
TENSOR_NAMES = (
    "x",
    "attn_norm",
    "Wq",
    "Wk",
    "Wv",
    "q_rot",
    "K",
    "V",
    "attention",
    "Wo",
    "r",
    "ffn_norm",
    "Wgate",
    "Wup",
    "product",
    "Wdown",
    "y",
)
TID = {name: index for index, name in enumerate(TENSOR_NAMES)}
WEIGHTS = tuple(name for name in TENSOR_NAMES if name in WEIGHT_SHAPES)
SCRATCH = tuple(
    name for name in TENSOR_NAMES if name not in (*WEIGHTS, "x", "y", "K", "V")
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numeric_policy(dtype: str) -> dict:
    _weight_size(dtype)
    return {
        "numeric_policy": "hf-llama-eager-bf16-v1"
        if dtype == "bfloat16"
        else "decoder-f32-v1",
        "weight_dtype": dtype,
        "activation_dtype": dtype,
        "kv_dtype": dtype,
        "accumulation_dtype": "float32",
        "logits_compute_dtype": dtype,
        "api_output_dtype": "float32",
    }


def _model_fields(model: dict, dtype: str) -> dict:
    return {
        **{k: v for k, v in model.items() if k != "dtype"},
        **numeric_policy(dtype),
    }


def _validate_policy(model: dict) -> str:
    dtype = model.get("weight_dtype", "float32")
    expected = numeric_policy(dtype)
    _require(
        "dtype" not in model
        and "matrix_kernel" not in model
        and all(model.get(k) == v for k, v in expected.items()),
        "inconsistent numeric policy",
    )
    return dtype


def _attention_bytes(spec: DecoderSpec, dtype: str) -> int:
    """Scores, then probabilities, for every query head and visible position."""
    return spec.num_attention_heads * spec.max_context * _weight_size(dtype)


def _attention_partial_bytes(spec: DecoderSpec, chunks: int) -> int:
    """FP32 partial P*V plus a maximum and a sum per query head and chunk."""
    return chunks * spec.num_attention_heads * (spec.head_dim + 2) * 4


def attention_chunks(spec: DecoderSpec, workers: int) -> int:
    """Context chunks per K/V head: the fewest that give every worker equal work."""
    return next(
        chunks
        for chunks in range(1, workers + 1)
        if spec.num_key_value_heads * chunks % workers == 0
    )


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _strides(shape: tuple[int, ...]) -> list[int]:
    result = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        result[index] = result[index + 1] * shape[index + 1]
    return result


def _shape(spec: DecoderSpec, name: str) -> tuple[int, ...]:
    d = spec.hidden_size
    f = spec.intermediate_size
    if name in WEIGHT_SHAPES:
        return WEIGHT_SHAPES[name](spec)
    if name in ("x", "q_rot", "attention", "r", "y"):
        return (d,)
    if name == "product":
        return (f,)
    if name in ("K", "V"):
        return (1, spec.max_context, spec.num_key_value_heads, spec.head_dim)
    raise ValueError(f"unknown tensor {name}")


def _storage_class(name: str) -> str:
    if name in WEIGHTS:
        return "weight"
    if name in ("K", "V"):
        return "kv"
    if name == "x":
        return "input"
    if name == "y":
        return "output"
    return "scratch"


def _weight_size(weight_dtype: str) -> int:
    _require(weight_dtype in WEIGHT_DTYPES, "unsupported weight dtype")
    return WEIGHT_DTYPES[weight_dtype]


def _descriptors(spec: DecoderSpec, weight_dtype: str = "float32") -> list[dict]:
    _weight_size(weight_dtype)
    return [
        {
            "id": TID[name],
            "name": name,
            "shape": list(_shape(spec, name)),
            "strides": _strides(_shape(spec, name)),
            "dtype": weight_dtype,
            "storage": _storage_class(name),
        }
        for name in TENSOR_NAMES
    ]


def _regions(
    spec: DecoderSpec, names: tuple[str, ...], alignment: int, element_bytes: int = 4
) -> tuple[list[dict], int]:
    cursor = 0
    entries = []
    for name in names:
        cursor += (-cursor) % alignment
        size = element_bytes * int(np.prod(_shape(spec, name)))
        entries.append(
            {
                "tensor": TID[name],
                "offset": cursor,
                "bytes": size,
                "alignment": alignment,
            }
        )
        cursor += size
    return entries, cursor


def _pack_array(value: np.ndarray, weight_dtype: str) -> bytes:
    if weight_dtype == "float32":
        return np.ascontiguousarray(value, dtype="<f4").tobytes()
    _weight_size(weight_dtype)
    bits = torch.from_numpy(np.ascontiguousarray(value)).to(torch.bfloat16)
    return np.ascontiguousarray(bits.view(torch.uint16).numpy(), dtype="<u2").tobytes()


def _pack_weights(
    spec: DecoderSpec, weights: dict[str, np.ndarray], weight_dtype: str = "float32"
) -> tuple[bytes, list[dict]]:
    entries, size = _regions(spec, WEIGHTS, 64, _weight_size(weight_dtype))
    packed = bytearray(size)
    for entry, name in zip(entries, WEIGHTS, strict=True):
        raw = _pack_array(weights[name], weight_dtype)
        packed[entry["offset"] : entry["offset"] + entry["bytes"]] = raw
    return bytes(packed), entries


def read_model(path: Path) -> DecoderSpec:
    document = json.loads(path.read_text())
    _require(
        document.get("kind") == "llama_decoder_block_reference"
        and document.get("version") == 1,
        "unsupported decoder model fixture",
    )
    names = {field.name for field in fields(DecoderSpec)}
    spec = DecoderSpec(**{name: document[name] for name in names})
    _require(document["head_to_kv"] == spec.head_to_kv, "head mapping mismatch")
    return spec


def read_weights(path: Path, spec: DecoderSpec) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        _require(set(source.files) == set(WEIGHTS), "weight names mismatch")
        weights = {name: source[name].copy() for name in WEIGHTS}
    for name, value in weights.items():
        _require(
            value.shape == _shape(spec, name)
            and value.dtype == np.float32
            and bool(np.isfinite(value).all()),
            f"invalid weight {name}",
        )
    return weights


def _graph(spec: DecoderSpec, workers: int, chunks: int, dtype: str) -> list[dict]:
    """One layer's tasks with their minimal data dependencies.

    Projections read a whole input vector, so each depends on every producer
    of that vector. Q/K/V tasks own one head, so RoPE and the cache write
    fit in the same task. Attention splits every K/V head's context into
    chunks; the remaining projections split their rows evenly over workers.
    """
    d, f, dh = spec.hidden_size, spec.intermediate_size, spec.head_dim
    nq, nkv = spec.num_attention_heads, spec.num_key_value_heads
    group = nq // nkv
    suffix = "_bf16" if dtype == "bfloat16" else "_f32"
    tasks: list[dict] = []

    def add(
        opcode: str, operands: dict, output: list, predecessors: list, work: int
    ) -> int:
        tasks.append(
            {
                "id": len(tasks),
                "opcode": opcode + suffix,
                "operands": operands,
                "output": output,
                "predecessors": sorted(set(predecessors)),
                "estimated_work": work,
            }
        )
        return len(tasks) - 1

    def rows(tensor: str, begin: int, end: int) -> dict:
        return {"tensor": TID[tensor], "rows": [begin, end]}

    # Q, K and V heads in the order of the concatenated [Q; K; V] projection.
    heads = []
    for index in range(nq + 2 * nkv):
        begin = index * dh
        if index < nq:
            output = [rows("q_rot", begin, begin + dh)]
        else:
            cache = "K" if index < nq + nkv else "V"
            head = (index - nq) % nkv
            output = [{"tensor": TID[cache], "position": "current", "head": head}]
        heads.append(add("qkv", {"rows": [begin, begin + dh]}, output, [], dh * d))

    chunk_work = 2 * spec.max_context * dh * group // chunks
    scores, values = [], []
    for kv_head in range(nkv):
        queries = [kv_head * group, (kv_head + 1) * group]
        inputs = heads[queries[0] : queries[1]]
        inputs += [heads[nq + kv_head], heads[nq + nkv + kv_head]]
        group_scores = [
            add(
                "attention_scores",
                {"kv_head": kv_head, "chunk": chunk},
                [{"workspace": "scores", "chunk": chunk, "heads": queries}],
                inputs,
                chunk_work,
            )
            for chunk in range(chunks)
        ]
        scores += group_scores
        values += [
            add(
                "attention_values",
                {"kv_head": kv_head, "chunk": chunk},
                [{"workspace": "partials", "chunk": chunk, "heads": queries}],
                group_scores,
                chunk_work,
            )
            for chunk in range(chunks)
        ]
    attention = [
        add(
            "output_projection",
            {"rows": [begin, end]},
            [rows("r", begin, end)] + ([rows("attention", 0, d)] if begin == 0 else []),
            values,
            (end - begin) * d,
        )
        for begin, end in _split(d, workers)
    ]
    product = [
        add(
            "gate_up",
            {"rows": [begin, end]},
            [rows("product", begin, end)],
            attention,
            2 * (end - begin) * d,
        )
        for begin, end in _split(f, workers)
    ]
    for begin, end in _split(d, workers):
        add(
            "down",
            {"rows": [begin, end]},
            [rows("y", begin, end)],
            product,
            (end - begin) * f,
        )
    return tasks


def _split(rows: int, parts: int) -> list[tuple[int, int]]:
    """Return at most ``parts`` contiguous, near-equal, nonempty row ranges."""
    count = min(rows, parts)
    return [(i * rows // count, (i + 1) * rows // count) for i in range(count)]


def _program(
    spec: DecoderSpec,
    dtype: str,
    workers: int,
    layers: int = 1,
    vocab_size: int | None = None,
) -> list[dict]:
    """Whole-token task graph in topological order.

    A stack (``vocab_size`` set) starts with embedding tasks and ends with final
    normalization and logits tasks, each split into one row range per worker.
    Every layer repeats the block graph; its first normalization depends on the
    previous layer's output tasks. Layer tasks name their layer so the executor
    selects that layer's weights, K/V, and alternating hidden buffers.
    """
    suffix = "_bf16" if dtype == "bfloat16" else "_f32"
    d = spec.hidden_size
    tasks: list[dict] = []

    def stage(
        opcode: str, name: str, rows: int, predecessors: list[int], work
    ) -> list[int]:
        ids = []
        for begin, end in _split(rows, workers):
            ids.append(len(tasks))
            tasks.append(
                {
                    "id": len(tasks),
                    "layer": None,
                    "opcode": opcode + suffix,
                    "operands": {"rows": [begin, end]},
                    "output": [{"stage": name, "rows": [begin, end]}],
                    "predecessors": sorted(predecessors),
                    "estimated_work": work(begin, end),
                }
            )
        return ids

    previous = (
        stage("embedding", "embedding", d, [], lambda begin, end: end - begin)
        if vocab_size is not None
        else []
    )
    template = _graph(spec, workers, attention_chunks(spec, workers), dtype)
    for layer in range(layers):
        offset = len(tasks)
        for task in template:
            repeated = copy.deepcopy(task)
            repeated["id"] += offset
            repeated["layer"] = layer
            repeated["predecessors"] = sorted(
                [index + offset for index in task["predecessors"]]
                if task["predecessors"]
                else previous
            )
            tasks.append(repeated)
        previous = [
            task["id"] for task in tasks[offset:] if task["opcode"].startswith("down")
        ]
    if vocab_size is not None:
        normalized = stage(
            "final_rms_norm",
            "final_norm",
            d,
            previous,
            lambda begin, end: d + end - begin,
        )
        stage(
            "logits",
            "logits",
            vocab_size,
            normalized,
            lambda begin, end: (end - begin) * d,
        )
    return tasks


def layer_view(plan: dict, layer: int = 0) -> dict:
    """Return a copy of ``plan`` restricted to one layer, renumbered from zero.

    Every layer of the current planner occupies its own consecutive waves, so
    the view shows the repeated block schedule for display and inspection.
    """
    selected = [task for task in plan["tasks"] if task["layer"] == layer]
    ids = {task["id"]: index for index, task in enumerate(selected)}
    first = min(task["wave"] for task in selected)
    last = max(task["wave"] for task in selected)
    tasks = []
    for task in selected:
        local = copy.deepcopy(task)
        local["id"] = ids[task["id"]]
        local["wave"] -= first
        local["predecessors"] = [ids[p] for p in task["predecessors"] if p in ids]
        tasks.append(local)
    waves = []
    for wave in plan["waves"][first : last + 1]:
        lanes = [[ids[i] for i in lane if i in ids] for lane in wave["workers"]]
        waves.append(
            {
                "id": wave["id"] - first,
                "workers": lanes,
                "estimated_work": [
                    sum(tasks[i]["estimated_work"] for i in lane) for lane in lanes
                ],
            }
        )
    return {**plan, "tasks": tasks, "waves": waves}


def _schedule(tasks: list[dict], workers: int) -> list[dict]:
    buckets: list[list[int]] = []
    for task in tasks:
        wave = (
            0
            if not task["predecessors"]
            else 1 + max(tasks[previous]["wave"] for previous in task["predecessors"])
        )
        task["wave"] = wave
        while len(buckets) <= wave:
            buckets.append([])
        buckets[wave].append(task["id"])
    result = []
    for wave, ids in enumerate(buckets):
        lanes: list[list[int]] = [[] for _ in range(workers)]
        loads = [0] * workers
        for task_id in sorted(
            ids, key=lambda index: (-tasks[index]["estimated_work"], index)
        ):
            worker = min(range(workers), key=lambda index: (loads[index], index))
            tasks[task_id]["worker"] = worker
            lanes[worker].append(task_id)
            loads[worker] += tasks[task_id]["estimated_work"]
        result.append({"id": wave, "workers": lanes, "estimated_work": loads})
    return result


def make_plan(
    spec: DecoderSpec,
    weights: dict[str, np.ndarray],
    target: dict,
    weight_dtype: str = "float32",
) -> tuple[dict, bytes]:
    _require(set(weights) == set(WEIGHTS), "weight names mismatch")
    for name, value in weights.items():
        _require(
            value.shape == _shape(spec, name)
            and value.dtype == np.float32
            and bool(np.isfinite(value).all()),
            f"invalid weight {name}",
        )
    packed, weight_regions = _pack_weights(spec, weights, weight_dtype)
    scratch_regions, scratch_bytes = _regions(
        spec, SCRATCH, 16, _weight_size(weight_dtype)
    )
    kv_regions, kv_bytes = _regions(spec, ("K", "V"), 16, _weight_size(weight_dtype))
    workers = target["launch"]["worker_count"]
    chunks = attention_chunks(spec, workers)
    tasks = _program(spec, weight_dtype, workers)
    waves = _schedule(tasks, workers)
    plan = {
        "header": {
            "format_version": FORMAT,
            "executor_abi": ABI,
            "generator": GENERATOR,
            "model_kind": "llama_decoder_block",
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
            "layers": 1,
            "attention_chunks": chunks,
            "tensors": _descriptors(spec, weight_dtype),
        },
        "storage": {
            "input_bytes": spec.hidden_size * _weight_size(weight_dtype),
            "output_bytes": spec.hidden_size * _weight_size(weight_dtype),
            "scratch_bytes": scratch_bytes,
            "attention_bytes": _attention_bytes(spec, weight_dtype),
            "attention_partial_bytes": _attention_partial_bytes(spec, chunks),
            "scratch": scratch_regions,
            "kv_bytes": kv_bytes,
            "kv": kv_regions,
            "weight_bytes": len(packed),
            "weights": weight_regions,
        },
        "tasks": tasks,
        "waves": waves,
        "integrity": {
            "weights_bytes": len(packed),
            "weights_sha256": sha256(packed).hexdigest(),
        },
    }
    validate(plan, packed)
    return plan, packed


def validate(plan: dict, packed: bytes) -> None:
    """Validate every declaration before a package is handed to the runtime."""
    header, target, model, storage = (
        plan[name] for name in ("header", "target", "model", "storage")
    )
    _require(
        header["format_version"] == FORMAT
        and header["executor_abi"] == ABI
        and header["model_kind"] == "llama_decoder_block",
        "unsupported decoder header; regenerate the package with the current planner",
    )
    weight_dtype = _validate_policy(model)
    weight_bytes = _weight_size(weight_dtype)
    names = {field.name for field in fields(DecoderSpec)}
    spec = DecoderSpec(**{name: model[name] for name in names if name != "dtype"})
    _require(
        model["head_to_kv"] == spec.head_to_kv and model["layers"] == 1,
        "unsupported model/head mapping",
    )
    _require(
        model["tensors"] == _descriptors(spec, weight_dtype),
        "invalid tensor descriptors",
    )
    workers = target["worker_count"]
    _require(
        type(workers) is int
        and workers > 0
        and workers == target["sm_count"]
        and target["cooperative_launch"] is True
        and type(target["threads_per_block"]) is int
        and target["threads_per_block"] > 0
        and isinstance(target["kernel_fingerprint"], str)
        and bool(target["kernel_fingerprint"]),
        "invalid target",
    )
    _require(
        storage["input_bytes"] == spec.hidden_size * weight_bytes
        and storage["output_bytes"] == spec.hidden_size * weight_bytes
        and storage["attention_bytes"] == _attention_bytes(spec, weight_dtype)
        and model["attention_chunks"] == attention_chunks(spec, workers)
        and storage["attention_partial_bytes"]
        == _attention_partial_bytes(spec, model["attention_chunks"]),
        "invalid IO or attention storage",
    )
    for names_in_class, key, alignment in (
        (WEIGHTS, "weights", 64),
        (SCRATCH, "scratch", 16),
        (("K", "V"), "kv", 16),
    ):
        expected, size = _regions(
            spec,
            names_in_class,
            alignment,
            weight_bytes,
        )
        _require(
            storage[key] == expected
            and storage[key[:-1] + "_bytes" if key == "weights" else key + "_bytes"]
            == size,
            f"invalid {key} storage",
        )
    _require(
        plan["integrity"]
        == {"weights_bytes": len(packed), "weights_sha256": sha256(packed).hexdigest()},
        "weight length or checksum mismatch",
    )
    expected_tasks = _program(spec, weight_dtype, workers)
    expected_waves = _schedule(expected_tasks, workers)
    _require(
        plan["tasks"] == expected_tasks and plan["waves"] == expected_waves,
        "invalid decoder task graph, ownership, dependencies, or schedule",
    )


def build(
    model_path: Path,
    weights_path: Path,
    target_path: Path,
    out: Path,
    weight_dtype: str = "float32",
) -> dict:
    spec = read_model(model_path)
    weights = read_weights(weights_path, spec)
    target = read_target(target_path)
    plan, packed = make_plan(spec, weights, target, weight_dtype)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (temporary / "weights.bin").write_bytes(packed)
        data = _json_bytes(plan)
        (temporary / "plan.json").write_bytes(data)
        report = summary(plan, sha256(data).hexdigest())
        (temporary / "planning_report.json").write_bytes(_json_bytes(report))
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def read_package(package: Path) -> tuple[dict, bytes, str]:
    data = (package / "plan.json").read_bytes()
    plan = json.loads(data)
    packed = (package / "weights.bin").read_bytes()
    if plan.get("header", {}).get("model_kind") == "llama_decoder_stack":
        from .decoder_stack import validate_stack

        validate_stack(plan, packed)
    else:
        validate(plan, packed)
    return plan, packed, sha256(data).hexdigest()


def summary(plan: dict, digest: str) -> dict:
    storage = plan["storage"]
    waves = plan["waves"]
    weight_dtype = plan["model"].get("weight_dtype", "float32")
    if plan["header"]["model_kind"] == "llama_decoder_stack":
        return {
            **numeric_policy(weight_dtype),
            "attention_bytes": storage.get("attention_bytes", 0),
            "plan_sha256": digest,
            "weights_sha256": plan["integrity"]["weights_sha256"],
            "layers": plan["model"]["layers"],
            "vocab_size": plan["model"]["vocab_size"],
            "attention_chunks": plan["model"]["attention_chunks"],
            "task_count": len(plan["tasks"]),
            "wave_count": len(waves),
            "layer_wave_count": len(layer_view(plan)["waves"]),
            "scratch_bytes": storage["scratch_bytes"],
            "kv_bytes": storage["kv_bytes"],
            "weight_bytes": storage["weight_bytes"],
            "worker_loads": [wave["estimated_work"] for wave in waves],
        }
    return {
        **numeric_policy(weight_dtype),
        "attention_bytes": storage.get("attention_bytes", 0),
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
