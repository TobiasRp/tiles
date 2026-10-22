"""Version-5 decoder package builder and independent package checks."""

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

FORMAT = 5
GENERATOR = "tileplan-decoder-0.5"
WEIGHT_DTYPES = {"float32": 4, "bfloat16": 2}
TENSOR_NAMES = (
    "x",
    "attn_norm",
    "u",
    "Wq",
    "q",
    "Wk",
    "k",
    "Wv",
    "v",
    "q_rot",
    "K",
    "V",
    "attention",
    "Wo",
    "r",
    "ffn_norm",
    "z",
    "Wgate",
    "gate",
    "Wup",
    "up",
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
    return spec.num_attention_heads * spec.max_context * 2 if dtype == "bfloat16" else 0


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
    kv = spec.num_key_value_heads * spec.head_dim
    if name in WEIGHT_SHAPES:
        return WEIGHT_SHAPES[name](spec)
    if name in ("x", "u", "q", "q_rot", "attention", "r", "z", "y"):
        return (d,)
    if name in ("k", "v"):
        return (kv,)
    if name in ("gate", "up", "product"):
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


def _graph(spec: DecoderSpec, tile_rows: int, dtype: str = "float32") -> list[dict]:
    """Canonical region ownership and minimal data dependencies."""
    d, f, dh = spec.hidden_size, spec.intermediate_size, spec.head_dim
    kv_width = spec.num_key_value_heads * dh
    tasks: list[dict] = []
    writers: dict[str, list[tuple[int, int, int]]] = {}

    def add(
        opcode: str,
        operands: dict,
        output: list[dict],
        predecessors: list[int],
        work: int,
    ) -> int:
        task_id = len(tasks)
        tasks.append(
            {
                "id": task_id,
                "opcode": opcode.replace("_f32", "_bf16")
                if dtype == "bfloat16"
                else opcode,
                "operands": operands,
                "output": output,
                "predecessors": sorted(set(predecessors)),
                "estimated_work": work,
            }
        )
        return task_id

    def owners(name: str, begin: int, end: int) -> list[int]:
        return [task_id for lo, hi, task_id in writers[name] if lo < end and begin < hi]

    def tiled_gemv(
        input_name: str,
        weight: str,
        output_name: str,
        rows: int,
        width: int,
        predecessors: list[int],
        addend: str | None = None,
    ) -> None:
        writers[output_name] = []
        for begin in range(0, rows, tile_rows):
            end = min(begin + tile_rows, rows)
            task_id = add(
                "gemv_f32",
                {
                    "input": TID[input_name],
                    "weight": TID[weight],
                    "output": TID[output_name],
                    "rows": [begin, end],
                    "addend": TID[addend] if addend else None,
                },
                [{"tensor": TID[output_name], "rows": [begin, end]}],
                predecessors,
                (end - begin) * width,
            )
            writers[output_name].append((begin, end, task_id))

    first = add(
        "rms_norm_f32",
        {
            "input": TID["x"],
            "scale": TID["attn_norm"],
            "output": TID["u"],
        },
        [{"tensor": TID["u"], "rows": [0, d]}],
        [],
        d,
    )
    writers["u"] = [(0, d, first)]
    tiled_gemv("u", "Wq", "q", d, d, [first])
    tiled_gemv("u", "Wk", "k", kv_width, d, [first])
    tiled_gemv("u", "Wv", "v", kv_width, d, [first])
    writers["q_rot"] = []
    for head in range(spec.num_attention_heads):
        begin, end = head * dh, (head + 1) * dh
        task_id = add(
            "rope_query_f32",
            {
                "input": TID["q"],
                "output": TID["q_rot"],
                "head": head,
            },
            [{"tensor": TID["q_rot"], "rows": [begin, end]}],
            owners("q", begin, end),
            dh,
        )
        writers["q_rot"].append((begin, end, task_id))
    cache_tasks = []
    for head in range(spec.num_key_value_heads):
        begin, end = head * dh, (head + 1) * dh
        task_id = add(
            "rope_cache_f32",
            {
                "key": TID["k"],
                "value": TID["v"],
                "key_cache": TID["K"],
                "value_cache": TID["V"],
                "kv_head": head,
            },
            [
                {"tensor": TID["K"], "position": "current", "head": head},
                {"tensor": TID["V"], "position": "current", "head": head},
            ],
            owners("k", begin, end) + owners("v", begin, end),
            2 * dh,
        )
        cache_tasks.append(task_id)
    writers["attention"] = []
    for head, kv_head in enumerate(spec.head_to_kv):
        begin, end = head * dh, (head + 1) * dh
        task_id = add(
            "attention_prefix_f32",
            {
                "query": TID["q_rot"],
                "key_cache": TID["K"],
                "value_cache": TID["V"],
                "output": TID["attention"],
                "head": head,
                "kv_head": kv_head,
            },
            [{"tensor": TID["attention"], "rows": [begin, end]}],
            owners("q_rot", begin, end) + [cache_tasks[kv_head]],
            2 * spec.max_context * dh,
        )
        writers["attention"].append((begin, end, task_id))
    tiled_gemv(
        "attention",
        "Wo",
        "r",
        d,
        d,
        [task_id for _, _, task_id in writers["attention"]],
        "x",
    )
    second = add(
        "rms_norm_f32",
        {
            "input": TID["r"],
            "scale": TID["ffn_norm"],
            "output": TID["z"],
        },
        [{"tensor": TID["z"], "rows": [0, d]}],
        [task_id for _, _, task_id in writers["r"]],
        d,
    )
    writers["z"] = [(0, d, second)]
    tiled_gemv("z", "Wgate", "gate", f, d, [second])
    tiled_gemv("z", "Wup", "up", f, d, [second])
    writers["product"] = []
    for begin in range(0, f, tile_rows):
        end = min(begin + tile_rows, f)
        task_id = add(
            "silu_product_f32",
            {
                "gate": TID["gate"],
                "up": TID["up"],
                "output": TID["product"],
                "rows": [begin, end],
            },
            [{"tensor": TID["product"], "rows": [begin, end]}],
            owners("gate", begin, end) + owners("up", begin, end),
            end - begin,
        )
        writers["product"].append((begin, end, task_id))
    tiled_gemv(
        "product",
        "Wdown",
        "y",
        d,
        f,
        [task_id for _, _, task_id in writers["product"]],
        "r",
    )
    return tasks


def _split(rows: int, parts: int) -> list[tuple[int, int]]:
    """Return at most ``parts`` contiguous, near-equal, nonempty row ranges."""
    count = min(rows, parts)
    return [(i * rows // count, (i + 1) * rows // count) for i in range(count)]


def _program(
    spec: DecoderSpec,
    tile_rows: int,
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
    template = _graph(spec, tile_rows, dtype)
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
            task["id"]
            for task in tasks[offset:]
            if task["operands"].get("weight") == TID["Wdown"]
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


def choose_tile_rows(spec: DecoderSpec, workers: int) -> tuple[int, list[dict]]:
    """Choose a balanced schedule with a small per-task dispatch penalty.

    This is an offline schedule heuristic, not a measured latency prediction.
    """
    candidates = []
    for tile_rows in (2, 4, 8, 16):
        tasks = _graph(spec, tile_rows)
        waves = _schedule(tasks, workers)
        critical_work = sum(max(wave["estimated_work"]) for wave in waves)
        score = critical_work + 8 * len(tasks)
        candidates.append(
            {
                "tile_rows": tile_rows,
                "task_count": len(tasks),
                "critical_work": critical_work,
                "score": score,
            }
        )
    best = min(candidates, key=lambda item: (item["score"], item["task_count"]))
    return best["tile_rows"], candidates


def make_plan(
    spec: DecoderSpec,
    weights: dict[str, np.ndarray],
    target: dict,
    tile_rows: int,
    weight_dtype: str = "float32",
) -> tuple[dict, bytes]:
    _require(type(tile_rows) is int and tile_rows > 0, "tile_rows must be positive")
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
    tasks = _program(spec, tile_rows, weight_dtype, workers)
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
            "tile_rows": tile_rows,
            "tensors": _descriptors(spec, weight_dtype),
        },
        "storage": {
            "input_bytes": spec.hidden_size * _weight_size(weight_dtype),
            "output_bytes": spec.hidden_size * _weight_size(weight_dtype),
            "scratch_bytes": scratch_bytes,
            "attention_bytes": _attention_bytes(spec, weight_dtype),
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
    tile_rows = model["tile_rows"]
    _require(type(tile_rows) is int and tile_rows > 0, "invalid tile rows")
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
        and storage["attention_bytes"] == _attention_bytes(spec, weight_dtype),
        "invalid IO storage",
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
    expected_tasks = _program(spec, tile_rows, weight_dtype, workers)
    expected_waves = _schedule(expected_tasks, workers)
    _require(
        plan["tasks"] == expected_tasks and plan["waves"] == expected_waves,
        "invalid decoder task graph, ownership, dependencies, or schedule",
    )


def build(
    model_path: Path,
    weights_path: Path,
    target_path: Path,
    tile_rows: int | str,
    out: Path,
    weight_dtype: str = "float32",
) -> dict:
    spec = read_model(model_path)
    weights = read_weights(weights_path, spec)
    target = read_target(target_path)
    candidates = None
    if tile_rows == "auto":
        tile_rows, candidates = choose_tile_rows(spec, target["launch"]["worker_count"])
    elif isinstance(tile_rows, str):
        tile_rows = int(tile_rows)
    plan, packed = make_plan(spec, weights, target, tile_rows, weight_dtype)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (temporary / "weights.bin").write_bytes(packed)
        data = _json_bytes(plan)
        (temporary / "plan.json").write_bytes(data)
        report = summary(plan, sha256(data).hexdigest())
        if candidates is not None:
            report["tile_search"] = candidates
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
            "tile_rows": plan["model"]["tile_rows"],
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
                "tile_rows",
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
        "peak_bytes": storage.get("attention_bytes", 0)
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
