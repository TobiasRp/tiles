"""Planning primitives shared by every decoder package kind.

A package is a plan (header, target, model, storage, tasks, waves, integrity)
plus packed weights. This module holds what block and stack packages have in
common: the numeric policy, tensor layout and packing, the per-layer task graph
with its wave schedule, the plan sections built from those, and package I/O.
"""

import copy
import json
import mmap
import os
import shutil
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .decoder_reference import WEIGHT_SHAPES, DecoderSpec
from .probe import ABI

FORMAT = 7
GENERATOR = "tileplan-decoder-0.7"
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


# Numeric policy and sizes


def weight_size(dtype: str) -> int:
    require(dtype in WEIGHT_DTYPES, "unsupported weight dtype")
    return WEIGHT_DTYPES[dtype]


def numeric_policy(dtype: str) -> dict:
    weight_size(dtype)
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


def attention_capacity(spec: DecoderSpec, dtype: str, target: dict) -> int:
    """Most time steps one attention task holds: the probabilities of all its
    query heads must fit the compiled worker's shared attention region."""
    group = spec.num_attention_heads // spec.num_key_value_heads
    return target["attention_shared_bytes"] // (group * weight_size(dtype))


# Longest prefix that one fused attention task per K/V head handles by
# default: the S2b probe's crossover against two split chunks on the RTX 3050.
FUSED_STEPS = 384


def attention_chunking(
    spec: DecoderSpec,
    dtype: str,
    target: dict,
    steps: int | None = None,
    fused: int | None = None,
) -> dict:
    """Chunk tasks per K/V head, the most time steps each covers, and the
    longest prefix handled by one fused task.

    The worker runs a prefix of at most ``fused`` steps as one fused task per
    K/V head, which also skips the barrier before the values wave. A longer
    prefix splits evenly into the fewest chunks of at most ``steps`` time
    steps, but at least two. ``steps`` defaults to the shared-memory capacity;
    the program holds the chunk tasks of a ``max_context`` prefix.
    """
    capacity = attention_capacity(spec, dtype, target)
    steps = capacity if steps is None else steps
    require(
        type(steps) is int and 0 < steps <= capacity,
        f"attention chunk steps must be in [1, {capacity}]",
    )
    fused = min(FUSED_STEPS, steps) if fused is None else fused
    require(
        type(fused) is int and 0 < fused <= steps,
        "attention fused steps must be in [1, attention chunk steps]",
    )
    longest = spec.max_context
    return {
        "attention_chunks": 1 if longest <= fused else max(2, -(-longest // steps)),
        "attention_chunk_steps": steps,
        "attention_fused_steps": fused,
    }


def block_spec(source: dict) -> DecoderSpec:
    """Rebuild a block spec from a reference fixture or a plan's model section."""
    spec = DecoderSpec(
        **{
            field.name: source[field.name]
            for field in fields(DecoderSpec)
            if field.name != "dtype"
        },
        dtype=source.get("dtype", "float32"),
    )
    require(source["head_to_kv"] == spec.head_to_kv, "head mapping mismatch")
    return spec


# Tensor layout and packing


def _strides(shape: tuple[int, ...]) -> list[int]:
    result = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        result[index] = result[index + 1] * shape[index + 1]
    return result


def tensor_shape(spec: DecoderSpec, name: str) -> tuple[int, ...]:
    if name in WEIGHT_SHAPES:
        return WEIGHT_SHAPES[name](spec)
    if name in ("x", "q_rot", "attention", "r", "y"):
        return (spec.hidden_size,)
    if name == "product":
        return (spec.intermediate_size,)
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


def descriptors(spec: DecoderSpec, dtype: str) -> list[dict]:
    weight_size(dtype)
    return [
        {
            "id": TID[name],
            "name": name,
            "shape": list(tensor_shape(spec, name)),
            "strides": _strides(tensor_shape(spec, name)),
            "dtype": dtype,
            "storage": _storage_class(name),
        }
        for name in TENSOR_NAMES
    ]


def regions(
    spec: DecoderSpec, names: tuple[str, ...], alignment: int, element_bytes: int
) -> tuple[list[dict], int]:
    """Aligned byte ranges for ``names`` and the total size they span."""
    cursor = 0
    entries = []
    for name in names:
        cursor += (-cursor) % alignment
        size = element_bytes * int(np.prod(tensor_shape(spec, name)))
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


def align(value: int, alignment: int = 64) -> int:
    return value + (-value) % alignment


def check_weight(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    require(
        value.shape == shape
        and value.dtype == np.float32
        and bool(np.isfinite(value).all()),
        f"invalid weight {name}",
    )


def pack_array(value: np.ndarray, dtype: str) -> bytes:
    if dtype == "float32":
        return np.ascontiguousarray(value, dtype="<f4").tobytes()
    weight_size(dtype)
    bits = torch.from_numpy(np.ascontiguousarray(value)).to(torch.bfloat16)
    return np.ascontiguousarray(bits.view(torch.uint16).numpy(), dtype="<u2").tobytes()


def pack_weights(
    spec: DecoderSpec, weights: dict[str, np.ndarray], dtype: str
) -> bytes:
    """One layer's weights, laid out as ``regions(spec, WEIGHTS, 64, ...)``."""
    entries, size = regions(spec, WEIGHTS, 64, weight_size(dtype))
    packed = bytearray(size)
    for entry, name in zip(entries, WEIGHTS, strict=True):
        raw = pack_array(weights[name], dtype)
        packed[entry["offset"] : entry["offset"] + entry["bytes"]] = raw
    return bytes(packed)


# Task graph and schedule


def _suffix(dtype: str) -> str:
    return "_bf16" if dtype == "bfloat16" else "_f32"


def _split(rows: int, parts: int) -> list[tuple[int, int]]:
    """Return at most ``parts`` contiguous, near-equal, nonempty row ranges."""
    count = min(rows, parts)
    return [(i * rows // count, (i + 1) * rows // count) for i in range(count)]


def _graph(spec: DecoderSpec, workers: int, chunking: dict, dtype: str) -> list[dict]:
    """One layer's tasks with their minimal data dependencies.

    Projections read a whole input vector, so each depends on every producer
    of that vector. Q/K/V tasks own one head, so RoPE and the cache write
    fit in the same task. Attention splits every K/V head's context into
    chunks of at most ``attention_chunk_steps`` time steps; the remaining
    projections split their rows evenly over workers.
    """
    d, f, dh = spec.hidden_size, spec.intermediate_size, spec.head_dim
    nq, nkv = spec.num_attention_heads, spec.num_key_value_heads
    group = nq // nkv
    suffix = _suffix(dtype)
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

    chunks = chunking["attention_chunks"]
    steps = min(chunking["attention_chunk_steps"], spec.max_context)
    chunk_work = 2 * steps * dh * group
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


def _program(
    spec: DecoderSpec,
    dtype: str,
    workers: int,
    chunking: dict,
    layers: int,
    vocab_size: int | None,
) -> list[dict]:
    """Whole-token task graph in topological order.

    A stack (``vocab_size`` set) starts with embedding tasks and ends with final
    normalization and logits tasks, each split into one row range per worker.
    Every layer repeats the block graph; its first normalization depends on the
    previous layer's output tasks. Layer tasks name their layer so the executor
    selects that layer's weights, K/V, and alternating hidden buffers.
    """
    suffix = _suffix(dtype)
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
    template = _graph(spec, workers, chunking, dtype)
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


def _schedule(tasks: list[dict], workers: int) -> list[dict]:
    """Assign each task a wave and worker; returns the waves."""
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


def _mark_conditional_barriers(
    tasks: list[dict], waves: list[dict], fused: int
) -> None:
    """Mark the barrier between each scores wave and its values wave.

    While the visible prefix has at most ``fused`` time steps, scores tasks
    finish attention alone and values tasks do nothing, so every worker skips
    that barrier at the same positions.
    """

    def only(wave: dict, opcode: str) -> bool:
        ids = [task for lane in wave["workers"] for task in lane]
        return bool(ids) and all(tasks[i]["opcode"].startswith(opcode) for i in ids)

    for wave, following in zip(waves, waves[1:]):
        if only(wave, "attention_scores") and only(following, "attention_values"):
            wave["conditional_barrier"] = {"skip_up_to_length": fused}


def scheduled_program(
    spec: DecoderSpec,
    dtype: str,
    workers: int,
    chunking: dict,
    layers: int = 1,
    vocab_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """The whole-token tasks, each annotated with its wave, and the waves."""
    tasks = _program(spec, dtype, workers, chunking, layers, vocab_size)
    waves = _schedule(tasks, workers)
    _mark_conditional_barriers(tasks, waves, chunking["attention_fused_steps"])
    return tasks, waves


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
        local = {
            "id": wave["id"] - first,
            "workers": lanes,
            "estimated_work": [
                sum(tasks[i]["estimated_work"] for i in lane) for lane in lanes
            ],
        }
        if "conditional_barrier" in wave:
            local["conditional_barrier"] = wave["conditional_barrier"]
        waves.append(local)
    return {**plan, "tasks": tasks, "waves": waves}


# Plan sections


def plan_target(probe: dict) -> dict:
    """The plan's record of a probed launch target."""
    return {
        "compute_capability": probe["device"]["compute_capability"],
        "sm_count": probe["device"]["sm_count"],
        "cooperative_launch": True,
        "worker_count": probe["launch"]["worker_count"],
        "threads_per_block": probe["launch"]["threads_per_block"],
        "kernel_fingerprint": probe["executor"]["fingerprint"],
        "attention_shared_bytes": probe["executor"]["attention_shared_bytes"],
    }


def validate_target(target: dict) -> None:
    workers = target["worker_count"]
    require(
        type(workers) is int
        and workers > 0
        and workers == target["sm_count"]
        and target["cooperative_launch"] is True
        and type(target["threads_per_block"]) is int
        and target["threads_per_block"] > 0
        and isinstance(target["kernel_fingerprint"], str)
        and bool(target["kernel_fingerprint"])
        and type(target["attention_shared_bytes"]) is int
        and target["attention_shared_bytes"] > 0,
        "invalid target",
    )


def model_section(
    model_fields: dict, block: DecoderSpec, dtype: str, chunking: dict
) -> dict:
    """Model fields (without the reference dtype) plus policy, attention
    chunking, and tensors."""
    return {
        **{key: value for key, value in model_fields.items() if key != "dtype"},
        **numeric_policy(dtype),
        **chunking,
        "tensors": descriptors(block, dtype),
    }


def storage_section(
    block: DecoderSpec,
    dtype: str,
    chunks: int,
    layers: int,
    input_bytes: int,
    output_bytes: int,
    weight_bytes: int | None = None,
) -> dict:
    """Buffers and regions; ``weight_bytes`` defaults to one layer's weights."""
    size = weight_size(dtype)
    weights, layer_weight_bytes = regions(block, WEIGHTS, 64, size)
    scratch, scratch_bytes = regions(block, SCRATCH, 16, size)
    kv, layer_kv_bytes = regions(block, ("K", "V"), 16, size)
    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "scratch_bytes": scratch_bytes,
        # Scores, then probabilities, for every query head and visible position.
        "attention_bytes": block.num_attention_heads * block.max_context * size,
        # FP32 partial P*V plus a maximum and a sum per query head and chunk.
        "attention_partial_bytes": chunks
        * block.num_attention_heads
        * (block.head_dim + 2)
        * 4,
        "scratch": scratch,
        "kv_bytes": layer_kv_bytes * layers,
        "kv": kv,
        "weight_bytes": layer_weight_bytes if weight_bytes is None else weight_bytes,
        "weights": weights,
    }


def assemble_plan(
    kind: str,
    target: dict,
    model: dict,
    storage: dict,
    program: tuple[list[dict], list[dict]],
    weights_sha256: str,
) -> dict:
    tasks, waves = program
    return {
        "header": {
            "format_version": FORMAT,
            "executor_abi": ABI,
            "generator": GENERATOR,
            "model_kind": kind,
        },
        "target": target,
        "model": model,
        "storage": storage,
        "tasks": tasks,
        "waves": waves,
        "integrity": {
            "weights_bytes": storage["weight_bytes"],
            "weights_sha256": weights_sha256,
        },
    }


def validate_plan(
    plan: dict, packed: bytes | mmap.mmap, rebuild: Callable[[str], dict]
) -> None:
    """Check ``plan`` against the plan its own declarations must produce.

    ``rebuild`` returns that plan for the weights' checksum, so any edited
    region, task, dependency, or schedule differs from it. Provenance under
    ``source`` is not part of the planned program.
    """
    digest = sha256(packed).hexdigest()
    require(
        plan["integrity"] == {"weights_bytes": len(packed), "weights_sha256": digest},
        "weight length or checksum mismatch",
    )
    require(
        {key: value for key, value in plan.items() if key != "source"}
        == rebuild(digest),
        "invalid decoder plan: declarations, task graph, or schedule differ "
        "from the planner's",
    )


def validate_header(plan: dict, kind: str) -> None:
    header = plan["header"]
    require(
        header["format_version"] == FORMAT
        and header["executor_abi"] == ABI
        and header["model_kind"] == kind,
        "unsupported decoder header; regenerate the package with the current planner",
    )


# Package I/O


@contextmanager
def new_package_dir(out: Path) -> Generator[Path]:
    """A scratch directory that becomes ``out`` only if the block completes."""
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        yield temporary
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def write_package(
    out: Path,
    plan: dict,
    packed: bytes,
    summarize: Callable[[dict, str], dict],
) -> dict:
    data = json_bytes(plan)
    report = summarize(plan, sha256(data).hexdigest())
    with new_package_dir(out) as directory:
        (directory / "weights.bin").write_bytes(packed)
        (directory / "plan.json").write_bytes(data)
        (directory / "planning_report.json").write_bytes(json_bytes(report))
    return report
