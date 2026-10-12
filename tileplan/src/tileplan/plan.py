"""Deterministic lowering, wave scheduling, and package emission."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .model import Model, WEIGHT_NAMES, read_weights
from .probe import ABI, read_target
from .validate import validate


FORMAT = 1
GENERATOR = "tileplan-0.1"
ALIGNMENT = 64


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _tensor(tid: int, name: str, shape: tuple[int, ...], storage: str) -> dict:
    strides = [1]
    for n in reversed(shape[1:]):
        strides.insert(0, strides[0] * n)
    return {"id": tid, "name": name, "shape": list(shape), "strides": strides, "dtype": "float32", "storage": storage}


def _pack_weights(weights: dict[str, np.ndarray]) -> tuple[bytes, list[dict]]:
    data = bytearray()
    entries = []
    for name in WEIGHT_NAMES:
        data.extend(bytes((-len(data)) % ALIGNMENT))
        offset = len(data)
        raw = weights[name].astype("<f4", copy=False).tobytes(order="C")
        data.extend(raw)
        entries.append({"tensor": name, "offset": offset, "bytes": len(raw), "alignment": ALIGNMENT})
    return bytes(data), entries


def _tasks(model: Model, tile_h1: int, tile_h2: int) -> list[dict]:
    tasks = []
    for wave, (rows, height, opcode, inp, weight, bias, out) in enumerate((
        (model.h, tile_h1, "gemv_bias_gelu_f32", "x", "W1", "b1", "h"),
        (model.o, tile_h2, "gemv_bias_f32", "h", "W2", "b2", "y"),
    )):
        predecessors = list(range(len(tasks))) if wave else []
        for begin in range(0, rows, height):
            end = min(begin + height, rows)
            tasks.append({"id": len(tasks), "opcode": opcode, "wave": wave, "worker": -1,
                          "rows": [begin, end], "input": inp, "weight": weight, "bias": bias,
                          "output": out, "predecessors": predecessors.copy()})
    return tasks


def _schedule(tasks: list[dict], worker_count: int, model: Model) -> list[dict]:
    waves = []
    for wave in (0, 1):
        workers = [[] for _ in range(worker_count)]
        loads = [0] * worker_count
        stage = [t for t in tasks if t["wave"] == wave]
        width = model.d if wave == 0 else model.h
        for task in sorted(stage, key=lambda t: (-(t["rows"][1] - t["rows"][0]) * width, t["id"])):
            worker = min(range(worker_count), key=lambda i: (loads[i], i))
            task["worker"] = worker
            workers[worker].append(task["id"])
            loads[worker] += (task["rows"][1] - task["rows"][0]) * width
        waves.append({"id": wave, "workers": workers, "estimated_work": loads})
    return waves


def make_plan(model: Model, weights: dict[str, np.ndarray], target: dict, tile_h1: int, tile_h2: int) -> tuple[dict, bytes]:
    if any(type(n) is not int or n < 1 for n in (tile_h1, tile_h2)):
        raise ValueError("tile heights must be positive integers")
    worker_count = target["launch"]["worker_count"]
    packed, entries = _pack_weights(weights)
    tasks = _tasks(model, tile_h1, tile_h2)
    waves = _schedule(tasks, worker_count, model)
    tensors = [
        _tensor(0, "x", (model.d,), "input"), _tensor(1, "W1", (model.h, model.d), "weight"),
        _tensor(2, "b1", (model.h,), "weight"), _tensor(3, "h", (model.h,), "scratch"),
        _tensor(4, "W2", (model.o, model.h), "weight"), _tensor(5, "b2", (model.o,), "weight"),
        _tensor(6, "y", (model.o,), "output"),
    ]
    plan = {
        "header": {"format_version": FORMAT, "executor_abi": ABI, "generator": GENERATOR,
                   "model_kind": "mlp_f32", "activation": "gelu_tanh"},
        "target": {"compute_capability": target["device"]["compute_capability"],
                   "sm_count": target["device"]["sm_count"], "cooperative_launch": True,
                   "worker_count": worker_count, "threads_per_block": target["launch"]["threads_per_block"],
                   "kernel_fingerprint": target["executor"]["fingerprint"]},
        "model": {"D": model.d, "H": model.h, "O": model.o, "dtype": "float32",
                  "tile_h1": tile_h1, "tile_h2": tile_h2, "tensors": tensors},
        "storage": {"input_bytes": model.d * 4, "output_bytes": model.o * 4,
                    "scratch_bytes": model.h * 4, "scratch": [{"tensor": "h", "offset": 0, "bytes": model.h * 4, "alignment": 4}],
                    "weight_bytes": len(packed), "weights": entries},
        "tasks": tasks, "waves": waves,
        "integrity": {"weights_bytes": len(packed), "weights_sha256": sha256(packed).hexdigest()},
    }
    validate(plan, packed)
    return plan, packed


def build(model_path: Path, weights_path: Path, target_path: Path, tile_h1: int, tile_h2: int, out: Path) -> dict:
    model = Model.read(model_path)
    weights = read_weights(weights_path, model)
    target = read_target(target_path)
    plan, packed = make_plan(model, weights, target, tile_h1, tile_h2)
    if out.exists():
        raise FileExistsError(f"package already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (temporary / "weights.bin").write_bytes(packed)
        plan_data = json_bytes(plan)
        (temporary / "plan.json").write_bytes(plan_data)
        report = summary(plan, sha256(plan_data).hexdigest())
        (temporary / "planning_report.json").write_bytes(json_bytes(report))
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return report


def read_package(package: Path) -> tuple[dict, bytes, str]:
    raw_plan = (package / "plan.json").read_bytes()
    plan = json.loads(raw_plan)
    packed = (package / "weights.bin").read_bytes()
    validate(plan, packed)
    return plan, packed, sha256(raw_plan).hexdigest()


def summary(plan: dict, plan_sha256: str) -> dict:
    counts = {name: sum(t["opcode"] == name for t in plan["tasks"]) for name in ("gemv_bias_gelu_f32", "gemv_bias_f32")}
    return {"task_counts": counts, "wave_count": len(plan["waves"]),
            "worker_loads": [w["estimated_work"] for w in plan["waves"]],
            "idle_workers": [sum(load == 0 for load in w["estimated_work"]) for w in plan["waves"]],
            "scratch_bytes": plan["storage"]["scratch_bytes"], "weight_bytes": plan["storage"]["weight_bytes"],
            "peak_bytes": plan["storage"]["input_bytes"] + plan["storage"]["output_bytes"] + plan["storage"]["scratch_bytes"] + plan["storage"]["weight_bytes"],
            "plan_sha256": plan_sha256, "weights_sha256": plan["integrity"]["weights_sha256"],
            "target": plan["target"]}
