"""Independent checks of the serialized MLP package contract."""

from hashlib import sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(plan: dict, packed_weights: bytes) -> None:
    header, model, target = plan["header"], plan["model"], plan["target"]
    storage, integrity = plan["storage"], plan["integrity"]
    _require(header["format_version"] == 1 and header["executor_abi"] == "tileinfer-mlp-v1"
             and header["model_kind"] == "mlp_f32" and header["activation"] == "gelu_tanh", "unsupported plan header")
    d, h, o = (model[k] for k in ("D", "H", "O"))
    t1, t2 = model["tile_h1"], model["tile_h2"]
    workers = target["worker_count"]
    _require(all(type(n) is int and n > 0 for n in (d, h, o, t1, t2, workers, target["sm_count"], target["threads_per_block"])), "invalid model or target sizes")
    _require(model["dtype"] == "float32" and workers == target["sm_count"] and target["cooperative_launch"] is True
             and isinstance(target["kernel_fingerprint"], str) and bool(target["kernel_fingerprint"]), "invalid target contract")
    shapes = {"x": [d], "W1": [h, d], "b1": [h], "h": [h], "W2": [o, h], "b2": [o], "y": [o]}
    classes = {"x": "input", "W1": "weight", "b1": "weight", "h": "scratch", "W2": "weight", "b2": "weight", "y": "output"}
    tensors = model["tensors"]
    _require(len(tensors) == 7 and [tensor["name"] for tensor in tensors] == list(shapes), "invalid tensor list")
    for tid, tensor in enumerate(tensors):
        name, shape = tensor["name"], tensor["shape"]
        strides = [1] if len(shape) == 1 else [shape[1], 1]
        _require(tensor == {"id": tid, "name": name, "shape": shapes[name], "strides": strides,
                            "dtype": "float32", "storage": classes[name]}, f"invalid tensor {name}")
    _require(storage["input_bytes"] == d * 4 and storage["output_bytes"] == o * 4
             and storage["scratch_bytes"] == h * 4 and storage["scratch"] == [{"tensor": "h", "offset": 0, "bytes": h * 4, "alignment": 4}], "invalid runtime buffers")
    _require(storage["weight_bytes"] == len(packed_weights) == integrity["weights_bytes"]
             and sha256(packed_weights).hexdigest() == integrity["weights_sha256"], "weight length or checksum mismatch")
    cursor = 0
    for entry, name in zip(storage["weights"], ("W1", "b1", "W2", "b2"), strict=True):
        size = 4
        for n in shapes[name]:
            size *= n
        offset = entry["offset"]
        _require(entry == {"tensor": name, "offset": offset, "bytes": size, "alignment": 64}
                 and type(offset) is int and offset >= cursor and offset % 64 == 0
                 and offset + size <= len(packed_weights), f"invalid weight region {name}")
        cursor = offset + size
    _require(cursor == len(packed_weights), "unexpected weight data after final entry")

    tasks, waves = plan["tasks"], plan["waves"]
    _require(len(waves) == 2 and [wave["id"] for wave in waves] == [0, 1], "MLP requires two waves")
    _require([task["id"] for task in tasks] == list(range(len(tasks))), "task IDs are not canonical")
    first = [task for task in tasks if task["wave"] == 0]
    second = [task for task in tasks if task["wave"] == 1]
    _require(len(first) == (h + t1 - 1) // t1 and len(second) == (o + t2 - 1) // t2
             and tasks == first + second, "wrong task count or order")
    for stage, stage_tasks, rows, height, names in (
        (0, first, h, t1, ("gemv_bias_gelu_f32", "x", "W1", "b1", "h")),
        (1, second, o, t2, ("gemv_bias_f32", "h", "W2", "b2", "y")),
    ):
        for index, task in enumerate(stage_tasks):
            begin, end = index * height, min((index + 1) * height, rows)
            expected = {"id": task["id"], "opcode": names[0], "wave": stage, "worker": task["worker"],
                        "rows": [begin, end], "input": names[1], "weight": names[2],
                        "bias": names[3], "output": names[4],
                        "predecessors": [producer["id"] for producer in first] if stage else []}
            _require(task == expected and type(task["worker"]) is int and 0 <= task["worker"] < workers,
                     f"invalid task {task['id']}")
    seen = []
    for stage, wave in enumerate(waves):
        lists, loads = wave["workers"], wave["estimated_work"]
        _require(len(lists) == len(loads) == workers, "incomplete worker coverage")
        width = d if stage == 0 else h
        for worker, ids in enumerate(lists):
            _require(all(type(tid) is int and 0 <= tid < len(tasks) for tid in ids), "invalid scheduled task ID")
            _require(all(tasks[tid]["wave"] == stage and tasks[tid]["worker"] == worker for tid in ids), "task/wave assignment mismatch")
            work = sum((tasks[tid]["rows"][1] - tasks[tid]["rows"][0]) * width for tid in ids)
            _require(loads[worker] == work, "incorrect worker work estimate")
            seen.extend(ids)
    _require(sorted(seen) == list(range(len(tasks))), "tasks are missing or scheduled twice")
