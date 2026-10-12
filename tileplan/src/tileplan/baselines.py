"""PyTorch eager and CUDA Graph measurements for the packaged MLP."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .check import compare
from .plan import read_package


def _weight_tensors(plan: dict, packed: bytes, device: torch.device) -> dict[str, torch.Tensor]:
    shapes = {entry["name"]: tuple(entry["shape"]) for entry in plan["model"]["tensors"]}
    weights = {}
    for entry in plan["storage"]["weights"]:
        start, size = entry["offset"], entry["bytes"]
        value = np.frombuffer(packed, dtype="<f4", count=size // 4, offset=start).copy().reshape(shapes[entry["tensor"]])
        weights[entry["tensor"]] = torch.from_numpy(value).to(device)
    return weights


def run(package: Path, cases_path: Path, out: Path, iterations: int = 100, rounds: int = 5) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for the baseline benchmark")
    if iterations < 1 or rounds < 1:
        raise ValueError("iterations and rounds must be positive")
    plan, packed, plan_hash = read_package(package)
    with np.load(cases_path, allow_pickle=False) as cases:
        x, expected_h, expected_y = (cases[name].copy() for name in ("x", "h", "y"))
    d, h, o = (plan["model"][key] for key in ("D", "H", "O"))
    if x.ndim != 2 or x.shape[1] != d or expected_h.shape != (len(x), h) or expected_y.shape != (len(x), o):
        raise ValueError("reference cases do not match the package")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    weights = _weight_tensors(plan, packed, device)

    def forward(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(value @ weights["W1"].T + weights["b1"], approximate="tanh")
        return hidden, hidden @ weights["W2"].T + weights["b2"]

    checks = []
    with torch.inference_mode():
        for idx, case in enumerate(x):
            hidden, output = forward(torch.from_numpy(case).to(device))
            checks.append({"case": idx, "h": compare(hidden.cpu().numpy(), expected_h[idx]),
                           "y": compare(output.cpu().numpy(), expected_y[idx])})
        if not all(item[name]["pass"] for item in checks for name in ("h", "y")):
            raise ValueError("PyTorch GPU baseline does not match saved reference")

        sample = torch.from_numpy(np.ascontiguousarray(x[1 if len(x) > 1 else 0])).to(device)
        for _ in range(10):
            forward(sample)
        torch.cuda.synchronize()

        static_input = torch.empty_like(sample)
        static_input.copy_(sample)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(10):
                forward(static_input)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_h, graph_y = forward(static_input)
        graph.replay()
        graph_checks = {"h": compare(graph_h.cpu().numpy(), expected_h[1 if len(x) > 1 else 0]),
                        "y": compare(graph_y.cpu().numpy(), expected_y[1 if len(x) > 1 else 0])}
        if not all(value["pass"] for value in graph_checks.values()):
            raise ValueError("PyTorch CUDA Graph replay does not match saved reference")

        def measure(call) -> float:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(iterations):
                call()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) * 1000 / iterations

        # Rotate which path runs first between rounds to reduce ordering bias.
        eager, replay = [], []
        for index in range(rounds):
            order = (("eager", lambda: forward(sample)), ("cuda_graph", graph.replay))
            if index % 2:
                order = order[::-1]
            for name, call in order:
                (eager if name == "eager" else replay).append(measure(call))

    sample_raw = np.ascontiguousarray(x[1 if len(x) > 1 else 0], dtype="<f4").tobytes()
    report = {"kind": "pytorch_baselines_only", "plan_sha256": plan_hash,
              "input_sha256": sha256(sample_raw).hexdigest(), "iterations_per_round": iterations,
              "rounds": rounds, "timing_unit": "microseconds_per_forward_batch_average",
              "eager": eager, "cuda_graph": replay, "checks": checks, "graph_check": graph_checks,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(), "float32_matmul_precision": torch.get_float32_matmul_precision(),
                              "tf32_allowed": torch.backends.cuda.matmul.allow_tf32}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
