"""File bridge and one-command driver for the future CUDA executor."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import baselines
from .check import CaseCheck, compare
from .model import Model
from .plan import build, read_package
from .probe import probe
from .reference import generate


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"executor stage failed ({command[1]}): {result.stderr.strip() or result.stdout.strip()}"
        )


def run_cases(executor: Path, package: Path, cases_path: Path, out: Path) -> dict:
    plan, _, plan_hash = read_package(package)
    with np.load(cases_path, allow_pickle=False) as cases:
        x, expected_h, expected_y = (cases[name].copy() for name in ("x", "h", "y"))
    d, h, o = (plan["model"][key] for key in ("D", "H", "O"))
    if (
        x.ndim != 2
        or x.shape[1] != d
        or expected_h.shape != (len(x), h)
        or expected_y.shape != (len(x), o)
    ):
        raise ValueError("reference cases do not match package dimensions")
    out.mkdir(parents=True, exist_ok=True)
    raw = np.ascontiguousarray(x, dtype="<f4").tobytes()
    (out / "inputs.bin").write_bytes(raw)
    _run(
        [
            str(executor),
            "--run-plan",
            str(package),
            "--inputs",
            str(out / "inputs.bin"),
            "--outputs",
            str(out / "outputs.bin"),
            "--hidden",
            str(out / "hidden.bin"),
            "--report",
            str(out / "runtime_cases.json"),
        ]
    )
    hidden_raw, output_raw = (
        (out / "hidden.bin").read_bytes(),
        (out / "outputs.bin").read_bytes(),
    )
    if len(hidden_raw) != len(x) * h * 4 or len(output_raw) != len(x) * o * 4:
        raise ValueError("executor wrote unexpected output lengths")
    hidden = np.frombuffer(hidden_raw, dtype="<f4").reshape(len(x), h)
    output = np.frombuffer(output_raw, dtype="<f4").reshape(len(x), o)
    checks: list[CaseCheck] = [
        {
            "case": i,
            "h": compare(hidden[i], expected_h[i]),
            "y": compare(output[i], expected_y[i]),
        }
        for i in range(len(x))
    ]
    report = {
        "pass": all(item[name]["pass"] for item in checks for name in ("h", "y")),
        "cases": checks,
        "plan_sha256": plan_hash,
        "input_sha256": sha256(raw).hexdigest(),
    }
    (out / "correctness.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    if not report["pass"]:
        raise ValueError("executor output failed reference comparison")
    return report


def benchmark_executor(
    executor: Path,
    package: Path,
    cases_path: Path,
    out: Path,
    iterations: int = 100,
    rounds: int = 5,
) -> dict:
    with np.load(cases_path, allow_pickle=False) as cases:
        case = np.ascontiguousarray(
            cases["x"][1 if len(cases["x"]) > 1 else 0], dtype="<f4"
        )
    raw = case.tobytes()
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_input.bin").write_bytes(raw)
    _run(
        [
            str(executor),
            "--benchmark",
            str(package),
            "--input",
            str(out / "benchmark_input.bin"),
            "--iterations",
            str(iterations),
            "--rounds",
            str(rounds),
            "--report",
            str(out / "runtime_benchmark.json"),
        ]
    )
    runtime = json.loads((out / "runtime_benchmark.json").read_text())
    if runtime.get("input_sha256") != sha256(raw).hexdigest():
        raise ValueError("executor benchmark used a different input")
    return runtime


def demo(
    out: Path,
    executor: Path | None,
    model: Model,
    seed: int,
    tile_h1: int,
    tile_h2: int,
    iterations: int,
    rounds: int,
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[3]
    if executor is None:
        cuda = repo / "tileinfer"
        for command in (
            ["cmake", "--preset", "release"],
            ["cmake", "--build", "--preset", "release"],
        ):
            subprocess.run(command, cwd=cuda, check=True)
        executor = cuda / "build" / "release" / "tile_runtime"
    fixture = out / "reference"
    generate(fixture, model, seed)
    probe(executor, out / "target.json")
    package = out / "package"
    build(
        fixture / "model.json",
        fixture / "weights.npz",
        out / "target.json",
        tile_h1,
        tile_h2,
        package,
    )
    run_cases(executor, package, fixture / "reference_cases.npz", out / "execution")
    base = baselines.run(
        package,
        fixture / "reference_cases.npz",
        out / "baselines.json",
        iterations,
        rounds,
    )
    runtime = benchmark_executor(
        executor,
        package,
        fixture / "reference_cases.npz",
        out / "execution",
        iterations,
        rounds,
    )
    report = {"correctness": "pass", "baselines": base, "runtime": runtime}
    (out / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    return report
