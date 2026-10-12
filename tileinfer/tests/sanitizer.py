"""Run GPU memory, race, and synchronization checks on a generated plan."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tileplan.check import compare
from tileplan.model import Model
from tileplan.plan import build
from tileplan.probe import probe
from tileplan.reference import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="tileinfer-sanitizer-") as directory:
        root = Path(directory)
        model = Model(33, 17, 9)
        reference = root / "reference"
        generate(reference, model, seed=43)
        target = root / "target.json"
        probe(args.executor, target)
        package = root / "package"
        build(
            reference / "model.json",
            reference / "weights.npz",
            target,
            7,
            4,
            package,
        )
        with np.load(reference / "reference_cases.npz") as cases:
            x = np.ascontiguousarray(cases["x"], dtype="<f4")
            expected_h = cases["h"].copy()
            expected_y = cases["y"].copy()
        inputs = root / "inputs.bin"
        inputs.write_bytes(x.tobytes())

        for tool in ("memcheck", "racecheck", "synccheck"):
            hidden = root / f"{tool}-hidden.bin"
            outputs = root / f"{tool}-outputs.bin"
            command = [
                str(args.sanitizer),
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                str(args.executor),
                "--run-plan",
                str(package),
                "--inputs",
                str(inputs),
                "--outputs",
                str(outputs),
                "--hidden",
                str(hidden),
                "--report",
                str(root / f"{tool}-report.json"),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise RuntimeError(f"{tool} failed:\n{result.stdout}\n{result.stderr}")
            actual_h = np.fromfile(hidden, dtype="<f4").reshape(expected_h.shape)
            actual_y = np.fromfile(outputs, dtype="<f4").reshape(expected_y.shape)
            if (
                not compare(actual_h, expected_h)["pass"]
                or not compare(actual_y, expected_y)["pass"]
            ):
                raise AssertionError(f"{tool} execution differs from PyTorch")
            print(f"{tool}: passed, outputs match PyTorch")


if __name__ == "__main__":
    main()
