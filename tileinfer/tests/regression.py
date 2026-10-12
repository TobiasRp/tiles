"""Exercise the compiled executor with real TilePlan packages and PyTorch fixtures."""

import argparse
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tileplan.model import Model
from tileplan.pipeline import run_cases
from tileplan.plan import build
from tileplan.probe import probe
from tileplan.reference import generate

parser = argparse.ArgumentParser()
parser.add_argument("--executor", type=Path, required=True)
arguments, remaining = parser.parse_known_args()
EXECUTOR = arguments.executor.resolve()


class RuntimeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="tileinfer-regression-")
        cls.root = Path(cls.temporary.name)
        cls.target = cls.root / "target.json"
        probe(EXECUTOR, cls.target)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def package(
        self, name: str, model: Model, tiles: tuple[int, int]
    ) -> tuple[Path, Path]:
        fixture = self.root / f"{name}-fixture"
        generate(fixture, model, seed=29)
        package = self.root / f"{name}-package"
        build(
            fixture / "model.json",
            fixture / "weights.npz",
            self.target,
            *tiles,
            package,
        )
        return fixture, package

    def test_reference_regression_few_tasks(self) -> None:
        fixture, package = self.package("few", Model(5, 3, 2), (8, 4))
        report = run_cases(
            EXECUTOR,
            package,
            fixture / "reference_cases.npz",
            self.root / "few-results",
        )
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["cases"]), 3)

    def test_reference_regression_many_tasks_and_tails(self) -> None:
        fixture, package = self.package("tails", Model(129, 65, 33), (4, 3))
        report = run_cases(
            EXECUTOR,
            package,
            fixture / "reference_cases.npz",
            self.root / "tails-results",
        )
        self.assertTrue(report["pass"])
        self.assertTrue(
            all(case["h"]["pass"] and case["y"]["pass"] for case in report["cases"])
        )

    def test_loader_rejects_corruption(self) -> None:
        fixture, package = self.package("invalid", Model(13, 9, 7), (4, 3))
        inputs = self.root / "one-input.bin"
        with np.load(fixture / "reference_cases.npz") as cases:
            inputs.write_bytes(cases["x"][0].tobytes())
        bad = self.root / "bad-package"

        def rejected(change: str, expected: str) -> None:
            if bad.exists():
                shutil.rmtree(bad)
            shutil.copytree(package, bad)
            if change == "weights":
                raw = bytearray((bad / "weights.bin").read_bytes())
                raw[0] ^= 1
                (bad / "weights.bin").write_bytes(raw)
            else:
                path = bad / "plan.json"
                plan = json.loads(path.read_text())
                if change == "target":
                    plan["target"]["kernel_fingerprint"] = "stale-kernel"
                elif change == "schedule":
                    plan["waves"][0]["workers"][0].append(
                        plan["waves"][0]["workers"][0][0]
                    )
                elif change == "dependencies":
                    plan["tasks"][-1]["predecessors"] = []
                path.write_text(json.dumps(plan))
            command = [
                str(EXECUTOR),
                "--run-plan",
                str(bad),
                "--inputs",
                str(inputs),
                "--outputs",
                str(self.root / "discard.bin"),
                "--report",
                str(self.root / "discard.json"),
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(expected, completed.stderr)

        for change, error in (
            ("weights", "checksum"),
            ("target", "target"),
            ("schedule", "twice"),
            ("dependencies", "dependencies"),
        ):
            with self.subTest(change=change):
                rejected(change, error)


if __name__ == "__main__":
    unittest.main(argv=["regression.py", *remaining])
