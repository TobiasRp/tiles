import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tileplan.model import Model, read_weights
from tileplan.plan import build, read_package
from tileplan.probe import normalize
from tileplan.reference import generate
from tileplan.validate import validate


def target() -> dict:
    return normalize(
        {
            "device": {
                "name": "fixture GPU",
                "compute_capability": [8, 6],
                "sm_count": 4,
                "cooperative_launch": True,
                "total_global_memory_bytes": 4_000_000_000,
            },
            "executor": {
                "abi": "tileinfer-mlp-v1",
                "fingerprint": "compiled-worker-fixture",
                "threads_per_block": 128,
                "active_blocks_per_sm": 2,
                "opcodes": ["gemv_bias_gelu_f32", "gemv_bias_f32"],
            },
        }
    )


class PlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Model(5, 7, 3)
        generate(self.root / "fixture", self.model, seed=19)
        (self.root / "target.json").write_text(json.dumps(target()))

    def package(self, name: str = "package") -> Path:
        out = self.root / name
        build(
            self.root / "fixture/model.json",
            self.root / "fixture/weights.npz",
            self.root / "target.json",
            3,
            2,
            out,
        )
        return out

    def test_fixture_is_deterministic_and_shapes_are_exact(self) -> None:
        generate(self.root / "other", self.model, seed=19)
        for name in ("W1", "b1", "W2", "b2"):
            np.testing.assert_array_equal(
                read_weights(self.root / "fixture/weights.npz", self.model)[name],
                read_weights(self.root / "other/weights.npz", self.model)[name],
            )
        with np.load(self.root / "fixture/reference_cases.npz") as cases:
            self.assertEqual(cases["x"].shape, (3, 5))
            self.assertEqual(cases["h"].shape, (3, 7))
            self.assertEqual(cases["y"].shape, (3, 3))
            self.assertTrue(np.all(cases["x"][0] == 0))

    def test_two_waves_tail_tiles_and_reproducible_package(self) -> None:
        first = self.package()
        second = self.package("again")
        self.assertEqual(
            (first / "plan.json").read_bytes(), (second / "plan.json").read_bytes()
        )
        self.assertEqual(
            (first / "weights.bin").read_bytes(), (second / "weights.bin").read_bytes()
        )
        plan, weights, _ = read_package(first)
        self.assertEqual(
            [task["rows"] for task in plan["tasks"]],
            [[0, 3], [3, 6], [6, 7], [0, 2], [2, 3]],
        )
        self.assertEqual(
            [task["predecessors"] for task in plan["tasks"][3:]], [[0, 1, 2], [0, 1, 2]]
        )
        self.assertEqual(len(plan["waves"][0]["workers"]), 4)
        self.assertTrue(any(not ids for ids in plan["waves"][0]["workers"]))
        self.assertEqual(len(weights), plan["storage"]["weight_bytes"])

    def test_verifier_rejects_corruption_and_schedule_hazards(self) -> None:
        package = self.package()
        plan, weights, _ = read_package(package)
        for mutation in (
            lambda p: p["tasks"][3].update(predecessors=[]),
            lambda p: p["tasks"][2].update(rows=[5, 7]),
            lambda p: p["waves"][0]["workers"][0].append(0),
            lambda p: p["storage"]["weights"][1].update(offset=0),
        ):
            bad = copy.deepcopy(plan)
            mutation(bad)
            with self.assertRaises(ValueError):
                validate(bad, weights)
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate(plan, weights[:-1] + bytes([weights[-1] ^ 1]))

    def test_rejects_invalid_fixture_and_target(self) -> None:
        bad = target()
        bad["launch"]["worker_count"] = 5
        (self.root / "target.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "target"):
            self.package()
        with self.assertRaises(ValueError):
            normalize({"device": {"cooperative_launch": False}, "executor": {}})


if __name__ == "__main__":
    unittest.main()
