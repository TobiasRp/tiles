import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tileplan.baselines import run
from tileplan.check import compare
from tileplan.model import Model
from tileplan.plan import build
from tileplan.probe import normalize
from tileplan.reference import generate


class ComparisonTest(unittest.TestCase):
    def test_tolerance_and_nonfinite_values(self) -> None:
        expected = np.array([0.0, 1.0], dtype=np.float32)
        self.assertTrue(compare(expected + 1e-5, expected)["pass"])
        self.assertFalse(compare(expected + 1e-2, expected)["pass"])
        self.assertFalse(
            compare(np.array([np.nan, 1], dtype=np.float32), expected)["pass"]
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_eager_and_graph_replay_match_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate(root / "fixture", Model(129, 65, 33), 17)
            target = normalize(
                {
                    "device": {
                        "name": "fixture",
                        "compute_capability": [8, 6],
                        "sm_count": 2,
                        "cooperative_launch": True,
                    },
                    "executor": {
                        "abi": "tileinfer-mlp-v1",
                        "fingerprint": "fixture",
                        "threads_per_block": 128,
                        "active_blocks_per_sm": 1,
                        "opcodes": ["gemv_bias_gelu_f32", "gemv_bias_f32"],
                    },
                }
            )
            (root / "target.json").write_text(json.dumps(target))
            build(
                root / "fixture/model.json",
                root / "fixture/weights.npz",
                root / "target.json",
                16,
                16,
                root / "package",
            )
            report = run(
                root / "package",
                root / "fixture/reference_cases.npz",
                root / "baseline.json",
                2,
                2,
            )
            self.assertEqual(len(report["eager"]), 2)
            self.assertEqual(len(report["cuda_graph"]), 2)
            self.assertTrue(
                all(
                    case["h"]["pass"] and case["y"]["pass"] for case in report["checks"]
                )
            )


if __name__ == "__main__":
    unittest.main()
