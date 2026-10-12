import stat
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tileplan.model import Model
from tileplan.pipeline import demo

MOCK_EXECUTOR = r"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from tileplan.plan import read_package

if sys.argv[1] == "--describe-target-json":
    print(json.dumps({"device": {"name": "mock GPU", "compute_capability": [8, 6],
                                 "sm_count": 2, "cooperative_launch": True},
                      "executor": {"abi": "tileinfer-mlp-v1", "fingerprint": "mock-only",
                                   "threads_per_block": 128, "active_blocks_per_sm": 1,
                                   "opcodes": ["gemv_bias_gelu_f32", "gemv_bias_f32"]}}))
    sys.exit(0)
args = sys.argv
package = Path(args[2])
plan, packed, _ = read_package(package)
if args[1] == "--benchmark":
    source = Path(args[args.index("--input") + 1]).read_bytes()
    report = {"input_sha256": hashlib.sha256(source).hexdigest(), "mock": True}
    Path(args[args.index("--report") + 1]).write_text(json.dumps(report))
    sys.exit(0)
if args[1] != "--run-plan":
    sys.exit(2)
shapes = {item["name"]: item["shape"] for item in plan["model"]["tensors"]}
weights = {}
for entry in plan["storage"]["weights"]:
    weights[entry["tensor"]] = np.frombuffer(packed, dtype="<f4", count=entry["bytes"] // 4,
                                             offset=entry["offset"]).reshape(shapes[entry["tensor"]])
x = np.fromfile(args[args.index("--inputs") + 1], dtype="<f4").reshape(-1, plan["model"]["D"])
z = x @ weights["W1"].T + weights["b1"]
h = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
y = h @ weights["W2"].T + weights["b2"]
np.asarray(h, dtype="<f4").tofile(args[args.index("--hidden") + 1])
np.asarray(y, dtype="<f4").tofile(args[args.index("--outputs") + 1])
Path(args[args.index("--report") + 1]).write_text(json.dumps({"mock": True}))
"""


class PipelineTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA baseline requires a GPU")
    def test_demo_file_bridge_with_mock_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock = root / "mock_executor"
            mock.write_text(f"#!{sys.executable}\n" + MOCK_EXECUTOR)
            mock.chmod(mock.stat().st_mode | stat.S_IXUSR)
            report = demo(root / "run", mock, Model(5, 7, 3), 19, 3, 2, 2, 2)
            self.assertEqual(report["correctness"], "pass")
            self.assertEqual(report["runtime"]["mock"], True)
            self.assertEqual(len(report["baselines"]["cuda_graph"]), 2)


if __name__ == "__main__":
    unittest.main()
