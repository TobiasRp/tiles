import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from tileplan.probe import normalize


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tileplan", *args],
        capture_output=True,
        text=True,
        check=False,
    )


class CliTest(unittest.TestCase):
    def test_offline_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            result = invoke(
                "reference", "--out", str(fixture), "--d", "5", "--h", "7", "--o", "3"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = {
                "device": {
                    "name": "test",
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
            (root / "target.json").write_text(json.dumps(normalize(raw)))
            result = invoke(
                "build",
                "--model",
                str(fixture / "model.json"),
                "--weights",
                str(fixture / "weights.npz"),
                "--target",
                str(root / "target.json"),
                "--tile-h1",
                "3",
                "--tile-h2",
                "2",
                "--out",
                str(root / "package"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout)["task_counts"]["gemv_bias_gelu_f32"], 3
            )
            for command in ("inspect", "verify"):
                result = invoke(command, str(root / "package"))
                self.assertEqual(result.returncode, 0, result.stderr)
            image = root / "schedule.svg"
            result = invoke(
                "visualize",
                str(root / "package"),
                "--out",
                str(image),
                "--note",
                "synthetic <target> & test",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            svg = image.read_text()
            self.assertEqual(
                len(
                    ElementTree.fromstring(svg).findall("{http://www.w3.org/2000/svg}g")
                ),
                5,
            )
            self.assertIn("FULL GRID BARRIER", svg)
            self.assertIn("synthetic &lt;target&gt; &amp; test", svg)
            result = invoke("visualize", str(root / "package"), "--out", str(image))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Worker 0", image.read_text())
            (root / "package/weights.bin").write_bytes(b"broken")
            result = invoke("verify", str(root / "package"))
            self.assertEqual(result.returncode, 2)
            self.assertIn("checksum", result.stderr)
            result = invoke("visualize", str(root / "package"), "--out", str(image))
            self.assertEqual(result.returncode, 2)

    def test_demo_stops_at_unimplemented_runtime(self) -> None:
        runtime = (
            Path(__file__).resolve().parents[2] / "tileinfer/build/release/tile_runtime"
        )
        if not runtime.exists():
            self.skipTest("C++ scaffold executable not built")
        with tempfile.TemporaryDirectory() as directory:
            result = invoke(
                "demo",
                "--executor",
                str(runtime),
                "--out",
                str(Path(directory) / "run"),
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("target probe failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
