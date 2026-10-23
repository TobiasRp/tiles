"""Integration checks for schedule and semantic tensor graph SVGs."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from tileplan.cli import main
from tileplan.decoder_plan import build, layer_view, read_package
from tileplan.decoder_stack import build_stack
from tileplan.probe import ABI, OPCODES, normalize

FIXTURES = Path(__file__).parent / "fixtures"


class VisualizationTest(unittest.TestCase):
    def test_schedule_shows_one_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = normalize(
                {
                    "device": {
                        "name": "fixture GPU",
                        "compute_capability": [8, 6],
                        "sm_count": 4,
                        "cooperative_launch": True,
                    },
                    "executor": {
                        "abi": ABI,
                        "opcodes": sorted(OPCODES),
                        "active_blocks_per_sm": 2,
                        "threads_per_block": 128,
                        "fingerprint": "fixture-worker",
                    },
                }
            )
            target_path = root / "target.json"
            target_path.write_text(json.dumps(target))
            stack_fixture = FIXTURES / "decoder_m4"
            package = root / "stack"
            build_stack(
                stack_fixture / "model.json",
                stack_fixture / "weights.npz",
                target_path,
                package,
            )
            schedule = root / "schedule.svg"
            self.assertEqual(
                main(["visualize", str(package), "--out", str(schedule)]), 0
            )
            svg = schedule.read_text()
            root_element = ElementTree.fromstring(svg)
            self.assertEqual(root_element.tag, "{http://www.w3.org/2000/svg}svg")
            # One layer of the stack: its six waves, each task card labeled.
            plan = layer_view(read_package(package)[0])
            for wave in range(len(plan["waves"])):
                self.assertIn(f"Wave {wave}", svg)
            self.assertIn("Full decoder: 3 layers", svg)
            for opcode in ("qkv_f32", "attention_scores_f32", "gate_up_f32"):
                self.assertIn(opcode, svg)

            block_fixture = FIXTURES / "decoder_m1"
            block = root / "block"
            build(
                block_fixture / "model.json",
                block_fixture / "weights.npz",
                target_path,
                block,
            )
            block_schedule = root / "block-schedule.svg"
            self.assertEqual(
                main(["visualize", str(block), "--out", str(block_schedule)]), 0
            )
            self.assertIn("One-token decoder block", block_schedule.read_text())


if __name__ == "__main__":
    unittest.main()
