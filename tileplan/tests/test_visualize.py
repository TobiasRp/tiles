"""Integration checks for schedule and semantic tensor graph SVGs."""

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from tileplan.cli import main
from tileplan.decoder_plan import build, read_package
from tileplan.decoder_stack import build_stack
from tileplan.probe import ABI, normalize

FIXTURES = Path(__file__).parent / "fixtures"


class VisualizationTest(unittest.TestCase):
    def test_stack_graph_shows_shapes_and_one_repeated_block(self) -> None:
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
                        "opcodes": [
                            "rms_norm_f32",
                            "gemv_f32",
                            "rope_query_f32",
                            "rope_cache_f32",
                            "attention_prefix_f32",
                            "silu_product_f32",
                        ],
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
                4,
                package,
            )
            schedule = root / "schedule.svg"
            graph = root / "tensor-graph.svg"
            flow = root / "tile-flow.svg"
            self.assertEqual(
                main(
                    [
                        "visualize",
                        str(package),
                        "--out",
                        str(schedule),
                        "--tensor-graph-out",
                        str(graph),
                        "--tile-flow-out",
                        str(flow),
                    ]
                ),
                0,
            )
            self.assertTrue(schedule.is_file())
            self.assertEqual(
                ElementTree.parse(graph).getroot().tag,
                "{http://www.w3.org/2000/svg}svg",
            )
            svg = graph.read_text()
            self.assertEqual(svg.count("Decoder block × 3"), 1)
            for label in (
                "Embedding lookup",
                "current layer hidden",
                "y feeds the next layer’s x",
                "Wgate: [64, 32]",
                "K, V [3, 6, 2, 8]",
                "logits [48]",
                "lm_head: [48, 32] (tied)",
            ):
                self.assertIn(label, svg)

            flow_root = ElementTree.parse(flow).getroot()
            flow_svg = flow.read_text()
            plan, _, _ = read_package(package)
            for wave in range(len(plan["waves"])):
                self.assertIn(f"Wave {wave}", flow_svg)
            self.assertIn("full grid barrier", flow_svg)
            self.assertIn("collapsed prerequisites", flow_svg)
            self.assertIn("SiLU(gate[rows]) × up[rows]", flow_svg)
            self.assertLess(len(flow_svg), 100_000)
            cards = [
                element.attrib["data-card"]
                for element in flow_root.iter()
                if "data-card" in element.attrib
            ]
            self.assertTrue(any(card.startswith("a") for card in cards))
            for element in flow_root.iter():
                source = element.attrib.get("data-from", "")
                target = element.attrib.get("data-to", "")
                if source.startswith("t") and target.startswith("t"):
                    self.assertIn(
                        int(source[1:]),
                        plan["tasks"][int(target[1:])]["predecessors"],
                    )

            block_fixture = FIXTURES / "decoder_m1"
            block = root / "block"
            build(
                block_fixture / "model.json",
                block_fixture / "weights.npz",
                target_path,
                4,
                block,
            )
            self.assertEqual(
                main(
                    [
                        "visualize",
                        str(block),
                        "--out",
                        str(root / "block-schedule.svg"),
                        "--tensor-graph-out",
                        str(root / "block-graph.svg"),
                    ]
                ),
                0,
            )
            block_svg = (root / "block-graph.svg").read_text()
            self.assertIn("Decoder block × 1", block_svg)
            self.assertNotIn("Embedding lookup", block_svg)
            self.assertNotIn("y feeds the next layer’s x", block_svg)


if __name__ == "__main__":
    unittest.main()
