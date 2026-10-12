"""Decoder package, schedule, and validation tests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tileplan.decoder_plan import build, read_package, summary, validate
from tileplan.probe import ABI, normalize

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m1"
OPCODES = [
    "rms_norm_f32",
    "gemv_f32",
    "rope_query_f32",
    "rope_cache_f32",
    "attention_prefix_f32",
    "silu_product_f32",
]


def target(workers: int = 4) -> dict:
    return normalize(
        {
            "device": {
                "name": "fixture RTX",
                "compute_capability": [8, 6],
                "sm_count": workers,
                "cooperative_launch": True,
            },
            "executor": {
                "abi": ABI,
                "opcodes": OPCODES,
                "active_blocks_per_sm": 2,
                "threads_per_block": 128,
                "fingerprint": "fixture-decoder-kernel",
            },
        }
    )


class DecoderPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "target.json").write_text(json.dumps(target()))

    def package(self, name: str = "package", tile_rows: int = 8) -> Path:
        out = self.root / name
        build(
            FIXTURE / "model.json",
            FIXTURE / "weights.npz",
            self.root / "target.json",
            tile_rows,
            out,
        )
        return out

    def test_versioned_package_and_earliest_balanced_waves(self) -> None:
        package = self.package()
        plan, weights, digest = read_package(package)
        self.assertEqual(plan["header"]["format_version"], 3)
        self.assertEqual(plan["header"]["model_kind"], "llama_decoder_block_f32")
        self.assertEqual(plan["model"]["weight_dtype"], "float32")
        self.assertEqual(len(plan["waves"]), 9)
        self.assertEqual(len(plan["model"]["tensors"]), 24)
        self.assertEqual(len(plan["storage"]["kv"]), 2)
        self.assertEqual(plan["storage"]["kv_bytes"], 768)
        self.assertEqual(plan["storage"]["scratch_bytes"], 1664)
        self.assertEqual(len(weights), plan["storage"]["weight_bytes"])
        self.assertEqual(summary(plan, digest)["peak_bytes"], 39808)
        for task in plan["tasks"]:
            expected_wave = (
                0
                if not task["predecessors"]
                else 1
                + max(
                    plan["tasks"][previous]["wave"] for previous in task["predecessors"]
                )
            )
            self.assertEqual(task["wave"], expected_wave)
        for wave in plan["waves"]:
            self.assertEqual(len(wave["workers"]), 4)
            self.assertLessEqual(
                max(wave["estimated_work"]) - min(wave["estimated_work"]),
                max(
                    (
                        plan["tasks"][task_id]["estimated_work"]
                        for lane in wave["workers"]
                        for task_id in lane
                    ),
                    default=0,
                ),
            )
        self.assertEqual(
            [
                task["operands"]["kv_head"]
                for task in plan["tasks"]
                if task["opcode"] == "attention_prefix_f32"
            ],
            [0, 0, 1, 1],
        )

    def test_auto_tiling_balances_compiled_target(self) -> None:
        (self.root / "target.json").write_text(json.dumps(target(20)))
        package = self.root / "auto"
        report = build(
            FIXTURE / "model.json",
            FIXTURE / "weights.npz",
            self.root / "target.json",
            "auto",
            package,
        )
        self.assertEqual(report["model"]["tile_rows"], 4)
        self.assertEqual(
            min(report["tile_search"], key=lambda item: item["score"])["tile_rows"],
            4,
        )
        self.assertEqual(report["idle_workers"][6], 0)

    def test_package_is_reproducible(self) -> None:
        first, second = self.package(), self.package("again")
        for name in ("plan.json", "weights.bin", "planning_report.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_bfloat16_weight_regions_and_descriptors(self) -> None:
        fp32 = self.package()
        bf16 = self.root / "bf16"
        build(
            FIXTURE / "model.json",
            FIXTURE / "weights.npz",
            self.root / "target.json",
            8,
            bf16,
            "bfloat16",
        )
        plan, packed, _ = read_package(bf16)
        fp32_plan, _, _ = read_package(fp32)
        self.assertLess(len(packed), fp32_plan["storage"]["weight_bytes"])
        self.assertEqual(plan["model"]["weight_dtype"], "bfloat16")
        self.assertTrue(
            all(
                tensor["dtype"]
                == ("bfloat16" if tensor["storage"] == "weight" else "float32")
                for tensor in plan["model"]["tensors"]
            )
        )
        for mutate in (
            lambda item: item["model"]["tensors"][3].update(dtype="float32"),
            lambda item: item["storage"]["weights"][1].update(offset=0),
            lambda item: item["storage"]["weights"][1].update(bytes=4),
        ):
            bad = copy.deepcopy(plan)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate(bad, packed)

    def test_rejects_operand_dependency_storage_and_checksum_corruption(self) -> None:
        plan, weights, _ = read_package(self.package())
        attention = next(
            index
            for index, task in enumerate(plan["tasks"])
            if task["opcode"] == "attention_prefix_f32"
        )
        for mutation in (
            lambda p: p["tasks"][attention]["operands"].update(kv_head=1),
            lambda p: p["tasks"][attention].update(predecessors=[]),
            lambda p: p["tasks"][attention].update(wave=0),
            lambda p: p["tasks"][attention]["output"][0].update(rows=[1, 9]),
            lambda p: p["storage"]["kv"][1].update(offset=0),
            lambda p: p["waves"][0]["workers"][0].append(0),
        ):
            bad = copy.deepcopy(plan)
            mutation(bad)
            with self.assertRaises(ValueError):
                validate(bad, weights)
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate(plan, weights[:-1] + bytes([weights[-1] ^ 1]))

    def test_target_abi_and_tiling_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "capabilities"):
            normalize(
                {
                    "device": {
                        "compute_capability": [8, 6],
                        "sm_count": 4,
                        "cooperative_launch": True,
                    },
                    "executor": {
                        "abi": "tileinfer-mlp-v1",
                        "opcodes": OPCODES,
                        "active_blocks_per_sm": 2,
                        "threads_per_block": 128,
                        "fingerprint": "old",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "tile_rows"):
            self.package("bad", tile_rows=0)


if __name__ == "__main__":
    unittest.main()
