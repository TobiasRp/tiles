"""Decoder package, schedule, and validation tests."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tileplan.decoder_block import build, read_model, validate
from tileplan.package import read_package
from tileplan.plan_core import attention_chunking, plan_target
from tileplan.probe import ABI, normalize
from tileplan.probe import OPCODES as PROBE_OPCODES

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m1"
OPCODES = sorted(PROBE_OPCODES)


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
                "attention_shared_bytes": 16384,
            },
        }
    )


class DecoderPlanningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "target.json").write_text(json.dumps(target()))

    def package(self, name: str = "package", **chunking) -> Path:
        out = self.root / name
        build(
            FIXTURE / "model.json",
            FIXTURE / "weights.npz",
            self.root / "target.json",
            out,
            **chunking,
        )
        return out

    def test_versioned_package_and_earliest_balanced_waves(self) -> None:
        package = self.package(chunk_steps=4, fused_steps=2)
        plan, weights, _ = read_package(package)
        self.assertEqual(plan["header"]["format_version"], 7)
        self.assertEqual({task["layer"] for task in plan["tasks"]}, {0})
        self.assertEqual(plan["header"]["model_kind"], "llama_decoder_block")
        self.assertEqual(plan["model"]["weight_dtype"], "float32")
        # Q/K/V heads, attention scores, attention values, Wo, gate/up, down.
        # The scores wave ends with the barrier skipped for fused prefixes.
        self.assertEqual(len(plan["waves"]), 6)
        self.assertEqual(
            [wave.get("conditional_barrier") for wave in plan["waves"]],
            [None, {"skip_up_to_length": 2}, None, None, None, None],
        )
        self.assertEqual(len(plan["model"]["tensors"]), 17)
        tensors = {item["id"]: item["name"] for item in plan["model"]["tensors"]}
        self.assertEqual(
            [tensors[entry["tensor"]] for entry in plan["storage"]["scratch"]],
            ["q_rot", "attention", "r", "product"],
        )
        self.assertEqual(len(plan["storage"]["kv"]), 2)
        self.assertEqual(plan["storage"]["kv_bytes"], 768)
        self.assertEqual(len(weights), plan["storage"]["weight_bytes"])
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
        # A 6-step context beyond the fused limit splits into two chunks.
        self.assertEqual(
            [
                (task["operands"]["kv_head"], task["operands"]["chunk"])
                for task in plan["tasks"]
                if task["opcode"] == "attention_scores_f32"
            ],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )

    def test_work_splits_over_every_worker(self) -> None:
        (self.root / "target.json").write_text(json.dumps(target(20)))
        report = build(
            FIXTURE / "model.json",
            FIXTURE / "weights.npz",
            self.root / "target.json",
            self.root / "wide",
        )
        # The 6-step context runs as one fused attention task per K/V head, and
        # each projection splits its rows into one range per worker.
        self.assertEqual(report["model"]["attention_chunks"], 1)
        self.assertEqual(report["idle_workers"][1:3], [18, 18])
        self.assertEqual(report["idle_workers"][3:], [0, 0, 0])

    def test_attention_chunking(self) -> None:
        spec = read_model(FIXTURE / "model.json")
        launch = plan_target(target())

        def chunking(dtype="float32", context=6, **options):
            return attention_chunking(
                replace(spec, max_context=context), dtype, launch, *options.values()
            )

        # Two query heads per K/V head: 16 KB holds 2,048 FP32 or 4,096 BF16 steps.
        self.assertEqual(
            chunking(),
            {
                "attention_chunks": 1,
                "attention_chunk_steps": 2048,
                "attention_fused_steps": 384,
            },
        )
        self.assertEqual(chunking("bfloat16")["attention_chunk_steps"], 4096)
        # Beyond the fused limit: at least two chunks, more past the capacity.
        self.assertEqual(chunking(context=385)["attention_chunks"], 2)
        self.assertEqual(chunking(context=4096)["attention_chunks"], 2)
        self.assertEqual(chunking(context=4097)["attention_chunks"], 3)
        self.assertEqual(chunking(context=6, steps=4, fused=4)["attention_chunks"], 2)
        for steps, fused in ((2049, None), (0, None), (8, 9), (8, 0)):
            with self.assertRaises(ValueError):
                chunking(steps=steps, fused=fused)

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
            bf16,
            "bfloat16",
        )
        plan, packed, _ = read_package(bf16)
        fp32_plan, _, _ = read_package(fp32)
        self.assertLess(len(packed), fp32_plan["storage"]["weight_bytes"])
        self.assertEqual(plan["model"]["weight_dtype"], "bfloat16")
        self.assertTrue(
            all(tensor["dtype"] == "bfloat16" for tensor in plan["model"]["tensors"])
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
            if task["opcode"] == "attention_scores_f32"
        )
        for mutation in (
            lambda p: p["tasks"][attention]["operands"].update(kv_head=1),
            lambda p: p["tasks"][attention].update(predecessors=[]),
            lambda p: p["tasks"][attention].update(wave=0),
            lambda p: p["tasks"][attention]["output"][0].update(chunk=1),
            lambda p: p["storage"]["kv"][1].update(offset=0),
            lambda p: p["waves"][0]["workers"][0].append(0),
        ):
            bad = copy.deepcopy(plan)
            mutation(bad)
            with self.assertRaises(ValueError):
                validate(bad, weights)
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate(plan, weights[:-1] + bytes([weights[-1] ^ 1]))

    def test_target_abi_is_checked(self) -> None:
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
                        "attention_shared_bytes": 16384,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
