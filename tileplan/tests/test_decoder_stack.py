"""Synthetic stack reference, compact plan, and package checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tileplan.decoder_plan import read_package
from tileplan.decoder_stack import (
    StackCache,
    StackSpec,
    build_stack,
    decode_token,
    fixed_stack_weights,
    generate_stack_fixture,
    recompute_stack_prefix,
    validate_stack,
)
from tileplan.probe import ABI, normalize

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m4"
OPCODES = [
    "rms_norm_f32",
    "gemv_f32",
    "rope_query_f32",
    "rope_cache_f32",
    "attention_prefix_f32",
    "silu_product_f32",
    "embedding_lookup_f32",
    "logits_f32",
]


def target() -> dict:
    return normalize(
        {
            "device": {
                "name": "fixture RTX",
                "compute_capability": [8, 6],
                "sm_count": 20,
                "cooperative_launch": True,
            },
            "executor": {
                "abi": ABI,
                "opcodes": OPCODES,
                "active_blocks_per_sm": 2,
                "threads_per_block": 128,
                "fingerprint": "fixture-stack-kernel",
            },
        }
    )


class DecoderStackTest(unittest.TestCase):
    def test_incremental_logits_match_full_causal_recomputation(self) -> None:
        for tied in (True, False):
            spec = StackSpec(tie_word_embeddings=tied)
            weights = fixed_stack_weights(spec)
            cache = StackCache.empty(spec)
            ids = [0, 7, 15, 3, 47]
            with torch.inference_mode():
                for position, token in enumerate(ids):
                    incremental = decode_token(token, weights, spec, cache)
                    full = recompute_stack_prefix(ids[: position + 1], weights, spec)
                    torch.testing.assert_close(
                        incremental["logits"],
                        full[-1],
                        atol=spec.block.atol,
                        rtol=spec.block.rtol,
                    )
                    self.assertEqual(cache.position, position + 1)
                cache.reset()
                reset = decode_token(ids[0], weights, spec, cache)
                fresh = decode_token(ids[0], weights, spec, StackCache.empty(spec))
                torch.testing.assert_close(reset["logits"], fresh["logits"])

    def test_compact_package_alias_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.json").write_text(json.dumps(target()))
            package = root / "package"
            build_stack(
                FIXTURE / "model.json",
                FIXTURE / "weights.npz",
                root / "target.json",
                7,
                package,
            )
            plan, packed, _ = read_package(package)
            self.assertEqual(plan["model"]["layers"], 3)
            self.assertEqual(plan["execution"]["repeat_block_count"], 3)
            self.assertEqual(plan["storage"]["scratch_bytes"], 1664)
            self.assertEqual(plan["storage"]["kv_bytes"], 3 * 768)
            self.assertTrue(plan["storage"]["stack"]["lm_head_alias"])
            self.assertEqual(plan["storage"]["stack"]["lm_head_offset"], 0)
            self.assertEqual(len(packed), plan["storage"]["weight_bytes"])
            self.assertLess(
                len(packed),
                3 * plan["storage"]["stack"]["layer_weight_bytes"]
                + 2 * plan["storage"]["stack"]["embedding_bytes"]
                + plan["storage"]["stack"]["final_norm_bytes"],
            )
            for mutate in (
                lambda item: item["storage"]["stack"].update(layer_kv_stride=0),
                lambda item: item["execution"].update(repeat_block_count=2),
                lambda item: item["tasks"][-1].update(predecessors=[]),
            ):
                bad = copy.deepcopy(plan)
                mutate(bad)
                with self.assertRaises(ValueError):
                    validate_stack(bad, packed)
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_stack(plan, packed[:-1] + bytes([packed[-1] ^ 1]))

            bf16_package = root / "bf16"
            build_stack(
                FIXTURE / "model.json",
                FIXTURE / "weights.npz",
                root / "target.json",
                7,
                bf16_package,
                "bfloat16",
            )
            bf16_plan, bf16_packed, _ = read_package(bf16_package)
            self.assertLess(len(bf16_packed), len(packed))
            self.assertEqual(bf16_plan["model"]["weight_dtype"], "bfloat16")
            self.assertEqual(bf16_plan["storage"]["stack"]["lm_head_offset"], 0)
            bad = copy.deepcopy(bf16_plan)
            bad["storage"]["stack"]["lm_head_offset"] = 64
            with self.assertRaises(ValueError):
                validate_stack(bad, bf16_packed)

    def test_frozen_fixture_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fixture"
            generate_stack_fixture(out)
            self.assertEqual(
                json.loads((FIXTURE / "model.json").read_text()),
                json.loads((out / "model.json").read_text()),
            )
            for name in ("weights.npz", "reference_cases.npz"):
                with np.load(FIXTURE / name) as saved, np.load(out / name) as current:
                    self.assertEqual(set(saved.files), set(current.files))
                    for key in saved.files:
                        np.testing.assert_array_equal(saved[key], current[key])


if __name__ == "__main__":
    unittest.main()
