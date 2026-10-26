"""Precision contracts, BF16 reference boundaries, and package migration."""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch
from test_decoder_planning import target

from tileplan.decoder_block import make_plan, read_model, validate
from tileplan.decoder_reference import (
    ReferenceCache,
    decode_one,
    fixed_tokens,
    fixed_weights,
    recompute_prefix,
    rms_norm,
)
from tileplan.decoder_stack import (
    StackSpec,
    fixed_stack_weights,
    make_stack_plan,
    validate_stack,
)


class PrecisionTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).parent / "fixtures/decoder_m1"
        self.spec = read_model(source / "model.json")
        with np.load(source / "weights.npz") as stored:
            self.weights = {name: stored[name].copy() for name in stored.files}

    def test_two_storage_policies_and_independent_validation(self):
        fp32, _ = make_plan(
            self.spec,
            self.weights,
            target(),
        )
        bf16, packed = make_plan(self.spec, self.weights, target(), "bfloat16")
        self.assertTrue(all(t["dtype"] == "bfloat16" for t in bf16["model"]["tensors"]))
        self.assertEqual(bf16["storage"]["kv_bytes"] * 2, fp32["storage"]["kv_bytes"])
        self.assertEqual(
            bf16["storage"]["attention_bytes"],
            self.spec.num_attention_heads * self.spec.max_context * 2,
        )
        for field, value in (
            ("activation_dtype", "float32"),
            ("kv_dtype", "float32"),
            ("accumulation_dtype", "bfloat16"),
            ("logits_compute_dtype", "float32"),
            ("api_output_dtype", "bfloat16"),
            ("numeric_policy", "unknown"),
            ("matrix_kernel", "simt"),
        ):
            bad = copy.deepcopy(bf16)
            bad["model"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(bad, packed)
        for mutation in (
            lambda p: p["storage"].update(attention_bytes=2),
            lambda p: p["storage"]["scratch"][1].update(offset=0),
            lambda p: p["tasks"][0].update(opcode="rms_norm_f32"),
        ):
            bad = copy.deepcopy(bf16)
            mutation(bad)
            with self.assertRaises(ValueError):
                validate(bad, packed)

    def test_retired_package_formats_are_rejected(self):
        for dtype in ("float32", "bfloat16"):
            plan, packed = make_plan(self.spec, self.weights, target(), dtype)
            for version in (2, 3):
                legacy = copy.deepcopy(plan)
                legacy["header"].update(
                    format_version=version,
                    executor_abi=f"tileinfer-decoder-v{version}",
                    model_kind="llama_decoder_block_f32",
                )
                legacy["model"]["dtype"] = "float32"
                legacy["storage"].pop("attention_bytes")
                with self.assertRaisesRegex(ValueError, "regenerate"):
                    validate(legacy, packed)
        spec = StackSpec(self.spec)
        weights = {
            name: value.numpy() for name, value in fixed_stack_weights(spec).items()
        }
        stack, packed = make_stack_plan(
            spec,
            weights,
            target(),
        )
        stack["header"].update(
            format_version=3,
            executor_abi="tileinfer-decoder-v3",
            model_kind="llama_decoder_stack_f32",
        )
        stack["model"]["dtype"] = "float32"
        stack["storage"].pop("attention_bytes")
        with self.assertRaisesRegex(ValueError, "regenerate"):
            validate_stack(stack, packed)

    def test_bf16_reference_keeps_real_bf16_tensors_and_cache(self):
        weights = {
            name: value.to(torch.bfloat16)
            for name, value in fixed_weights(self.spec).items()
        }
        token = fixed_tokens(self.spec)[1].to(torch.bfloat16)
        cache = ReferenceCache.empty(self.spec, torch.bfloat16)
        with torch.inference_mode():
            result = decode_one(token, weights, self.spec, cache)
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in result.values()))
        self.assertEqual(cache.k.dtype, torch.bfloat16)
        self.assertEqual(cache.v.dtype, torch.bfloat16)
        self.assertEqual(cache.position, 1)
        full = recompute_prefix(token[None], weights, self.spec)
        torch.testing.assert_close(result["y"], full["y"][0], atol=1e-2, rtol=1e-2)
        with self.assertRaisesRegex(ValueError, "cache shape"):
            decode_one(token.float(), weights, self.spec, cache)

    def test_rms_norm_rounds_before_scale(self):
        generator = torch.Generator().manual_seed(251)
        x = torch.randn(4096, generator=generator).to(torch.bfloat16)
        scale = torch.randn(4096, generator=generator).to(torch.bfloat16)
        normalized = x.float() * torch.rsqrt(x.float().square().mean() + 1e-5)
        expected = normalized.to(torch.bfloat16) * scale
        actual = rms_norm(x, scale, 1e-5)
        self.assertTrue(torch.equal(actual, expected))
        incorrectly_fused = (normalized * scale.float()).to(torch.bfloat16)
        self.assertGreater(int((actual != incorrectly_fused).sum()), 500)


if __name__ == "__main__":
    unittest.main()
