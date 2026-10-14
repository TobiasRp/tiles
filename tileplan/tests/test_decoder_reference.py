"""Milestone-1 decoder semantics and frozen fixture checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tileplan.decoder_reference import (
    DecoderSpec,
    ReferenceCache,
    apply_rope,
    decode_one,
    fixed_tokens,
    fixed_weights,
    generate_fixture,
    recompute_prefix,
    rope_inverse_frequencies,
)

FIXTURE = Path(__file__).parent / "fixtures" / "decoder_m1"


class DecoderReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = DecoderSpec()
        self.weights = fixed_weights(self.spec)
        self.tokens = fixed_tokens(self.spec)

    def assert_trace_close(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(
            actual, expected, atol=self.spec.atol, rtol=self.spec.rtol
        )

    def test_spec_freezes_gqa_rope_and_cache_layout(self) -> None:
        self.assertEqual(self.spec.head_to_kv, [0, 0, 1, 1])
        self.assertEqual(
            self.spec.kv_cache_layout, "[layers,max_context,kv_heads,head_dim]"
        )
        self.assertEqual(ReferenceCache.empty(self.spec).k.shape, (1, 6, 2, 8))
        self.assertEqual(self.spec.rope_type, "llama3")
        self.assertEqual(self.spec.rope_pairing, "split_half")
        self.assertEqual(self.spec.rms_norm_eps, 1e-5)
        self.assertEqual((self.spec.atol, self.spec.rtol), (2e-5, 2e-5))

    def test_rope_uses_split_half_and_llama3_frequency_branches(self) -> None:
        frequencies = rope_inverse_frequencies(self.spec)
        base = [self.spec.rope_theta ** (-i / self.spec.head_dim) for i in (0, 2, 4, 6)]
        self.assertAlmostEqual(float(frequencies[0]), base[0], places=7)
        self.assertAlmostEqual(float(frequencies[1]), base[1], places=7)
        self.assertLess(float(frequencies[2]), base[2])
        self.assertGreater(float(frequencies[2]), base[2] / self.spec.rope_factor)
        self.assertAlmostEqual(
            float(frequencies[3]), base[3] / self.spec.rope_factor, places=9
        )
        values = torch.zeros((1, self.spec.head_dim), dtype=torch.float32)
        values[0, 0] = 1
        rotated = apply_rope(values, torch.tensor(1), self.spec)
        self.assertAlmostEqual(float(rotated[0, 0]), math.cos(1), places=6)
        self.assertAlmostEqual(float(rotated[0, 4]), math.sin(1), places=6)
        self.assertEqual(float(rotated[0, 1]), 0)

    def test_each_incremental_step_matches_full_prefix_recomputation(self) -> None:
        cache = ReferenceCache.empty(self.spec)
        with torch.inference_mode():
            for p, token in enumerate(self.tokens):
                step = decode_one(token, self.weights, self.spec, cache)
                full = recompute_prefix(self.tokens[: p + 1], self.weights, self.spec)
                for name, actual in step.items():
                    self.assert_trace_close(actual, full[name][-1])
                self.assert_trace_close(cache.k[0, : p + 1], full["k"])
                self.assert_trace_close(cache.v[0, : p + 1], full["v"])
                self.assertEqual(cache.position, p + 1)

    def test_future_tokens_do_not_change_past_outputs(self) -> None:
        with torch.inference_mode():
            original = recompute_prefix(self.tokens, self.weights, self.spec)
            changed = self.tokens.clone()
            changed[-1] += 100
            modified = recompute_prefix(changed, self.weights, self.spec)
        self.assert_trace_close(original["y"][:-1], modified["y"][:-1])
        self.assertFalse(torch.allclose(original["y"][-1], modified["y"][-1]))

    def test_capacity_and_reset(self) -> None:
        cache = ReferenceCache.empty(self.spec)
        with torch.inference_mode():
            for token in self.tokens:
                decode_one(token, self.weights, self.spec, cache)
            decode_one(self.tokens[1], self.weights, self.spec, cache)
            self.assertEqual(cache.position, self.spec.max_context)
            with self.assertRaisesRegex(ValueError, "capacity"):
                decode_one(self.tokens[2], self.weights, self.spec, cache)
            cache.reset()
            reset_step = decode_one(self.tokens[1], self.weights, self.spec, cache)
            fresh_step = decode_one(
                self.tokens[1], self.weights, self.spec, ReferenceCache.empty(self.spec)
            )
        self.assert_trace_close(reset_step["y"], fresh_step["y"])
        self.assertEqual(cache.position, 1)

    def test_committed_fixture_is_reproducible_and_matches_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "decoder_m1"
            generate_fixture(regenerated)
            self.assertEqual(
                json.loads((FIXTURE / "model.json").read_text()),
                json.loads((regenerated / "model.json").read_text()),
            )
            for filename in ("weights.npz", "reference_cases.npz"):
                with (
                    np.load(FIXTURE / filename) as saved,
                    np.load(regenerated / filename) as current,
                ):
                    self.assertEqual(set(saved.files), set(current.files))
                    for name in saved.files:
                        np.testing.assert_array_equal(saved[name], current[name])
            with np.load(FIXTURE / "reference_cases.npz") as cases:
                self.assertEqual(cases["x"].shape, (5, 32))
                self.assertEqual(cases["q"].shape, (5, 4, 8))
                self.assertEqual(cases["k_cache"].shape, (1, 6, 2, 8))
                self.assertEqual(cases["v_cache"].shape, (1, 6, 2, 8))
                with torch.inference_mode():
                    full = recompute_prefix(
                        torch.from_numpy(cases["x"]), self.weights, self.spec
                    )
                for name in full:
                    self.assert_trace_close(torch.from_numpy(cases[name]), full[name])


if __name__ == "__main__":
    unittest.main()
