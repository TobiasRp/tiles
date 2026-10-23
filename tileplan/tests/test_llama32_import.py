"""Checkpoint name, shape, dtype, and exact BF16 packing checks."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from tileplan.decoder_plan import read_package
from tileplan.decoder_reference import DecoderSpec
from tileplan.decoder_stack import StackSpec, fixed_stack_weights, make_stack_plan
from tileplan.llama32_import import (
    PINNED_REVISION,
    _name_map,
    pack_safetensors,
    read_llama32_config,
)
from tileplan.probe import ABI, OPCODES, normalize


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
                "opcodes": sorted(OPCODES),
                "active_blocks_per_sm": 2,
                "threads_per_block": 128,
                "fingerprint": "fixture-stack-kernel",
            },
        }
    )


class Llama32ImportTest(unittest.TestCase):
    def test_config_accepts_only_pinned_architecture(self) -> None:
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "mlp_bias": False,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            self.assertEqual(read_llama32_config(path, 4096).block.max_context, 4096)
            config["tie_word_embeddings"] = False
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
                read_llama32_config(path, 4096)

    def test_packs_exact_bf16_bits_and_aliases_output(self) -> None:
        spec = StackSpec(DecoderSpec(), layers=2, vocab_size=48)
        weights = fixed_stack_weights(spec)
        bf16 = {name: value.to(torch.bfloat16) for name, value in weights.items()}
        expected, packed = make_stack_plan(
            spec,
            {name: value.float().numpy() for name, value in bf16.items()},
            target(),
            "bfloat16",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n")
            names = _name_map(spec)
            save_file(
                {hf_name: bf16[name].contiguous() for name, hf_name in names.items()},
                root / "model.safetensors",
            )
            package = root / "package"
            report = pack_safetensors(root, spec, target(), package, PINNED_REVISION)
            plan, actual, _ = read_package(package)
            self.assertEqual(actual, packed)
            self.assertEqual(plan["storage"], expected["storage"])
            self.assertEqual(report["revision"], PINNED_REVISION)
            self.assertTrue(plan["storage"]["stack"]["lm_head_alias"])
            self.assertEqual(plan["source"]["revision_evidence"], "explicit")

            _, expected_float32 = make_stack_plan(
                spec,
                {name: value.float().numpy() for name, value in bf16.items()},
                target(),
                "float32",
            )
            float32_package = root / "package-float32"
            pack_safetensors(
                root, spec, target(), float32_package, PINNED_REVISION, "float32"
            )
            float32_plan, float32_bytes, _ = read_package(float32_package)
            self.assertEqual(float32_bytes, expected_float32)
            self.assertEqual(float32_plan["model"]["activation_dtype"], "float32")
            # Both precisions keep a score workspace per query head and position.
            model = float32_plan["model"]
            self.assertEqual(
                float32_plan["storage"]["attention_bytes"],
                model["num_attention_heads"] * model["max_context"] * 4,
            )

            # An unexpected tensor cannot silently be dropped.
            package2 = root / "package2"
            extra = {
                hf_name: bf16[name].contiguous() for name, hf_name in names.items()
            }
            extra["unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
            save_file(extra, root / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "names mismatch"):
                pack_safetensors(root, spec, target(), package2, PINNED_REVISION)


if __name__ == "__main__":
    unittest.main()
