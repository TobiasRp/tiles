"""Cross-check the Python decoder package against the C++ loader on the GPU host."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from tileplan.decoder_plan import build
from tileplan.decoder_reference import (
    DecoderSpec,
    ReferenceCache,
    decode_one,
    fixed_weights,
)
from tileplan.probe import probe


def run(executor: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executor), *args], text=True, capture_output=True, check=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).parents[2] / "tileplan/tests/fixtures/decoder_m1"
    with tempfile.TemporaryDirectory(prefix="decoder-regression-") as directory:
        root = Path(directory)
        probe(args.executor, root / "target.json")
        package = root / "package"
        build(
            fixture / "model.json",
            fixture / "weights.npz",
            root / "target.json",
            4,
            package,
        )
        result = run(args.executor, "--inspect-plan", str(package))
        if result.returncode:
            raise AssertionError(result.stderr)
        report = json.loads(result.stdout)
        if (
            report["wave_count"] != 9
            or report["scratch_bytes"] != 1664
            or report["kv_bytes"] != 768
            or report["task_count"] != 92
        ):
            raise AssertionError(report)
        result = run(args.executor, "--state-smoke", str(package))
        if result.returncode or json.loads(result.stdout) != {"state_smoke": True}:
            raise AssertionError(result.stderr or result.stdout)

        with np.load(fixture / "reference_cases.npz", allow_pickle=False) as cases:
            hidden = cases["x"].copy()
        request = {"tokens": [{"hidden": row.tolist()} for row in hidden]}
        request["tokens"].append({"reset": True})
        request["tokens"].append({"hidden": hidden[0].tolist()})
        request_path = root / "inputs.json"
        request_path.write_text(json.dumps(request))
        result = run(args.executor, "--run-block", str(package), str(request_path))
        if result.returncode:
            raise AssertionError(result.stderr)
        observed = json.loads(result.stdout)
        spec = DecoderSpec()
        weights = fixed_weights(spec)
        cache = ReferenceCache.empty(spec)
        traces = []
        with torch.inference_mode():
            for row in hidden:
                traces.append(decode_one(torch.from_numpy(row), weights, spec, cache))
            cache.reset()
            traces.append(decode_one(torch.from_numpy(hidden[0]), weights, spec, cache))
        for index, (actual, expected) in enumerate(
            zip(observed["steps"], traces, strict=True)
        ):
            if actual["position"] != (index if index < len(hidden) else 0):
                raise AssertionError("wrong decoder position")
            for field in ("q", "k", "v", "attention", "attn_residual", "ffn", "y"):
                np.testing.assert_allclose(
                    np.asarray(actual[field], dtype=np.float32),
                    expected[field].numpy().reshape(-1),
                    atol=spec.atol,
                    rtol=spec.rtol,
                    err_msg=f"token {index}: {field}",
                )
        if observed["position"] != 1 or not observed["valid"]:
            raise AssertionError("reset failed")
        capacity = {"tokens": [{"hidden": hidden[0].tolist()}] * (spec.max_context + 1)}
        request_path.write_text(json.dumps(capacity))
        result = run(args.executor, "--run-block", str(package), str(request_path))
        if result.returncode == 0 or "capacity reached" not in result.stderr:
            raise AssertionError("capacity check failed")

        tail_package = root / "tail-package"
        build(
            fixture / "model.json",
            fixture / "weights.npz",
            root / "target.json",
            7,
            tail_package,
        )
        request_path.write_text(json.dumps({"tokens": request["tokens"][:2]}))
        result = run(args.executor, "--run-block", str(tail_package), str(request_path))
        if result.returncode:
            raise AssertionError(result.stderr)
        for actual, expected in zip(
            json.loads(result.stdout)["steps"], traces[:2], strict=True
        ):
            for field in ("q", "k", "v", "attention", "attn_residual", "ffn", "y"):
                np.testing.assert_allclose(
                    actual[field],
                    expected[field].numpy().reshape(-1),
                    atol=spec.atol,
                    rtol=spec.rtol,
                    err_msg=f"tail tile {field}",
                )

        bf16_package = root / "bf16-package"
        build(
            fixture / "model.json",
            fixture / "weights.npz",
            root / "target.json",
            7,
            bf16_package,
            "bfloat16",
        )
        bf16_plan = json.loads((bf16_package / "plan.json").read_text())
        if bf16_plan["model"]["weight_dtype"] != "bfloat16":
            raise AssertionError("missing BF16 weight dtype")
        if any(
            tensor["dtype"]
            != ("bfloat16" if tensor["storage"] == "weight" else "float32")
            for tensor in bf16_plan["model"]["tensors"]
        ):
            raise AssertionError("incorrect per-tensor dtype")
        bf16_blob = (bf16_package / "weights.bin").read_bytes()
        wq_region = next(
            region
            for region in bf16_plan["storage"]["weights"]
            if region["tensor"] == 3
        )
        expected_wq = weights["Wq"].to(torch.bfloat16).view(torch.uint16).numpy()
        actual_wq = np.frombuffer(
            bf16_blob,
            dtype="<u2",
            count=expected_wq.size,
            offset=wq_region["offset"],
        ).reshape(expected_wq.shape)
        np.testing.assert_array_equal(actual_wq, expected_wq)
        quantized = {
            name: value.to(torch.bfloat16).float() for name, value in weights.items()
        }
        bf16_cache = ReferenceCache.empty(spec)
        with torch.inference_mode():
            bf16_traces = [
                decode_one(torch.from_numpy(row), quantized, spec, bf16_cache)
                for row in hidden[:2]
            ]
        request_path.write_text(json.dumps({"tokens": request["tokens"][:2]}))
        result = run(args.executor, "--run-block", str(bf16_package), str(request_path))
        if result.returncode:
            raise AssertionError(result.stderr)
        for actual, expected in zip(
            json.loads(result.stdout)["steps"], bf16_traces, strict=True
        ):
            for field in ("q", "k", "v", "attention", "attn_residual", "ffn", "y"):
                np.testing.assert_allclose(
                    actual[field],
                    expected[field].numpy().reshape(-1),
                    atol=spec.atol,
                    rtol=spec.rtol,
                    err_msg=f"BF16 {field}",
                )
        original_bf16 = bf16_plan
        bad = copy.deepcopy(original_bf16)
        bad["model"]["tensors"][3]["dtype"] = "float32"
        (bf16_package / "plan.json").write_text(json.dumps(bad))
        if run(args.executor, "--inspect-plan", str(bf16_package)).returncode == 0:
            raise AssertionError("C++ loader accepted a bad BF16 tensor dtype")

        original = json.loads((package / "plan.json").read_text())
        mutations = {
            "operand": lambda p: p["tasks"][0]["operands"].update(scale=15),
            "dependency": lambda p: p["tasks"][-1].update(predecessors=[]),
            "cache bounds": lambda p: p["storage"]["kv"][1].update(offset=0),
            "wave assignment": lambda p: p["tasks"][-1].update(wave=0),
        }
        for name, mutation in mutations.items():
            bad = copy.deepcopy(original)
            mutation(bad)
            (package / "plan.json").write_text(json.dumps(bad))
            result = run(args.executor, "--inspect-plan", str(package))
            if result.returncode == 0:
                raise AssertionError(f"C++ loader accepted invalid {name}")
        (package / "plan.json").write_text(json.dumps(original))
        weight_path = package / "weights.bin"
        weights = weight_path.read_bytes()
        weight_path.write_bytes(weights[:-1] + bytes([weights[-1] ^ 1]))
        result = run(args.executor, "--inspect-plan", str(package))
        if result.returncode == 0:
            raise AssertionError("C++ loader accepted corrupted weights")
    print("decoder package, GPU block, state, and rejection checks passed")


if __name__ == "__main__":
    main()
