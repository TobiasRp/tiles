"""GPU and loader regression for a synthetic full decoder stack."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from tileplan.decoder_stack import (
    StackCache,
    StackSpec,
    build_stack,
    decode_token,
    fixed_stack_weights,
    generate_stack_fixture,
)
from tileplan.probe import probe


def run(executor: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executor), *args], capture_output=True, text=True, check=False
    )


def check_sequence(
    executor: Path,
    package: Path,
    root: Path,
    token_ids: list[int],
    expected: list[dict],
    spec: StackSpec,
) -> None:
    request = {
        "trace_blocks": True,
        "tokens": token_ids + [{"reset": True}, token_ids[0]],
    }
    input_path = root / "stack_inputs.json"
    input_path.write_text(json.dumps(request))
    result = run(executor, "check", str(package), str(input_path))
    if result.returncode:
        raise AssertionError(result.stderr)
    report = json.loads(result.stdout)
    traces = expected + [expected[0]]
    for index, (step, reference) in enumerate(
        zip(report["steps"], traces, strict=True)
    ):
        position = index if index < len(token_ids) else 0
        if step["position"] != position:
            raise AssertionError("wrong stack token position")
        np.testing.assert_allclose(
            step["logits"],
            reference["logits"].numpy(),
            atol=spec.block.atol,
            rtol=spec.block.rtol,
            err_msg=f"logits at token {index}",
        )
        np.testing.assert_allclose(
            step["blocks"],
            np.stack([block["y"].numpy() for block in reference["blocks"]]),
            atol=spec.block.atol,
            rtol=spec.block.rtol,
            err_msg=f"block outputs at token {index}",
        )
        for layer in range(spec.layers):
            for field in ("k", "v"):
                np.testing.assert_allclose(
                    np.asarray(step[field][layer]).reshape(
                        spec.block.num_key_value_heads, spec.block.head_dim
                    ),
                    reference["blocks"][layer][field].numpy(),
                    atol=spec.block.atol,
                    rtol=spec.block.rtol,
                    err_msg=f"{field} layer {layer} token {index}",
                )
    if report["position"] != 1 or not report["valid"]:
        raise AssertionError("stack reset failed")
    input_path.write_text(
        json.dumps({"tokens": [token_ids[0]] * (spec.block.max_context + 1)})
    )
    result = run(executor, "check", str(package), str(input_path))
    if result.returncode == 0 or "capacity reached" not in result.stderr:
        raise AssertionError("stack capacity check failed")
    input_path.write_text(json.dumps({"tokens": [spec.vocab_size]}))
    result = run(executor, "check", str(package), str(input_path))
    if result.returncode == 0 or "outside vocabulary" not in result.stderr:
        raise AssertionError("token range check failed")


def check_generation(
    executor: Path,
    package: Path,
    root: Path,
    prompt: list[int],
    weights: dict,
    spec: StackSpec,
) -> None:
    input_path = root / "generate_inputs.json"
    input_path.write_text(json.dumps({"tokens": prompt}))
    cache = StackCache.empty(spec)
    with torch.inference_mode():
        for token in prompt:
            step = decode_token(token, weights, spec, cache)
        expected = []
        for _ in range(2):
            token = int(step["logits"].argmax())
            expected.append(token)
            step = decode_token(token, weights, spec, cache)
    result = run(
        executor,
        "generate",
        str(package),
        str(input_path),
        "--max-new-tokens",
        "2",
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    actual = json.loads(result.stdout)
    if (
        actual["generated"] != expected
        or actual["generated_count"] != 2
        or actual["position"] != len(prompt) + 2
        or actual["stop_reason"] != "limit"
    ):
        raise AssertionError(actual)
    invalid_count = run(
        executor,
        "generate",
        str(package),
        str(input_path),
        "--max-new-tokens",
        "2trailing",
    )
    if invalid_count.returncode == 0 or "invalid integer" not in invalid_count.stderr:
        raise AssertionError("CLI accepted a partial integer")
    eos = run(
        executor,
        "generate",
        str(package),
        str(input_path),
        "--max-new-tokens",
        "2",
        "--eos-id",
        str(expected[0]),
    )
    if eos.returncode:
        raise AssertionError(eos.stderr)
    stopped = json.loads(eos.stdout)
    if (
        stopped["generated"] != expected[:1]
        or stopped["position"] != len(prompt) + 1
        or stopped["stop_reason"] != "eos"
    ):
        raise AssertionError(stopped)
    bench = run(
        executor,
        "bench",
        str(package),
        str(input_path),
        "--new-tokens",
        "2",
        "--warmup",
        "0",
        "--rounds",
        "1",
    )
    if bench.returncode:
        raise AssertionError(bench.stderr)
    measurement = json.loads(bench.stdout)
    if measurement["generated"] != expected or measurement["mean_us"] <= 0:
        raise AssertionError(measurement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--sequence-smoke", type=Path)
    parser.add_argument("--ring-smoke", type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).parents[2] / "tileplan/tests/fixtures/decoder_m4"
    with tempfile.TemporaryDirectory(prefix="stack-regression-") as directory:
        root = Path(directory)
        probe(args.executor, root / "target.json")
        spec = StackSpec()
        with np.load(fixture / "reference_cases.npz", allow_pickle=False) as cases:
            ids = cases["token_ids"].tolist()
            expected_logits = cases["logits"].copy()
            expected_k = cases["k_cache"].copy()
            expected_v = cases["v_cache"].copy()
        weights = fixed_stack_weights(spec)
        cache = StackCache.empty(spec)
        with torch.inference_mode():
            traces = [decode_token(token, weights, spec, cache) for token in ids]
        np.testing.assert_allclose(
            np.stack([trace["logits"].numpy() for trace in traces]),
            expected_logits,
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.stack([layer.k.numpy() for layer in cache.layers]), expected_k
        )
        np.testing.assert_allclose(
            np.stack([layer.v.numpy() for layer in cache.layers]), expected_v
        )
        for tile_rows in (4, 7):
            package = root / f"package-{tile_rows}"
            build_stack(
                fixture / "model.json",
                fixture / "weights.npz",
                root / "target.json",
                tile_rows,
                package,
            )
            result = run(args.executor, "--inspect-plan", str(package))
            if result.returncode:
                raise AssertionError(result.stderr)
            report = json.loads(result.stdout)
            if report["layers"] != spec.layers or report["kv_bytes"] != 2304:
                raise AssertionError(report)
            check_sequence(args.executor, package, root, ids, traces, spec)
            check_generation(args.executor, package, root, ids[:2], weights, spec)
            if args.sequence_smoke and tile_rows == 4:
                context_result = run(args.sequence_smoke, str(package))
                if context_result.returncode:
                    raise AssertionError(context_result.stderr)
        if args.ring_smoke:
            for ring_package in (root / "package-4",):
                ring_result = run(args.ring_smoke, str(ring_package))
                if ring_result.returncode:
                    raise AssertionError(ring_result.stderr)
        # Old packages retain their 128-thread target; the current worker executes
        # the same records using its own block size and freshly queried occupancy.
        legacy_path = package / "plan.json"
        current = json.loads(legacy_path.read_text())
        legacy = copy.deepcopy(current)
        legacy["header"].update(format_version=2, executor_abi="tileinfer-decoder-v2")
        legacy["target"].update(
            threads_per_block=128,
            kernel_fingerprint="decoder-wave-abi2-ops-v3-stack-sm86-b128",
        )
        legacy_path.write_text(json.dumps(legacy))
        check_sequence(args.executor, package, root, ids, traces, spec)
        check_generation(args.executor, package, root, ids[:2], weights, spec)
        legacy_path.write_text(json.dumps(current))
        original_weight_bytes = report["weight_bytes"]

        bf16_package = root / "bf16-stack"
        build_stack(
            fixture / "model.json",
            fixture / "weights.npz",
            root / "target.json",
            7,
            bf16_package,
            "bfloat16",
        )
        bf16_plan = json.loads((bf16_package / "plan.json").read_text())
        if bf16_plan["storage"]["stack"]["lm_head_offset"] != 0:
            raise AssertionError("tied BF16 output is not aliased")
        if bf16_plan["storage"]["weight_bytes"] >= original_weight_bytes:
            raise AssertionError("BF16 weights did not reduce storage")
        bf16_weights = {
            name: value.to(torch.bfloat16).float() for name, value in weights.items()
        }
        bf16_cache = StackCache.empty(spec)
        with torch.inference_mode():
            bf16_traces = [
                decode_token(token, bf16_weights, spec, bf16_cache) for token in ids
            ]
        check_sequence(args.executor, bf16_package, root, ids, bf16_traces, spec)
        check_generation(args.executor, bf16_package, root, ids[:2], bf16_weights, spec)
        if args.sequence_smoke:
            context_result = run(args.sequence_smoke, str(bf16_package))
            if context_result.returncode:
                raise AssertionError(context_result.stderr)

        if args.ring_smoke:
            ring_result = run(args.ring_smoke, str(bf16_package))
            if ring_result.returncode:
                raise AssertionError(ring_result.stderr)
        original = json.loads((package / "plan.json").read_text())
        for name, mutate in {
            "layer stride": lambda plan: plan["storage"]["stack"].update(
                layer_weight_stride=0
            ),
            "stage": lambda plan: plan["execution"].update(prelude="logits_f32"),
        }.items():
            bad = copy.deepcopy(original)
            mutate(bad)
            (package / "plan.json").write_text(json.dumps(bad))
            result = run(args.executor, "--inspect-plan", str(package))
            if result.returncode == 0:
                raise AssertionError(f"C++ loader accepted invalid {name}")

        untied = StackSpec(layers=2, tie_word_embeddings=False)
        untied_fixture = root / "untied-fixture"
        generate_stack_fixture(untied_fixture, untied)
        untied_package = root / "untied-package"
        build_stack(
            untied_fixture / "model.json",
            untied_fixture / "weights.npz",
            root / "target.json",
            7,
            untied_package,
        )
        untied_weights = fixed_stack_weights(untied)
        untied_cache = StackCache.empty(untied)
        with np.load(untied_fixture / "reference_cases.npz") as cases:
            untied_ids = cases["token_ids"].tolist()
        with torch.inference_mode():
            untied_traces = [
                decode_token(token, untied_weights, untied, untied_cache)
                for token in untied_ids
            ]
        check_sequence(
            args.executor,
            untied_package,
            root,
            untied_ids,
            untied_traces,
            untied,
        )
        untied_bf16_package = root / "untied-bf16-package"
        build_stack(
            untied_fixture / "model.json",
            untied_fixture / "weights.npz",
            root / "target.json",
            7,
            untied_bf16_package,
            "bfloat16",
        )
        untied_bf16_plan = json.loads((untied_bf16_package / "plan.json").read_text())
        if untied_bf16_plan["storage"]["stack"]["lm_head_offset"] == 0:
            raise AssertionError("untied BF16 output incorrectly aliases embedding")
        untied_bf16_weights = {
            name: value.to(torch.bfloat16).float()
            for name, value in untied_weights.items()
        }
        untied_bf16_cache = StackCache.empty(untied)
        with torch.inference_mode():
            untied_bf16_traces = [
                decode_token(token, untied_bf16_weights, untied, untied_bf16_cache)
                for token in untied_ids
            ]
        check_sequence(
            args.executor,
            untied_bf16_package,
            root,
            untied_ids,
            untied_bf16_traces,
            untied,
        )
    print(
        "stack logits, per-layer cache, tails, reset, capacity, and loader checks passed"
    )


if __name__ == "__main__":
    main()
