"""Check warp GEMV, shared head scores, tails, and alternative legal schedules."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from loader_regression import rearrange
from stack_regression import check_generation, check_sequence
from tileplan.decoder_block import build
from tileplan.decoder_reference import (
    DecoderSpec,
    ReferenceCache,
    decode_one,
    fixed_tokens,
    fixed_weights,
    generate_fixture,
)
from tileplan.decoder_stack import (
    StackCache,
    StackSpec,
    build_stack,
    decode_token,
    fixed_stack_weights,
    generate_stack_fixture,
)
from tileplan.probe import probe
from tileplan.timeline import align, budget


def check_timeline(executor: Path, package: Path, position: int) -> None:
    """Trace a fixed-position launch and check its records and bubble budget."""
    plan = json.loads((package / "plan.json").read_text())
    trace = json.loads(
        subprocess.check_output(
            [
                str(executor),
                "--timeline-stack",
                str(package),
                str(position),
                "1",
                "1",
                "2",
            ],
            text=True,
            timeout=60,
        )
    )
    workers = plan["target"]["worker_count"]
    barriers = trace["instruction_tasks"].count(-1)
    # Skipped conditional barriers still record one instruction per worker.
    if (
        len(trace["samples"]) != 2
        or sorted(t for t in trace["instruction_tasks"] if t >= 0)
        != list(range(len(plan["tasks"])))
        or barriers != workers * (len(plan["waves"]) - 1)
    ):
        raise AssertionError("timeline does not cover every task and barrier once")
    for sample in trace["samples"]:
        if any(
            end < start
            for start, end in zip(
                sample["start_cycle"], sample["end_cycle"], strict=True
            )
        ):
            raise AssertionError("timeline instruction ends before it starts")
    events, _ = align(trace)
    report = budget(plan, events, position, 200.0)
    parts = report["ideal_stream_us"] + sum(report["components_us"].values())
    if abs(parts - report["traced_kernel_us"]) > 1e-6:
        raise AssertionError("bubble budget does not sum to the traced kernel time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--block-executor", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="kernel-regression-") as directory:
        root = Path(directory)
        probe(args.executor, root / "target.json")
        # A long context split into up to five uneven 16-step chunks after an
        # 8-step fused prefix, and the widest supported head in three chunks.
        specs = (
            DecoderSpec(
                hidden_size=32, intermediate_size=93, head_dim=8, max_context=67
            ),
            DecoderSpec(
                hidden_size=256,
                intermediate_size=333,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=128,
                max_context=9,
            ),
        )
        chunking = ((16, 8), (4, 4))
        for number, spec in enumerate(specs):
            fixture = root / f"fixture-{number}"
            generate_fixture(fixture, spec)
            weights = fixed_weights(spec)
            weights["Wq"] *= 8
            weights["Wk"] *= 8
            np.savez(
                fixture / "weights.npz",
                **{
                    name: value.float().cpu().numpy() for name, value in weights.items()
                },
            )
            tokens = fixed_tokens(spec).repeat((spec.max_context + 4) // 5, 1)[
                : spec.max_context
            ]
            for dtype in ("float32", "bfloat16"):
                package = root / f"block-{number}-{dtype}"
                build(
                    fixture / "model.json",
                    fixture / "weights.npz",
                    root / "target.json",
                    package,
                    dtype,
                    *chunking[number],
                )
                path = package / "plan.json"
                plan = json.loads(path.read_text())
                rearrange(plan)
                path.write_text(json.dumps(plan))
                quantized = {
                    name: value.to(device="cuda", dtype=torch.bfloat16)
                    if dtype == "bfloat16"
                    else value
                    for name, value in weights.items()
                }
                cache = ReferenceCache.empty(
                    spec, quantized["Wq"].dtype, quantized["Wq"].device
                )
                with torch.inference_mode():
                    expected = [
                        decode_one(token.to(quantized["Wq"]), quantized, spec, cache)
                        for token in tokens
                    ]
                inputs = root / "block-inputs.json"
                inputs.write_text(
                    json.dumps(
                        {
                            "tokens": [{"hidden": token.tolist()} for token in tokens]
                            + [{"reset": True}, {"hidden": tokens[1].tolist()}]
                        }
                    )
                )
                cache.reset()
                with torch.inference_mode():
                    expected.append(
                        decode_one(
                            tokens[1].to(quantized["Wq"]), quantized, spec, cache
                        )
                    )
                run = subprocess.run(
                    [
                        str(args.block_executor),
                        "--run-block",
                        str(package),
                        str(inputs),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if run.returncode:
                    raise AssertionError(run.stderr)
                actual = json.loads(run.stdout)["steps"]
                for index, (step, reference) in enumerate(
                    zip(actual, expected, strict=True)
                ):
                    for field in (
                        "q",
                        "k",
                        "v",
                        "attention",
                        "attn_residual",
                        "ffn",
                        "y",
                    ):
                        np.testing.assert_allclose(
                            step[field],
                            reference[field].float().cpu().numpy().reshape(-1),
                            atol=1e-2 if dtype == "bfloat16" else spec.atol,
                            rtol=1e-2 if dtype == "bfloat16" else spec.rtol,
                            err_msg=f"{number}/{dtype}/{index}/{field}",
                        )
        spec = StackSpec(
            block=specs[0], layers=2, vocab_size=53, tie_word_embeddings=False
        )
        fixture = root / "stack-fixture"
        generate_stack_fixture(fixture, spec)
        weights = fixed_stack_weights(spec)
        ids = [0, 1, 52, 7]
        for dtype in ("float32", "bfloat16"):
            package = root / f"stack-{dtype}"
            # Positions 0-1 run fused attention, 2-3 split it.
            build_stack(
                fixture / "model.json",
                fixture / "weights.npz",
                root / "target.json",
                package,
                dtype,
                2,
                2,
            )
            quantized = {
                name: value.to(device="cuda", dtype=torch.bfloat16)
                if dtype == "bfloat16"
                else value
                for name, value in weights.items()
            }
            cache = StackCache.empty(
                spec, quantized["embedding"].dtype, quantized["embedding"].device
            )
            with torch.inference_mode():
                traces = [decode_token(token, quantized, spec, cache) for token in ids]
            check_sequence(args.executor, package, root, ids, traces, spec)
            check_generation(args.executor, package, root, ids[:2], quantized, spec)
            for position in (1, 2):  # fused, then split
                check_timeline(args.block_executor, package, position)
            # Zero logits tie across the whole vocabulary. Sampling must keep
            # the smallest token ID in every precision.
            tied_weights = root / "ties-weights.npz"
            np.savez(
                tied_weights,
                **{
                    name: np.zeros_like(value.numpy())
                    for name, value in weights.items()
                },
            )
            tied_package = root / f"ties-{dtype}"
            build_stack(
                fixture / "model.json",
                tied_weights,
                root / "target.json",
                tied_package,
                dtype,
            )
            tied_input = root / "ties-input.json"
            tied_input.write_text(json.dumps({"tokens": [1]}))
            tied = json.loads(
                subprocess.check_output(
                    [
                        str(args.executor),
                        "generate",
                        str(tied_package),
                        str(tied_input),
                        "--max-new-tokens",
                        "2",
                    ],
                    text=True,
                )
            )
            if tied["generated"] != [0, 0]:
                raise AssertionError("greedy tie did not select the smallest token ID")
    print(
        "long/extreme-score attention, wide heads, GEMV/logits tails, and legal schedules passed"
    )


if __name__ == "__main__":
    main()
