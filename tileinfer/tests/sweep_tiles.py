"""Run a frozen synthetic shape/tile sweep with independent pilot and final rounds."""

import argparse
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from tileplan.decoder_reference import DecoderSpec
from tileplan.decoder_stack import (
    StackCache,
    StackSpec,
    build_stack,
    decode_token,
    fixed_stack_weights,
    generate_stack_fixture,
)
from tileplan.probe import probe

SHAPES = ((64, 1, 1), (256, 4, 2), (1024, 16, 4))
TILES = (4, 16, 64)
POSITIONS = (0, 15, 63)
DTYPES = ("float32", "bfloat16")
PILOT = {"warmup": 3, "iterations": 20, "rounds": 3}
FINAL = {"warmup": 5, "iterations": 50, "rounds": 5}


class Selection(TypedDict):
    hidden_size: int
    weight_dtype: str
    tile_rows: int
    pilot_scores_us: dict[int, float]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executor", type=Path, required=True, help="tile_runtime_test"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    target = probe(args.executor, args.out / "target.json")
    protocol = {
        "shapes": SHAPES,
        "intermediate_multiplier": 4,
        "head_dim": 64,
        "layers": 2,
        "vocab_size": 1024,
        "max_context": 65,
        "token_id": 1,
        "tiles": TILES,
        "positions": POSITIONS,
        "dtypes": DTYPES,
        "pilot": PILOT,
        "final": FINAL,
        "selection": "minimum geometric mean of pilot means across all three positions",
        "order": "seeded permutation per phase, seed 20261013",
        "metric": "CUDA-event batch average at fixed context; no individual tail percentiles",
        "reference": "CPU PyTorch; stored float32 or BF16 weights converted to float32",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    packages = {}
    expected = {}
    for hidden, heads, kv_heads in SHAPES:
        spec = StackSpec(
            block=DecoderSpec(
                hidden_size=hidden,
                intermediate_size=4 * hidden,
                num_attention_heads=heads,
                num_key_value_heads=kv_heads,
                head_dim=64,
                max_context=65,
            ),
            layers=2,
            vocab_size=1024,
        )
        fixture = args.out / f"fixture-{hidden}"
        generate_stack_fixture(fixture, spec)
        weights = fixed_stack_weights(spec)
        for dtype in DTYPES:
            quantized = {
                name: value.to(torch.bfloat16).float() if dtype == "bfloat16" else value
                for name, value in weights.items()
            }
            cache = StackCache.empty(spec)
            with torch.inference_mode():
                for position in range(max(POSITIONS) + 1):
                    step = decode_token(1, quantized, spec, cache)
                    if position in POSITIONS:
                        logits = step["logits"]
                        expected[hidden, dtype, position] = (
                            int(logits.argmax()),
                            float(logits.max()),
                        )
            for tile in TILES:
                package = args.out / f"package-{hidden}-{dtype}-{tile}"
                build_stack(
                    fixture / "model.json",
                    fixture / "weights.npz",
                    args.out / "target.json",
                    tile,
                    package,
                    dtype,
                )
                packages[hidden, dtype, tile] = package
    pilot = []
    final = []
    selected: list[Selection] = []
    randomizer = random.Random(20261013)

    def measure(key: tuple[int, str, int, int], phase: str) -> dict:
        hidden, dtype, tile, position = key
        parameters = PILOT if phase == "pilot" else FINAL
        run = subprocess.run(
            [
                str(args.executor),
                "--benchmark-stack",
                str(packages[hidden, dtype, tile]),
                str(position),
                "1",
                str(parameters["warmup"]),
                str(parameters["iterations"]),
                str(parameters["rounds"]),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if run.returncode:
            raise AssertionError(run.stderr)
        result = json.loads(run.stdout)
        token, logit = expected[hidden, dtype, position]
        if result["top_token"] != token or not np.isclose(
            result["top_logit"], logit, atol=2e-5, rtol=2e-5
        ):
            raise AssertionError(
                (key, result["top_token"], result["top_logit"], token, logit)
            )
        row = {
            "hidden_size": hidden,
            "weight_dtype": dtype,
            "tile_rows": tile,
            **result,
        }
        print(
            f"{phase}: d={hidden} {dtype} tile={tile} context={position + 1}: {result['mean_us']:.3f} us",
            flush=True,
        )
        return row

    def save() -> None:
        (args.out / "report.json").write_text(
            json.dumps(
                {
                    "protocol": protocol,
                    "target": target,
                    "pilot": pilot,
                    "selected": selected,
                    "final": final,
                },
                indent=2,
            )
            + "\n"
        )

    order = [
        (hidden, dtype, tile, position)
        for hidden, _, _ in SHAPES
        for dtype in DTYPES
        for tile in TILES
        for position in POSITIONS
    ]
    randomizer.shuffle(order)
    for key in order:
        pilot.append(measure(key, "pilot"))
        save()
    for hidden, _, _ in SHAPES:
        for dtype in DTYPES:
            scores = {
                tile: math.exp(
                    statistics.mean(
                        math.log(row["mean_us"])
                        for row in pilot
                        if row["hidden_size"] == hidden
                        and row["weight_dtype"] == dtype
                        and row["tile_rows"] == tile
                    )
                )
                for tile in TILES
            }
            tile = min(TILES, key=lambda candidate: (scores[candidate], candidate))
            selected.append(
                {
                    "hidden_size": hidden,
                    "weight_dtype": dtype,
                    "tile_rows": tile,
                    "pilot_scores_us": scores,
                }
            )
    order = [
        (row["hidden_size"], row["weight_dtype"], row["tile_rows"], position)
        for row in selected
        for position in POSITIONS
    ]
    randomizer.shuffle(order)
    for key in order:
        final.append(measure(key, "final"))
        save()
    print(
        f"Completed {len(pilot)} pilot and {len(final)} independent final cases: {args.out / 'report.json'}"
    )


if __name__ == "__main__":
    main()
