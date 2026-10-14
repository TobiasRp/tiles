"""Controlled BF16 SIMT/MMA tile pilots and independent final checkpoint rounds."""

import argparse
import gc
import json
import math
import os
import random
import statistics
import subprocess
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from benchmark_decode import _baseline
from llama32_checkpoint import TOLERANCE_CALIBRATION
from tileplan.decoder_plan import _json_bytes
from tileplan.decoder_reference import DecoderSpec
from tileplan.decoder_stack import StackSpec, make_stack_plan_metadata
from tileplan.probe import probe

POSITIONS = (0, 63, 255)
TILES = (64, 256)
KERNELS = ("simt", "mma")
PILOT = {"warmup": 1, "iterations": 2, "rounds": 2}
FINAL = {"warmup": 3, "iterations": 5, "rounds": 5}


def telemetry() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--executor", type=Path, required=True, help="tile_runtime_test"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.package / "plan.json").read_text())
    model = source["model"]
    if (
        source["header"]["format_version"] != 4
        or model["activation_dtype"] != "bfloat16"
    ):
        raise ValueError("requires a version-4 BF16-compute checkpoint package")
    block = DecoderSpec(
        **{
            field.name: model[field.name]
            for field in fields(DecoderSpec)
            if field.name != "dtype"
        }
    )
    if block.max_context <= max(POSITIONS):
        raise ValueError("package capacity is too small for the frozen context suite")
    spec = StackSpec(
        block, model["layers"], model["vocab_size"], model["tie_word_embeddings"]
    )
    target = probe(args.executor, args.out / "target.json")
    protocol = {
        "positions": POSITIONS,
        "tiles": TILES,
        "kernels": KERNELS,
        "pilot": PILOT,
        "final": FINAL,
        "seed": 20261013,
        "selection": "minimum geometric mean across pilot context means for each kernel",
        "timing": "CUDA-event batch averages; setup/prefix/readback excluded",
        "prefix": "sequential one-token calls",
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
        "clocks_locked": False,
        "weights_sha256": source["integrity"]["weights_sha256"],
        "compiler_version": subprocess.check_output(
            ["/usr/local/cuda/bin/nvcc", "--version"], text=True
        ).strip(),
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "executable_sha256": sha256(args.executor.read_bytes()).hexdigest(),
    }
    packages = {}
    for kernel in KERNELS:
        for tile in TILES:
            package = args.out / f"package-{kernel}-{tile}"
            package.mkdir()
            plan = make_stack_plan_metadata(
                spec,
                target,
                tile,
                "bfloat16",
                source["integrity"]["weights_sha256"],
                kernel,
            )
            plan["source"] = source["source"]
            (package / "plan.json").write_bytes(_json_bytes(plan))
            os.link(args.package / "weights.bin", package / "weights.bin")
            packages[kernel, tile] = package
    before = telemetry()
    # Release PyTorch weights before launching TileInfer subprocesses.
    baseline = _baseline(
        args.checkpoint,
        list(POSITIONS),
        1,
        FINAL["warmup"],
        FINAL["iterations"],
        FINAL["rounds"],
        "bfloat16",
    )
    reference = {row["position"]: row for row in baseline["rows"]}
    gc.collect()
    torch.cuda.empty_cache()
    rng = random.Random(20261013)
    pilot, final = [], []

    def measure(kernel, tile, position, parameters):
        output = subprocess.check_output(
            [
                str(args.executor),
                "--benchmark-stack",
                str(packages[kernel, tile]),
                str(position),
                "1",
                str(parameters["warmup"]),
                str(parameters["iterations"]),
                str(parameters["rounds"]),
            ],
            text=True,
        )
        row = json.loads(output)
        expected = reference[position]
        limits = TOLERANCE_CALIBRATION["bfloat16"]["logits"]
        if row["top_token"] != expected["next_token"] or abs(
            row["top_logit"] - expected["top_logit"]
        ) > limits["atol"] + limits["rtol"] * abs(expected["top_logit"]):
            raise AssertionError(
                f"checkpoint output mismatch for {kernel}/{tile}/{position}"
            )
        row.update(tile_rows=tile)
        return row

    order = [
        (kernel, tile, position)
        for kernel in KERNELS
        for tile in TILES
        for position in POSITIONS
    ]
    rng.shuffle(order)
    for kernel, tile, position in order:
        row = measure(kernel, tile, position, PILOT)
        pilot.append(row)
        print(
            f"pilot {kernel} tile={tile} context={position + 1}: {row['mean_us']:.1f} us",
            flush=True,
        )
    selected = {}
    for kernel in KERNELS:
        scores = {
            tile: math.exp(
                statistics.mean(
                    math.log(row["mean_us"])
                    for row in pilot
                    if row["matrix_kernel"] == kernel and row["tile_rows"] == tile
                )
            )
            for tile in TILES
        }
        selected[kernel] = {
            "tile_rows": min(scores, key=lambda tile: scores[tile]),
            "pilot_scores_us": scores,
        }
    order = [
        (kernel, selected[kernel]["tile_rows"], position)
        for kernel in KERNELS
        for position in POSITIONS
    ]
    rng.shuffle(order)
    for kernel, tile, position in order:
        row = measure(kernel, tile, position, FINAL)
        final.append(row)
        print(
            f"final {kernel} tile={tile} context={position + 1}: {row['mean_us']:.1f} us",
            flush=True,
        )
    report = {
        "protocol": protocol,
        "target": target,
        "telemetry_before": before,
        "telemetry_after": telemetry(),
        "baseline": baseline,
        "pilot": pilot,
        "selected": selected,
        "final": final,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.out / "report.json")


if __name__ == "__main__":
    main()
