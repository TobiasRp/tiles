"""Generate fresh packages and run all four Compute Sanitizer checks."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from tileplan.decoder_block import build
from tileplan.decoder_reference import DecoderSpec, generate_fixture
from tileplan.decoder_stack import StackSpec, build_stack, generate_stack_fixture
from tileplan.probe import probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-api-errors", choices=("all", "no"), default="no")
    parser.add_argument(
        "--keep", action="store_true", help="keep the run directory after a pass"
    )
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    out = args.out or build_dir / "sanitizers"
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=out))
    executor = build_dir / "tile_runtime"
    probe(executor, root / "target.json")
    fixture_root = Path(__file__).parents[2] / "tileplan/tests/fixtures"
    tail = root / "tail-fixture"
    generate_fixture(
        tail, DecoderSpec(hidden_size=32, intermediate_size=93, head_dim=8)
    )
    with np.load(tail / "reference_cases.npz") as source:
        inputs = source["x"][:3].tolist()
    (root / "block.json").write_text(
        json.dumps({"tokens": [{"hidden": value} for value in inputs]})
    )
    (root / "prompt.json").write_text(json.dumps({"tokens": [0, 1]}))
    (root / "check.json").write_text(
        json.dumps({"trace_blocks": True, "tokens": [0, 1, {"reset": True}, 1]})
    )
    cases = {}
    for dtype in ("float32", "bfloat16"):
        block = root / f"block-{dtype}"
        stack = root / f"stack-{dtype}"
        # One-step chunks: position 0 runs fused attention and skips the
        # conditional barrier, later positions split it into two or more chunks.
        build(
            tail / "model.json",
            tail / "weights.npz",
            root / "target.json",
            block,
            dtype,
            1,
            1,
        )
        source = fixture_root / "decoder_m4"
        build_stack(
            source / "model.json",
            source / "weights.npz",
            root / "target.json",
            stack,
            dtype,
            1,
            1,
        )
        generation = [
            str(executor),
            "generate",
            str(stack),
            str(root / "prompt.json"),
            "--max-new-tokens",
            "2",
        ]
        baseline = subprocess.run(
            generation, capture_output=True, text=True, check=True, timeout=30
        )
        eos = json.loads(baseline.stdout)["generated"][0]
        cases[f"{dtype}-generate-eos"] = [
            *generation,
            "--eos-id",
            str(eos),
        ]
        cases[f"{dtype}-check-reset"] = [
            str(executor),
            "check",
            str(stack),
            str(root / "check.json"),
        ]
        cases[f"{dtype}-block-tail"] = [
            str(build_dir / "tile_runtime_test"),
            "--run-block",
            str(block),
            str(root / "block.json"),
        ]
        for position, regime in ((0, "fused"), (2, "split")):
            cases[f"{dtype}-timeline-{regime}"] = [
                str(build_dir / "tile_runtime_test"),
                "--timeline-stack",
                str(stack),
                str(position),
                "1",
                "1",
                "1",
            ]
        cases[f"{dtype}-tickets"] = [
            str(build_dir / "sequence_context_smoke"),
            str(stack),
        ]
        cases[f"{dtype}-ring"] = [
            str(build_dir / "sequence_ring_smoke"),
            str(stack),
        ]
    # Prefill and generation over several launches of up to 64 steps, with
    # split attention beyond 32 steps.
    long_fixture = root / "long-fixture"
    generate_stack_fixture(
        long_fixture, StackSpec(DecoderSpec(max_context=160), layers=2)
    )
    long_stack = root / "long-stack"
    build_stack(
        long_fixture / "model.json",
        long_fixture / "weights.npz",
        root / "target.json",
        long_stack,
        "bfloat16",
        64,
        32,
    )
    (root / "long-prompt.json").write_text(
        json.dumps({"tokens": [(7 * index + 3) % 47 for index in range(70)]})
    )
    cases["bfloat16-multi-launch-generate"] = [
        str(executor),
        "generate",
        str(long_stack),
        str(root / "long-prompt.json"),
        "--max-new-tokens",
        "85",
    ]
    ops_cases = root / "ops-cases.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("ops_unit.py")),
            "--executor",
            str(build_dir / "ops_unit"),
            "--write-cases",
            str(ops_cases),
        ],
        check=True,
    )
    cases["isolated-operators"] = [str(build_dir / "ops_unit"), str(ops_cases)]
    cases["ring-copy"] = [str(build_dir / "input_ring_copy_smoke")]
    results = []
    if args.report_api_errors == "no":
        print(
            "CUDA API-error reporting disabled for the documented CCCL pinned-pool initialization issue; use --report-api-errors all to enable it.",
            flush=True,
        )
    for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
        for name, command in cases.items():
            label = f"{tool}-{name}"
            log = root / f"{label}.log"
            options = []
            if tool == "initcheck" and name.endswith("-ring"):
                # API inspection synchronizes copies, which cannot complete while
                # this test intentionally holds their stream. Kernel reads remain checked.
                options = ["--check-api-memory-access", "no"]
            full_command = [
                str(args.sanitizer),
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                "--report-api-errors",
                args.report_api_errors,
                "--log-file",
                str(log),
                *options,
                *command,
            ]
            with (root / f"{label}.stdout").open("w") as output:
                process = subprocess.Popen(
                    full_command,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    _, stderr = process.communicate(
                        timeout=600
                        if name.endswith(("-ring", "-multi-launch-generate"))
                        else 180
                    )
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    _, stderr = process.communicate()
                    stderr += (
                        "\nSanitizer test timed out; its process group was terminated."
                    )
                    returncode = 124
            results.append(
                {
                    "case": label,
                    "returncode": returncode,
                    "log": str(log),
                    "command": full_command,
                    "stderr": stderr,
                }
            )
            print(
                f"{label}: {'passed' if returncode == 0 else 'FAILED'}",
                flush=True,
            )
    summary = {
        "api_error_reporting": args.report_api_errors,
        "gated_initcheck_api_memory_access": "disabled only for the controlled ring cases",
        "target": json.loads((root / "target.json").read_text()),
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(result["returncode"] for result in results):
        raise SystemExit(f"Sanitizer failures; see {root}")
    if args.keep:
        print(f"All {len(results)} checks passed; logs: {root}")
    else:
        # A full run holds several GB of packages; failed runs stay for inspection.
        shutil.rmtree(root)
        print(f"All {len(results)} checks passed")


if __name__ == "__main__":
    main()
