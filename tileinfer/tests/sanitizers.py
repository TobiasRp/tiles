"""Generate fresh packages and run all four Compute Sanitizer checks."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tileplan.decoder_plan import build
from tileplan.decoder_reference import DecoderSpec, generate_fixture
from tileplan.decoder_stack import build_stack
from tileplan.probe import probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--sanitizer", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report-api-errors", choices=("all", "no"), default="no")
    parser.add_argument(
        "--matrix-kernel",
        choices=("simt", "mma"),
        help="restrict stack/block cases to one matrix implementation; isolated operators remain included",
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
        tail, DecoderSpec(hidden_size=40, intermediate_size=93, head_dim=10)
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
    for dtype, matrix_kernel in (
        ("float32", "simt"),
        ("bfloat16", "simt"),
        ("bfloat16", "mma"),
    ):
        if args.matrix_kernel is not None and matrix_kernel != args.matrix_kernel:
            continue
        block = root / f"block-{dtype}-{matrix_kernel}"
        stack = root / f"stack-{dtype}-{matrix_kernel}"
        build(
            tail / "model.json",
            tail / "weights.npz",
            root / "target.json",
            7,
            block,
            dtype,
            matrix_kernel,
        )
        source = fixture_root / "decoder_m4"
        build_stack(
            source / "model.json",
            source / "weights.npz",
            root / "target.json",
            7,
            stack,
            dtype,
            matrix_kernel,
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
        cases[f"{dtype}-{matrix_kernel}-generate-eos"] = [
            *generation,
            "--eos-id",
            str(eos),
        ]
        cases[f"{dtype}-{matrix_kernel}-check-reset"] = [
            str(executor),
            "check",
            str(stack),
            str(root / "check.json"),
        ]
        cases[f"{dtype}-{matrix_kernel}-block-tail"] = [
            str(build_dir / "tile_runtime_test"),
            "--run-block",
            str(block),
            str(root / "block.json"),
        ]
        cases[f"{dtype}-{matrix_kernel}-tickets"] = [
            str(build_dir / "sequence_context_smoke"),
            str(stack),
        ]
        cases[f"{dtype}-{matrix_kernel}-ring"] = [
            str(build_dir / "sequence_ring_smoke"),
            str(stack),
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
                        timeout=600 if name.endswith("-ring") else 180
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
        "matrix_kernel_filter": args.matrix_kernel,
        "api_error_reporting": args.report_api_errors,
        "gated_initcheck_api_memory_access": "disabled only for the controlled ring cases",
        "target": json.loads((root / "target.json").read_text()),
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(result["returncode"] for result in results):
        raise SystemExit(f"Sanitizer failures; see {root}")
    print(f"All {len(results)} checks passed; logs: {root}")


if __name__ == "__main__":
    main()
