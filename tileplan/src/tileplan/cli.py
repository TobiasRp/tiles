"""Decoder-only reference, planning, inspection, and visualization commands."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .decoder_block import build
from .decoder_reference import generate_fixture
from .decoder_stack import build_stack, generate_stack_fixture
from .llama32_import import import_llama32
from .package import read_package, summary
from .probe import probe
from .visualize import visualize


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tileplan")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference", help="write a decoder PyTorch fixture")
    reference.add_argument("--out", type=_path, required=True)
    reference.add_argument("--seed", type=int, default=19)
    stack_reference = commands.add_parser(
        "stack-reference", help="write a deterministic full decoder fixture"
    )
    stack_reference.add_argument("--out", type=_path, required=True)
    stack_reference.add_argument("--seed", type=int, default=37)
    target = commands.add_parser("probe", help="query the compiled decoder worker")
    target.add_argument("--executor", type=_path, required=True)
    target.add_argument("--out", type=_path, required=True)
    planner = commands.add_parser("build", help="write a decoder block package")
    for name in ("model", "weights", "target", "out"):
        planner.add_argument(f"--{name}", type=_path, required=True)
    planner.add_argument(
        "--dtype",
        "--weight-dtype",
        dest="weight_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    stack_planner = commands.add_parser(
        "build-stack", help="write a decoder stack package"
    )
    for name in ("model", "weights", "target", "out"):
        stack_planner.add_argument(f"--{name}", type=_path, required=True)
    stack_planner.add_argument(
        "--dtype",
        "--weight-dtype",
        dest="weight_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    importer = commands.add_parser(
        "import-llama32", help="pack the pinned Llama 3.2 1B base checkpoint"
    )
    for name in ("checkpoint", "target", "out"):
        importer.add_argument(f"--{name}", type=_path, required=True)
    importer.add_argument("--max-context", type=int, default=4096)
    importer.add_argument("--revision", help="pinned Hugging Face commit SHA")
    importer.add_argument(
        "--dtype",
        dest="weight_dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    for name in ("inspect", "verify"):
        command = commands.add_parser(name, help=f"{name} a decoder package")
        command.add_argument("package", type=_path)
    image = commands.add_parser("visualize", help="render one layer's schedule as SVG")
    image.add_argument("package", type=_path)
    image.add_argument("--out", type=_path, required=True)
    image.add_argument("--note", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "reference":
            generate_fixture(args.out, seed=args.seed)
            result = {"reference": str(args.out)}
        elif args.command == "stack-reference":
            generate_stack_fixture(args.out, seed=args.seed)
            result = {"reference": str(args.out)}
        elif args.command == "probe":
            result = probe(args.executor, args.out)
        elif args.command == "build":
            result = build(
                args.model,
                args.weights,
                args.target,
                args.out,
                args.weight_dtype,
            )
        elif args.command == "build-stack":
            result = build_stack(
                args.model,
                args.weights,
                args.target,
                args.out,
                args.weight_dtype,
            )
        elif args.command == "import-llama32":
            result = import_llama32(
                args.checkpoint,
                args.target,
                args.out,
                args.max_context,
                args.revision,
                args.weight_dtype,
            )
        elif args.command in {"inspect", "verify"}:
            plan, _, digest = read_package(args.package)
            result = (
                summary(plan, digest)
                if args.command == "inspect"
                else {"valid": True, "plan_sha256": digest}
            )
        else:
            result = {
                "visualization": str(visualize(args.package, args.out, note=args.note))
            }
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"tileplan {args.command}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0
