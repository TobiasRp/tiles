"""Command-line entry points for each offline planning stage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import baselines
from .model import Model
from .pipeline import demo
from .plan import build, read_package, summary
from .probe import probe
from .reference import generate


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tileplan")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser(
        "reference", help="generate deterministic PyTorch MLP cases"
    )
    reference.add_argument("--out", type=_path, required=True)
    reference.add_argument("--seed", type=int, default=7)
    for name, default in (("d", 129), ("h", 65), ("o", 33)):
        reference.add_argument(f"--{name}", type=int, default=default)
    target = commands.add_parser(
        "probe", help="query final compiled CUDA worker capabilities"
    )
    target.add_argument("--executor", type=_path, required=True)
    target.add_argument("--out", type=_path, required=True)
    planner = commands.add_parser("build", help="write a validated tile-plan package")
    for name in ("model", "weights", "target", "out"):
        planner.add_argument(f"--{name}", type=_path, required=True)
    planner.add_argument("--tile-h1", type=int, required=True)
    planner.add_argument("--tile-h2", type=int, required=True)
    for name in ("inspect", "verify"):
        command = commands.add_parser(name, help=f"{name} a package without CUDA")
        command.add_argument("package", type=_path)
    benchmark = commands.add_parser(
        "benchmark", help="measure PyTorch eager and CUDA Graph baselines"
    )
    benchmark.add_argument("--package", type=_path, required=True)
    benchmark.add_argument("--cases", type=_path, required=True)
    benchmark.add_argument("--out", type=_path, required=True)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--rounds", type=int, default=5)
    driver = commands.add_parser(
        "demo", help="build, plan, execute, check, and benchmark"
    )
    driver.add_argument("--out", type=_path, required=True)
    driver.add_argument("--executor", type=_path)
    driver.add_argument("--seed", type=int, default=7)
    for name, default in (
        ("d", 129),
        ("h", 65),
        ("o", 33),
        ("tile-h1", 16),
        ("tile-h2", 16),
        ("iterations", 100),
        ("rounds", 5),
    ):
        driver.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    try:
        if args.command == "reference":
            generate(args.out, Model(args.d, args.h, args.o), args.seed)
            result = {"reference": str(args.out)}
        elif args.command == "probe":
            result = probe(args.executor, args.out)
        elif args.command == "build":
            result = build(
                args.model,
                args.weights,
                args.target,
                args.tile_h1,
                args.tile_h2,
                args.out,
            )
        elif args.command in {"inspect", "verify"}:
            plan, _, digest = read_package(args.package)
            result = (
                summary(plan, digest)
                if args.command == "inspect"
                else {"valid": True, "plan_sha256": digest}
            )
        elif args.command == "benchmark":
            result = baselines.run(
                args.package, args.cases, args.out, args.iterations, args.rounds
            )
        else:
            result = demo(
                args.out,
                args.executor,
                Model(args.d, args.h, args.o),
                args.seed,
                args.tile_h1,
                args.tile_h2,
                args.iterations,
                args.rounds,
            )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"tileplan {args.command}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return 0
