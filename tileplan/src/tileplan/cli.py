"""Command boundary for the planned end-to-end pipeline."""

import argparse


_COMMANDS = {
    "reference": "create the PyTorch MLP fixture and reference values",
    "probe": "query the compiled CUDA worker and write target.json",
    "build": "lower the MLP into a validated tile-plan package",
    "inspect": "show the schedule and memory plan",
    "verify": "validate a package without running inference",
    "benchmark": "compare the executor with PyTorch eager and graph replay",
    "demo": "build, plan, execute, check, and benchmark in one run",
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="tileplan")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in _COMMANDS.items():
        command = commands.add_parser(name, help=description)
        if name in {"reference", "demo"}:
            command.add_argument("--out", required=True)
    args = parser.parse_args()
    parser.exit(2, f"{args.command}: implementation pending after scaffold review\n")
    return 2
