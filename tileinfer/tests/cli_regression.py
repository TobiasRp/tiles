"""CLI syntax must fail before a CUDA probe, even without a visible GPU."""

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", required=True)
    args = parser.parse_args()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cases = (
        (
            ["generate", "missing", "missing", "--max-new-tokens", "2tail"],
            "invalid integer",
        ),
        (
            [
                "generate",
                "missing",
                "missing",
                "--max-new-tokens",
                "2",
                "--max-new-tokens",
                "3",
            ],
            "duplicate option",
        ),
        (
            ["generate", "missing", "missing", "--eos-id", "0"],
            "missing required option",
        ),
        (
            [
                "generate",
                "missing",
                "missing",
                "--max-new-tokens",
                "2",
                "--eos-id",
                "-1",
            ],
            "nonnegative",
        ),
        (["generate", "missing", "missing", "--unknown", "1"], "unknown option"),
        (
            ["bench", "missing", "missing", "--new-tokens", "2", "--warmup", "0"],
            "missing required option",
        ),
        (
            [
                "bench",
                "missing",
                "missing",
                "--new-tokens",
                "2",
                "--warmup",
                "0",
                "--rounds",
                "0",
            ],
            "invalid generation",
        ),
        (
            ["generate", "missing", "missing", "--max-new-tokens"],
            "missing option value",
        ),
    )
    for command, expected in cases:
        result = subprocess.run(
            [args.executor, *command],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 2 or expected not in result.stderr:
            raise AssertionError((command, result.stderr))
    print(f"{len(cases)} pre-CUDA CLI rejection checks passed")


if __name__ == "__main__":
    main()
