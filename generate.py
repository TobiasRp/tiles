#!/usr/bin/env python3
"""Build, plan, and generate text with the local Llama 3.2 1B checkpoint."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "tileplan"
INFER = ROOT / "tileinfer"
CHECKPOINT = ROOT / "checkpoints" / "Llama-3.2-1B"
EXECUTOR = INFER / "build" / "release" / "tile_runtime"
VENV = PLAN / ".venv"
# Package format accepted by the runtime built from this checkout.
FORMAT = 4


def run(command, *, cwd):
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--max-context", type=int, help="context size for a new package (default: 4096)"
    )
    parser.add_argument(
        "--package",
        type=Path,
        help="existing Llama 3.2 1B package or run directory; skips TilePlan",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        help="new-package dtype (default: bfloat16); must match when reusing a package",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not CHECKPOINT.is_dir():
        parser.error(f"checkpoint is missing: {CHECKPOINT}")

    package = None
    if args.package is not None:
        if args.max_context is not None:
            parser.error("--max-context only applies when creating a new package")
        selected = args.package.expanduser().resolve()
        package = (
            selected if (selected / "plan.json").is_file() else selected / "package"
        )
        if not (package / "plan.json").is_file():
            parser.error(f"no package/plan.json found at {selected}")
        plan = json.loads((package / "plan.json").read_text())
        if plan.get("source", {}).get("repository") != "meta-llama/Llama-3.2-1B":
            parser.error(f"not a Llama 3.2 1B checkpoint package: {package}")
        model = plan.get("model", {})
        if plan["header"]["format_version"] != FORMAT:
            parser.error(
                f"package format {plan['header']['format_version']} is retired; "
                f"regenerate a version-{FORMAT} package"
            )
        if args.dtype is not None and args.dtype != model.get(
            "weight_dtype", "float32"
        ):
            parser.error("--dtype conflicts with the existing package")
        max_context = model.get("max_context")
        if type(max_context) is not int or max_context < 2:
            parser.error(f"invalid max_context in {package / 'plan.json'}")
    else:
        max_context = args.max_context if args.max_context is not None else 4096
        if max_context < 2:
            parser.error("--max-context must be at least 2")

    if Path(sys.prefix).resolve() != VENV.resolve():
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        print(
            "Preparing checkpoint Python dependencies...", file=sys.stderr, flush=True
        )
        run([uv, "sync", "--locked", "--group", "checkpoint"], cwd=PLAN)
        python = VENV / "bin" / "python"
        os.execv(python, [str(python), str(__file__), *sys.argv[1:]])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    prompt_tokens = tokenizer(args.prompt, add_special_tokens=True)["input_ids"]
    if len(prompt_tokens) + args.max_new_tokens > max_context:
        parser.error(
            f"prompt and generated tokens exceed the {max_context}-token context"
        )

    print("Building TileInfer...", file=sys.stderr, flush=True)
    cache = INFER / "build" / "release" / "CMakeCache.txt"
    configure = ["cmake", "--preset", "release"]
    if (
        cache.exists()
        and f"CMAKE_HOME_DIRECTORY:INTERNAL={INFER}" not in cache.read_text()
    ):
        configure.insert(1, "--fresh")
    run(configure, cwd=INFER)
    run(["cmake", "--build", "--preset", "release"], cwd=INFER)

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="llama32-", dir=runs))
    inputs = run_dir / "inputs.json"
    inputs.write_text(json.dumps({"tokens": prompt_tokens}) + "\n")

    if package is None:
        target = run_dir / "target.json"
        package = run_dir / "package"
        print(
            f"Probing worker and importing checkpoint into {run_dir}...",
            file=sys.stderr,
            flush=True,
        )
        run(
            [
                sys.executable,
                "-m",
                "tileplan",
                "probe",
                "--executor",
                EXECUTOR,
                "--out",
                target,
            ],
            cwd=PLAN,
        )
        run(
            [
                sys.executable,
                "-m",
                "tileplan",
                "import-llama32",
                "--checkpoint",
                CHECKPOINT,
                "--target",
                target,
                "--max-context",
                max_context,
                "--tile-rows",
                "auto",
                "--dtype",
                args.dtype or "bfloat16",
                "--out",
                package,
            ],
            cwd=PLAN,
        )
        run([sys.executable, "-m", "tileplan", "verify", package], cwd=PLAN)
    else:
        print(f"Using existing package: {package}", file=sys.stderr, flush=True)

    command = [
        str(EXECUTOR),
        "generate",
        str(package),
        str(inputs),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if tokenizer.eos_token_id is not None:
        command.extend(["--eos-id", str(tokenizer.eos_token_id)])

    print("Running TileInfer...", file=sys.stderr, flush=True)
    start = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True
    )
    wall_seconds = time.perf_counter() - start
    (run_dir / "generation.json").write_text(result.stdout)
    generation = json.loads(result.stdout)
    response = tokenizer.decode(generation["generated"], skip_special_tokens=True)

    print(response)
    print(
        f"TileInfer GPU selection + decode: {generation['compute_us'] / 1000:.2f} ms "
        f"({generation['generated_count']} generated tokens)"
    )
    print(
        f"TileInfer invocation: {wall_seconds:.2f} s "
        "(model load, prefill, generation, and readback)"
    )
    print(f"Run files: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"generate: {error}", file=sys.stderr)
        sys.exit(1)
