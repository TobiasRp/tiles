"""Record one decode token of each runtime with Nsight Systems.

usage (repository root, tileplan environment with the checkpoint group):
  python experiments/showcase/kernel_trace.py --out DIR --package PACKAGE \
    --executor tile_runtime_test --checkpoint CHECKPOINT --probe rt_probe \
    --gguf GGUF --positions 0 4095

For every runtime and context it runs a short fixed-context decode under
``nsys profile``, exports SQLite, and keeps the kernels of the last token:
TileInfer's last launch (one kernel), the kernels of the last CUDA graph
launch for PyTorch graph replay and llama.cpp, and the kernels launched
inside the last NVTX ``token`` range for PyTorch eager. Each
``kernels-RUNTIME-pPOSITION.json`` lists the kernels' start and duration
relative to the token's first kernel, and the idle GPU time between the
previous token's last kernel and this token's first. Tracing adds a little
time per launch, so these spans are for the picture, not for the comparison
table.
"""

import argparse
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIMES = ("tileinfer", "pytorch-graph", "pytorch-eager", "llamacpp")


def pytorch_child(checkpoint: Path, dtype: str, position: int) -> None:
    """Decode at `position` with graph replay, then eager in NVTX ranges."""
    path = REPOSITORY / "tileinfer/tests/controlled_decode.py"
    spec = importlib.util.spec_from_file_location("controlled_decode", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arms = module.PyTorchArms(checkpoint, dtype, synthetic=False)
    arms.prepare(position, token=1, warmup=2)
    torch = arms.torch
    assert arms.graph is not None
    # Back-to-back calls without a host sync in between, as the benchmark runs them.
    with torch.inference_mode():
        for _ in range(3):
            arms.graph.replay()
        torch.cuda.synchronize()
        for _ in range(3):
            with torch.cuda.nvtx.range("token"):
                arms.forward()
        torch.cuda.synchronize()


def profile(base: Path, command: list[str]) -> Path:
    subprocess.run(
        [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--export=sqlite",
            "-o",
            str(base),
            *command,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return base.with_suffix(".sqlite")


def token_kernels(database: Path, mode: str) -> list[list[tuple[int, int, str]]]:
    """(start ns, end ns, name) of the kernels of the last two tokens."""
    connection = sqlite3.connect(database)
    kernels = (
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName "
    )
    if mode == "single":
        rows = connection.execute(
            kernels + "WHERE k.end - k.start > 1000000 ORDER BY k.start DESC LIMIT 2"
        ).fetchall()
        tokens = [[row] for row in rows]
    elif mode == "graph":
        # Kernels of one graph launch share the launch's correlation ID.
        correlations = connection.execute(
            "SELECT correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId > 0 "
            "GROUP BY correlationId ORDER BY MAX(start) DESC LIMIT 2"
        ).fetchall()
        tokens = [
            connection.execute(kernels + "WHERE k.correlationId = ?", c).fetchall()
            for c in correlations
        ]
    else:
        ranges = connection.execute(
            "SELECT n.start, n.end FROM NVTX_EVENTS n "
            "WHERE n.text = 'token' ORDER BY n.start DESC LIMIT 2"
        ).fetchall()
        tokens = [
            connection.execute(
                kernels + "WHERE k.correlationId IN (SELECT correlationId FROM "
                "CUPTI_ACTIVITY_KIND_RUNTIME WHERE start BETWEEN ? AND ?)",
                r,
            ).fetchall()
            for r in ranges
        ]
    connection.close()
    if len(tokens) < 2 or not all(tokens):
        raise RuntimeError(f"fewer than two traced tokens in {database}")
    return [sorted(rows) for rows in reversed(tokens)]


def summarize(
    runtime: str, position: int, tokens: list[list[tuple[int, int, str]]]
) -> dict:
    """The last token's kernels, and the idle GPU time since the previous token."""
    previous, rows = tokens
    origin = rows[0][0]
    # Kernels on different streams can overlap: busy time is their union.
    busy, reach = 0, origin
    for start, end, _ in rows:
        busy += max(end - max(start, reach), 0)
        reach = max(reach, end)
    span = reach - origin
    return {
        "runtime": runtime,
        "position": position,
        "context_length": position + 1,
        "kernel_count": len(rows),
        "span_us": span / 1e3,
        "busy_us": busy / 1e3,
        "idle_us": (span - busy) / 1e3,
        "gap_before_us": (origin - max(end for _, end, _ in previous)) / 1e3,
        "kernels": [
            {
                "start_us": (start - origin) / 1e3,
                "duration_us": (end - start) / 1e3,
                "name": name,
            }
            for start, end, name in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 4095])
    parser.add_argument("--runtimes", nargs="+", choices=RUNTIMES, default=RUNTIMES)
    parser.add_argument("--keep", action="store_true", help="keep the Nsight files")
    parser.add_argument("--pytorch-child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    dtype = json.loads((args.package / "plan.json").read_text())["model"][
        "weight_dtype"
    ]
    if args.pytorch_child is not None:
        pytorch_child(args.checkpoint, dtype, args.pytorch_child)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for position in args.positions:
        commands = {
            "tileinfer": (
                "single",
                [str(args.executor), "--benchmark-stack", str(args.package)]
                + [str(position), "1", "2", "3", "1"],
            ),
            "pytorch": (
                "",
                [sys.executable, __file__, "--pytorch-child", str(position)]
                + ["--out", str(args.out), "--package", str(args.package)]
                + ["--executor", str(args.executor), "--checkpoint"]
                + [str(args.checkpoint), "--probe", str(args.probe)]
                + ["--gguf", str(args.gguf)],
            ),
            "llamacpp": (
                "graph",
                [str(args.probe), "decode", "--model", str(args.gguf)]
                + ["--fa", "on", "--kv", "f16", "--positions", str(position)]
                + ["--warmup", "2", "--iterations", "3", "--rounds", "1"],
            ),
        }
        for program, (mode, command) in commands.items():
            wanted = [r for r in args.runtimes if r.startswith(program)]
            if not wanted:
                continue
            print(f"context {position + 1}: {program}", file=sys.stderr, flush=True)
            base = args.out / f"{program}-p{position}"
            database = profile(base, command)
            for runtime in wanted:
                kind = {"pytorch-graph": "graph", "pytorch-eager": "nvtx"}.get(
                    runtime, mode
                )
                result = summarize(runtime, position, token_kernels(database, kind))
                path = args.out / f"kernels-{runtime}-p{position}.json"
                path.write_text(json.dumps(result, indent=1) + "\n")
                print(
                    f"  {runtime}: {result['kernel_count']} kernels, "
                    f"{result['span_us']:.0f} us span, {result['idle_us']:.0f} us idle, "
                    f"{result['gap_before_us']:.0f} us since the previous token"
                )
            if not args.keep:
                database.unlink()
                base.with_suffix(".nsys-rep").unlink()


if __name__ == "__main__":
    main()
