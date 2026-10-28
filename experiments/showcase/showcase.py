"""One command for the S6 showcase: from the checkpoint to a report on the RTX 3050.

usage (from the repository root, on the CUDA machine):
  python3 experiments/showcase/showcase.py run --out runs/s6-2026-10-27
  python3 experiments/showcase/showcase.py sanitize --out runs/s6-2026-10-27

``run`` builds the worker and the llama.cpp probe, probes the compiled
worker, imports and verifies the final package (reference tolerances on the
checkpoint), runs CTest, the llama.cpp parity check, the streaming roofline,
the controlled comparison (TileInfer against PyTorch eager, graph replay and
SDPA graph replay, and against llama.cpp with f16 and BF16 K/V, each session
in both spawn orders), the generation case, the timeline trace and the
Nsight kernel traces, then draws the figures and writes ``report.md``.

Each step stores the hashes of its inputs in ``OUT/steps/`` and is skipped
when an earlier run finished it with the same inputs, so a rerun continues
where a failed one stopped and a changed worker reruns everything after it.
``--force STEP...`` reruns steps anyway.

Compute Sanitizer takes minutes, so it is not part of ``run``. ``sanitize``
runs it and records the commit and worker hash; the report states whether
that pass covers the measured build.

The GPU state is recorded, not set: lock the clocks first if wanted (one
setting per call: ``sudo nvidia-smi -pm 1``, ``-lgc 1950,1950``, ``-lmc
6801,6801``); the report shows the clocks the telemetry saw.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
TILEINFER = REPOSITORY / "tileinfer"
RUNTIMES = REPOSITORY / "experiments/runtimes"
BUILD = TILEINFER / "build/release"
PROBE = RUNTIMES / "probe/build-cuda/bin/rt_probe"
GGUF = RUNTIMES / "models/Llama-3.2-1B-bf16.gguf"
UV = ["uv", "run", "--locked", "--project", str(REPOSITORY / "tileplan")]
UV_TORCH = [*UV, "--group", "checkpoint"]
POSITIONS = [0, 255, 1023, 2047, 4095]
TRACED = [0, 4095]
# "1, 2, 3," generates 64 tokens without a BF16 near-tie (runtime comparison).
PROMPT = [128000, 16, 11, 220, 17, 11, 220, 18, 11]
NEW_TOKENS = 64
KV_TYPES = ("f16", "bf16")
WEIGHT_BYTES = 2_471_628_800
KV_BYTES_PER_POSITION = 32_768  # 16 layers x K and V x 8 heads x 64 x BF16
NEAR_TIE = 0.125  # one BF16 unit at the top logits' magnitude


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def source_id(*paths: str) -> str:
    """Committed tree, uncommitted diff and untracked files of the paths."""
    parts = [
        subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", f"HEAD:{path}"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "untracked"
        for path in paths
    ]
    parts.append(git("diff", "HEAD", "--", *paths))
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths)
    parts += [f"{name} {sha256(REPOSITORY / name)}" for name in untracked.split()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def checkpoint_id(checkpoint: Path) -> str:
    """Names, sizes and modification times; hashing 2.5 GB per run is not needed."""
    entries = [
        f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}"
        for p in sorted(checkpoint.iterdir())
        if p.is_file()
    ]
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


class Runner:
    """Runs steps, logs their output, and skips those whose inputs are unchanged."""

    def __init__(self, out: Path, force: list[str]) -> None:
        self.out = out
        self.force = set(force)
        (out / "steps").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(exist_ok=True)
        self.env = dict(os.environ)
        self.env["PATH"] = os.pathsep.join(
            ["/usr/local/cuda/bin", str(Path.home() / ".local/bin"), self.env["PATH"]]
        )

    def sh(
        self, name: str, command: list[str], cwd: Path = REPOSITORY, check: bool = True
    ) -> str:
        """Run a command, append its output to logs/NAME.log, return stdout."""
        with (self.out / "logs" / f"{name}.log").open("a") as log:
            log.write(f"$ {' '.join(command)}\n")
            log.flush()
            result = subprocess.run(
                command, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=log
            )
            stdout = result.stdout.decode()
            log.write(stdout)
        if check and result.returncode:
            raise RuntimeError(f"step {name} failed; see {log.name}")
        return stdout

    def step(self, name: str, key: dict, action: Callable[[], object]) -> dict:
        """Run `action` unless a finished stamp has the same key; a dict it
        returns is stored and returned again on a skip."""
        stamp = self.out / "steps" / f"{name}.json"
        if stamp.exists() and name not in self.force:
            previous = json.loads(stamp.read_text())
            if previous["key"] == key:
                print(f"[skip] {name}", flush=True)
                return previous["result"]
        print(f"[run]  {name}", flush=True)
        started = time.time()
        result = action()
        result = result if isinstance(result, dict) else {}
        stamp.write_text(
            json.dumps(
                {
                    "key": key,
                    "result": result,
                    "seconds": round(time.time() - started, 1),
                    "finished": datetime.datetime.now(datetime.UTC).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        return result


def fresh(path: Path) -> Path:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    return path


def environment() -> dict:
    def query(*command: str) -> str:
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()

    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "gpu": query(
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,"
            "power.limit,persistence_mode",
            "--format=csv,noheader",
        ),
        "clocks": query("nvidia-smi", "-q", "-d", "CLOCK"),
        "compute_apps": query(
            "nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"
        ),
        "llamacpp_pin": (RUNTIMES / "scripts/llamacpp.pin").read_text().strip(),
    }


def build(runner: Runner) -> None:
    """Incremental builds; cheap when nothing changed."""
    runner.sh("build", ["cmake", "--build", "--preset", "release"], TILEINFER)
    if not (RUNTIMES / "third_party/build-cuda/bin").is_dir():
        runner.sh("build", ["scripts/fetch_llamacpp.sh"], RUNTIMES)
        runner.sh("build", ["scripts/build_llamacpp.sh", "cuda"], RUNTIMES)
    runner.sh("build", ["scripts/build_probe.sh", "cuda"], RUNTIMES)


def controlled(
    runner: Runner, name: str, arms: list[str], order: list[str], extra: list[str]
) -> dict:
    out = fresh(runner.out / name)
    runner.sh(
        name,
        [
            *UV_TORCH,
            "python",
            str(TILEINFER / "tests/controlled_decode.py"),
            *arms,
            *extra,
            "--order",
            *order,
            "--reference",
            "tileinfer",
            "--positions",
            *map(str, POSITIONS),
            "--out",
            str(out),
        ],
    )
    return {"report": str(out / "report.json")}


def generation(runner: Runner, package: Path, checkpoint: Path) -> dict:
    """Two passes in opposite order: TileInfer with PyTorch, then llama.cpp."""
    out = runner.out / "generate"
    out.mkdir(exist_ok=True)
    prompt = out / "prompt.json"
    prompt.write_text(json.dumps({"tokens": PROMPT}) + "\n")

    def tileinfer(index: int) -> None:
        runner.sh(
            "generate",
            [
                *UV_TORCH,
                "python",
                str(TILEINFER / "tests/benchmark_generate.py"),
                "--checkpoint",
                str(checkpoint),
                "--package",
                str(package),
                "--executor",
                str(BUILD / "tile_runtime"),
                "--inputs",
                str(prompt),
                "--new-tokens",
                str(NEW_TOKENS),
                "--warmup",
                "3",
                "--rounds",
                "7",
                "--out",
                str(out / f"tileinfer-p{index}.json"),
            ],
        )

    def llamacpp(index: int, kv: str) -> None:
        stdout = runner.sh(
            "generate",
            [
                str(PROBE),
                "greedy",
                "--model",
                str(GGUF),
                "--fa",
                "on",
                "--kv",
                kv,
                "--ids",
                ",".join(map(str, PROMPT)),
                "--new-tokens",
                str(NEW_TOKENS),
                "--warmup",
                "3",
                "--rounds",
                "7",
            ],
        )
        (out / f"llamacpp-{kv}-p{index}.json").write_text(stdout)

    tileinfer(1)
    for kv in KV_TYPES:
        llamacpp(1, kv)
    for kv in reversed(KV_TYPES):
        llamacpp(2, kv)
    tileinfer(2)
    return {"dir": str(out)}


def sanitize(args: argparse.Namespace) -> None:
    runner = Runner(args.out, [])
    build(runner)
    started = time.time()
    runner.sh(
        "sanitize",
        ["cmake", "--build", "--preset", "release", "--target", "sanitizers"],
        TILEINFER,
    )
    record = {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source": source_id("tileinfer", "tileplan"),
        "worker_sha256": sha256(BUILD / "tile_runtime"),
        "passed": True,
        "seconds": round(time.time() - started),
        "finished": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    (args.out / "sanitizer.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Compute Sanitizer passed at {record['commit'][:9]}")


def run(args: argparse.Namespace) -> None:
    runner = Runner(args.out, args.force)
    out = args.out
    checkpoint = args.checkpoint.resolve()
    (out / "environment-start.json").write_text(
        json.dumps(environment(), indent=2) + "\n"
    )
    build(runner)
    worker = sha256(BUILD / "tile_runtime")
    harness = sha256(BUILD / "tile_runtime_test")
    target = out / "target.json"
    runner.step(
        "probe",
        {"worker": worker},
        lambda: runner.sh(
            "probe",
            [*UV, "python", "-m", "tileplan", "probe", "--executor"]
            + [str(BUILD / "tile_runtime"), "--out", str(fresh(target))],
        ),
    )
    package = out / "package-4k"

    def import_package() -> dict:
        stdout = runner.sh(
            "import",
            [*UV_TORCH, "python", "-m", "tileplan", "import-llama32"]
            + ["--checkpoint", str(checkpoint), "--target", str(target)]
            + ["--max-context", "4096", "--out", str(fresh(package))],
        )
        return json.loads(stdout)

    imported = runner.step(
        "import",
        {
            "target": sha256(target),
            "tileplan": source_id("tileplan/src"),
            "checkpoint": checkpoint_id(checkpoint),
        },
        import_package,
    )
    plan = imported["plan_sha256"]
    runner.step(
        "verify",
        {"plan": plan},
        lambda: runner.sh(
            "verify", [*UV, "python", "-m", "tileplan", "verify", str(package)]
        ),
    )

    def checkpoint_check() -> dict:
        """Calibrated tolerances must pass; a greedy mismatch only on a near-tie.

        The script also fails on any greedy mismatch. The 4K package has a known
        one at a one-unit BF16 tie of the reference (S2b in docs/development.md), so
        a mismatch is accepted where the reference's top-two margin is at most
        NEAR_TIE, and the report lists it.
        """
        folder = fresh(out / "checkpoint")
        runner.sh(
            "checkpoint",
            [*UV_TORCH, "python", str(TILEINFER / "tests/llama32_checkpoint.py")]
            + ["--checkpoint", str(checkpoint), "--package", str(package)]
            + ["--executor", str(BUILD / "tile_runtime"), "--out", str(folder)],
            check=False,
        )
        report = json.loads((folder / "report.json").read_text())
        if not all(c["within_tolerance"] for c in report["checks"].values()):
            raise RuntimeError("checkpoint tolerances failed; see logs/checkpoint.log")
        margins = report["reference_top_two_margins"]
        mismatches = [
            {"step": i, "reference_margin": margins[i]}
            for i, match in enumerate(report["top1_matches"])
            if not match
        ]
        if any(m["reference_margin"] > NEAR_TIE for m in mismatches):
            raise RuntimeError(f"greedy mismatch beyond a near-tie: {mismatches}")
        return {"steps": report["steps"], "greedy_mismatches": mismatches}

    runner.step(
        "checkpoint",
        {"plan": plan, "worker": worker, "test": source_id("tileinfer/tests")},
        checkpoint_check,
    )
    runner.step(
        "ctest",
        {"source": source_id("tileinfer", "tileplan"), "worker": worker},
        lambda: runner.sh("ctest", ["ctest", "--preset", "release"], TILEINFER),
    )

    if not GGUF.exists():
        runner.sh(
            "gguf",
            ["uv", "run", "--no-sync", "rtbench", "convert"]
            + ["--checkpoint", str(checkpoint)],
            RUNTIMES,
        )
    gguf = runner.step(
        "gguf",
        {"checkpoint": checkpoint_id(checkpoint)},
        lambda: {"sha256": sha256(GGUF)},
    )["sha256"]
    probe = sha256(PROBE)

    def parity() -> dict:
        results = {}
        for kv in KV_TYPES:
            path = out / f"parity-{kv}.json"
            runner.sh(
                "parity",
                ["uv", "run", "--no-sync", "rtbench", "parity", "--backend", "cuda"]
                + ["--gguf", str(GGUF), "--fa", "on", "--kv", kv]
                + ["--checkpoint", str(checkpoint), "--out", str(path)],
                RUNTIMES,
            )
            results[kv] = str(path)
        return results

    runner.step("parity", {"gguf": gguf, "probe": probe}, parity)
    stream = out / "stream.json"

    def roofline() -> None:
        probe = str(BUILD / "bandwidth_probe")
        stream.write_text(runner.sh("stream", [probe, "stream", str(WEIGHT_BYTES)]))
        (out / "rowdot.json").write_text(runner.sh("stream", [probe, "rowdot"]))

    runner.step("stream", {"probe": sha256(BUILD / "bandwidth_probe")}, roofline)

    tile_arm = ["--arm", f"tileinfer={BUILD / 'tile_runtime_test'},{package}"]
    decode_key = {
        "plan": plan,
        "harness": harness,
        "probe": probe,
        "gguf": gguf,
        "protocol": source_id("tileinfer/tests/controlled_decode.py"),
    }
    pytorch = ["--pytorch", str(checkpoint), "--sdpa"]
    pytorch_arms = ["eager", "graph", "graph-sdpa"]
    llama = ["--llamacpp", f"{PROBE},{GGUF},on", "--llamacpp-kv", *KV_TYPES]
    llama_arms = [f"llamacpp-{kv}" for kv in KV_TYPES]
    sessions = {
        "decode-pytorch-tileinfer-first": (pytorch, ["tileinfer", *pytorch_arms]),
        "decode-pytorch-pytorch-first": (pytorch, [*pytorch_arms, "tileinfer"]),
        "decode-llamacpp-tileinfer-first": (llama, ["tileinfer", *llama_arms]),
        "decode-llamacpp-llamacpp-first": (llama, [*llama_arms, "tileinfer"]),
    }
    for name, (extra, order) in sessions.items():
        runner.step(
            name,
            decode_key,
            lambda name=name, extra=extra, order=order: controlled(
                runner, name, tile_arm, order, extra
            ),
        )
    runner.step(
        "generate",
        {"plan": plan, "worker": worker, "probe": probe, "gguf": gguf},
        lambda: generation(runner, package, checkpoint),
    )
    runner.step(
        "timeline",
        {"plan": plan, "harness": harness, "stream": sha256(stream)},
        lambda: runner.sh(
            "timeline",
            [*UV, "python", str(TILEINFER / "tests/profile_timeline.py")]
            + ["--executor", str(BUILD / "tile_runtime_test")]
            + ["--package", str(package), "--stream-report", str(stream)]
            + ["--positions", *map(str, TRACED)]
            + ["--out", str(fresh(out / "timeline"))],
        ),
    )
    runner.step(
        "kernels",
        {**decode_key, "tracer": source_id("experiments/showcase/kernel_trace.py")},
        lambda: runner.sh(
            "kernels",
            [*UV_TORCH, "python", str(Path(__file__).with_name("kernel_trace.py"))]
            + ["--out", str(fresh(out / "kernels")), "--package", str(package)]
            + ["--executor", str(BUILD / "tile_runtime_test")]
            + ["--checkpoint", str(checkpoint), "--probe", str(PROBE)]
            + ["--gguf", str(GGUF), "--positions", *map(str, TRACED)],
        ),
    )
    (out / "environment-end.json").write_text(
        json.dumps(environment(), indent=2) + "\n"
    )
    runner.sh(
        "figures",
        [*UV, "python", str(REPOSITORY / "docs/blog/make_s6_figures.py")]
        + [str(out), str(out / "figures")],
    )
    write_report(out, worker)
    print(out / "report.md")


# --- report -------------------------------------------------------------


def floor_ms(context: int, gbps: float) -> float:
    return (WEIGHT_BYTES + KV_BYTES_PER_POSITION * context) / gbps / 1e6


def comparison(out: Path) -> dict[int, dict[str, float]]:
    """Median µs per arm and context, averaged over both spawn orders."""
    cells: dict[int, dict[str, list[float]]] = {}
    for report in sorted(out.glob("decode-*/report.json")):
        for context in json.loads(report.read_text())["contexts"]:
            row = cells.setdefault(context["context_length"], {})
            for arm, summary in context["summary"].items():
                row.setdefault(arm, []).append(summary["median_us"])
    return {
        length: {arm: statistics.mean(v) for arm, v in row.items()}
        for length, row in sorted(cells.items())
    }


def differences(out: Path) -> dict[int, dict[str, tuple[float, float]]]:
    """Paired difference against TileInfer: mean over sessions, widest half-width."""
    cells: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for report in sorted(out.glob("decode-*/report.json")):
        for context in json.loads(report.read_text())["contexts"]:
            row = cells.setdefault(context["context_length"], {})
            for arm, stats in context["paired_vs_tileinfer"].items():
                row.setdefault(arm, []).append(
                    (stats["mean_us"], stats["ci95_half_width_us"])
                )
    return {
        length: {
            arm: (statistics.mean(d for d, _ in v), max(h for _, h in v))
            for arm, v in row.items()
        }
        for length, row in sorted(cells.items())
    }


def parity_rows(out: Path) -> list[str]:
    """First tokens per context, from the TileInfer-first session of each pairing."""
    rows = []
    for report in sorted(out.glob("decode-*-tileinfer-first/report.json")):
        for context in json.loads(report.read_text())["contexts"]:
            reference = context["pytorch"]
            tokens, logits = context["top_tokens"], context["top_logits"]
            if reference:
                rows.append(
                    f"| {context['context_length']:,} | PyTorch eager | "
                    f"{reference['next_token']} | {reference['top_logit']:.3f} | "
                    f"TileInfer {tokens['tileinfer']} ({logits['tileinfer']:+.3f}); "
                    f"SDPA {reference['sdpa']['next_token']} "
                    f"({reference['sdpa']['top_logit']:+.3f}) |"
                )
            elif "llamacpp-f16" in tokens:
                others = "; ".join(
                    f"{arm} {tokens[arm]} ({logits[arm]:+.3f})"
                    for arm in tokens
                    if arm != "tileinfer"
                )
                rows.append(
                    f"| {context['context_length']:,} | TileInfer | "
                    f"{tokens['tileinfer']} | {logits['tileinfer']:.3f} | {others} |"
                )
        if rows:
            rows.append("")
    return rows


def telemetry_range(out: Path) -> str:
    sm, mem = [], []
    for report in out.glob("decode-*/report.json"):
        for context in json.loads(report.read_text())["contexts"]:
            for block in context["blocks"]:
                t = block.get("telemetry", {})
                if t.get("samples"):
                    sm += t["sm_mhz"]
                    mem += t["mem_mhz"]
    if not sm:
        return "no telemetry"
    return (
        f"SM {min(sm):.0f} to {max(sm):.0f} MHz, memory {min(mem):.0f} to "
        f"{max(mem):.0f} MHz during the timed blocks"
    )


def generation_rows(out: Path) -> list[str]:
    folder = out / "generate"
    rows = []
    tile = [json.loads(p.read_text()) for p in sorted(folder.glob("tileinfer-p?.json"))]
    if tile:
        arms = {
            "TileInfer": [t["runtime"]["us_per_token"] for t in tile],
            "PyTorch eager": [t["reference"]["eager"]["us_per_token"] for t in tile],
            "PyTorch graph": [t["reference"]["graph"]["us_per_token"] for t in tile],
        }
        for kv in KV_TYPES:
            passes = [
                json.loads(p.read_text()) for p in folder.glob(f"llamacpp-{kv}-p*.json")
            ]
            arms[f"llama.cpp FA, {kv} K/V"] = [
                statistics.median(g["decode_us_per_token"])
                + statistics.median(g["sample_us_per_token"])
                for g in passes
            ]
        for name, values in arms.items():
            us = statistics.mean(values)
            rows.append(
                f"| {name} | {us / 1e3:.3f} | {1e6 / us:.1f} | "
                + ", ".join(f"{v / 1e3:.3f}" for v in values)
                + " |"
            )
    return rows


def kernel_rows(out: Path) -> list[str]:
    rows = []
    for path in sorted((out / "kernels").glob("kernels-*.json")):
        k = json.loads(path.read_text())
        rows.append(
            f"| {k['runtime']} | {k['context_length']:,} | {k['kernel_count']} | "
            f"{k['span_us'] / 1e3:.2f} | {max(k['idle_us'], 0):.0f} | "
            f"{k['gap_before_us']:.0f} |"
        )
    return rows


def checkpoint_line(out: Path) -> str:
    result = json.loads((out / "steps/checkpoint.json").read_text())["result"]
    mismatches = result["greedy_mismatches"]
    if not mismatches:
        return f"greedy selection matches the reference at all {result['steps']} steps."
    where = ", ".join(
        f"step {m['step']} (reference top-two margin {m['reference_margin']:.3f})"
        for m in mismatches
    )
    return (
        f"greedy selection matches the reference at {result['steps'] - len(mismatches)} "
        f"of {result['steps']} steps; it differs at the BF16 near-tie {where}."
    )


def sanitizer_line(out: Path, worker: str) -> str:
    path = out / "sanitizer.json"
    if not path.exists():
        return (
            "**Compute Sanitizer: no record in this run directory.** Run "
            "`showcase.py sanitize` before citing these numbers."
        )
    record = json.loads(path.read_text())
    covered = record["worker_sha256"] == worker and record["source"] == source_id(
        "tileinfer", "tileplan"
    )
    state = "covers" if covered else "**does not cover**"
    return (
        f"Compute Sanitizer passed at `{record['commit'][:9]}` "
        f"({record['finished'][:16]}); that pass {state} the measured worker."
    )


def write_report(out: Path, worker: str) -> None:
    start = json.loads((out / "environment-start.json").read_text())
    stream = json.loads((out / "stream.json").read_text())
    gbps = max(row["gbps"] for row in stream["stream"])
    imported = json.loads((out / "steps/import.json").read_text())["result"]
    table = comparison(out)
    paired = differences(out)
    columns = [
        ("tileinfer", "TileInfer"),
        ("llamacpp-f16", "llama.cpp f16 K/V"),
        ("llamacpp-bf16", "llama.cpp BF16 K/V"),
        ("graph", "PyTorch graph"),
        ("graph-sdpa", "PyTorch graph, SDPA"),
        ("eager", "PyTorch eager"),
    ]
    lines = [
        "# S6 showcase report",
        "",
        f"Generated {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC by "
        "`experiments/showcase/showcase.py run`.",
        "",
        f"- Commit `{start['commit'][:12]}`"
        + (" (dirty tree)" if start["dirty"] else ""),
        f"- GPU: {start['gpu']}",
        f"- Clocks: {telemetry_range(out)}",
        f"- Package plan `{imported['plan_sha256'][:12]}`, weights "
        f"`{imported['weights_sha256'][:12]}`, worker `{worker[:12]}`",
        f"- GGUF `{json.loads((out / 'steps/gguf.json').read_text())['result']['sha256'][:12]}`, "
        + ", ".join(
            line
            for line in start["llamacpp_pin"].splitlines()
            if line and not line.startswith("#")
        ),
        f"- Streaming roofline {gbps:.1f} GB/s (`bandwidth_probe stream`)",
        f"- {sanitizer_line(out, worker)}",
        "- CTest and the calibrated checkpoint tolerances passed in this run "
        f"(`logs/ctest.log`, `checkpoint/report.json`); {checkpoint_line(out)}",
        "",
        "## Decode latency, ms per token",
        "",
        "Fixed context, logits ready, no sampling. Medians of the S6a protocol "
        "(interleaved blocks, decoded prefixes), averaged over both spawn orders. "
        "% is the share of the streaming floor (weights and K/V bytes at the "
        "roofline) over the measured time.",
        "",
        "| Context | Floor | " + " | ".join(label for _, label in columns) + " |",
        "| ---: | ---: |" + " ---: |" * len(columns),
    ]
    for length, row in table.items():
        floor = floor_ms(length, gbps)
        cells = [
            f"{row[arm] / 1e3:.3f} ({100 * floor * 1e3 / row[arm]:.0f}%)"
            if arm in row
            else "–"
            for arm, _ in columns
        ]
        lines.append(f"| {length:,} | {floor:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Paired difference against TileInfer in µs (mean over sessions, widest "
        "95% half-width):",
        "",
        "| Context | " + " | ".join(label for _, label in columns[1:]) + " |",
        "| ---: |" + " ---: |" * (len(columns) - 1),
    ]
    for length, row in paired.items():
        cells = [
            f"{row[arm][0]:+.0f} ± {row[arm][1]:.0f}" if arm in row else "–"
            for arm, _ in columns[1:]
        ]
        lines.append(f"| {length:,} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "llama.cpp rounds are host wall time around each synchronous "
        "`llama_decode` (its API returns when the logits are ready); TileInfer and "
        "PyTorch rounds are CUDA-event batches of 20 back-to-back calls.",
        "",
        "## Parity at the measured contexts",
        "",
        "| Context | Reference | Next token | Top logit | Others: token (top logit) |",
        "| ---: | --- | ---: | ---: | --- |",
        *parity_rows(out),
        "Teacher-forced llama.cpp parity against Transformers: `parity-f16.json`, "
        "`parity-bf16.json`.",
        "",
        f"## Generation, {len(PROMPT)}-token prompt, {NEW_TOKENS} new tokens, greedy",
        "",
        "| Runtime | ms per token | tokens/s | passes (ms) |",
        "| --- | ---: | ---: | --- |",
        *generation_rows(out),
        "",
        "## One token in Nsight Systems",
        "",
        "| Runtime | Context | Kernels | Span, ms | Idle inside, µs | Idle before, µs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *kernel_rows(out),
        "",
        "Spans are under tracing and are for the figures only. Idle before: from "
        "the previous token's last kernel to this token's first, with calls issued "
        "as in the benchmarks (llama.cpp's decode is synchronous).",
        "",
        "## Figures",
        "",
        *[f"- `figures/{p.name}`" for p in sorted((out / "figures").glob("*.svg"))],
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run", help="build, measure, draw, report")
    runs.add_argument("--out", type=Path, required=True)
    runs.add_argument(
        "--checkpoint", type=Path, default=REPOSITORY / "checkpoints/Llama-3.2-1B"
    )
    runs.add_argument("--force", nargs="+", default=[], metavar="STEP")
    runs.add_argument(
        "--report-only", action="store_true", help="redraw figures and report"
    )
    sanitizes = commands.add_parser("sanitize", help="Compute Sanitizer, separately")
    sanitizes.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command == "sanitize":
        sanitize(args)
    elif args.report_only:
        runner = Runner(args.out, [])
        runner.sh(
            "figures",
            [*UV, "python", str(REPOSITORY / "docs/blog/make_s6_figures.py")]
            + [str(args.out), str(args.out / "figures")],
        )
        write_report(args.out, sha256(BUILD / "tile_runtime"))
        print(args.out / "report.md")
    else:
        run(args)


if __name__ == "__main__":
    sys.exit(main())
