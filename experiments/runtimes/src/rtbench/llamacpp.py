"""Run llama.cpp (llama-bench and rt_probe) and turn the output into records."""

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .paths import llamacpp_bin, llamacpp_pin, probe_bin
from .schema import Record

MODEL_NAME = "Llama-3.2-1B"


def version_label() -> str:
    pin = llamacpp_pin()
    return f"llama.cpp {pin['LLAMACPP_TAG']} ({pin['LLAMACPP_COMMIT'][:9]})"


def device_label(backend: str) -> str:
    if backend == "cuda":
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        )
        return out.splitlines()[0].strip()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        return f"{cpu} ({backend})"
    return f"{platform.processor() or platform.machine()} ({backend})"


def default_threads(backend: str) -> int:
    """Threads for CPU backends: performance cores on Apple silicon, else all cores."""
    if backend != "cpu":
        return 0
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout)
    return os.cpu_count() or 1


def _env(graphs: bool) -> dict[str, str]:
    env = dict(os.environ)
    if not graphs:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    return env


def _flags(
    backend: str, kv: str, fa: str, graphs: bool, threads: int = 0
) -> dict[str, str]:
    flags = {"backend": backend, "kv": kv, "fa": fa}
    if backend == "cpu":
        flags["threads"] = str(threads or default_threads(backend))
    if backend == "cuda":
        flags["graphs"] = "on" if graphs else "off"
    return flags


def _weight_dtype(gguf: Path) -> str:
    """Weight dtype named by the GGUF file (convert_gguf.sh writes <model>-bf16.gguf)."""
    return "bfloat16" if "bf16" in gguf.name.lower() else "other"


def run_probe(backend: str, args: list[str], graphs: bool = True) -> dict:
    result = subprocess.run(
        [str(probe_bin(backend)), *args],
        capture_output=True,
        text=True,
        env=_env(graphs),
    )
    if result.returncode != 0:
        raise RuntimeError(f"rt_probe failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _probe_common(
    backend: str, gguf: Path, kv: str, fa: str, ngl: int, threads: int
) -> list[str]:
    args = ["--model", str(gguf), "--kv", kv, "--fa", fa, "--ngl", str(ngl)]
    threads = threads or default_threads(backend)
    if threads:
        args += ["--threads", str(threads)]
    return args


def decode_records(
    backend: str,
    gguf: Path,
    positions: list[int],
    *,
    kv: str = "bf16",
    fa: str = "auto",
    ngl: int = 99,
    threads: int = 0,
    token: int = 1,
    warmup: int = 5,
    iterations: int = 20,
    rounds: int = 3,
    graphs: bool = True,
) -> list[Record]:
    """Fixed-context decode, both without and with greedy sampling."""
    data = run_probe(
        backend,
        [
            "decode",
            *_probe_common(backend, gguf, kv, fa, ngl, threads),
            "--positions",
            ",".join(map(str, positions)),
            "--token",
            str(token),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--rounds",
            str(rounds),
        ],
        graphs,
    )
    records = []
    for row in data["rows"]:
        for sampling in ("none", "greedy"):
            records.append(
                Record(
                    runtime="llama.cpp",
                    runtime_version=version_label(),
                    model=MODEL_NAME,
                    weight_dtype=_weight_dtype(gguf),
                    kv_dtype=kv,
                    device=device_label(backend),
                    test="decode_fixed_ctx",
                    sampling=sampling,
                    context_length=row["context_length"],
                    samples_us=row[f"{sampling}_samples_us"],
                    outputs={
                        "next_token": row["next_token"],
                        "top_logit": row["top_logit"],
                    },
                    flags=_flags(backend, kv, fa, graphs, threads)
                    | {"tool": "rt_probe"},
                    notes="wall clock, one llama_decode + synchronize per call; "
                    "greedy adds llama_sampler_sample (logits read on host)",
                )
            )
    return records


def generate(
    backend: str,
    gguf: Path,
    ids: list[int],
    new_tokens: int,
    *,
    kv: str = "bf16",
    fa: str = "auto",
    ngl: int = 99,
    threads: int = 0,
    warmup: int = 1,
    rounds: int = 3,
    graphs: bool = True,
    forced: list[int] | None = None,
) -> dict:
    """Greedy generation from token ids; tokens plus per-round timings.

    With `forced`, those tokens are fed instead of the sampled ones (teacher
    forcing); `tokens` still holds what llama.cpp would have chosen.
    """
    return run_probe(
        backend,
        [
            "greedy",
            *_probe_common(backend, gguf, kv, fa, ngl, threads),
            "--ids",
            ",".join(map(str, ids)),
            "--new-tokens",
            str(new_tokens),
            "--warmup",
            str(warmup),
            "--rounds",
            str(rounds),
            *(["--force", ",".join(map(str, forced))] if forced else []),
        ],
        graphs,
    )


def generate_records(
    backend: str,
    gguf: Path,
    ids: list[int],
    new_tokens: int,
    *,
    kv: str = "bf16",
    fa: str = "auto",
    threads: int = 0,
    graphs: bool = True,
    data: dict,
) -> list[Record]:
    common: dict[str, Any] = {
        "runtime": "llama.cpp",
        "runtime_version": version_label(),
        "model": MODEL_NAME,
        "weight_dtype": _weight_dtype(gguf),
        "kv_dtype": kv,
        "device": device_label(backend),
        "test": "generate",
        "context_length": len(ids),
        "flags": _flags(backend, kv, fa, graphs, threads)
        | {"tool": "rt_probe", "new_tokens": str(new_tokens)},
    }
    decode = data["decode_us_per_token"]
    sample = data["sample_us_per_token"]
    return [
        Record(
            sampling="none",
            samples_us=decode,
            notes="mean decode time per generated token; sampling excluded",
            **common,
        ),
        Record(
            sampling="greedy",
            samples_us=[d + s for d, s in zip(decode, sample, strict=True)],
            notes="decode + greedy sampling per generated token",
            **common,
        ),
    ]


def bench_records(
    backend: str,
    gguf: Path,
    depths: list[int],
    *,
    n_gen: int = 16,
    kv: str = "bf16",
    fa: str = "auto",
    ngl: int = 99,
    threads: int = 0,
    repetitions: int = 3,
    graphs: bool = True,
) -> list[Record]:
    """llama-bench token generation at KV depths (decode only, no sampling)."""
    fa_flag = {"on": "1", "off": "0", "auto": "auto"}[fa]
    cmd = [
        str(llamacpp_bin(backend, "llama-bench")),
        "-m",
        str(gguf),
        "-p",
        "0",
        "-n",
        str(n_gen),
        "-d",
        ",".join(map(str, depths)),
        "-r",
        str(repetitions),
        "-fa",
        fa_flag,
        "-ngl",
        str(ngl),
        "-ctk",
        kv,
        "-ctv",
        kv,
        "-o",
        "json",
    ]
    threads = threads or default_threads(backend)
    if threads:
        cmd += ["-t", str(threads)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env(graphs))
    if result.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{result.stderr}")
    rows = json.loads(result.stdout)
    records = []
    for row in rows:
        if row["n_gen"] == 0:
            continue
        per_token_us = [ns / row["n_gen"] / 1000 for ns in row["samples_ns"]]
        records.append(
            Record(
                runtime="llama.cpp",
                runtime_version=version_label(),
                model=MODEL_NAME,
                weight_dtype=_weight_dtype(gguf),
                kv_dtype=kv,
                device=device_label(backend),
                test="decode_fixed_ctx",
                sampling="none",
                # The cache grows by n_gen during the test; report where it starts.
                context_length=row["n_depth"] + 1,
                samples_us=per_token_us,
                flags=_flags(backend, kv, fa, graphs, threads)
                | {"tool": "llama-bench", "n_gen": str(n_gen)},
                notes=f"llama-bench tg{row['n_gen']} @ d{row['n_depth']}; "
                f"context grows to {row['n_depth'] + row['n_gen']} during the test",
            )
        )
    return records
