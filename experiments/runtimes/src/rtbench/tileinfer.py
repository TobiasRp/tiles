"""Read existing TileInfer and PyTorch benchmark reports as records."""

import json
from pathlib import Path

from .schema import Record

MODEL_NAME = "Llama-3.2-1B"


def _record(path_info: dict, **fields) -> Record:
    return Record(
        model=MODEL_NAME,
        weight_dtype=path_info["weight_dtype"],
        kv_dtype=path_info["weight_dtype"],
        device=path_info["device"],
        **fields,
    )


def decode_records(report_path: Path, *, sampling: str = "none") -> list[Record]:
    """tileinfer/tests/benchmark_decode.py output.

    `sampling` states what that run included. Today TileInfer's fixed-context
    benchmark excludes its argmax kernel ("none"); pass "greedy" for a build
    that folds sampling into the measured token.
    """
    report = json.loads(report_path.read_text())
    baseline = report["baseline"]
    info = {
        "weight_dtype": baseline["activation_dtype"],
        "device": baseline["device"],
    }
    records = []
    torch_version = f"torch {baseline['torch_version']}"
    for row in baseline["rows"]:
        for name, key in (("pytorch-eager", "eager"), ("pytorch-graph", "graph")):
            records.append(
                _record(
                    info,
                    runtime=name,
                    runtime_version=torch_version,
                    test="decode_fixed_ctx",
                    sampling="none",
                    context_length=row["context_length"],
                    samples_us=row[f"{key}_samples_us"],
                    flags={"attn": "eager"},
                    notes="CUDA events, batch average; logits ready, no sampling",
                )
            )
    for position, runtime in zip(report["positions"], report["runtime"], strict=True):
        samples = runtime.get("samples_us") or [runtime["mean_us"]]
        records.append(
            _record(
                info,
                runtime="tileinfer",
                runtime_version="tileinfer (see report)",
                test="decode_fixed_ctx",
                sampling=sampling,
                context_length=position + 1,
                samples_us=samples,
                notes=report["timing"],
            )
        )
    return records


def generate_records(report_path: Path) -> list[Record]:
    """tileinfer/tests/benchmark_generate.py output (selection included)."""
    report = json.loads(report_path.read_text())
    reference = report["reference"]
    info = {
        "weight_dtype": reference["weight_dtype"],
        "device": reference["device"],
    }
    flags = {"new_tokens": str(report["new_tokens"])}
    records = []
    for name, key in (("pytorch-eager", "eager"), ("pytorch-graph", "graph")):
        per_token = [s / report["new_tokens"] for s in reference[key]["samples_us"]]
        records.append(
            _record(
                info,
                runtime=name,
                runtime_version=f"torch {reference['torch_version']}",
                test="generate",
                sampling="greedy",
                context_length=report["prompt_tokens"],
                samples_us=per_token,
                flags=flags,
                notes=reference["baseline"],
            )
        )
    runtime = report["runtime"]
    samples = [s / report["new_tokens"] for s in runtime.get("samples_us", [])]
    records.append(
        _record(
            info,
            runtime="tileinfer",
            runtime_version="tileinfer (see report)",
            test="generate",
            sampling="greedy",
            context_length=report["prompt_tokens"],
            samples_us=samples or [runtime["us_per_token"]],
            flags=flags,
            notes=report["timing"],
        )
    )
    return records
