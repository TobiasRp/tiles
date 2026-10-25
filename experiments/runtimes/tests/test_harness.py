import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rtbench import llamacpp, parity, report, tileinfer
from rtbench.schema import Record, read_records, write_records


def make_record(**overrides: Any) -> Record:
    fields: dict[str, Any] = dict(
        runtime="llama.cpp",
        runtime_version="v",
        model="m",
        weight_dtype="bfloat16",
        kv_dtype="bf16",
        device="d",
        test="decode_fixed_ctx",
        sampling="none",
        context_length=1,
        samples_us=[1000.0, 3000.0],
        flags={"fa": "on"},
    )
    return Record(**(fields | overrides))


def test_record_roundtrip(tmp_path: Path) -> None:
    records = [make_record(), make_record(sampling="greedy", samples_us=[2000.0])]
    write_records(tmp_path / "r.jsonl", records)
    assert read_records(tmp_path / "r.jsonl") == records
    assert records[0].mean_us == 2000.0


def test_record_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        make_record(sampling="argmax")
    with pytest.raises(ValueError):
        make_record(samples_us=[])


def test_report_keeps_sampling_variants_apart() -> None:
    records = [
        make_record(samples_us=[2000.0]),
        make_record(sampling="greedy", samples_us=[2500.0]),
        make_record(runtime="tileinfer", flags={}, samples_us=[1500.0]),
    ]
    text = report.report(records)
    assert "decode only (no sampling)" in text
    assert "decode + greedy sampling" in text
    none_table = report.table(records, "decode_fixed_ctx", "none")
    assert "llama.cpp[fa=on]" in none_table and "tileinfer" in none_table
    assert "2.500" not in none_table
    assert "2.500" in report.table(records, "decode_fixed_ctx", "greedy")


def test_parity_compare_splits_ties_from_real_mismatches() -> None:
    ref = parity.Reference(
        ids=[1],
        tokens=[5, 6, 7, 8],
        top_logits=[10.0, 9.0, 8.0, 7.0],
        gaps=[1.0, 1.0, 0.0, 1.0],
    )
    same = parity.compare(ref, [5, 6, 7, 8], [10.05, 9.0, 8.0, 7.0])
    assert not same.tie_mismatches and not same.real_mismatches
    assert same.logits_within_tolerance
    mixed = parity.compare(ref, [5, 9, 9, 8], [10.0, 9.0, 8.0, 7.0])
    assert mixed.tie_mismatches == [2] and mixed.real_mismatches == [1]
    off = parity.compare(ref, [5, 6, 7, 8], [11.0, 9.0, 8.0, 7.0])
    assert not off.logits_within_tolerance and off.max_logit_error == 1.0


def test_tileinfer_decode_adapter(tmp_path: Path) -> None:
    report_json = {
        "positions": [0, 63],
        "timing": "CUDA event batch average",
        "baseline": {
            "activation_dtype": "bfloat16",
            "device": "RTX 3050",
            "torch_version": "2.14.1",
            "rows": [
                {
                    "context_length": 1,
                    "eager_samples_us": [14000.0],
                    "graph_samples_us": [12800.0],
                },
                {
                    "context_length": 64,
                    "eager_samples_us": [14100.0],
                    "graph_samples_us": [12900.0],
                },
            ],
        },
        "runtime": [{"mean_us": 13000.0}, {"mean_us": 13100.0}],
    }
    path = tmp_path / "decode.json"
    path.write_text(json.dumps(report_json))
    records = tileinfer.decode_records(path, sampling="greedy")
    ours = [r for r in records if r.runtime == "tileinfer"]
    assert [r.context_length for r in ours] == [1, 64]
    assert all(r.sampling == "greedy" for r in ours)
    assert {r.runtime for r in records} == {
        "tileinfer",
        "pytorch-eager",
        "pytorch-graph",
    }


def test_llama_bench_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [
        {"n_gen": 16, "n_depth": 255, "samples_ns": [16_000_000, 32_000_000]},
        {"n_gen": 0, "n_depth": 255, "samples_ns": [1]},
    ]
    monkeypatch.setattr(llamacpp, "llamacpp_bin", lambda backend, name: Path(name))
    monkeypatch.setattr(llamacpp, "device_label", lambda backend: "dev")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(rows), ""),
    )
    gguf = tmp_path / "model-bf16.gguf"
    (record,) = llamacpp.bench_records("cpu", gguf, [255])
    assert record.samples_us == [1000.0, 2000.0]
    assert record.context_length == 255 + 1
    assert record.weight_dtype == "bfloat16"
    assert record.flags["tool"] == "llama-bench"
