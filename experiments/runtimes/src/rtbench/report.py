"""Merge records into a markdown table: one row per context, one column per runtime."""

from collections import defaultdict

from .schema import Record


def _ms(us: float) -> str:
    return f"{us / 1000:.3f}"


def table(records: list[Record], test: str, sampling: str) -> str:
    """Rows are context lengths, columns runtimes (ms per token)."""
    selected = [r for r in records if r.test == test and r.sampling == sampling]
    if not selected:
        return f"_no {test} records with sampling={sampling}_\n"
    columns = sorted({r.runtime_label for r in selected})
    cells: dict[int, dict[str, float]] = defaultdict(dict)
    for r in selected:
        cells[r.context_length][r.runtime_label] = r.mean_us
    lines = [
        "| Context | " + " | ".join(columns) + " |",
        "| ---: | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for context in sorted(cells):
        row = [_ms(cells[context][c]) if c in cells[context] else "" for c in columns]
        lines.append(f"| {context} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def report(records: list[Record]) -> str:
    """ms/token tables for every (test, sampling) pair present, with provenance."""
    out = []
    for test in ("decode_fixed_ctx", "generate", "prefill"):
        for sampling in ("none", "greedy"):
            if any(r.test == test and r.sampling == sampling for r in records):
                title = {
                    "none": "decode only (no sampling)",
                    "greedy": "decode + greedy sampling",
                }
                out.append(f"### {test}: {title[sampling]}, ms per token\n")
                out.append(table(records, test, sampling))
    versions = sorted(
        {f"{r.runtime}: {r.runtime_version} on {r.device}" for r in records}
    )
    out.append("### Provenance\n")
    out.extend(f"- {v}" for v in versions)
    return "\n".join(out) + "\n"
