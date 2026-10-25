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
    # Records of the same runtime and context (for example two passes) are averaged.
    pooled: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in selected:
        pooled[r.context_length][r.runtime_label].append(r.mean_us)
    cells = {
        context: {label: sum(v) / len(v) for label, v in by_label.items()}
        for context, by_label in pooled.items()
    }
    lines = [
        "| Context | " + " | ".join(columns) + " |",
        "| ---: | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for context in sorted(cells):
        row = [_ms(cells[context][c]) if c in cells[context] else "" for c in columns]
        lines.append(f"| {context} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def crosscheck(records: list[Record], reference: str = "pytorch-eager") -> str:
    """Next token and top logit per context against one runtime's outputs.

    Uses the decode-only rows. A runtime is "ok" when its token matches and
    its top logit is within the BF16 tolerance used by benchmark_decode.py.
    """
    atol, rtol = 0.15, 0.03
    rows = [
        r
        for r in records
        if r.test == "decode_fixed_ctx" and r.sampling == "none" and r.outputs
    ]
    refs = {
        r.context_length: r.outputs
        for r in rows
        if r.runtime_label.startswith(reference)
    }
    if not refs:
        return f"_no {reference} outputs to check against_\n"
    lines = [
        f"| Context | Runtime | Token | Top logit | Error vs {reference} | Verdict |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    seen = set()
    for r in sorted(rows, key=lambda r: (r.context_length, r.runtime_label)):
        ref = refs.get(r.context_length)
        if ref is None or r.runtime_label.startswith(reference):
            continue
        if (
            r.context_length,
            r.runtime_label,
        ) in seen:  # the same point from another pass
            continue
        seen.add((r.context_length, r.runtime_label))
        error = abs(r.outputs["top_logit"] - ref["top_logit"])
        ok = r.outputs["next_token"] == ref[
            "next_token"
        ] and error <= atol + rtol * abs(ref["top_logit"])
        lines.append(
            f"| {r.context_length} | {r.runtime_label} | {int(r.outputs['next_token'])} "
            f"| {r.outputs['top_logit']:.3f} | {error:.3f} | {'ok' if ok else 'MISMATCH'} |"
        )
    return "\n".join(lines) + "\n"


def report(records: list[Record]) -> str:
    """ms/token tables for every (test, sampling) pair present, with provenance."""
    out = []
    for test in ("decode_fixed_ctx", "generate"):
        for sampling in ("none", "greedy"):
            if any(r.test == test and r.sampling == sampling for r in records):
                title = {
                    "none": "decode only (no sampling)",
                    "greedy": "decode + greedy sampling",
                }
                out.append(f"### {test}: {title[sampling]}, ms per token\n")
                out.append(table(records, test, sampling))
    if any(r.outputs for r in records):
        out.append("### Cross-check at the swept contexts\n")
        out.append(crosscheck(records))
    versions = sorted(
        {f"{r.runtime}: {r.runtime_version} on {r.device}" for r in records}
    )
    out.append("### Provenance\n")
    out.extend(f"- {v}" for v in versions)
    return "\n".join(out) + "\n"
