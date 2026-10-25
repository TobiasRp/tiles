"""Common result record shared by every runtime in the comparison."""

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

TESTS = ("decode_fixed_ctx", "generate")
# "none": logits are ready, no next token chosen. "greedy": next token id known.
SAMPLING = ("none", "greedy")


@dataclass
class Record:
    runtime: str
    runtime_version: str
    model: str
    weight_dtype: str
    kv_dtype: str
    device: str
    test: str
    sampling: str
    context_length: int
    samples_us: list[float]
    flags: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Model outputs at this point (next_token, top_logit), for cross-checks.
    outputs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.test not in TESTS:
            raise ValueError(f"unknown test {self.test!r}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"unknown sampling {self.sampling!r}")
        if not self.samples_us:
            raise ValueError("a record needs at least one sample")

    @property
    def mean_us(self) -> float:
        return statistics.mean(self.samples_us)

    @property
    def key(self) -> tuple[str, str, str, int]:
        """Rows that may be compared against each other."""
        return (self.test, self.sampling, self.runtime_label, self.context_length)

    @property
    def runtime_label(self) -> str:
        """Runtime plus the flags that tell configurations apart.

        Constant per sweep (backend, token counts) and default values (CUDA
        graphs on) are left out; llama-bench rows are marked as such.
        """
        shown = [
            f"{k}={self.flags[k]}" for k in ("fa", "kv", "threads") if k in self.flags
        ]
        if self.flags.get("graphs") == "off":
            shown.append("graphs=off")
        if self.flags.get("tool") == "llama-bench":
            shown.append("llama-bench")
        return f"{self.runtime} {' '.join(shown)}" if shown else self.runtime

    def to_json(self) -> dict:
        data = asdict(self)
        data["mean_us"] = self.mean_us
        return data


def write_records(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for record in records:
            out.write(json.dumps(record.to_json()) + "\n")


def read_records(path: Path) -> list[Record]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        data.pop("mean_us", None)
        records.append(Record(**data))
    return records
