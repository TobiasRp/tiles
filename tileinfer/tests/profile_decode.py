"""Profile one fixed-context TileInfer worker launch with Nsight Compute."""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--ncu", type=Path, default=Path("/usr/local/cuda/bin/ncu"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.package / "plan.json").read_text())
    if args.position < 0 or args.position >= plan["model"]["max_context"]:
        parser.error("position outside package context capacity")
    command = [
        str(args.ncu),
        "--target-processes",
        "all",
        "--kernel-name",
        "regex:decoder_worker",
        "--launch-skip",
        str(args.position),
        "--launch-count",
        "1",
        "--metrics",
        ",".join(METRICS),
        "--csv",
        str(args.executor),
        "--benchmark-stack",
        str(args.package),
        str(args.position),
        str(args.token_id),
        "0",
        "1",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = completed.stdout.splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header is None:
        raise RuntimeError(
            f"Nsight Compute returned no metric table: {completed.stderr}"
        )
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    if len(rows) != len(METRICS):
        raise RuntimeError(
            f"expected one worker and {len(METRICS)} counters, got {len(rows)} rows"
        )
    counters = {row["Metric Name"]: float(row["Metric Value"]) for row in rows}
    if set(counters) != set(METRICS):
        raise RuntimeError(f"missing profiler metrics: {set(METRICS) - set(counters)}")
    weight_bytes = plan["storage"]["weight_bytes"]
    runtime_start = completed.stdout.find("{")
    runtime = json.JSONDecoder().raw_decode(completed.stdout[runtime_start:])[0]
    report = {
        "position": args.position,
        "context_length": args.position + 1,
        "plan_sha256": runtime["plan_sha256"],
        "weight_bytes": weight_bytes,
        "dram_read_bytes": int(counters["dram__bytes_read.sum"]),
        "dram_write_bytes": int(counters["dram__bytes_write.sum"]),
        "l2_bytes": int(counters["lts__t_bytes.sum"]),
        "sm_throughput_percent": counters[
            "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        ],
        "dram_read_to_weight_ratio": counters["dram__bytes_read.sum"] / weight_bytes,
        "grid": rows[0]["Grid Size"],
        "block": rows[0]["Block Size"],
        "note": "Nsight Compute replay perturbs latency; use benchmark_decode.py for timing",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
