"""Trace one decoder launch per context length and write a bubble budget.

Runs ``tile_runtime_test --timeline-stack`` once for all positions, in
increasing order so the prefix is filled once; keeps each position's traced
launch with the median CUDA-event time, and decomposes it with
``tileplan.timeline``. The ideal stream uses the best rate from a
``bandwidth_probe stream`` report. Writes ``timeline-report.json``, the raw
trace (``trace-pPOSITION.json``), and a whole-token and a one-layer Gantt
chart per position.
"""

import argparse
import json
import subprocess
from pathlib import Path

from tileplan.timeline import align, budget, layer_window, render_timeline_svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--stream-report", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[63])
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--synthetic-prefix",
        action="store_true",
        help="fill K/V with deterministic values instead of decoding the prefix",
    )
    parser.add_argument(
        "--layer", type=int, default=8, help="layer for the zoomed chart"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.package / "plan.json").read_text())
    stream = json.loads(args.stream_report.read_text())["stream"]
    bandwidth = max(row["gbps"] for row in stream)
    report = {
        "package": str(args.package),
        "bandwidth_gbps": bandwidth,
        "bandwidth_source": str(args.stream_report),
        "positions": [],
    }
    report["synthetic_prefix"] = args.synthetic_prefix
    positions = sorted(set(args.positions))
    completed = subprocess.run(
        [
            str(args.executor),
            "--timeline-stack",
            str(args.package),
            ",".join(map(str, positions)),
            str(args.token_id),
            str(args.warmup),
            str(args.samples),
        ]
        + (["--synthetic-prefix"] if args.synthetic_prefix else []),
        capture_output=True,
        text=True,
        check=True,
        timeout=3600,
    )
    traces = [json.loads(line) for line in completed.stdout.splitlines()]
    if [trace["position"] for trace in traces] != positions:
        raise RuntimeError("timeline output does not cover every position")
    for position, trace in zip(positions, traces, strict=True):
        (args.out / f"trace-p{position}.json").write_text(json.dumps(trace) + "\n")
        times = [sample["kernel_us"] for sample in trace["samples"]]
        median = sorted(range(len(times)), key=times.__getitem__)[len(times) // 2]
        events, info = align(trace, median)
        result = budget(plan, events, position, bandwidth)
        result.update(info)
        result["kernel_samples_us"] = times
        waves = result.pop("waves")
        (args.out / f"waves-p{position}.json").write_text(
            json.dumps(waves, indent=1) + "\n"
        )
        report["positions"].append(result)
        context = position + 1
        (args.out / f"timeline-p{position}.svg").write_text(
            render_timeline_svg(
                plan, events, title=f"Worker timeline, one token at context {context}"
            )
        )
        (args.out / f"timeline-p{position}-layer{args.layer}.svg").write_text(
            render_timeline_svg(
                plan,
                events,
                title=f"Layer {args.layer} at context {context}",
                window=layer_window(plan, events, args.layer),
            )
        )
        components = ", ".join(
            f"{name} {value:.0f}" for name, value in result["components_us"].items()
        )
        print(
            f"context {context}: traced {result['traced_kernel_us']:.0f} us, ideal "
            f"{result['ideal_stream_us']:.0f} us, bubbles {result['bubble_us']:.0f} us "
            f"({components})"
        )
    (args.out / "timeline-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
