"""Render the S3 post's figures from ``bandwidth_probe ring`` runs.

usage: python docs/blog/make_s3_figures.py RUN_DIR OUT_DIR

RUN_DIR holds ring-*.json, repeated runs of the committed probe, and
ring-v1-*.json, runs of its first version, whose ring recomputed every
stage's source address (same sweep, same JSON). Repeated runs are averaged.
"""

import json
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

from make_figures import GRID, MUTED, NEUTRAL, SERIES, SURFACE, WIDTH, Svg, bar, mean

Key = tuple[str, float, str, int, int]


def load(paths: list[Path]) -> dict[Key, float]:
    """Mean microseconds per (program, gap, prefetch, depth, stage bytes)."""
    samples: dict[Key, list[float]] = defaultdict(list)
    for path in paths:
        runs = json.loads(path.read_text())["ring"]["runs"]
        if not all(run["bitwise_equal"] for run in runs):
            raise SystemExit(f"{path}: a ring run differs from project_rows")
        for run in runs:
            key = (
                run["program"],
                run["gap_us"],
                run["prefetch"],
                run["depth"],
                run["stage_bytes"],
            )
            samples[key].append(run["us"])
    return {key: mean(values) for key, values in samples.items()}


def ideal_us(paths: list[Path], program: str) -> float:
    runs = json.loads(paths[0].read_text())["ring"]["runs"]
    return next(run["ideal_us"] for run in runs if run["program"] == program)


def place_labels(svg: Svg, x: float, labels: list[tuple[float, str]]) -> None:
    placed: list[float] = []
    for label_y, label in sorted(labels):
        label_y = max(label_y, placed[-1] + 15) if placed else label_y
        placed.append(label_y)
        svg.add(svg.text(x, label_y + 4, label, size=11.5))


def gap_absorption(times: dict[Key, float], ideal: float, out: Path) -> None:
    """Time above the floor per layer as post-barrier busy work grows."""
    gaps = sorted({key[1] for key in times if key[0] == "layer"})
    series = [
        ("direct loads (S2)", SERIES[0], ("direct", 0, 0)),
        ("ring filled before barrier", SERIES[1], ("across_barriers", 4, 16384)),
        ("ring filled after barrier", SERIES[2], ("after_barrier", 4, 16384)),
        ("L2 prefetch, 32 KB", SERIES[3], ("l2", 0, 32768)),
    ]
    direct = [times[("layer", gap, "direct", 0, 0)] - ideal for gap in gaps]
    svg = Svg(
        410,
        "Only prefetch issued after the barrier hides post-barrier work",
        "One Llama layer's four GEMV waves on 20 workers; rings are 4 × 16 KB; lower is better",
    )
    left, right, top, bottom = 64, 200, 102, 362
    hi = 80.0

    def x(gap: float) -> float:
        return left + gap / gaps[-1] * (WIDTH - left - right)

    def y(us: float) -> float:
        return bottom - us / hi * (bottom - top)

    svg.legend(24, 80, [(color, name) for name, color, _ in series])
    for tick in range(0, int(hi) + 1, 20):
        svg.add(
            f'<line x1="{left}" x2="{WIDTH - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                left - 8, y(tick) + 4, f"{tick} µs", size=11, color=MUTED, anchor="end"
            )
        )
    for gap in gaps:
        svg.add(
            svg.text(
                x(gap), bottom + 18, f"{gap:g}", size=11, color=MUTED, anchor="middle"
            )
        )
    svg.add(
        svg.text(
            (left + WIDTH - right) / 2,
            bottom + 38,
            "busy work after each barrier (µs); y: time above the bandwidth floor per layer",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    # References: the gap fully hidden (flat) and fully exposed (4 barriers per layer).
    references = [
        ("gap fully hidden", [direct[0] for _ in gaps]),
        ("gap fully exposed", [direct[0] + 4 * gap for gap in gaps]),
    ]
    labels: list[tuple[float, str]] = []
    for name, values in references:
        points = " ".join(
            f"{x(g):.1f},{y(v):.1f}" for g, v in zip(gaps, values, strict=True)
        )
        svg.add(
            f'<polyline points="{points}" fill="none" stroke="{NEUTRAL}" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        labels.append((y(values[-1]), name))
    for name, color, (prefetch, depth, stage) in series:
        values = [times[("layer", gap, prefetch, depth, stage)] - ideal for gap in gaps]
        points = " ".join(
            f"{x(g):.1f},{y(v):.1f}" for g, v in zip(gaps, values, strict=True)
        )
        svg.add(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        for gap, value in zip(gaps, values, strict=True):
            svg.add(
                f'<circle cx="{x(gap):.1f}" cy="{y(value):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{escape(name)}, {gap:g} µs gap: {value:.1f} µs above the floor</title></circle>'
            )
        labels.append((y(values[-1]), f"{name.split(',')[0]} {values[-1]:.0f}"))
    place_labels(svg, WIDTH - right + 10, labels)
    svg.write(out / "s3-gap-absorption.svg")


def stage_size(
    first: dict[Key, float], final: dict[Key, float], ideal: float, out: Path
) -> None:
    """Layer bandwidth per ring geometry, before and after the cursor fix."""
    configs = sorted(
        {(key[3], key[4]) for key in final if key[2] == "across_barriers"},
        key=lambda config: (config[1], config[0]),
    )
    bytes_per_layer = ideal * 212.9 * 1e3

    def gbps(times: dict[Key, float], prefetch: str, depth: int, stage: int) -> float:
        return bytes_per_layer / times[("layer", 0.0, prefetch, depth, stage)] / 1e3

    direct = gbps(final, "direct", 0, 0)
    row_height = 34
    top = 100
    bottom = top + row_height * len(configs)
    svg = Svg(
        bottom + 56,
        "Ring bandwidth depends on stage size, not ring depth",
        "One layer's GEMV waves, no gap, ring across barriers; longer is better",
    )
    left, right = 150, 90
    hi = 240.0

    def x(value: float) -> float:
        return left + value / hi * (WIDTH - left - right)

    svg.legend(
        24,
        80,
        [
            (SERIES[1], "first ring: address recomputed per stage"),
            (SERIES[0], "committed ring: address advanced per stage"),
        ],
    )
    for tick in range(0, int(hi) + 1, 40):
        svg.add(
            f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top - 6}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        if abs(x(tick) - x(direct)) > 30:
            svg.add(
                svg.text(
                    x(tick),
                    bottom + 18,
                    f"{tick}",
                    size=11,
                    color=MUTED,
                    anchor="middle",
                )
            )
    svg.add(
        svg.text(
            (left + WIDTH - right) / 2,
            bottom + 38,
            "GB/s",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    # The reference sits behind the bars; value labels clear it.
    svg.add(
        f'<line x1="{x(direct):.1f}" x2="{x(direct):.1f}" y1="{top - 6}" y2="{bottom}" stroke="{NEUTRAL}" stroke-width="2" stroke-dasharray="5 4"/>'
    )
    svg.add(
        svg.text(
            x(direct), bottom + 18, f"direct {direct:.0f}", size=11.5, anchor="middle"
        )
    )
    for index, (depth, stage) in enumerate(configs):
        row = top + index * row_height
        svg.add(
            svg.text(
                left - 10,
                row + 17,
                f"{depth} × {stage // 1024} KB",
                size=11.5,
                color=MUTED,
                anchor="end",
            )
        )
        for offset, (times, color, name) in enumerate(
            ((first, SERIES[1], "first ring"), (final, SERIES[0], "committed ring"))
        ):
            value = gbps(times, "across_barriers", depth, stage)
            y = row + 2 + offset * 14
            svg.add(
                bar(
                    left,
                    y,
                    x(value) - left,
                    12,
                    color,
                    f"{name}, {depth} × {stage // 1024} KB: {value:.0f} GB/s",
                    end=True,
                )
            )
            label_x = max(x(value), x(direct)) + 6
            svg.add(svg.text(label_x, y + 10, f"{value:.0f}", size=11))
    svg.write(out / "s3-stage-size.svg")


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    final_paths = sorted(run.glob("ring-[0-9]*.json"))
    first_paths = sorted(run.glob("ring-v1-*.json"))
    final = load(final_paths)
    ideal = ideal_us(final_paths, "layer")
    gap_absorption(final, ideal, out)
    stage_size(load(first_paths), final, ideal, out)


if __name__ == "__main__":
    main()
