"""Render the blog's baseline figures from an S0 run directory.

usage: python docs/blog/make_figures.py RUN_DIR OUT_DIR

RUN_DIR holds the S0 artifacts: stream.json and rowdot.json from
``bandwidth_probe``, context-sweep.json from ``benchmark_decode.py``,
timeline/timeline-report.json from ``profile_timeline.py``, traces/ with raw
``--timeline-stack`` output, and package-4k/plan.json. The SVGs are static,
light-surface figures; every mark carries a native tooltip.
"""

import json
import math
import sys
from html import escape
from pathlib import Path

from tileplan.timeline import STREAMING, align, layer_window, phase

# Reference categorical palette, fixed order (validated: four slots adjacent,
# three slots all-pairs, light surface). Aqua and yellow sit below 3:1, so
# every figure is direct-labeled and backed by a table in the post.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, TEXT, MUTED, GRID, NEUTRAL = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#e6e5e1",
    "#c9c8c2",
)
FONT = "font-family=\"Inter, 'Helvetica Neue', Arial, sans-serif\""
WIDTH = 760
LABELS = {
    "embedding": "embedding",
    "attn_norm": "attention norm",
    "qkv": "QKV",
    "rope": "RoPE + K/V write",
    "attention": "attention",
    "wo": "Wo",
    "ffn_norm": "FFN norm",
    "gate_up": "gate/up",
    "silu": "SiLU·up",
    "down": "down",
    "final_norm": "final norm",
    "lm_head": "lm_head",
}


class Svg:
    def __init__(self, height: int, title: str, subtitle: str) -> None:
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" role="img" aria-label="{escape(title)}">',
            f"<title>{escape(title)}</title>",
            f'<rect width="100%" height="100%" fill="{SURFACE}"/>',
            self.text(24, 32, title, size=17, weight=600),
            self.text(24, 54, subtitle, size=12.5, color=MUTED),
        ]

    @staticmethod
    def text(x, y, value, *, size=12, color=TEXT, anchor="start", weight=400) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'
        )

    def add(self, item: str) -> None:
        self.parts.append(item)

    def legend(self, x: float, y: float, entries: list[tuple[str, str]]) -> None:
        for color, label in entries:
            self.add(
                f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{color}"/>'
            )
            self.add(self.text(x + 15, y, label, size=12, color=MUTED))
            x += 22 + 6.8 * len(label)

    def write(self, path: Path) -> None:
        path.write_text("\n".join([*self.parts, "</svg>"]) + "\n")


def nice_step(span: float, target: int = 5) -> float:
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


def bar(
    x: float, y: float, width: float, height: float, color: str, tip: str, *, end: bool
) -> str:
    """Horizontal bar segment; only the data end is rounded (4px)."""
    width = max(width, 0.0)
    if end and width > 4:
        path = (
            f"M{x:.1f},{y:.1f} h{width - 4:.1f} a4,4 0 0 1 4,4 v{height - 8:.1f} "
            f"a4,4 0 0 1 -4,4 h{-(width - 4):.1f} z"
        )
        shape = f'<path d="{path}" fill="{color}">'
    else:
        shape = f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height}" fill="{color}">'
    closing = "</path>" if shape.startswith("<path") else "</rect>"
    return f"{shape}<title>{escape(tip)}</title>{closing}"


def context_sweep(run: Path, out: Path) -> None:
    sweep = json.loads((run / "context-sweep.json").read_text())
    report = json.loads((run / "timeline/timeline-report.json").read_text())
    floor = {p["position"] + 1: p["ideal_stream_us"] / 1e3 for p in report["positions"]}
    rows = sweep["baseline"]["rows"]
    contexts = [row["context_length"] for row in rows]
    series = [
        ("TileInfer", SERIES[0], [r["mean_us"] / 1e3 for r in sweep["runtime"]]),
        ("PyTorch CUDA Graph", SERIES[1], [r["graph_mean_us"] / 1e3 for r in rows]),
        ("PyTorch eager", SERIES[2], [r["eager_mean_us"] / 1e3 for r in rows]),
    ]
    svg = Svg(
        400,
        "One-token decode latency vs. context length",
        "Llama 3.2 1B, BF16, RTX 3050; fixed position, GPU-resident; lower is better",
    )
    left, right, top, bottom = 64, 170, 92, 352
    lo, hi = 10.0, 24.0

    def x(context: int) -> float:
        return left + math.log2(context) / 12 * (WIDTH - left - right)

    def y(ms: float) -> float:
        return bottom - (ms - lo) / (hi - lo) * (bottom - top)

    svg.legend(left, 80, [(c, n) for n, c, _ in series])
    for tick in range(int(lo), int(hi) + 1, 2):
        svg.add(
            f'<line x1="{left}" x2="{WIDTH - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                left - 8, y(tick) + 4, f"{tick} ms", size=11, color=MUTED, anchor="end"
            )
        )
    for context in contexts:
        svg.add(
            svg.text(
                x(context),
                bottom + 18,
                f"{context:,}",
                size=11,
                color=MUTED,
                anchor="middle",
            )
        )
    svg.add(
        svg.text(
            (left + WIDTH - right) / 2,
            bottom + 38,
            "context length (tokens, log scale)",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    floor_points = " ".join(
        f"{x(c):.1f},{y(floor[c]):.1f}" for c in contexts if c in floor
    )
    svg.add(
        f'<polyline points="{floor_points}" fill="none" stroke="{NEUTRAL}" stroke-width="2"/>'
    )
    svg.add(
        svg.text(
            WIDTH - right + 10,
            y(floor[contexts[-1]]) + 4,
            "bandwidth floor",
            size=11.5,
            color=MUTED,
        )
    )
    ends = []
    for name, color, values in series:
        points = " ".join(
            f"{x(c):.1f},{y(v):.1f}" for c, v in zip(contexts, values, strict=True)
        )
        svg.add(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for c, v in zip(contexts, values, strict=True):
            svg.add(
                f'<circle cx="{x(c):.1f}" cy="{y(v):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>{escape(name)}, context {c:,}: {v:.2f} ms</title></circle>'
            )
        ends.append((y(values[-1]), f"{name} {values[-1]:.1f} ms"))
    # End labels top to bottom, pushed apart so they never overlap.
    ends.sort()
    placed: list[float] = []
    for label_y, label in ends:
        label_y = max(label_y, placed[-1] + 15) if placed else label_y
        placed.append(label_y)
        svg.add(svg.text(WIDTH - right + 10, label_y + 4, label, size=11.5))
    svg.write(out / "context-sweep.svg")


def loss_budget(run: Path, out: Path) -> None:
    report = json.loads((run / "timeline/timeline-report.json").read_text())
    parts = [
        ("streaming_excess", "GEMVs below roofline", SERIES[0]),
        ("small_phases", "small phases", SERIES[1]),
        ("attention_excess", "attention", SERIES[2]),
        ("barrier_sync", "barrier latency", SERIES[3]),
    ]
    positions = report["positions"]
    svg = Svg(
        110 + 40 * len(positions),
        "Time per token above the bandwidth floor",
        "Traced worker kernel minus ideal streaming of unique weight and K/V bytes at 212.9 GB/s",
    )
    left, top, row, thick = 96, 100, 40, 22
    scale = (WIDTH - left - 190) / 10.5  # px per ms
    svg.legend(left, 80, [(c, label) for _, label, c in parts])
    for tick in range(0, 11, 2):
        gx = left + tick * scale
        svg.add(
            f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 6}" y2="{top + row * len(positions) - 10}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                gx,
                top + row * len(positions) + 6,
                f"{tick} ms",
                size=11,
                color=MUTED,
                anchor="middle",
            )
        )
    for i, p in enumerate(positions):
        y0 = top + i * row
        svg.add(
            svg.text(
                left - 10,
                y0 + 15,
                f"context {p['position'] + 1:,}",
                size=11.5,
                color=MUTED,
                anchor="end",
            )
        )
        x0 = left
        active = [(k, label, c, p["components_us"][k] / 1e3) for k, label, c in parts]
        for j, (_, label, color, ms) in enumerate(active):
            width = ms * scale
            last = j == len(active) - 1
            svg.add(
                bar(
                    x0,
                    y0,
                    width - (0 if last else 2),
                    thick,
                    color,
                    f"context {p['position'] + 1:,}: {label} {ms:.2f} ms",
                    end=last,
                )
            )
            x0 += width
        svg.add(
            svg.text(
                x0 + 8,
                y0 + 15,
                f"+{p['bubble_us'] / 1e3:.2f} ms over {p['ideal_stream_us'] / 1e3:.2f} ms ideal",
                size=11.5,
            )
        )
    svg.write(out / "loss-budget.svg")


def worker_bandwidth(run: Path, out: Path) -> None:
    rows = [
        r
        for r in json.loads((run / "rowdot.json").read_text())["rowdot"]
        if r.get("tile") == 0
    ]
    roofline = max(
        r["gbps"] for r in json.loads((run / "stream.json").read_text())["stream"]
    )
    counts = sorted({r["active"] for r in rows})
    by_count = {n: [r["gbps"] for r in rows if r["active"] == n] for n in counts}
    svg = Svg(
        380,
        "GEMV bandwidth vs. active workers",
        "Worker row-dot on the five Llama projection shapes; line = mean, band = min to max",
    )
    left, right, top, bottom = 64, 150, 76, 330

    def x(n: float) -> float:
        return left + (n - 1) / 19 * (WIDTH - left - right)

    def y(gbps: float) -> float:
        return bottom - gbps / 240 * (bottom - top)

    for tick in range(0, 241, 40):
        svg.add(
            f'<line x1="{left}" x2="{WIDTH - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                left - 8, y(tick) + 4, f"{tick}", size=11, color=MUTED, anchor="end"
            )
        )
    svg.add(svg.text(left - 8, top - 10, "GB/s", size=11, color=MUTED, anchor="end"))
    for n in (1, 4, 8, 12, 16, 20):
        svg.add(
            svg.text(x(n), bottom + 18, str(n), size=11, color=MUTED, anchor="middle")
        )
    svg.add(
        svg.text(
            (left + WIDTH - right) / 2,
            bottom + 38,
            "active workers (one per SM)",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    upper = " ".join(f"{x(n):.1f},{y(max(by_count[n])):.1f}" for n in counts)
    lower = " ".join(f"{x(n):.1f},{y(min(by_count[n])):.1f}" for n in reversed(counts))
    svg.add(
        f'<polygon points="{upper} {lower}" fill="{SERIES[0]}" fill-opacity="0.12"/>'
    )
    svg.add(
        f'<line x1="{left}" x2="{WIDTH - right}" y1="{y(roofline):.1f}" y2="{y(roofline):.1f}" stroke="{NEUTRAL}" stroke-width="2"/>'
    )
    svg.add(
        svg.text(
            WIDTH - right + 10,
            y(roofline) + 4,
            f"stream roofline {roofline:.0f}",
            size=11.5,
            color=MUTED,
        )
    )
    means = [sum(by_count[n]) / len(by_count[n]) for n in counts]
    points = " ".join(
        f"{x(n):.1f},{y(m):.1f}" for n, m in zip(counts, means, strict=True)
    )
    svg.add(
        f'<polyline points="{points}" fill="none" stroke="{SERIES[0]}" stroke-width="2" stroke-linejoin="round"/>'
    )
    for n, m in zip(counts, means, strict=True):
        svg.add(
            f'<circle cx="{x(n):.1f}" cy="{y(m):.1f}" r="4" fill="{SERIES[0]}" stroke="{SURFACE}" stroke-width="2"><title>{n} workers: mean {m:.0f} GB/s (min {min(by_count[n]):.0f}, max {max(by_count[n]):.0f})</title></circle>'
        )
    svg.add(
        svg.text(
            WIDTH - right + 10,
            y(means[-1]) + 18,
            f"20 workers {means[-1]:.0f}",
            size=11.5,
        )
    )
    svg.add(
        f'<line x1="{x(8):.1f}" x2="{x(8):.1f}" y1="{y(means[7]) + 8:.1f}" y2="{bottom}" stroke="{MUTED}" stroke-width="1"/>'
    )
    svg.add(
        svg.text(
            x(8) + 6,
            bottom - 10,
            f"8 workers already reach {means[7]:.0f} GB/s",
            size=11.5,
        )
    )
    svg.add(
        svg.text(x(1) + 8, y(means[0]) + 4, f"one worker {means[0]:.0f}", size=11.5)
    )
    svg.write(out / "worker-bandwidth.svg")


def phase_breakdown(run: Path, out: Path) -> None:
    report = json.loads((run / "timeline/timeline-report.json").read_text())
    phases = next(p for p in report["positions"] if p["position"] == 63)["phases"]
    names = sorted(phases, key=lambda n: -phases[n]["duration_us"])
    svg = Svg(
        124 + 28 * len(names),
        "Where a token's time goes, by phase (context 64)",
        "Per-token time summed over all layers; ideal = unique bytes at 212.9 GB/s",
    )
    left, top, row, thick = 124, 96, 28, 18
    scale = (WIDTH - left - 150) / 5500
    svg.legend(left, 78, [(SERIES[0], "ideal streaming"), (SERIES[1], "above ideal")])
    for tick in range(0, 5501, 1000):
        gx = left + tick * scale
        svg.add(
            f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 6}" y2="{top + row * len(names) - 8}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                gx,
                top + row * len(names) + 8,
                f"{tick / 1000:g} ms",
                size=11,
                color=MUTED,
                anchor="middle",
            )
        )
    for i, name in enumerate(names):
        p = phases[name]
        y0 = top + i * row
        ideal = min(p["ideal_us"], p["duration_us"])
        above = p["duration_us"] - ideal
        svg.add(
            svg.text(
                left - 10, y0 + 13, LABELS[name], size=11.5, color=MUTED, anchor="end"
            )
        )
        if ideal * scale > 1:
            svg.add(
                bar(
                    left,
                    y0,
                    ideal * scale - (2 if above * scale > 1 else 0),
                    thick,
                    SERIES[0],
                    f"{LABELS[name]}: ideal {ideal:.0f} µs",
                    end=above * scale <= 1,
                )
            )
        svg.add(
            bar(
                left + ideal * scale,
                y0,
                above * scale,
                thick,
                SERIES[1],
                f"{LABELS[name]}: above ideal {above:.0f} µs",
                end=True,
            )
        )
        svg.add(
            svg.text(
                left + p["duration_us"] * scale + 8,
                y0 + 13,
                f"{p['duration_us']:,.0f} µs",
                size=11.5,
            )
        )
    svg.write(out / "phase-breakdown.svg")


def group(task: dict) -> str:
    name = phase(task)
    if name in STREAMING:
        return "weight streaming"
    return "attention" if name == "attention" else "small phases"


GROUPS = {
    "weight streaming": SERIES[0],
    "attention": SERIES[1],
    "small phases": SERIES[2],
    "barrier wait": NEUTRAL,
}


def timeline(run: Path, out: Path, position: int, layer: int) -> None:
    plan = json.loads((run / "package-4k/plan.json").read_text())
    trace = json.loads((run / f"traces/trace-p{position}.json").read_text())
    times = [s["kernel_us"] for s in trace["samples"]]
    events, _ = align(
        trace, sorted(range(len(times)), key=times.__getitem__)[len(times) // 2]
    )
    begin, end = layer_window(plan, events, layer)
    tasks = plan["tasks"]
    workers = plan["target"]["worker_count"]
    span = end - begin
    left, top, row = 74, 96, 13
    plot = WIDTH - left - 24
    svg = Svg(
        top + workers * row + 50,
        f"One decoder layer on 20 workers, context {position + 1:,}",
        f"Layer {layer} of 16, {span / 1e3:.0f} µs; each row is one worker (thread block), time runs right",
    )
    svg.legend(left, 80, [(c, n) for n, c in GROUPS.items()])
    for w in range(workers):
        if w % 4 == 0:
            svg.add(
                svg.text(
                    left - 8,
                    top + w * row + 10,
                    f"w{w}",
                    size=10,
                    color=MUTED,
                    anchor="end",
                )
            )
    for e in events:
        if e.end < begin or e.start > end:
            continue
        name = "barrier wait" if e.task < 0 else group(tasks[e.task])
        x0 = left + (max(e.start, begin) - begin) / span * plot
        x1 = left + (min(e.end, end) - begin) / span * plot
        label = "barrier wait" if e.task < 0 else LABELS[phase(tasks[e.task])]
        svg.add(
            f'<rect x="{x0:.2f}" y="{top + e.worker * row + 1}" width="{max(x1 - x0 - 0.6, 0.4):.2f}" height="{row - 2}" fill="{GROUPS[name]}">'
            f"<title>worker {e.worker}: {escape(label)}, {(e.end - e.start) / 1e3:.1f} µs</title></rect>"
        )
    step = nice_step(span / 1e3)
    tick = 0.0
    axis = top + workers * row + 16
    while tick <= span / 1e3:
        gx = left + tick * 1e3 / span * plot
        svg.add(
            svg.text(gx, axis, f"{tick:g} µs", size=10.5, color=MUTED, anchor="middle")
        )
        tick += step
    svg.write(out / f"timeline-layer-ctx{position + 1}.svg")


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    context_sweep(run, out)
    loss_budget(run, out)
    worker_bandwidth(run, out)
    phase_breakdown(run, out)
    timeline(run, out, 63, 8)
    timeline(run, out, 4095, 8)


if __name__ == "__main__":
    main()
