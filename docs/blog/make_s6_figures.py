"""Render the S6 showcase figures from a showcase run directory.

usage: python docs/blog/make_s6_figures.py RUN_DIR OUT_DIR

RUN_DIR is written by ``experiments/showcase/showcase.py run``: the
controlled sessions in decode-*/report.json, timeline/ (timeline-report.json
and the raw trace-pPOSITION.json), kernels/ from the Nsight traces, and
stream.json, rowdot.json, target.json and package-4k/plan.json. The schedule
illustrations read only the plan and the target, so they describe the
package that was measured. The SVGs use the palette and conventions of
make_figures.py; every mark carries a native tooltip.
"""

import json
import statistics
import sys
from html import escape
from pathlib import Path

from make_figures import (
    GRID,
    MUTED,
    NEUTRAL,
    SERIES,
    SURFACE,
    TEXT,
    WIDTH,
    Svg,
    nice_step,
)
from make_s3_figures import place_labels

from tileplan.timeline import (
    STREAMING,
    Event,
    align,
    layer_window,
    phase,
    streamed_bytes,
)

WEIGHT_BYTES = 2_471_628_800
KV_BYTES_PER_POSITION = 32_768
ZOOM_LAYER = 8
# Phase groups of the worker timelines (make_figures.GROUPS).
GROUPS = {
    "weight streaming": SERIES[0],
    "attention": SERIES[1],
    "small phases": SERIES[2],
    "barrier wait": NEUTRAL,
}
LABELS = {
    "embedding": "embedding",
    "qkv": "QKV",
    "attention": "attention",
    "wo": "Wo",
    "ffn": "FFN",
    "ffn_reduce": "FFN reduce",
    "final_norm": "final norm",
    "lm_head": "lm_head",
}
ARMS = (
    ("tileinfer", "TileInfer", SERIES[0], ""),
    ("llamacpp-f16", "llama.cpp, f16 K/V", SERIES[1], ""),
    ("llamacpp-bf16", "llama.cpp, BF16 K/V", SERIES[1], "5 4"),
    ("graph", "PyTorch graph", SERIES[2], ""),
    ("graph-sdpa", "PyTorch graph, SDPA", SERIES[2], "5 4"),
    ("eager", "PyTorch eager", SERIES[3], ""),
)
ARROW = (
    '<defs><marker id="ah" markerUnits="userSpaceOnUse" markerWidth="9" '
    'markerHeight="9" refX="8" refY="4.5" orient="auto">'
    f'<path d="M0,0 L9,4.5 L0,9 z" fill="{MUTED}"/></marker></defs>'
)
FILL = ("#e8f1fc", "#fdeee7", "#e6f6ef", "#fdf3dc")  # SERIES at ~10%


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def floor_ms(context: int, gbps: float) -> float:
    return (WEIGHT_BYTES + KV_BYTES_PER_POSITION * context) / gbps / 1e6


def roofline(run: Path) -> float:
    return max(row["gbps"] for row in load(run / "stream.json")["stream"])


def mb(count: float) -> str:
    if count >= 1e5:
        return f"{count / 1e6:.1f} MB"
    return f"{count / 1e3:.0f} KB" if count >= 1e3 else f"{count:.0f} B"


def box(
    svg: Svg,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    *,
    fill: str = SURFACE,
    stroke: str = NEUTRAL,
    dash: str = "",
) -> None:
    """A rounded box with a bold first line and muted detail lines."""
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dashed}/>'
    )
    for i, line in enumerate(lines):
        svg.add(
            svg.text(
                x + width / 2,
                y + 20 + 15 * i,
                line,
                size=12.5 if i == 0 else 11,
                color=TEXT if i == 0 else MUTED,
                anchor="middle",
                weight=600 if i == 0 else 400,
            )
        )


def arrow(svg: Svg, x1: float, y1: float, x2: float, y2: float) -> None:
    svg.add(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{MUTED}" stroke-width="1.5" marker-end="url(#ah)"/>'
    )


# --- comparison -----------------------------------------------------------


def comparison_table(run: Path) -> dict[int, dict[str, float]]:
    """Median µs per arm and context, averaged over both spawn orders."""
    cells: dict[int, dict[str, list[float]]] = {}
    for report in sorted(run.glob("decode-*/report.json")):
        for context in load(report)["contexts"]:
            row = cells.setdefault(context["context_length"], {})
            for arm, summary in context["summary"].items():
                row.setdefault(arm, []).append(summary["median_us"])
    return {
        length: {arm: statistics.mean(values) for arm, values in row.items()}
        for length, row in sorted(cells.items())
    }


def comparison(run: Path, out: Path) -> None:
    table = comparison_table(run)
    gbps = roofline(run)
    contexts = list(table)
    values = [v / 1e3 for row in table.values() for v in row.values()]
    low = min(floor_ms(contexts[0], gbps), min(values)) - 0.3
    high = max(values) + 0.4
    svg = Svg(
        440,
        "Decode latency against context on the RTX 3050",
        "Llama 3.2 1B, BF16 weights, batch 1, ms per token with logits ready; "
        "labels on TileInfer give floor ÷ time",
    )
    left, right, top, bottom = 56, 170, 80, 390

    def x(context: float) -> float:
        return left + context / contexts[-1] * (WIDTH - left - right)

    def y(ms: float) -> float:
        return bottom - (ms - low) / (high - low) * (bottom - top)

    step = nice_step(high - low)
    tick = step * (low // step + 1)
    while tick < high:
        svg.add(
            f'<line x1="{left}" x2="{WIDTH - right}" y1="{y(tick):.1f}" '
            f'y2="{y(tick):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                left - 8, y(tick) + 4, f"{tick:g}", size=11, color=MUTED, anchor="end"
            )
        )
        tick += step
    svg.add(svg.text(left - 8, top - 10, "ms", size=11, color=MUTED, anchor="end"))
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
            "context length (tokens in the K/V cache, including the new one)",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    floor = " ".join(f"{x(c):.1f},{y(floor_ms(c, gbps)):.1f}" for c in contexts)
    svg.add(
        f'<polyline points="{floor}" fill="none" stroke="{NEUTRAL}" stroke-width="2" '
        'stroke-dasharray="2 4"/>'
    )
    labels = [(y(floor_ms(contexts[-1], gbps)), f"floor at {gbps:.0f} GB/s")]
    for arm, label, color, dash in ARMS:
        points = [(c, table[c][arm] / 1e3) for c in contexts if arm in table[c]]
        if not points:
            continue
        width = 3 if arm == "tileinfer" else 2
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        line = " ".join(f"{x(c):.1f},{y(ms):.1f}" for c, ms in points)
        svg.add(
            f'<polyline points="{line}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dashed}/>'
        )
        for c, ms in points:
            svg.add(
                f'<circle cx="{x(c):.1f}" cy="{y(ms):.1f}" r="3.5" fill="{color}" '
                f'stroke="{SURFACE}" stroke-width="1.5"><title>{escape(label)}, context '
                f"{c:,}: {ms:.3f} ms ({100 * floor_ms(c, gbps) / ms:.0f}% of the floor)"
                "</title></circle>"
            )
        labels.append((y(points[-1][1]), f"{label} {points[-1][1]:.2f}"))
    # Context 256 sits next to context 1 on a linear axis; label one of them.
    for c in contexts:
        if "tileinfer" in table[c] and c != 256:
            ms = table[c]["tileinfer"] / 1e3
            svg.add(
                svg.text(
                    x(c) + (4 if c == contexts[0] else 0),
                    y(ms) + 18,
                    f"{100 * floor_ms(c, gbps) / ms:.0f}%",
                    size=10.5,
                    color=SERIES[0],
                    anchor="start" if c == contexts[0] else "middle",
                    weight=600,
                )
            )
    place_labels(svg, WIDTH - right + 10, labels)
    svg.write(out / "s6-comparison.svg")


# --- worker timeline -------------------------------------------------------


def group(task: dict) -> str:
    name = phase(task)
    if name in STREAMING:
        return "weight streaming"
    return "attention" if name == "attention" else "small phases"


def traced(run: Path, position: int) -> tuple[dict, list[Event], dict]:
    plan = load(run / "package-4k/plan.json")
    trace = load(run / f"timeline/trace-p{position}.json")
    times = [s["kernel_us"] for s in trace["samples"]]
    events, _ = align(
        trace, sorted(range(len(times)), key=times.__getitem__)[len(times) // 2]
    )
    report = next(
        p
        for p in load(run / "timeline/timeline-report.json")["positions"]
        if p["position"] == position
    )
    return plan, events, report


def gantt(
    svg: Svg,
    plan: dict,
    events: list[Event],
    window: tuple[float, float],
    left: float,
    top: float,
    row: float,
) -> None:
    tasks = plan["tasks"]
    begin, end = window
    plot = WIDTH - left - 24
    for e in events:
        if e.end < begin or e.start > end:
            continue
        name = "barrier wait" if e.task < 0 else group(tasks[e.task])
        label = "barrier wait" if e.task < 0 else LABELS[phase(tasks[e.task])]
        x0 = left + (max(e.start, begin) - begin) / (end - begin) * plot
        x1 = left + (min(e.end, end) - begin) / (end - begin) * plot
        svg.add(
            f'<rect x="{x0:.2f}" y="{top + e.worker * row + 0.5:.1f}" '
            f'width="{max(x1 - x0 - 0.3, 0.3):.2f}" height="{row - 1:.1f}" '
            f'fill="{GROUPS[name]}"><title>worker {e.worker}: {escape(label)}, '
            f"{(e.end - e.start) / 1e3:.1f} µs</title></rect>"
        )


def time_axis(svg: Svg, left: float, y: float, span_us: float, unit: str) -> None:
    plot = WIDTH - left - 24
    scale = 1e3 if unit == "ms" else 1
    step = nice_step(span_us / scale)
    tick = 0.0
    while tick <= span_us / scale + 1e-9:
        svg.add(
            svg.text(
                left + tick * scale / span_us * plot,
                y,
                f"{tick:g} {unit}",
                size=10.5,
                color=MUTED,
                anchor="middle",
            )
        )
        tick += step


def timeline(run: Path, out: Path, position: int) -> None:
    plan, events, report = traced(run, position)
    tasks = plan["tasks"]
    model = plan["model"]
    workers = plan["target"]["worker_count"]
    start = min(e.start for e in events)
    finish = max(e.end for e in events)
    left = 52
    plot = WIDTH - left - 24
    whole_top, whole_row = 112, 7
    zoom_top, zoom_row = whole_top + workers * whole_row + 110, 12
    budget_top = zoom_top + workers * zoom_row + 64
    svg = Svg(
        budget_top + 120,
        f"One token on 20 workers, context {position + 1:,}",
        f"One launch, {len(tasks):,} tile tasks in {len(plan['waves'])} waves; each "
        "row is a worker (one thread block per SM), time runs right",
    )
    svg.legend(left, 80, [(c, n) for n, c in GROUPS.items()])

    def at(t: float) -> float:
        return left + (t - start) / (finish - start) * plot

    # Layer boundaries: the first instruction of each layer.
    firsts: dict[int | str, float] = {}
    for e in events:
        if e.task >= 0:
            layer = tasks[e.task]["layer"]
            key = layer if layer is not None else phase(tasks[e.task])
            firsts[key] = min(firsts.get(key, e.end), e.start)
    for key, t in firsts.items():
        if isinstance(key, int) and key % 4 == 0 or key == "lm_head":
            label = f"L{key}" if isinstance(key, int) else "lm_head"
            svg.add(
                f'<line x1="{at(t):.1f}" x2="{at(t):.1f}" y1="{whole_top - 6}" '
                f'y2="{whole_top + workers * whole_row}" stroke="{TEXT}" stroke-width="0.8" '
                'stroke-opacity="0.5"/>'
            )
            svg.add(svg.text(at(t) + 3, whole_top - 9, label, size=10.5, color=MUTED))
    gantt(svg, plan, events, (start, finish), left, whole_top, whole_row)
    svg.add(
        svg.text(left - 8, whole_top + 10, "w0", size=10, color=MUTED, anchor="end")
    )
    svg.add(
        svg.text(
            left - 8,
            whole_top + workers * whole_row,
            f"w{workers - 1}",
            size=10,
            color=MUTED,
            anchor="end",
        )
    )
    time_axis(
        svg, left, whole_top + workers * whole_row + 16, (finish - start) / 1e3, "ms"
    )

    begin, end = layer_window(plan, events, ZOOM_LAYER)
    zoom_bottom = zoom_top + workers * zoom_row
    svg.add(
        f'<path d="M{at(begin):.1f},{whole_top + workers * whole_row + 22} '
        f"L{left},{zoom_top - 40} M{at(end):.1f},{whole_top + workers * whole_row + 22} "
        f'L{WIDTH - 24},{zoom_top - 40}" stroke="{NEUTRAL}" stroke-width="1" fill="none"/>'
    )
    svg.add(
        f'<rect x="{at(begin):.1f}" y="{whole_top - 2}" width="{at(end) - at(begin):.1f}" '
        f'height="{workers * whole_row + 4}" fill="none" stroke="{TEXT}" stroke-width="1"/>'
    )
    svg.add(
        svg.text(
            left,
            zoom_top - 52,
            f"Layer {ZOOM_LAYER}, {(end - begin) / 1e3:.0f} µs; each wave with its "
            "streamed bytes per task",
            size=12.5,
            weight=600,
        )
    )
    # Caption every wave of the zoomed layer with its bytes per task.
    waves: dict[int, list[Event]] = {}
    for e in events:
        if e.task >= 0 and tasks[e.task]["layer"] == ZOOM_LAYER:
            waves.setdefault(tasks[e.task]["wave"], []).append(e)

    def zoom_at(t: float) -> float:
        return left + (t - begin) / (end - begin) * plot

    # One "phase bytes" caption per wave on three staggered rows, so narrow
    # waves (QKV, the two attention waves, Wo) do not overlap.
    for index, (_, members) in enumerate(sorted(waves.items())):
        w0 = min(e.start for e in members)
        w1 = max(e.end for e in members)
        loads = [streamed_bytes(tasks[e.task], model, position) for e in members]
        loads = [b for b in loads if b]
        opcode = tasks[members[0].task]["opcode"]
        name = LABELS[phase(tasks[members[0].task])]
        if opcode.startswith("attention"):
            name = "scores" if "scores" in opcode else "values"
        detail = mb(statistics.mean(loads)) if loads else "no weights"
        cx = (zoom_at(w0) + zoom_at(w1)) / 2
        svg.add(
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{zoom_top - 33 + 12 * (index % 3) + 3}" '
            f'y2="{zoom_top}" stroke="{NEUTRAL}" stroke-width="0.8"/>'
        )
        svg.add(
            svg.text(
                cx + 3,
                zoom_top - 33 + 12 * (index % 3),
                f"{name} {detail}",
                size=10.5,
                anchor="end" if cx > WIDTH - 140 else "start",
            )
        )
    gantt(svg, plan, events, (begin, end), left, zoom_top, zoom_row)
    for w in range(0, workers, 4):
        svg.add(
            svg.text(
                left - 8,
                zoom_top + w * zoom_row + 10,
                f"w{w}",
                size=10,
                color=MUTED,
                anchor="end",
            )
        )
    time_axis(svg, left, zoom_bottom + 16, (end - begin) / 1e3, "µs")

    components = report["components_us"]
    bubble = report["bubble_us"]
    svg.add(
        svg.text(
            left,
            budget_top,
            f"Where the token's time goes: traced {report['traced_kernel_us'] / 1e3:.2f} ms "
            f"= ideal streaming {report['ideal_stream_us'] / 1e3:.2f} ms + {bubble:.0f} µs",
            size=12.5,
            weight=600,
        )
    )
    parts = [
        ("streaming_excess", "GEMV below the roofline", SERIES[0]),
        ("attention_excess", "attention", SERIES[1]),
        ("small_phases", "small phases", SERIES[2]),
        ("barrier_sync", "barrier latency", NEUTRAL),
    ]
    x = left
    for key, label, color in parts:
        width = components.get(key, 0) / bubble * plot
        svg.add(
            f'<rect x="{x:.1f}" y="{budget_top + 14}" width="{max(width - 1, 0):.1f}" '
            f'height="22" fill="{color}"><title>{label}: {components.get(key, 0):.0f} µs'
            "</title></rect>"
        )
        if width > 60:
            svg.add(
                svg.text(
                    x + 6,
                    budget_top + 52,
                    f"{label} {components.get(key, 0):.0f} µs",
                    size=11,
                    color=MUTED,
                )
            )
        x += width
    svg.add(
        svg.text(
            left,
            budget_top + 76,
            "Ideal streaming: the token's weight and K/V bytes at the measured "
            f"{report['bandwidth_gbps']:.0f} GB/s; the bar splits the {bubble:.0f} µs above it.",
            size=11,
            color=MUTED,
        )
    )
    svg.write(out / f"s6-timeline-ctx{position + 1}.svg")


# --- kernel strips ---------------------------------------------------------


KERNEL_ROWS = (
    ("tileinfer", "TileInfer", SERIES[0]),
    ("pytorch-graph", "PyTorch graph", SERIES[2]),
    ("llamacpp", "llama.cpp, f16 K/V", SERIES[1]),
    ("pytorch-eager", "PyTorch eager", SERIES[3]),
)


def kernels(run: Path, out: Path, position: int) -> None:
    rows = []
    for runtime, label, color in KERNEL_ROWS:
        path = run / f"kernels/kernels-{runtime}-p{position}.json"
        if path.exists():
            rows.append((load(path), label, color))
    # Each strip starts at the previous token's last kernel.
    total = max(
        k["span_us"] + max(k.get("gap_before_us", 0.0), 0.0) for k, _, _ in rows
    )
    left, label_width, strip = 24, 150, 26
    plot_left = left + label_width
    plot = WIDTH - plot_left - 24
    top = 92
    zoom_top = top + len(rows) * (strip + 22) + 56
    window = 400.0  # µs
    svg = Svg(
        zoom_top + len(rows) * (strip + 22) + 30,
        f"One token as the GPU sees it, context {position + 1:,}",
        "Nsight Systems, from the previous token's last kernel (grey: idle) through "
        "the next token; shades alternate per kernel",
    )

    def strip_row(
        y: float, k: dict, color: str, begin: float, width_us: float, labels: bool
    ) -> None:
        svg.add(
            f'<rect x="{plot_left}" y="{y}" width="{plot}" height="{strip}" fill="{GRID}" '
            'fill-opacity="0.5"/>'
        )
        for i, kernel in enumerate(k["kernels"]):
            s = kernel["start_us"] - begin
            e = s + kernel["duration_us"]
            if e < 0 or s > width_us:
                continue
            x0 = plot_left + max(s, 0) / width_us * plot
            x1 = plot_left + min(e, width_us) / width_us * plot
            svg.add(
                f'<rect x="{x0:.2f}" y="{y}" width="{max(x1 - x0, 0.25):.2f}" height="{strip}" '
                f'fill="{color}" fill-opacity="{1.0 if i % 2 == 0 else 0.55}">'
                f"<title>{escape(kernel['name'][:80])}: {kernel['duration_us']:.1f} µs"
                "</title></rect>"
            )
        if labels and k["kernel_count"] == 1 and plan_waves:
            svg.add(
                svg.text(
                    plot_left + max(-begin, 0) / width_us * plot + 8,
                    y + strip / 2 + 4,
                    f"one kernel; its {plan_waves} waves meet at grid barriers inside",
                    size=11,
                    color=SURFACE,
                    weight=600,
                )
            )

    plan_waves = len(load(run / "package-4k/plan.json")["waves"])
    for index, (k, label, color) in enumerate(rows):
        y = top + index * (strip + 22)
        svg.add(svg.text(left, y + 12, label, size=12, weight=600))
        svg.add(
            svg.text(
                left,
                y + 26,
                f"{k['kernel_count']} kernel{'s' if k['kernel_count'] > 1 else ''}, "
                f"{max(k['idle_us'], 0):.0f} µs idle inside",
                size=10.5,
                color=MUTED,
            )
        )
        gap = max(k.get("gap_before_us", 0.0), 0.0)
        svg.add(
            svg.text(
                left,
                y + 39,
                f"{gap:,.0f} µs idle before it",
                size=10.5,
                color=MUTED,
            )
        )
        strip_row(y, k, color, -gap, total, True)
        # The idle time since the previous token's last kernel, hatched grey.
        svg.add(
            f'<rect x="{plot_left}" y="{y}" width="{gap / total * plot:.2f}" '
            f'height="{strip}" fill="{NEUTRAL}" fill-opacity="0.8"><title>idle since '
            f"the previous token: {gap:.0f} µs</title></rect>"
        )
    time_axis_at = top + len(rows) * (strip + 22) - 4
    step = nice_step(total / 1e3)
    tick = 0.0
    while tick <= total / 1e3 + 1e-9:
        svg.add(
            svg.text(
                plot_left + tick * 1e3 / total * plot,
                time_axis_at,
                f"{tick:g} ms",
                size=10.5,
                color=MUTED,
                anchor="middle",
            )
        )
        tick += step
    svg.add(
        svg.text(
            left,
            zoom_top - 16,
            f"The same tokens, {window:.0f} µs from the middle of each",
            size=12.5,
            weight=600,
        )
    )
    for index, (k, label, color) in enumerate(rows):
        y = zoom_top + index * (strip + 22)
        middle = k["span_us"] / 2
        svg.add(svg.text(left, y + 17, label, size=12))
        strip_row(y, k, color, middle, window, False)
        inside = [
            kernel
            for kernel in k["kernels"]
            if kernel["start_us"] < middle + window
            and kernel["start_us"] + kernel["duration_us"] > middle
        ]
        svg.add(
            svg.text(
                plot_left + plot,
                y + strip + 13,
                f"{len(inside)} kernel{'s' if len(inside) != 1 else ''} in this window",
                size=10.5,
                color=MUTED,
                anchor="end",
            )
        )
    svg.write(out / f"s6-kernels-ctx{position + 1}.svg")


# --- schedule illustrations --------------------------------------------------


def layer_tasks(plan: dict, layer: int) -> dict[int, list[dict]]:
    waves: dict[int, list[dict]] = {}
    for task in plan["tasks"]:
        if task["layer"] == layer:
            waves.setdefault(task["wave"], []).append(task)
    return dict(sorted(waves.items()))


def pipeline(run: Path, out: Path) -> None:
    plan = load(run / "package-4k/plan.json")
    target = load(run / "target.json")
    executor = target["executor"]
    model = plan["model"]
    svg = Svg(
        330,
        "From checkpoint to one launch",
        "Python plans once per model and GPU, offline; C++/CUDA executes the "
        "plan per request and makes no scheduling decision",
    )
    svg.add(ARROW)
    top, height, width, gap = 104, 128, 130, 17
    boxes = [
        (
            [
                "checkpoint",
                "Llama 3.2 1B",
                f"{plan['integrity']['weights_bytes'] / 1e9:.2f} GB BF16",
                f"rev {plan['source']['revision'][:7]}",
            ],
            FILL[3],
            SERIES[3],
        ),
        (
            [
                "tileplan probe",
                "the compiled worker:",
                f"{executor['active_blocks_per_sm']} block/SM × {executor['threads_per_block']}",
                f"{executor['registers_per_thread']} registers",
                f"{target['launch']['worker_count']} workers",
            ],
            FILL[0],
            SERIES[0],
        ),
        (
            [
                "tileplan import",
                f"{len(plan['tasks']):,} tile tasks",
                f"in {len(plan['waves'])} waves,",
                "each on a fixed",
                "worker; validated",
            ],
            FILL[0],
            SERIES[0],
        ),
        (
            [
                "package",
                f"plan.json (v{plan['header']['format_version']})",
                "weights.bin",
                "target.json",
                f"context ≤ {model['max_context']:,}",
            ],
            SURFACE,
            NEUTRAL,
        ),
        (
            [
                "tile_runtime",
                "loads, re-validates,",
                "1 cooperative launch",
                "per ≤ 64 tokens",
                "20 instruction streams",
            ],
            FILL[1],
            SERIES[1],
        ),
    ]
    x = 24
    for i, (lines, fill, stroke) in enumerate(boxes):
        box(svg, x, top, width, height, lines, fill=fill, stroke=stroke)
        if i < len(boxes) - 1:
            arrow(
                svg,
                x + width + 1,
                top + height / 2,
                x + width + gap - 1,
                top + height / 2,
            )
        x += width + gap
    brace = top + height + 26
    split = 24 + 3 * (width + gap) + width / 2
    for x0, x1, label in (
        (24, split - 6, "Python, offline: probe, plan, validate, pack"),
        (split + 6, WIDTH - 24, "C++/CUDA, online"),
    ):
        svg.add(
            f'<path d="M{x0},{brace} v8 H{x1} v-8" fill="none" stroke="{MUTED}" stroke-width="1"/>'
        )
        svg.add(
            svg.text(
                (x0 + x1) / 2,
                brace + 26,
                label,
                size=11.5,
                color=MUTED,
                anchor="middle",
            )
        )
    svg.add(
        svg.text(
            24,
            brace + 56,
            "The probe queries the occupancy of the actual worker binary; the plan "
            "records its fingerprint, and the runtime rejects any other worker.",
            size=11,
            color=MUTED,
        )
    )
    svg.write(out / "s6-pipeline.svg")


def layer_tiles(run: Path, out: Path) -> None:
    plan = load(run / "package-4k/plan.json")
    model = plan["model"]
    waves = layer_tasks(plan, ZOOM_LAYER)
    d, f = model["hidden_size"], model["intermediate_size"]
    q_rows = (model["num_attention_heads"] + 2 * model["num_key_value_heads"]) * model[
        "head_dim"
    ]
    rows_of = {
        "qkv": (
            q_rows,
            f"[Q; K; V] rows, {q_rows:,} × {d:,}",
            "RMSNorm(x) recomputed in every task",
        ),
        "wo": (d, f"Wo rows, {d:,} × {d:,}", "plus the residual add"),
        "ffn": (
            f,
            f"gate/up rows and Wdown columns, {f:,} of them",
            "SiLU(gate)·up, then an FP32 down partial over all 2,048 outputs",
        ),
        "ffn_reduce": (
            d,
            f"output rows, {d:,}",
            "sum of the 20 partials in a fixed order, + residual",
        ),
    }
    left, plot = 24, WIDTH - 48
    bars = [(w, tasks) for w, tasks in waves.items() if phase(tasks[0]) in rows_of]
    top = 96
    gap = 74
    attention_top = top + len(bars) * gap + 20
    svg = Svg(
        attention_top + 200,
        "How one layer becomes tiles",
        f"Layer {ZOOM_LAYER}; each bar is one wave's output, cut into one task per "
        "worker (shades alternate per task, numbers are workers)",
    )
    for index, (_, tasks) in enumerate(bars):
        name = phase(tasks[0])
        total, title, detail = rows_of[name]
        y = top + index * gap
        loads = [streamed_bytes(t, model, 0) for t in tasks]
        sizes = sorted(
            {t["operands"]["rows"][1] - t["operands"]["rows"][0] for t in tasks}
        )
        size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} or {sizes[-1]}"
        summary = f"{LABELS[name]}: {len(tasks)} tasks of {size} rows"
        if any(loads):
            summary += f", {mb(statistics.mean(loads))} of weights each"
        svg.add(svg.text(left, y, summary, size=12, weight=600))
        svg.add(svg.text(left, y + 15, f"{title}; {detail}", size=10.5, color=MUTED))
        for t in sorted(tasks, key=lambda t: t["operands"]["rows"][0]):
            r0, r1 = t["operands"]["rows"]
            x0 = left + r0 / total * plot
            x1 = left + r1 / total * plot
            color = SERIES[2] if name == "ffn_reduce" else SERIES[0]
            svg.add(
                f'<rect x="{x0:.1f}" y="{y + 22}" width="{x1 - x0 - 0.6:.1f}" height="22" '
                f'fill="{color}" fill-opacity="{0.9 if t["worker"] % 2 == 0 else 0.55}">'
                f"<title>worker {t['worker']}: rows {r0}–{r1}</title></rect>"
            )
            svg.add(
                svg.text(
                    (x0 + x1) / 2,
                    y + 37,
                    str(t["worker"]),
                    size=9.5,
                    color=SURFACE,
                    anchor="middle",
                )
            )
    attention = [
        t for tasks in waves.values() for t in tasks if phase(t) == "attention"
    ]
    kv_heads = model["num_key_value_heads"]
    chunks = model["attention_chunks"]
    fused = model["attention_fused_steps"]
    chunk_steps = model["attention_chunk_steps"]
    group_size = model["num_attention_heads"] // kv_heads
    svg.add(
        svg.text(
            left,
            attention_top,
            f"Attention: one task per K/V head ({group_size} query heads each) and context chunk",
            size=12,
            weight=600,
        )
    )
    svg.add(
        svg.text(
            left,
            attention_top + 15,
            "Each task rotates its Q and K rows (RoPE) and the chunk-0 task writes the "
            "K cache row; V is written by QKV",
            size=10.5,
            color=MUTED,
        )
    )
    cell, cell_gap = 34, 4
    for panel, (title, active) in enumerate(
        (
            (f"context ≤ {fused}: {kv_heads} fused tasks, barrier skipped", 1),
            (
                f"context > {fused}: {kv_heads * chunks} tasks, scores → barrier → values",
                chunks,
            ),
        )
    ):
        x0 = left + panel * (plot / 2 + 12)
        y0 = attention_top + 44
        svg.add(svg.text(x0, y0 - 6, title, size=11, color=TEXT))
        for chunk in range(chunks):
            for head in range(kv_heads):
                on = chunk < active
                worker = next(
                    (
                        t["worker"]
                        for t in attention
                        if t["opcode"].startswith("attention_scores")
                        and t["operands"]["kv_head"] == head
                        and t["operands"]["chunk"] == chunk
                    ),
                    None,
                )
                cx = x0 + head * (cell + cell_gap)
                cy = y0 + 6 + chunk * (cell + cell_gap)
                svg.add(
                    f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="3" '
                    f'fill="{SERIES[1] if on else GRID}" fill-opacity="{0.85 if on else 1}">'
                    f"<title>K/V head {head}, chunk {chunk}: worker {worker}"
                    f"{'' if on else ' (idle at this length)'}</title></rect>"
                )
                if on:
                    svg.add(
                        svg.text(
                            cx + cell / 2,
                            cy + cell / 2 + 4,
                            str(worker),
                            size=10,
                            color=SURFACE,
                            anchor="middle",
                        )
                    )
        svg.add(
            svg.text(
                x0,
                y0 + 6 + chunks * (cell + cell_gap) + 14,
                "rows: chunks of ≤ "
                + (f"{fused} steps" if active == 1 else f"{chunk_steps:,} steps")
                + "; columns: K/V heads; numbers are workers",
                size=10.5,
                color=MUTED,
            )
        )
    svg.write(out / "s6-layer-tiles.svg")


def waves_figure(run: Path, out: Path) -> None:
    plan = load(run / "package-4k/plan.json")
    waves = layer_tasks(plan, ZOOM_LAYER)
    fused = plan["model"]["attention_fused_steps"]
    text = {
        "qkv": ("QKV", "norm, [Q; K; V]", "V → cache"),
        "attention_scores": ("scores", "RoPE, K → cache", "q·K, softmax"),
        "attention_values": ("values", "softmax · V", "chunk partials"),
        "wo": ("Wo", "merge, Wo rows", "+ residual"),
        "ffn": ("FFN", "norm, gate/up", "down partial"),
        "ffn_reduce": ("FFN reduce", "sum partials", "+ residual"),
    }
    svg = Svg(
        316,
        "One layer as waves and barriers",
        "A wave depends only on earlier waves and ends at a grid barrier; "
        "the plan fixes both",
    )
    svg.add(ARROW)
    count = len(waves)
    width, gap, top, height = 100, (WIDTH - 48 - count * 100) / (count - 1), 110, 92
    x = 24
    for i, (wave, tasks) in enumerate(waves.items()):
        key = tasks[0]["opcode"].rsplit("_", 1)[0].replace("output_projection", "wo")
        title, line1, line2 = text[key]
        group_name = group(tasks[0])
        stroke = GROUPS[group_name]
        fill = (
            FILL[0]
            if group_name == "weight streaming"
            else FILL[1]
            if group_name == "attention"
            else FILL[2]
        )
        box(
            svg,
            x,
            top,
            width,
            height,
            [title, f"{len(tasks)} tasks", line1, line2],
            fill=fill,
            stroke=stroke,
        )
        meta = plan["waves"][wave]
        if i < count - 1:
            bx = x + width + gap / 2
            conditional = "conditional_barrier" in meta
            svg.add(
                f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{top - 14}" y2="{top + height + 14}" '
                f'stroke="{TEXT}" stroke-width="2"'
                + (' stroke-dasharray="4 3"' if conditional else "")
                + "/>"
            )
            if conditional:
                svg.add(
                    svg.text(
                        bx,
                        top + height + 32,
                        f"skipped while context ≤ {fused}",
                        size=10.5,
                        color=MUTED,
                        anchor="middle",
                    )
                )
                svg.add(
                    svg.text(
                        bx,
                        top + height + 46,
                        "(scores and values then fuse)",
                        size=10.5,
                        color=MUTED,
                        anchor="middle",
                    )
                )
        x += width + gap
    svg.add(
        svg.text(24, top - 24, "black lines: grid barriers", size=10.5, color=MUTED)
    )
    for i, line in enumerate(
        (
            f"A token is {len(plan['waves'])} waves: embedding, "
            f"{plan['model']['layers']} × {count} layer waves, final norm and lm_head.",
            f"All {plan['target']['worker_count']} workers pass every barrier; a worker "
            "without a task in a wave waits there.",
        )
    ):
        svg.add(svg.text(24, top + height + 78 + 15 * i, line, size=11, color=MUTED))
    svg.write(out / "s6-waves.svg")


def assignment(run: Path, out: Path, position: int) -> None:
    plan = load(run / "package-4k/plan.json")
    model = plan["model"]
    workers = plan["target"]["worker_count"]
    waves = layer_tasks(plan, ZOOM_LAYER)
    tail = {}
    for task in plan["tasks"]:
        if task["layer"] is None and phase(task) == "lm_head":
            tail.setdefault(task["wave"], []).append(task)
    columns = [*waves.items(), *tail.items()]
    left, top, row = 52, 120, 14
    cell_width = (WIDTH - left - 24) / len(columns)
    svg = Svg(
        top + workers * row + 74,
        f"The static assignment, context {position + 1:,}",
        f"Streamed weight or K/V bytes per worker in layer {ZOOM_LAYER}'s waves and "
        "lm_head; bar length is relative to the wave's busiest worker",
    )
    for c, (_, tasks) in enumerate(columns):
        loads = dict.fromkeys(range(workers), 0)
        for t in tasks:
            loads[t["worker"]] += streamed_bytes(t, model, position)
        busiest = max(loads.values())
        busy = [v for v in loads.values() if v]
        x0 = left + c * cell_width
        name = phase(tasks[0])
        label = LABELS[name]
        if tasks[0]["opcode"].startswith("attention"):
            label = "scores" if "scores" in tasks[0]["opcode"] else "values"
        svg.add(
            svg.text(
                x0 + cell_width / 2,
                top - 28,
                label,
                size=11,
                anchor="middle",
                weight=600,
            )
        )
        balance = (
            f"max/mean {busiest / statistics.mean(busy):.2f}" if busy else "no bytes"
        )
        svg.add(
            svg.text(
                x0 + cell_width / 2,
                top - 14,
                balance,
                size=10,
                color=MUTED,
                anchor="middle",
            )
        )
        color = GROUPS[group(tasks[0])]
        for w in range(workers):
            y = top + w * row
            svg.add(
                f'<rect x="{x0 + 4:.1f}" y="{y + 1}" width="{cell_width - 8:.1f}" height="{row - 2}" '
                f'fill="{GRID}" fill-opacity="0.6"/>'
            )
            if loads[w]:
                svg.add(
                    f'<rect x="{x0 + 4:.1f}" y="{y + 1}" width="{(cell_width - 8) * loads[w] / busiest:.1f}" '
                    f'height="{row - 2}" fill="{color}"><title>worker {w}: {mb(loads[w])}</title></rect>'
                )
    for w in range(0, workers, 4):
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
    svg.add(
        svg.text(
            left,
            top + workers * row + 24,
            "Grey: the worker has no task in that wave and waits at the barrier. "
            "The planner balances rows, not time,",
            size=11,
            color=MUTED,
        )
    )
    svg.add(
        svg.text(
            left,
            top + workers * row + 40,
            "so the trace, not this chart, shows where workers wait.",
            size=11,
            color=MUTED,
        )
    )
    svg.write(out / f"s6-assignment-ctx{position + 1}.svg")


def token_loop(run: Path, out: Path) -> None:
    plan = load(run / "package-4k/plan.json")
    workers = plan["target"]["worker_count"]
    logits = [t for t in plan["tasks"] if phase(t) == "lm_head"]
    svg = Svg(
        360,
        "The token loop inside one launch",
        "Up to 64 steps per cooperative launch; greedy selection runs in the worker, "
        "so the host only launches and reads the result",
    )
    svg.add(ARROW)
    top = 96
    svg.add(
        f'<rect x="24" y="{top}" width="{WIDTH - 48}" height="196" rx="8" fill="none" '
        f'stroke="{NEUTRAL}" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    svg.add(
        svg.text(
            36, top + 20, "one launch: steps 1 … 64", size=11.5, color=MUTED, weight=600
        )
    )
    y = top + 44
    steps = [
        (["every worker", f"combines all {len(logits)}"], FILL[1], SERIES[1]),
        (["embedding", "token → x"], FILL[2], SERIES[2]),
        ([f"{plan['model']['layers']} layers", "6 waves each"], FILL[0], SERIES[0]),
        (["final norm", "+ lm_head"], FILL[0], SERIES[0]),
        ([f"{len(logits)} candidates", "max logit, lowest ID"], FILL[1], SERIES[1]),
    ]
    width, gap = 118, 22
    x = 40
    for i, (lines, fill, stroke) in enumerate(steps):
        box(svg, x, y, width, 62, lines, fill=fill, stroke=stroke)
        if i < len(steps) - 1:
            arrow(svg, x + width + 1, y + 31, x + width + gap - 1, y + 31)
        x += width + gap
    loop_y = y + 92
    svg.add(
        f'<path d="M{x - gap - width / 2:.1f},{y + 62} V{loop_y} H{40 + width / 2} V{y + 64}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#ah)"/>'
    )
    svg.add(
        svg.text(
            (40 + x - gap) / 2,
            loop_y + 18,
            "grid barrier → next step; worker 0 records the token, the count "
            "and the EOS flag",
            size=11,
            color=MUTED,
            anchor="middle",
        )
    )
    svg.add(
        svg.text(
            24,
            top + 226,
            f"Each logits task writes its rows' best candidate to its own slot ({workers} "
            "slots, 8 bytes each). All workers read the same slots, so they",
            size=11,
            color=MUTED,
        )
    )
    svg.add(
        svg.text(
            24,
            top + 242,
            "agree on the token and on stopping without a counter. EOS ends the loop; "
            "a non-finite logit sets the error word.",
            size=11,
            color=MUTED,
        )
    )
    svg.add(
        svg.text(
            24,
            top + 258,
            "64 steps, not the whole request: the display's kernel watchdog would kill a "
            "launch of tens of seconds.",
            size=11,
            color=MUTED,
        )
    )
    svg.write(out / "s6-token-loop.svg")


def hardware(run: Path, out: Path) -> None:
    target = load(run / "target.json")
    executor = target["executor"]
    rowdot = load(run / "rowdot.json")
    plan = load(run / "package-4k/plan.json")
    model = plan["model"]
    gbps = roofline(run)
    rows = [r for r in rowdot["rowdot"] if r.get("tile") == 0]
    barrier = next(r["us"] for r in rowdot["rowdot"] if r.get("shape") == "barrier")
    counts = sorted({r["active"] for r in rows})
    means = {
        n: statistics.mean(r["gbps"] for r in rows if r["active"] == n) for n in counts
    }
    layer_bytes = (
        (model["num_attention_heads"] + 2 * model["num_key_value_heads"])
        * model["head_dim"]
        * model["hidden_size"]
        + model["hidden_size"] ** 2
        + 3 * model["hidden_size"] * model["intermediate_size"]
    ) * 2
    svg = Svg(
        420,
        "What the RTX 3050 asks of the design",
        f"{target['device']['name']}, {target['device']['sm_count']} SMs, "
        f"{rowdot['l2_bytes'] / 2**20:.1f} MB L2, compute capability "
        f"{'.'.join(map(str, target['device']['compute_capability']))}",
    )
    left, right, top, bottom = 56, 420, 92, 330

    def x(n: float) -> float:
        return left + (n - 1) / 19 * (right - left)

    def y(value: float) -> float:
        return bottom - value / 240 * (bottom - top)

    svg.add(
        svg.text(
            left - 32,
            top - 14,
            "GEMV bandwidth against active workers",
            size=12,
            weight=600,
        )
    )
    for tick in range(0, 241, 60):
        svg.add(
            f'<line x1="{left}" x2="{right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="{GRID}"/>'
        )
        svg.add(
            svg.text(
                left - 8, y(tick) + 4, str(tick), size=10.5, color=MUTED, anchor="end"
            )
        )
    for n in (1, 4, 8, 12, 16, 20):
        svg.add(
            svg.text(x(n), bottom + 16, str(n), size=10.5, color=MUTED, anchor="middle")
        )
    svg.add(
        svg.text(
            (left + right) / 2,
            bottom + 34,
            "workers, one per SM (GB/s)",
            size=10.5,
            color=MUTED,
            anchor="middle",
        )
    )
    svg.add(
        f'<line x1="{left}" x2="{right}" y1="{y(gbps):.1f}" y2="{y(gbps):.1f}" stroke="{NEUTRAL}" stroke-width="2" stroke-dasharray="2 4"/>'
    )
    svg.add(
        svg.text(
            right,
            y(gbps) - 6,
            f"stream roofline {gbps:.0f}",
            size=10.5,
            color=MUTED,
            anchor="end",
        )
    )
    points = " ".join(f"{x(n):.1f},{y(means[n]):.1f}" for n in counts)
    svg.add(
        f'<polyline points="{points}" fill="none" stroke="{SERIES[0]}" stroke-width="2.5"/>'
    )
    for n in counts:
        svg.add(
            f'<circle cx="{x(n):.1f}" cy="{y(means[n]):.1f}" r="3" fill="{SERIES[0]}"><title>{n} workers: {means[n]:.0f} GB/s</title></circle>'
        )
    if 8 in means:
        svg.add(
            svg.text(
                x(8) + 6, y(means[8]) + 22, f"8 workers: {means[8]:.0f} GB/s", size=11
            )
        )
    facts = [
        (
            "Bandwidth, not compute",
            f"{gbps:.0f} GB/s floors a token at {WEIGHT_BYTES / gbps / 1e6:.1f} ms;",
        ),
        ("", f"8 SMs already stream {means.get(8, 0):.0f} GB/s"),
        (
            "One block per SM",
            f"{executor['threads_per_block']} threads, {executor['registers_per_thread']} registers,",
        ),
        (
            "",
            f"{executor['static_shared_bytes'] / 1e3:.1f} KB static shared memory (probed)",
        ),
        ("Cheap grid barriers", f"{barrier:.2f} µs each (cooperative launch);"),
        ("", f"{len(plan['waves'])} per token"),
        (
            "No cache to hide in",
            f"{rowdot['l2_bytes'] / 2**20:.1f} MB L2 against {layer_bytes / 1e6:.0f} MB",
        ),
        ("", "of weights per layer"),
        ("No TMA, no PDL", "plain loads; a shared-memory prefetch"),
        ("", "ring was probed (S3): ≤ 0.1 ms, not built"),
        ("A display watchdog", "kills long kernels: ≤ 64 tokens"),
        ("", "per launch"),
    ]
    fx = right + 40
    fy = top
    for title, line in facts:
        if title:
            fy += 10
            svg.add(svg.text(fx, fy, title, size=12, weight=600))
            fy += 15
        svg.add(svg.text(fx, fy, line, size=11, color=MUTED))
        fy += 14
    svg.write(out / "s6-rtx3050.svg")


def prior_art(run: Path, out: Path) -> None:
    plan = load(run / "package-4k/plan.json")
    opcodes = len({t["opcode"] for t in plan["tasks"]})
    waves = len(plan["waves"])
    rows = [
        (
            "",
            "Hazy Research, No Bubbles",
            "Mirage Persistent Kernel",
            "TileInfer (this repo)",
        ),
        ("GPU", "H100, B200", "—", "RTX 3050 8 GB (sm_86)"),
        (
            "Schedule",
            "instruction sequence per SM",
            "compiled from the model",
            "offline Python plan, validated",
        ),
        (
            "Dependencies",
            "counters in global memory",
            "event-driven task graph",
            f"grid barriers between {waves} waves",
        ),
        (
            "Overlap",
            "pages: next weights load early",
            "—",
            "none; prefetch probed, not built",
        ),
        (
            "Granularity",
            "7 fused instruction kinds",
            "—",
            f"{len(plan['tasks']):,} row tiles, {opcodes} opcodes",
        ),
        (
            "Hardware",
            "TMA, large shared memory",
            "—",
            "plain loads, 1 block × 1024 per SM",
        ),
        (
            "Bottleneck",
            "µs overheads vs a 0.3–0.7 ms floor",
            "—",
            "DRAM rate through every barrier",
        ),
    ]
    left, top, row_height = 24, 84, 30
    widths = [104, 200, 176, 232]
    svg = Svg(
        top + len(rows) * row_height + 40,
        "Megakernels for batch-1 decode: three designs",
        "Same idea, one launch per forward pass; the bottleneck moves with the GPU",
    )
    for r, cells in enumerate(rows):
        y = top + r * row_height
        if r % 2 == 1:
            svg.add(
                f'<rect x="{left}" y="{y - 19}" width="{sum(widths)}" height="{row_height}" fill="{GRID}" fill-opacity="0.45"/>'
            )
        x = left
        for c, cell in enumerate(cells):
            header = r == 0 or c == 0
            svg.add(
                svg.text(
                    x + 8,
                    y,
                    cell,
                    size=11.5 if header else 11,
                    color=TEXT if header or c == 3 else MUTED,
                    weight=600 if header else 400,
                )
            )
            x += widths[c]
    svg.add(
        svg.text(
            left,
            top + len(rows) * row_height + 10,
            "Hazy and Mirage cells summarize their public write-ups (— : not compared here); none of their numbers is measured here.",
            size=10.5,
            color=MUTED,
        )
    )
    svg.write(out / "s6-prior-art.svg")


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pipeline(run, out)
    layer_tiles(run, out)
    waves_figure(run, out)
    token_loop(run, out)
    hardware(run, out)
    prior_art(run, out)
    for position in (0, 4095):
        assignment(run, out, position)
    if list(run.glob("decode-*/report.json")):
        comparison(run, out)
    for position in (0, 4095):
        if (run / f"timeline/trace-p{position}.json").exists():
            timeline(run, out, position)
        if (run / f"kernels/kernels-tileinfer-p{position}.json").exists():
            kernels(run, out, position)


if __name__ == "__main__":
    main()
