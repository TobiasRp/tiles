"""Align, decompose, and draw per-instruction TileInfer worker timelines.

``tile_runtime_test --timeline-stack`` records, for every encoded worker
instruction, its start and end SM cycle and the coarse ``%globaltimer`` value
at its start. Cycle counters are per SM, so each worker's cycles are mapped to
nanoseconds by a least-squares fit against its own global-timer samples.

The bubble budget partitions the traced kernel into waves, delimited by the
first worker leaving each grid barrier. A conditional barrier that the
position skips does not delimit a wave; its two plan waves count as one, as
the fused attention tasks run them. A wave's duration is the time an
ideal stream would need for its unique bytes at the measured roofline
bandwidth, plus its excess work until the last worker finishes, plus the
barrier latency after that. Grouped by decoder phase, ideal streaming,
excess, and barrier latency sum to the traced kernel time. As a diagnostic,
each wave's excess is further split at the average worker's finish time;
the part after it is the wave tail, when some workers are already idle.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from html import escape

PHASES = (
    "embedding",
    "qkv",
    "attention",
    "wo",
    "gate_up",
    "down",
    "ffn",
    "ffn_reduce",
    "final_norm",
    "lm_head",
)
COLORS = {
    "embedding": "#94a3b8",
    "qkv": "#22c55e",
    "attention": "#f472b6",
    "wo": "#16a34a",
    "gate_up": "#15803d",
    "down": "#166534",
    "ffn": "#15803d",
    "ffn_reduce": "#166534",
    "final_norm": "#1d4ed8",
    "lm_head": "#0f766e",
    "barrier": "#e2e8f0",
}
# Phases whose time is dominated by streaming weights or the K/V cache.
# gate_up and down are the FFN phases of format-8 (S2c) packages. The FFN
# reduction streams nothing, but it finishes the down projection that S2c's
# down wave held, so its excess counts here too.
STREAMING = ("qkv", "wo", "gate_up", "down", "ffn", "ffn_reduce", "lm_head")


@dataclass(frozen=True)
class Event:
    """One executed instruction on the common nanosecond axis."""

    worker: int
    task: int  # plan task ID, or -1 for a barrier
    start: float
    end: float


@dataclass(frozen=True)
class Wave:
    """One wave's interval, split as described in the module docstring."""

    wave: int
    phase: str
    bytes: int
    duration_ns: float
    ideal_ns: float
    work_ns: float
    tail_ns: float
    sync_ns: float
    busy_workers: int


def phase(task: dict) -> str:
    """Name the decoder phase a plan task belongs to."""
    return {
        "qkv": "qkv",
        "attention_scores": "attention",
        "attention_values": "attention",
        "output_projection": "wo",
        "gate_up": "gate_up",
        "down": "down",
        "ffn": "ffn",
        "ffn_reduce": "ffn_reduce",
        "embedding": "embedding",
        "final_rms_norm": "final_norm",
        "logits": "lm_head",
    }[task["opcode"].rsplit("_", 1)[0]]


def active_chunks(model: dict, position: int) -> int:
    """Attention chunk tasks with work at ``position`` (ops::active_chunks)."""
    length = position + 1
    if length <= model["attention_fused_steps"]:
        return 1
    return max(2, -(-length // model["attention_chunk_steps"]))


def executed_barriers(plan: dict, position: int) -> list[bool]:
    """Whether each barrier between consecutive waves synchronizes at ``position``."""
    return [
        position + 1 > wave.get("conditional_barrier", {}).get("skip_up_to_length", 0)
        for wave in plan["waves"][:-1]
    ]


def streamed_bytes(task: dict, model: dict, position: int) -> int:
    """Weight or K/V bytes a task reads; activations are omitted."""
    element = 2 if model["weight_dtype"] == "bfloat16" else 4
    opcode = task["opcode"].rsplit("_", 1)[0]
    d, f = model["hidden_size"], model["intermediate_size"]
    if opcode.startswith("attention"):
        # Split chunks: scores read the chunk's K rows of one K/V head, values
        # its V rows. One fused chunk: the scores task reads both.
        length, chunks = position + 1, active_chunks(model, position)
        chunk = task["operands"]["chunk"]
        if chunk >= chunks or (chunks == 1 and opcode == "attention_values"):
            return 0
        steps = length * (chunk + 1) // chunks - length * chunk // chunks
        return (2 if chunks == 1 else 1) * steps * model["head_dim"] * element
    begin, end = task["operands"]["rows"]
    width = {"embedding": 1, "gate_up": 2 * d, "down": f, "ffn": 3 * d}.get(opcode, d)
    if opcode in ("final_rms_norm", "ffn_reduce"):
        return 0
    return (end - begin) * width * element


def align(trace: dict, sample: int = 0) -> tuple[list[Event], dict]:
    """Map one traced launch onto nanoseconds relative to its first instruction."""
    record = trace["samples"][sample]
    workers = trace["instruction_workers"]
    tasks = trace["instruction_tasks"]
    by_worker: dict[int, list[int]] = defaultdict(list)
    for index, worker in enumerate(workers):
        by_worker[worker].append(index)
    fits = {}
    for worker, indices in by_worker.items():
        cycles = [
            record["start_cycle"][i] - record["start_cycle"][indices[0]]
            for i in indices
        ]
        nanoseconds = [
            record["start_ns"][i] - record["start_ns"][indices[0]] for i in indices
        ]
        count = len(indices)
        mean_c = sum(cycles) / count
        mean_n = sum(nanoseconds) / count
        spread = sum((c - mean_c) ** 2 for c in cycles)
        slope = (
            sum(
                (c - mean_c) * (n - mean_n)
                for c, n in zip(cycles, nanoseconds, strict=True)
            )
            / spread
            if spread
            else 0.0
        )
        origin = record["start_ns"][indices[0]] + mean_n - slope * mean_c
        fits[worker] = (record["start_cycle"][indices[0]], origin, slope)
    events = []
    for index, (worker, task) in enumerate(zip(workers, tasks, strict=True)):
        first_cycle, origin, slope = fits[worker]
        start = origin + slope * (record["start_cycle"][index] - first_cycle)
        end = origin + slope * (record["end_cycle"][index] - first_cycle)
        events.append(Event(worker, task, start, end))
    base = min(event.start for event in events)
    events = [Event(e.worker, e.task, e.start - base, e.end - base) for e in events]
    slopes = [fit[2] for fit in fits.values()]
    info = {
        "kernel_us": record["kernel_us"],
        "traced_us": max(e.end for e in events) / 1e3,
        "sm_clock_ghz": [round(1 / s, 4) for s in (min(slopes), max(slopes))],
    }
    return events, info


def budget(
    plan: dict, events: list[Event], position: int, bandwidth_gbps: float
) -> dict:
    """Decompose the traced kernel time into ideal streaming and named losses."""
    tasks = plan["tasks"]
    model = plan["model"]
    wave_count = len(plan["waves"])
    workers = plan["target"]["worker_count"]
    barriers: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        if event.task < 0:
            barriers[event.worker].append(event)
    # Plan waves between two executed barriers form one traced wave.
    executed = executed_barriers(plan, position)
    first_waves, segment_of, releases = [0], [], []
    for wave in range(wave_count):
        segment_of.append(len(first_waves) - 1)
        if wave + 1 < wave_count and executed[wave]:
            group = [barriers[worker][wave] for worker in range(workers)]
            releases.append(min(e.end for e in group))
            first_waves.append(wave + 1)
    kernel_end = max(e.end for e in events)
    starts = [0.0, *releases]
    ends = [*releases, kernel_end]

    by_wave: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        if event.task >= 0:
            by_wave[segment_of[tasks[event.task]["wave"]]].append(event)
    waves = []
    for wave, first_wave in enumerate(first_waves):
        members = by_wave[wave]
        names = [phase(tasks[e.task]) for e in members]
        unique = sum(streamed_bytes(tasks[e.task], model, position) for e in members)
        finish = {w: starts[wave] for w in range(workers)}
        for e in members:
            finish[e.worker] = max(finish[e.worker], e.end)
        last = max(finish.values())
        mean = sum(finish.values()) / workers
        ideal = unique / bandwidth_gbps  # bytes / (GB/s) = ns
        waves.append(
            Wave(
                wave=first_wave,
                phase=max(set(names), key=names.count),
                bytes=unique,
                duration_ns=ends[wave] - starts[wave],
                ideal_ns=ideal,
                work_ns=mean - starts[wave] - ideal,
                tail_ns=last - mean,
                sync_ns=ends[wave] - last,
                busy_workers=len({e.worker for e in members}),
            )
        )

    phases = {}
    for name in PHASES:
        group = [w for w in waves if w.phase == name]
        if not group:
            continue
        duration = sum(w.duration_ns for w in group) / 1e3
        phases[name] = {
            "waves": len(group),
            "bytes": sum(w.bytes for w in group),
            "duration_us": duration,
            "ideal_us": sum(w.ideal_ns for w in group) / 1e3,
            "work_us": sum(w.work_ns for w in group) / 1e3,
            "tail_us": sum(w.tail_ns for w in group) / 1e3,
            "sync_us": sum(w.sync_ns for w in group) / 1e3,
            "achieved_gbps": sum(w.bytes for w in group) / duration / 1e3
            if duration
            else 0.0,
        }
    total = kernel_end / 1e3
    ideal = sum(w.ideal_ns for w in waves) / 1e3
    sync = sum(w.sync_ns for w in waves) / 1e3
    for values in phases.values():
        values["excess_us"] = (
            values["duration_us"] - values["ideal_us"] - values["sync_us"]
        )
    stream_excess = sum(phases[p]["excess_us"] for p in STREAMING if p in phases)
    attention = phases["attention"]["excess_us"] if "attention" in phases else 0.0
    small = sum(
        values["excess_us"]
        for name, values in phases.items()
        if name not in STREAMING and name != "attention"
    )
    barrier_wait = sum(e.end - e.start for e in events if e.task < 0) / workers / 1e3
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "position": position,
        "traced_kernel_us": total,
        "ideal_stream_us": ideal,
        "bubble_us": total - ideal,
        "components_us": {
            "barrier_sync": sync,
            "streaming_excess": stream_excess,
            "attention_excess": attention,
            "small_phases": small,
        },
        "mean_barrier_wait_us": barrier_wait,
        "phases": phases,
        "waves": [asdict(w) for w in waves],
    }


def render_timeline_svg(
    plan: dict,
    events: list[Event],
    *,
    title: str,
    window: tuple[float, float] | None = None,
) -> str:
    """Draw one row per worker; tasks are colored by phase, barriers in gray."""
    tasks = plan["tasks"]
    workers = plan["target"]["worker_count"]
    begin, end = window or (0.0, max(e.end for e in events))
    span = max(end - begin, 1.0)
    left, top, row, width = 92, 92, 16, 1500
    plot = width - left - 24
    height = top + workers * row + 70

    def x(time: float) -> float:
        return left + (min(max(time, begin), end) - begin) / span * plot

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="20" '
        f'font-weight="700" fill="#0f172a">{escape(title)}</text>',
    ]
    for index, name in enumerate((*PHASES, "barrier")):
        lx = left + index * 112
        svg.append(
            f'<rect x="{lx}" y="50" width="12" height="12" fill="{COLORS[name]}"/>'
            f'<text x="{lx + 16}" y="61" font-family="sans-serif" font-size="11" '
            f'fill="#334155">{name}</text>'
        )
    for worker in range(workers):
        y = top + worker * row
        svg.append(
            f'<text x="{left - 8}" y="{y + 11}" text-anchor="end" font-family="sans-serif" '
            f'font-size="10" fill="#475569">worker {worker}</text>'
        )
    for event in events:
        if event.end < begin or event.start > end:
            continue
        name = "barrier" if event.task < 0 else phase(tasks[event.task])
        x0, x1 = x(event.start), x(event.end)
        y = top + event.worker * row
        svg.append(
            f'<rect x="{x0:.2f}" y="{y + 1}" width="{max(x1 - x0, 0.3):.2f}" '
            f'height="{row - 2}" fill="{COLORS[name]}"/>'
        )
    step = 10 ** max(0, len(str(int(span))) - 1)
    while span / step < 4:
        step /= 2
    tick = (begin // step + 1) * step
    axis = top + workers * row + 18
    while tick < end:
        svg.append(
            f'<line x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" y2="{axis - 12}" '
            'stroke="#cbd5e1" stroke-width="0.5"/>'
            f'<text x="{x(tick):.1f}" y="{axis}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10" fill="#475569">'
            f"{tick / 1e3:.0f} µs</text>"
        )
        tick += step
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def layer_window(plan: dict, events: list[Event], layer: int) -> tuple[float, float]:
    """Time span of one layer's tasks."""
    tasks = plan["tasks"]
    selected = [e for e in events if e.task >= 0 and tasks[e.task]["layer"] == layer]
    return min(e.start for e in selected), max(e.end for e in selected)
