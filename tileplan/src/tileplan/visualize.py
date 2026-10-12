"""Render a validated tile plan as a self-contained SVG schedule."""

from html import escape
from pathlib import Path

from .plan import read_package


def render_schedule_svg(plan: dict, *, plan_sha256: str, note: str = "") -> str:
    """Show waves, logical workers, task order, row ownership, and the barrier."""
    workers = plan["target"]["worker_count"]
    model = plan["model"]
    storage = plan["storage"]
    peak_bytes = sum(
        storage[name]
        for name in ("input_bytes", "weight_bytes", "scratch_bytes", "output_bytes")
    )
    tasks = {task["id"]: task for task in plan["tasks"]}
    column_width = 220
    lane_x = 215
    width = lane_x + workers * column_width + 48
    card_height = 58
    card_gap = 10
    stage_tops = [185]
    stage_heights = [
        max(172, 87 + max(map(len, wave["workers"])) * (card_height + card_gap))
        for wave in plan["waves"]
    ]
    stage_tops.append(stage_tops[0] + stage_heights[0] + 76)
    height = stage_tops[1] + stage_heights[1] + 78

    def label(value: object) -> str:
        return escape(str(value), quote=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">TilePlan MLP task schedule</title>',
        '<desc id="description">Two waves of row-tile tasks assigned to logical workers, '
        "with a full barrier between the MLP layers.</desc>",
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<text x="48" y="49" font-family="sans-serif" font-size="27" '
        'font-weight="700" fill="#172338">TilePlan schedule</text>',
        f'<text x="48" y="79" font-family="sans-serif" font-size="15" '
        f'fill="#475569">MLP float32 · D={model["D"]} · H={model["H"]} · '
        f"O={model['O']} · tiles={model['tile_h1']}/{model['tile_h2']} · "
        f"{workers} logical workers</text>",
        f'<text x="48" y="105" font-family="sans-serif" font-size="13" '
        f'fill="#64748b">Plan SHA-256: {label(plan_sha256[:16])}… · '
        f"Kernel: {label(plan['target']['kernel_fingerprint'])}</text>",
        f'<text x="48" y="128" font-family="sans-serif" font-size="13" '
        f'fill="#64748b">Buffers: input {storage["input_bytes"]:,} B · '
        f"weights {storage['weight_bytes']:,} B · scratch {storage['scratch_bytes']:,} B · "
        f"output {storage['output_bytes']:,} B · peak {peak_bytes:,} B</text>",
    ]
    if note:
        svg.append(
            f'<text x="48" y="153" font-family="sans-serif" font-size="13" '
            f'font-weight="600" fill="#9a3412">{label(note)}</text>'
        )

    for wave_index, wave in enumerate(plan["waves"]):
        top = stage_tops[wave_index]
        stage_height = stage_heights[wave_index]
        layer = "W1 × x + b1; GELU" if wave_index == 0 else "W2 × h + b2; output y"
        color = "#dbeafe" if wave_index == 0 else "#dcfce7"
        stroke = "#93c5fd" if wave_index == 0 else "#86efac"
        svg.extend(
            [
                f'<rect x="32" y="{top}" width="{width - 64}" height="{stage_height}" '
                'rx="15" fill="white" stroke="#cbd5e1"/>',
                f'<text x="49" y="{top + 34}" font-family="sans-serif" font-size="20" '
                f'font-weight="700" fill="#172338">Wave {wave_index}</text>',
                f'<text x="49" y="{top + 58}" font-family="sans-serif" font-size="13" '
                f'fill="#475569">{layer}</text>',
            ]
        )
        for worker, ids in enumerate(wave["workers"]):
            x = lane_x + worker * column_width
            svg.append(
                f'<text x="{x + 8}" y="{top + 34}" font-family="sans-serif" '
                f'font-size="15" font-weight="700" fill="#334155">Worker {worker}</text>'
            )
            svg.append(
                f'<text x="{x + 8}" y="{top + 54}" font-family="sans-serif" '
                f'font-size="12" fill="#64748b">estimated work: '
                f"{wave['estimated_work'][worker]:,}</text>"
            )
            if not ids:
                svg.append(
                    f'<text x="{x + 8}" y="{top + 111}" font-family="sans-serif" '
                    'font-size="13" fill="#94a3b8">idle this wave</text>'
                )
            for position, task_id in enumerate(ids):
                task = tasks[task_id]
                begin, end = task["rows"]
                y = top + 72 + position * (card_height + card_gap)
                work = (end - begin) * (model["D"] if wave_index == 0 else model["H"])
                svg.extend(
                    [
                        f"<g><title>Task {task_id}: {label(task['opcode'])}, "
                        f"rows [{begin}, {end}), estimated work {work}</title>",
                        f'<rect x="{x + 7}" y="{y}" width="{column_width - 17}" '
                        f'height="{card_height}" rx="9" fill="{color}" stroke="{stroke}"/>',
                        f'<text x="{x + 20}" y="{y + 24}" font-family="sans-serif" '
                        f'font-size="15" font-weight="700" fill="#172338">Task {task_id}</text>',
                        f'<text x="{x + 20}" y="{y + 44}" font-family="sans-serif" '
                        f'font-size="13" fill="#334155">rows [{begin}, {end}) · work {work:,}</text>',
                        "</g>",
                    ]
                )

    barrier_y = stage_tops[0] + stage_heights[0] + 38
    svg.extend(
        [
            f'<line x1="48" x2="{width - 48}" y1="{barrier_y}" y2="{barrier_y}" '
            'stroke="#64748b" stroke-width="2" stroke-dasharray="7 6"/>',
            f'<rect x="{width / 2 - 155}" y="{barrier_y - 14}" width="310" '
            'height="28" rx="14" fill="#f5f7fb"/>',
            f'<text x="{width / 2}" y="{barrier_y + 5}" text-anchor="middle" '
            'font-family="sans-serif" font-size="13" font-weight="700" '
            'fill="#475569">FULL GRID BARRIER · all h rows ready</text>',
            f'<text x="48" y="{height - 30}" font-family="sans-serif" font-size="12" '
            'fill="#64748b">Cards follow each worker’s task order. '
            "Workers are logical blocks; vertical position is not elapsed time.</text>",
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def visualize(package: Path, out: Path, *, note: str = "") -> Path:
    """Validate a package, then write its schedule to an SVG file."""
    plan, _, digest = read_package(package)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_schedule_svg(plan, plan_sha256=digest, note=note))
    return out
