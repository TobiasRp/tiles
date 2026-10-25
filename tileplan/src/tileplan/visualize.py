"""Render one decoder layer's wave schedule as a self-contained SVG."""

from html import escape
from pathlib import Path

from .package import read_package
from .plan_core import layer_view

COLORS = {
    "qkv_f32": "#dcfce7",
    "attention_scores_f32": "#fce7f3",
    "attention_values_f32": "#fbcfe8",
    "output_projection_f32": "#bbf7d0",
    "gate_up_f32": "#dbeafe",
    "down_f32": "#bfdbfe",
}

COLORS.update(
    {key.replace("_f32", "_bf16"): value for key, value in list(COLORS.items())}
)


def _describe(plan: dict, part: dict) -> str:
    """Short text for one output region of a task."""
    if "workspace" in part:
        return f"{part['workspace']} heads {part['heads']} chunk {part['chunk']}"
    name = plan["model"]["tensors"][part["tensor"]]["name"]
    return (
        f"{name} {part['rows']}" if "rows" in part else f"{name} p/head {part['head']}"
    )


def render_schedule_svg(plan: dict, *, plan_sha256: str, note: str = "") -> str:
    tasks = {task["id"]: task for task in plan["tasks"]}
    waves = plan["waves"]
    width = 1210
    top = 222
    columns = 4
    card_width = 270
    card_height = 64
    gap_x = 16
    gap_y = 12
    wave_heights = []
    for wave in waves:
        count = sum(len(ids) for ids in wave["workers"])
        wave_heights.append(
            75 + ((count + columns - 1) // columns) * (card_height + gap_y) + 10
        )
    height = top + sum(wave_heights) + 52
    model, storage = plan["model"], plan["storage"]
    stack = plan["header"]["model_kind"] == "llama_decoder_stack"
    title = (
        f"Full decoder: {model['layers']} layers"
        if stack
        else "One-token decoder block"
    )
    stage_text = (
        f"Embedding lookup → these {len(waves)} waves × {model['layers']} layers "
        "→ final RMSNorm → logits. Scratch is reused each layer."
        if stack
        else "Each task is placed in its earliest legal wave. "
        "Larger tasks go to the least-loaded worker."
    )

    def label(value: object) -> str:
        return escape(str(value), quote=True)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{label(title)} schedule</title>',
        f'<desc id="description">{label(stage_text)} '
        "Cards show logical worker assignment and estimated work.</desc>",
        '<rect width="100%" height="100%" fill="#f5f7fb"/>',
        '<text x="40" y="51" font-family="sans-serif" font-size="28" '
        f'font-weight="700" fill="#172338">{label(title)}</text>',
        f'<text x="40" y="80" font-family="sans-serif" font-size="15" fill="#475569">'
        f"d={model['hidden_size']} · f={model['intermediate_size']} · "
        f"Q/KV heads={model['num_attention_heads']}/{model['num_key_value_heads']} · "
        f"context={model['max_context']} · attention chunks={model['attention_chunks']}</text>",
        f'<text x="40" y="107" font-family="sans-serif" font-size="14" fill="#475569">'
        f"{len(plan['tasks'])} tasks · {len(waves)} waves · "
        f"{plan['target']['worker_count']} logical workers · "
        f"scratch {storage['scratch_bytes']:,} B · KV {storage['kv_bytes']:,} B · "
        f"weights {storage['weight_bytes']:,} B</text>",
        f'<text x="40" y="132" font-family="sans-serif" font-size="12" fill="#64748b">'
        f"plan {label(plan_sha256[:16])}… · "
        f"kernel {label(plan['target']['kernel_fingerprint'])}</text>",
        f'<text x="40" y="158" font-family="sans-serif" font-size="13" '
        f'fill="#64748b">{label(stage_text)}</text>',
    ]
    if note:
        svg.append(
            f'<text x="40" y="185" font-family="sans-serif" font-size="13" '
            f'font-weight="600" fill="#9a3412">{label(note)}</text>'
        )
    for wave in waves:
        count = sum(len(ids) for ids in wave["workers"])
        wave_height = wave_heights[wave["id"]]
        active = sum(bool(ids) for ids in wave["workers"])
        svg.extend(
            [
                f'<rect x="24" y="{top}" width="{width - 48}" height="{wave_height}" '
                'rx="12" fill="white" stroke="#cbd5e1"/>',
                f'<text x="40" y="{top + 30}" font-family="sans-serif" font-size="18" '
                f'font-weight="700" fill="#172338">Wave {wave["id"]}</text>',
                f'<text x="170" y="{top + 29}" font-family="sans-serif" font-size="13" '
                f'fill="#475569">{count} tasks · {active} active workers · '
                f"estimated work {sum(wave['estimated_work']):,}</text>",
            ]
        )
        entries = [
            (worker, task_id)
            for worker, ids in enumerate(wave["workers"])
            for task_id in ids
        ]
        for index, (worker, task_id) in enumerate(entries):
            task = tasks[task_id]
            x = 40 + (index % columns) * (card_width + gap_x)
            y = top + 50 + (index // columns) * (card_height + gap_y)
            color = COLORS[task["opcode"]]
            output = ", ".join(_describe(plan, part) for part in task["output"])
            svg.extend(
                [
                    f"<g><title>Task {task_id}: {label(task['opcode'])}; "
                    f"predecessors {label(task['predecessors'])}; "
                    f"estimated work {task['estimated_work']}</title>",
                    f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" '
                    f'rx="8" fill="{color}" stroke="#cbd5e1"/>',
                    f'<text x="{x + 10}" y="{y + 19}" font-family="sans-serif" '
                    f'font-size="13" font-weight="700" fill="#172338">'
                    f"T{task_id} · worker {worker} · {label(task['opcode'])}</text>",
                    f'<text x="{x + 10}" y="{y + 42}" font-family="sans-serif" '
                    f'font-size="12" fill="#334155">{label(output)}</text></g>',
                ]
            )
        top += wave_height
    svg.extend(
        [
            f'<text x="40" y="{height - 24}" font-family="sans-serif" font-size="12" '
            'fill="#64748b">Worker IDs are logical CUDA blocks. '
            "Wave boundaries are full cooperative-grid barriers.</text>",
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def visualize(
    package: Path,
    out: Path,
    *,
    note: str = "",
) -> Path:
    plan, _, digest = read_package(package)
    # A stack repeats one layer schedule; draw that layer rather than every wave.
    block = layer_view(plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_schedule_svg(block, plan_sha256=digest, note=note))
    return out
