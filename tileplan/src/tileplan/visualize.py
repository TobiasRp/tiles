"""Render decoder schedules and tensor operations to self-contained SVGs."""

from html import escape
from pathlib import Path

from .decoder_plan import read_package
from .tile_flow import render_tile_flow_svg

COLORS = {
    "rms_norm_f32": "#dbeafe",
    "gemv_f32": "#dcfce7",
    "rope_query_f32": "#fef3c7",
    "rope_cache_f32": "#fde68a",
    "attention_prefix_f32": "#fce7f3",
    "silu_product_f32": "#e9d5ff",
}

COLORS.update(
    {key.replace("_f32", "_bf16"): value for key, value in list(COLORS.items())}
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
    stack = plan["header"]["model_kind"] in (
        "llama_decoder_stack_f32",
        "llama_decoder_stack",
    )
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
        f"context={model['max_context']} · tile rows={model['tile_rows']}</text>",
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
            output = ", ".join(
                f"{plan['model']['tensors'][part['tensor']]['name']}"
                + (f" {part['rows']}" if "rows" in part else f" p/head {part['head']}")
                for part in task["output"]
            )
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


def render_tensor_graph_svg(plan: dict, *, plan_sha256: str) -> str:
    """Draw the semantic decoder DAG, independent of its row-tile schedule."""
    model = plan["model"]
    stack = plan["header"]["model_kind"] in (
        "llama_decoder_stack_f32",
        "llama_decoder_stack",
    )
    layers = model["layers"]
    d = model["hidden_size"]
    f = model["intermediate_size"]
    vocab = model.get("vocab_size")
    tensors = {item["name"]: item for item in model["tensors"]}

    def shape(name: str) -> str:
        return "[" + ", ".join(str(size) for size in tensors[name]["shape"]) + "]"

    nodes: list[dict] = []

    def add(
        key: str, row: int, column: int, title: str, output: str, detail: str, kind: str
    ) -> None:
        nodes.append(
            {
                "key": key,
                "row": row,
                "column": column,
                "title": title,
                "output": output,
                "detail": detail,
                "kind": kind,
            }
        )

    if stack:
        add("token", 0, 1, "Token ID", "token: scalar", f"vocabulary: {vocab}", "io")
        add(
            "embedding",
            1,
            1,
            "Embedding lookup",
            f"x {shape('x')}",
            f"embedding: [{vocab}, {d}]",
            "io",
        )
        start = 3
    else:
        add("input", 0, 1, "Hidden input", f"x {shape('x')}", "one token", "io")
        start = 2

    add(
        "block_input",
        start - 1,
        1,
        "Layer input",
        f"x {shape('x')}",
        "current layer hidden",
        "io",
    )

    cache_shape = (
        f"[{layers}, {model['max_context']}, "
        f"{model['num_key_value_heads']}, {model['head_dim']}]"
    )
    add(
        "attn_norm",
        start,
        1,
        "Attention RMSNorm",
        f"u {shape('u')}",
        f"scale: {shape('attn_norm')}",
        "norm",
    )
    for key, column, title, output, weight in (
        ("q", 0, "Q projection", "q", "Wq"),
        ("k", 1, "K projection", "k", "Wk"),
        ("v", 2, "V projection", "v", "Wv"),
    ):
        add(
            key,
            start + 1,
            column,
            title,
            f"{output} {shape(output)}",
            f"{weight}: {shape(weight)}",
            "gemv",
        )
    add(
        "q_rope",
        start + 2,
        0,
        "Query RoPE",
        f"q_rot {shape('q_rot')}",
        f"{model['num_attention_heads']} query heads",
        "rope",
    )
    add(
        "kv_cache",
        start + 2,
        1,
        "Key RoPE + KV write",
        f"K, V {cache_shape}",
        "write current position",
        "rope",
    )
    add(
        "attention",
        start + 3,
        1,
        "Grouped-query attention",
        f"attention {shape('attention')}",
        f"{model['num_attention_heads']} Q / {model['num_key_value_heads']} KV heads",
        "attention",
    )
    add(
        "attn_out",
        start + 4,
        1,
        "Output projection + x",
        f"r {shape('r')}",
        f"Wo: {shape('Wo')}",
        "residual",
    )
    add(
        "ffn_norm",
        start + 5,
        1,
        "FFN RMSNorm",
        f"z {shape('z')}",
        f"scale: {shape('ffn_norm')}",
        "norm",
    )
    add(
        "gate",
        start + 6,
        0,
        "Gate projection",
        f"gate {shape('gate')}",
        f"Wgate: {shape('Wgate')}",
        "gemv",
    )
    add(
        "up",
        start + 6,
        2,
        "Up projection",
        f"up {shape('up')}",
        f"Wup: {shape('Wup')}",
        "gemv",
    )
    add(
        "product",
        start + 7,
        1,
        "SiLU(gate) × up",
        f"product {shape('product')}",
        "elementwise",
        "activation",
    )
    add(
        "down",
        start + 8,
        1,
        "Down projection + r",
        f"y {shape('y')}",
        f"Wdown: {shape('Wdown')}",
        "residual",
    )
    if stack:
        add(
            "final_norm",
            start + 9,
            1,
            "Final RMSNorm",
            f"hidden [{d}]",
            f"scale: [{d}]",
            "norm",
        )
        add(
            "logits",
            start + 10,
            1,
            "Vocabulary projection",
            f"logits [{vocab}]",
            f"lm_head: [{vocab}, {d}]"
            + (" (tied)" if model["tie_word_embeddings"] else ""),
            "io",
        )

    edges = [
        ("embedding" if stack else "input", "block_input"),
        ("block_input", "attn_norm"),
        ("attn_norm", "q"),
        ("attn_norm", "k"),
        ("attn_norm", "v"),
        ("q", "q_rope"),
        ("k", "kv_cache"),
        ("v", "kv_cache"),
        ("q_rope", "attention"),
        ("kv_cache", "attention"),
        ("attention", "attn_out"),
        ("attn_out", "ffn_norm"),
        ("ffn_norm", "gate"),
        ("ffn_norm", "up"),
        ("gate", "product"),
        ("up", "product"),
        ("product", "down"),
    ]
    if stack:
        edges.extend(
            [("token", "embedding"), ("down", "final_norm"), ("final_norm", "logits")]
        )

    width, card_width, card_height = 1040, 270, 78
    centers = (185, 520, 855)
    top, row_gap = 154, 126
    positions = {
        node["key"]: (centers[node["column"]], top + node["row"] * row_gap)
        for node in nodes
    }
    height = max(y for _, y in positions.values()) + card_height + 70
    block_top = positions["block_input"][1] - 38
    block_bottom = positions["down"][1] + card_height + 27
    fill = {
        "io": "#dbeafe",
        "norm": "#ede9fe",
        "gemv": "#dcfce7",
        "rope": "#fef3c7",
        "attention": "#fce7f3",
        "residual": "#fee2e2",
        "activation": "#e9d5ff",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Decoder tensor operation graph</title>',
        '<desc id="description">Tensor operations and shapes for one token. '
        f"The decoder block repeats {layers} time(s).</desc>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="40" y="48" font-family="sans-serif" font-size="28" '
        'font-weight="700" fill="#172338">Tensor operation graph</text>',
        f'<text x="40" y="79" font-family="sans-serif" font-size="15" '
        f'fill="#475569">{escape(str(layers))} decoder layer(s) · '
        f"d={d} · f={f} · max context={model['max_context']} · "
        f"weights={escape(model.get('weight_dtype', 'float32'))}</text>",
        f'<text x="40" y="104" font-family="sans-serif" font-size="12" '
        f'fill="#64748b">Plan {escape(plan_sha256[:16])}… · '
        "Shapes are for batch one and one decoded token.</text>",
        f'<rect x="24" y="{block_top}" width="{width - 48}" '
        f'height="{block_bottom - block_top}" rx="16" fill="#ffffff" '
        'stroke="#94a3b8" stroke-width="2"/>',
        f'<text x="45" y="{block_top + 25}" font-family="sans-serif" '
        'font-size="17" font-weight="700" fill="#334155">'
        f"Decoder block × {layers}</text>",
    ]
    if stack and layers > 1:
        svg.append(
            f'<text x="750" y="{block_top + 25}" font-family="sans-serif" '
            'font-size="12" fill="#475569">y feeds the next layer’s x</text>'
        )

    for source, target in edges:
        sx, sy = positions[source]
        tx, ty = positions[target]
        sy += card_height
        svg.append(
            f'<path d="M {sx} {sy} C {sx} {sy + 42}, {tx} {ty - 42}, {tx} {ty - 8}" '
            'fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    entry_x, entry_y = positions["block_input"]
    out_x, out_y = positions["attn_out"]
    down_x, down_y = positions["down"]
    svg.extend(
        [
            f'<path d="M {entry_x - card_width // 2} {entry_y + 40} H 18 '
            f'V {out_y + 40} H {out_x - card_width // 2 - 8}" '
            'fill="none" stroke="#e11d48" stroke-width="2" '
            'stroke-dasharray="6 5" marker-end="url(#arrow)"/>',
            f'<text x="32" y="{out_y + 30}" font-family="sans-serif" '
            f'font-size="12" fill="#be123c">skip x [{d}]</text>',
            f'<path d="M {out_x + card_width // 2} {out_y + 40} H 1022 '
            f'V {down_y + 40} H {down_x + card_width // 2 + 8}" '
            'fill="none" stroke="#e11d48" stroke-width="2" '
            'stroke-dasharray="6 5" marker-end="url(#arrow)"/>',
            f'<text x="899" y="{down_y + 30}" font-family="sans-serif" '
            f'font-size="12" fill="#be123c">skip r [{d}]</text>',
        ]
    )

    for node in nodes:
        cx, y = positions[node["key"]]
        x = cx - card_width // 2
        svg.extend(
            [
                f"<g><title>{escape(node['title'])}: {escape(node['output'])}; "
                f"{escape(node['detail'])}</title>",
                f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" '
                f'rx="10" fill="{fill[node["kind"]]}" stroke="#94a3b8"/>',
                f'<text x="{x + 12}" y="{y + 24}" font-family="sans-serif" '
                f'font-size="16" font-weight="700" fill="#172338">'
                f"{escape(node['title'])}</text>",
                f'<text x="{x + 12}" y="{y + 45}" font-family="sans-serif" '
                f'font-size="13" fill="#334155">{escape(node["output"])}</text>',
                f'<text x="{x + 12}" y="{y + 64}" font-family="sans-serif" '
                f'font-size="12" fill="#475569">{escape(node["detail"])}</text></g>',
            ]
        )
    svg.extend(
        [
            f'<text x="40" y="{height - 25}" font-family="sans-serif" '
            'font-size="12" fill="#64748b">Arrows carry the labeled tensors; '
            "dashed arrows are residual connections. The cache stores the visible prefix.</text>",
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def visualize(
    package: Path,
    out: Path,
    *,
    note: str = "",
    tensor_graph_out: Path | None = None,
    tile_flow_out: Path | None = None,
) -> Path:
    plan, _, digest = read_package(package)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_schedule_svg(plan, plan_sha256=digest, note=note))
    if tensor_graph_out is not None:
        tensor_graph_out.parent.mkdir(parents=True, exist_ok=True)
        tensor_graph_out.write_text(render_tensor_graph_svg(plan, plan_sha256=digest))
    if tile_flow_out is not None:
        tile_flow_out.parent.mkdir(parents=True, exist_ok=True)
        tile_flow_out.write_text(render_tile_flow_svg(plan, plan_sha256=digest))
    return out
