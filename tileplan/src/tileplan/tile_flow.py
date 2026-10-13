"""Render a bounded, plan-derived example of decoder tile dependencies."""

from collections import defaultdict
from html import escape


def render_tile_flow_svg(plan: dict, *, plan_sha256: str) -> str:
    """Show one Q/K/V-to-attention and one gate/up-to-MLP tile path.

    Gray cards collapse direct prerequisites omitted from the example. They are
    derived from the selected tasks' predecessor IDs, not inferred from the
    high-level operator graph.
    """
    tasks = plan["tasks"]
    model = plan["model"]
    tensors = {item["id"]: item for item in model["tensors"]}
    ids = {item["name"]: item["id"] for item in model["tensors"]}

    def output_name(task: dict) -> str:
        return tensors[task["output"][0]["tensor"]]["name"]

    def first(opcode: str, *, weight: str | None = None) -> dict:
        for task in tasks:
            if task["opcode"] == opcode and (
                weight is None or task["operands"].get("weight") == ids[weight]
            ):
                return task
        raise ValueError(f"plan has no {opcode} task for {weight}")

    def predecessor(task: dict, name: str) -> dict:
        for task_id in task["predecessors"]:
            candidate = tasks[task_id]
            if output_name(candidate) == name:
                return candidate
        raise ValueError(f"task {task['id']} has no {name} predecessor")

    norm = first("rms_norm_f32")
    query_rope = first("rope_query_f32")
    query_head = query_rope["operands"]["head"]
    kv_head = model["head_to_kv"][query_head]
    cache = next(
        task
        for task in tasks
        if task["opcode"] == "rope_cache_f32" and task["operands"]["kv_head"] == kv_head
    )
    attention = next(
        task
        for task in tasks
        if task["opcode"] == "attention_prefix_f32"
        and task["operands"]["head"] == query_head
    )
    output_projection = first("gemv_f32", weight="Wo")
    ffn_norm = next(
        task
        for task in tasks
        if task["opcode"] == "rms_norm_f32" and task["operands"]["output"] == ids["z"]
    )
    product = first("silu_product_f32")
    down = first("gemv_f32", weight="Wdown")
    selected = [
        norm,
        predecessor(query_rope, "q"),
        predecessor(cache, "k"),
        predecessor(cache, "v"),
        query_rope,
        cache,
        attention,
        output_projection,
        ffn_norm,
        predecessor(product, "gate"),
        predecessor(product, "up"),
        product,
        down,
    ]
    shown = {task["id"] for task in selected}
    if len(shown) != len(selected):
        raise ValueError("example paths unexpectedly share a task")
    rank = {task["id"]: index for index, task in enumerate(selected)}
    output_rank = {output_name(task): index for index, task in enumerate(selected)}

    def region(task: dict) -> str:
        parts = []
        for part in task["output"]:
            name = tensors[part["tensor"]]["name"]
            if "rows" in part:
                lo, hi = part["rows"]
                parts.append(f"{name}[{lo}:{hi}]")
            else:
                parts.append(f"{name}[p, kv{part['head']}]")
        return ", ".join(parts)

    def computation(task: dict) -> str:
        opcode = task["opcode"]
        operands = task["operands"]
        if opcode == "rms_norm_f32":
            source = tensors[operands["input"]]["name"]
            scale = tensors[operands["scale"]]["name"]
            return f"RMSNorm({source}, {scale})"
        if opcode == "gemv_f32":
            weight = tensors[operands["weight"]]["name"]
            source = tensors[operands["input"]]["name"]
            addend = operands["addend"]
            suffix = f" + {tensors[addend]['name']} slice" if addend is not None else ""
            return f"{weight}[rows, :] @ {source}{suffix}"
        if opcode == "rope_query_f32":
            return f"RoPE(q head {operands['head']}, position p)"
        if opcode == "rope_cache_f32":
            return f"RoPE(k); write k/v for KV head {operands['kv_head']}"
        if opcode == "attention_prefix_f32":
            return f"Q head {operands['head']} reads KV {operands['kv_head']} at 0..p"
        if opcode == "silu_product_f32":
            return "SiLU(gate[rows]) × up[rows]"
        raise ValueError(f"unsupported example opcode {opcode}")

    titles = {
        "u": "Attention RMSNorm",
        "q": "Q projection tile",
        "k": "K projection tile",
        "v": "V projection tile",
        "q_rot": "Query RoPE",
        "K": "RoPE + KV cache write",
        "attention": "Attention head",
        "r": "Output projection tile",
        "z": "FFN RMSNorm",
        "gate": "Gate projection tile",
        "up": "Up projection tile",
        "product": "SiLU product tile",
        "y": "Down projection tile",
    }
    cards: list[dict] = []
    links: list[tuple[str, str, bool]] = []
    for task in selected:
        name = output_name(task)
        cards.append(
            {
                "key": f"t{task['id']}",
                "wave": task["wave"],
                "order": rank[task["id"]],
                "title": titles[name],
                "meta": f"T{task['id']} · worker {task['worker']}",
                "output": region(task),
                "detail": computation(task),
                "aggregate": False,
            }
        )
        omitted: dict[str, list[int]] = defaultdict(list)
        for predecessor_id in task["predecessors"]:
            if predecessor_id in shown:
                links.append((f"t{predecessor_id}", f"t{task['id']}", False))
            else:
                omitted[output_name(tasks[predecessor_id])].append(predecessor_id)
        for predecessor_name, predecessor_ids in omitted.items():
            waves = {tasks[task_id]["wave"] for task_id in predecessor_ids}
            if len(waves) != 1:
                raise ValueError("collapsed prerequisites cross waves")
            key = f"a{task['id']}-{predecessor_name}"
            count = len(predecessor_ids)
            cards.append(
                {
                    "key": key,
                    "wave": waves.pop(),
                    "order": output_rank.get(predecessor_name, len(selected)) + 0.5,
                    "title": (
                        f"+{count} other {predecessor_name} "
                        f"tile{'s' if count != 1 else ''}"
                    ),
                    "meta": "collapsed prerequisites",
                    "output": f"{predecessor_name}: remaining slices",
                    "detail": f"all required before T{task['id']}",
                    "aggregate": True,
                }
            )
            links.append((key, f"t{task['id']}", True))

    by_wave: dict[int, list[dict]] = defaultdict(list)
    for card in cards:
        by_wave[card["wave"]].append(card)
    wave_count = len(plan["waves"])
    if set(by_wave) != set(range(wave_count)):
        raise ValueError("example does not cover every decoder wave")
    for wave_cards in by_wave.values():
        wave_cards.sort(key=lambda card: (card["order"], card["key"]))

    width = 64 + wave_count * 273
    card_width, card_height, card_gap = 245, 100, 17
    top = 226
    height = (
        top
        + max(len(wave_cards) for wave_cards in by_wave.values())
        * (card_height + card_gap)
        + 98
    )
    positions = {
        card["key"]: (46 + wave * 273, top + index * (card_height + card_gap))
        for wave, wave_cards in by_wave.items()
        for index, card in enumerate(wave_cards)
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Example decoder tile dependencies</title>',
        '<desc id="description">Selected real plan tasks span all decoder waves. '
        "Gray cards summarize omitted predecessor tasks. Every wave ends at a "
        "cooperative grid barrier.</desc>",
        '<defs><marker id="tile-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="40" y="46" font-family="sans-serif" font-size="27" '
        'font-weight="700" fill="#172338">Example tile flow</text>',
        f'<text x="40" y="76" font-family="sans-serif" font-size="14" '
        f'fill="#475569">Q head {query_head} / KV head {kv_head} · '
        f"tile rows {model['tile_rows']} · {plan['target']['worker_count']} workers · "
        f"block × {model['layers']} layer{'s' if model['layers'] != 1 else ''}</text>",
        f'<text x="40" y="103" font-family="sans-serif" font-size="12" '
        f'fill="#64748b">Plan {escape(plan_sha256[:16])}… · '
        "Cards are actual tasks unless gray and dashed.</text>",
        '<text x="40" y="133" font-family="sans-serif" font-size="13" '
        'fill="#334155">Blue arrows: declared tile dependencies. Gray cards: '
        "other direct prerequisites. Each wave boundary is a full grid barrier.</text>",
        '<text x="40" y="158" font-family="sans-serif" font-size="13" '
        'font-weight="700" fill="#9a3412">A ready tile still waits for the '
        "preceding wave to finish; this diagram does not claim a latency gain.</text>",
    ]
    for wave in range(wave_count):
        x = 36 + wave * 273
        svg.extend(
            [
                f'<rect x="{x}" y="185" width="263" height="{height - 251}" '
                'rx="10" fill="#ffffff" stroke="#cbd5e1"/>',
                f'<text x="{x + 12}" y="209" font-family="sans-serif" '
                'font-size="16" font-weight="700" fill="#172338">'
                f"Wave {wave}</text>",
            ]
        )
        if wave + 1 < wave_count:
            svg.append(
                f'<path d="M {x + 268} 185 V {height - 66}" '
                'stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 6"/>'
            )

    for source, target, aggregate in links:
        sx, sy = positions[source]
        tx, ty = positions[target]
        sx += card_width
        sy += card_height // 2
        ty += card_height // 2
        color = "#94a3b8" if aggregate else "#2563eb"
        dash = ' stroke-dasharray="5 5"' if aggregate else ""
        svg.append(
            f'<path data-from="{source}" data-to="{target}" '
            f'd="M {sx} {sy} C {sx + 18} {sy}, {tx - 18} {ty}, {tx - 7} {ty}" '
            f'fill="none" stroke="{color}" stroke-width="2"{dash} '
            'marker-end="url(#tile-arrow)"/>'
        )

    for card in cards:
        x, y = positions[card["key"]]
        aggregate = card["aggregate"]
        svg.extend(
            [
                f'<g data-card="{card["key"]}"><title>{escape(card["title"])}; '
                f"{escape(card['meta'])}; {escape(card['output'])}; "
                f"{escape(card['detail'])}</title>",
                f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" '
                f'rx="9" fill="{"#e2e8f0" if aggregate else "#dbeafe"}" '
                'stroke="#64748b" '
                + ('stroke-dasharray="5 4"' if aggregate else "")
                + "/>",
                f'<text x="{x + 10}" y="{y + 22}" font-family="sans-serif" '
                f'font-size="14" font-weight="700" fill="#172338">'
                f"{escape(card['title'])}</text>",
                f'<text x="{x + 10}" y="{y + 43}" font-family="sans-serif" '
                f'font-size="11" fill="#475569">{escape(card["meta"])}</text>',
                f'<text x="{x + 10}" y="{y + 64}" font-family="sans-serif" '
                f'font-size="12" fill="#1e293b">{escape(card["output"])}</text>',
                f'<text x="{x + 10}" y="{y + 84}" font-family="sans-serif" '
                f'font-size="11" fill="#475569">{escape(card["detail"])}</text></g>',
            ]
        )
    svg.extend(
        [
            f'<text x="40" y="{height - 28}" font-family="sans-serif" '
            'font-size="12" fill="#64748b">The diagram shows one block template; '
            "stack packages reuse it for each layer. Gray prerequisites are counted "
            "from the actual plan.</text>",
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"
