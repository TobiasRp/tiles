"""Render the blog's Llama 3.2 1B architecture figures.

usage: python docs/blog/make_architecture_figures.py OUT_DIR

Two static figures: the whole one-token decoder, and one decoder layer op by
op. Every number comes from the pinned model config (docs/architecture.md)
and is derived below, so the figures cannot drift from the shapes.
"""

import sys
from html import escape
from pathlib import Path

from make_figures import GRID, MUTED, NEUTRAL, SERIES, SURFACE, TEXT, WIDTH, Svg

HIDDEN, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM = 2048, 16, 32, 8, 64
FFN, VOCAB, THETA = 8192, 128_256, 500_000
BF16 = 2  # bytes per weight

Q_DIM, KV_DIM = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
ATTN_PARAMS = HIDDEN * (2 * Q_DIM + 2 * KV_DIM)
MLP_PARAMS = 3 * HIDDEN * FFN
LAYER_PARAMS = ATTN_PARAMS + MLP_PARAMS
EMBED_PARAMS = VOCAB * HIDDEN
NORM_PARAMS = (2 * LAYERS + 1) * HIDDEN
TOTAL_PARAMS = LAYERS * LAYER_PARAMS + EMBED_PARAMS + NORM_PARAMS

# (fill, stroke) per op family. Strokes are the figures' categorical colors;
# fills are the same hues at ~10% so text stays on a near-white surface.
WEIGHT = ("#e8f1fc", SERIES[0])
TABLE = ("#e6f6ef", SERIES[2])
STATE = ("#fdeee7", SERIES[1])
SMALL = ("#f1f0ed", "#8b8a85")
PLAIN = (SURFACE, NEUTRAL)
BAND = "#f7f7f5"

ARROW = (
    '<defs><marker id="ah" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" '
    f'refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{MUTED}"/></marker></defs>'
)


def millions(count: int) -> str:
    return f"{count / 1e6:.1f}M"


def megabytes(params: int) -> str:
    return f"{params * BF16 / 1e6:.1f} MB"


def n(value: int) -> str:
    return f"{value:,}"


def box(
    svg: Svg,
    cx: float,
    top: float,
    width: float,
    height: float,
    style: tuple[str, str],
    lines: list[str],
    tip: str = "",
) -> None:
    """Rounded box with centered text; the first line is the bold name."""
    fill, stroke = style
    title = f"<title>{escape(tip)}</title>" if tip else ""
    svg.add(
        f'<rect x="{cx - width / 2:.1f}" y="{top:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="6" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.5">{title}</rect>'
    )
    sizes = [12.5] + [11] * (len(lines) - 1)
    total = sum(size + 4 for size in sizes) - 4
    y = top + (height - total) / 2
    for i, (line, size) in enumerate(zip(lines, sizes)):
        y += size
        svg.add(
            svg.text(
                cx,
                y - 1,
                line,
                size=size,
                weight=600 if i == 0 else 400,
                color=TEXT if i == 0 else MUTED,
                anchor="middle",
            )
        )
        y += 4


def arrow(svg: Svg, *points: tuple[float, float], width: float = 1.5, dash: str = "") -> None:
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(
        f'<path d="{path}" fill="none" stroke="{MUTED}" stroke-width="{width}"'
        f'{extra} marker-end="url(#ah)"/>'
    )


def line(svg: Svg, *points: tuple[float, float], color: str = MUTED, width: float = 1.5, dash: str = "") -> None:
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')


def label(svg: Svg, x: float, y: float, value: str, *, anchor: str = "start", size: float = 11) -> None:
    svg.add(svg.text(x, y, value, size=size, color=MUTED, anchor=anchor))


def operator(svg: Svg, cx: float, cy: float, kind: str, tip: str) -> None:
    """Circled + (residual add) or × (elementwise multiply)."""
    svg.add(
        f'<circle cx="{cx}" cy="{cy}" r="10.5" fill="{SURFACE}" stroke="{TEXT}" stroke-width="1.5">'
        f"<title>{escape(tip)}</title></circle>"
    )
    if kind == "+":
        line(svg, (cx - 5, cy), (cx + 5, cy), color=TEXT)
        line(svg, (cx, cy - 5), (cx, cy + 5), color=TEXT)
    else:
        line(svg, (cx - 4, cy - 4), (cx + 4, cy + 4), color=TEXT)
        line(svg, (cx - 4, cy + 4), (cx + 4, cy - 4), color=TEXT)


def overview(out: Path) -> None:
    svg = Svg(
        664,
        "Llama 3.2 1B: one-token decoder",
        "One token in, one vector of vocabulary logits out; the hidden state is a 2,048-wide vector",
    )
    svg.add(ARROW)
    svg.legend(
        24,
        80,
        [
            (SERIES[2], "embedding table"),
            (SERIES[0], "decoder layers"),
            (SERIES[1], "KV cache"),
            ("#8b8a85", "normalization"),
        ],
    )
    cx, width = 250, 230

    # Input pill and embedding.
    box(svg, cx, 104, 200, 28, PLAIN, ["token id t, position p"])
    arrow(svg, (cx, 132), (cx, 158))
    box(
        svg, cx, 158, width, 50, TABLE,
        ["Token embedding", f"{n(VOCAB)} × {n(HIDDEN)} lookup · {millions(EMBED_PARAMS)}"],
        "Row t of the embedding table: no arithmetic, one 2,048-wide row read.",
    )
    arrow(svg, (cx, 208), (cx, 238))
    label(svg, cx + 8, 227, f"x₀  [{n(HIDDEN)}]")

    # The layer stack, with the first, second and last layer drawn.
    svg.add(
        f'<rect x="120" y="238" width="260" height="212" rx="10" fill="{WEIGHT[0]}" '
        f'fill-opacity="0.45" stroke="{SERIES[0]}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    svg.add(svg.text(134, 257, f"{LAYERS} × decoder layer", size=12.5, weight=600))
    tops = {0: 272, 1: 320, 15: 402}
    for index, top in tops.items():
        box(
            svg, cx, top, 190, 32, WEIGHT,
            [f"Layer {index}"],
            f"Same graph as every layer; layer {index} has its own {millions(LAYER_PARAMS)} weights.",
        )
    arrow(svg, (cx, 304), (cx, 320))
    arrow(svg, (cx, 352), (cx, 370))
    svg.add(svg.text(cx, 388, "⋮", size=18, color=MUTED, anchor="middle"))
    arrow(svg, (cx, 392), (cx, 402))
    arrow(svg, (cx, 450), (cx, 480))
    label(svg, cx + 8, 470, f"x₁₆  [{n(HIDDEN)}]")

    # Head.
    box(svg, cx, 480, width, 36, SMALL, ["Final RMSNorm", f"scale [{n(HIDDEN)}]"])
    arrow(svg, (cx, 516), (cx, 542))
    box(
        svg, cx, 542, width, 50, TABLE,
        ["lm_head", f"{n(HIDDEN)} → {n(VOCAB)} · shares the embedding"],
        "Output projection reuses the embedding matrix: no second copy of the weights.",
    )
    arrow(svg, (cx, 592), (cx, 624))
    label(svg, cx + 8, 613, "one dot product per vocabulary entry")
    box(svg, cx, 624, 260, 28, PLAIN, [f"logits  [{n(VOCAB)}] → next token"])

    # Tied weights.
    line(svg, (135, 183), (72, 183), (72, 567), (135, 567), color=SERIES[2], dash="5 4")
    svg.add(
        f'<text transform="translate(60,375) rotate(-90)" {_font()} font-size="11" fill="{MUTED}" '
        'text-anchor="middle">tied weights: one matrix, two uses</text>'
    )

    # KV cache beside the stack.
    box(
        svg, 595, 272, 250, 100, STATE,
        [
            "KV cache",
            f"{LAYERS} layers × {KV_HEADS} KV heads × T × {HEAD_DIM}",
            "keys and values, BF16",
            "each layer appends position p,",
            "then attends over positions 0…p",
        ],
        "State that persists between tokens; everything else is recomputed per token.",
    )
    svg.add(
        f'<path d="M380,322 L470,322" fill="none" stroke="{MUTED}" stroke-width="1.5" '
        'marker-end="url(#ah)" marker-start="url(#ah)"/>'
    )
    label(svg, 425, 314, "r / w", anchor="middle")

    # Config and parameter budget.
    rows = [
        ("hidden width", n(HIDDEN)),
        ("layers", str(LAYERS)),
        ("query / KV heads", f"{Q_HEADS} / {KV_HEADS} × {HEAD_DIM}"),
        ("FFN width", n(FFN)),
        ("vocabulary", n(VOCAB)),
        ("RoPE base θ", n(THETA)),
        ("RMSNorm ε", "1e-5"),
    ]
    svg.add(
        f'<rect x="470" y="104" width="250" height="124" rx="6" fill="{BAND}" stroke="{GRID}" stroke-width="1.5"/>'
    )
    for i, (name, value) in enumerate(rows):
        svg.add(svg.text(484, 124 + 16 * i, name, size=11.5, color=MUTED))
        svg.add(svg.text(706, 124 + 16 * i, value, size=11.5, anchor="end", weight=600))

    layer_total = LAYERS * LAYER_PARAMS
    budget = [
        f"{TOTAL_PARAMS / 1e9:.2f}B parameters, {TOTAL_PARAMS * BF16 / 1e9:.2f} GB in BF16",
        f"layers {millions(layer_total)} ({layer_total / TOTAL_PARAMS:.0%}) · "
        f"{millions(LAYER_PARAMS)} each",
        f"embedding {millions(EMBED_PARAMS)} ({EMBED_PARAMS / TOTAL_PARAMS:.0%})",
    ]
    for i, text in enumerate(budget):
        svg.add(svg.text(470, 404 + 17 * i, text, size=11.5, color=MUTED if i else TEXT, weight=400 if i else 600))
    svg.write(out)


def _font() -> str:
    return "font-family=\"Inter, 'Helvetica Neue', Arial, sans-serif\""


def decoder_layer(out: Path) -> None:
    svg = Svg(
        1112,
        "Llama 3.2 1B: one decoder layer",
        "Every op for one token at position p; the left rail is the residual stream",
    )
    svg.add(ARROW)
    svg.legend(
        24,
        80,
        [
            (SERIES[0], "weight matmul (streams BF16 weights)"),
            ("#8b8a85", "norm / elementwise"),
            (SERIES[1], "attention and KV cache"),
        ],
    )
    rail, cq, ck, cv, cm = 56, 240, 405, 570, 405
    band_x, band_w = 96, 628

    def fork(y: float, left_edge: float) -> None:
        svg.add(f'<circle cx="{rail}" cy="{y}" r="3.5" fill="{TEXT}"/>')
        arrow(svg, (rail, y), (left_edge, y))

    def band(top: float, bottom: float, name: str) -> None:
        svg.add(
            f'<rect x="{band_x}" y="{top}" width="{band_w}" height="{bottom - top}" rx="10" '
            f'fill="{BAND}" stroke="{GRID}" stroke-width="1.5"/>'
        )
        svg.add(svg.text(band_x + 12, top + 17, name, size=10.5, weight=600, color=MUTED))

    # Backgrounds go first so the rail and ops draw over them.
    attn_band = (128, 668)
    mlp_band = (682, 1034)
    band(*attn_band, "ATTENTION SUB-LAYER")
    band(*mlp_band, "MLP SUB-LAYER")

    # Residual rail.
    line(svg, (rail, 100), (rail, 1060), color=TEXT, width=3)
    svg.add(
        f'<path d="M{rail - 6},1052 L{rail},1064 L{rail + 6},1052 z" fill="{TEXT}"/>'
    )
    svg.add(svg.text(rail + 12, 112, f"x  [{n(HIDDEN)}] from the previous layer", size=11, color=MUTED))

    # --- Attention ---
    norm_top, norm_h = 142, 40
    box(
        svg, cm, norm_top, 200, norm_h, SMALL,
        ["RMSNorm", "x / √(mean(x²) + ε) · w"],
        f"attn_norm: scale [{n(HIDDEN)}], ε = 1e-5, float32 reduction",
    )
    fork(norm_top + norm_h / 2, cm - 100)
    split = 204
    line(svg, (cm, norm_top + norm_h), (cm, split))
    line(svg, (cq, split), (cv, split))
    label(svg, cm + 8, split - 6, f"u  [{n(HIDDEN)}]")

    proj_top, proj_h = 224, 58
    projections = [
        (cq, "Wq", f"{n(HIDDEN)} → {n(Q_DIM)}", HIDDEN * Q_DIM, f"{Q_HEADS} query heads × {HEAD_DIM}"),
        (ck, "Wk", f"{n(HIDDEN)} → {n(KV_DIM)}", HIDDEN * KV_DIM, f"{KV_HEADS} KV heads × {HEAD_DIM}"),
        (cv, "Wv", f"{n(HIDDEN)} → {n(KV_DIM)}", HIDDEN * KV_DIM, f"{KV_HEADS} KV heads × {HEAD_DIM}"),
    ]
    for cx, name, shape, params, heads in projections:
        arrow(svg, (cx, split), (cx, proj_top))
        box(
            svg, cx, proj_top, 146, proj_h, WEIGHT,
            [name, f"{shape} · {megabytes(params)}", heads],
            f"{name} [out, in] BF16 matrix, bias-free: {megabytes(params)} read per token",
        )

    rope_top, rope_h = 322, 40
    for cx, name in ((cq, "q"), (ck, "k")):
        arrow(svg, (cx, proj_top + proj_h), (cx, rope_top))
        box(
            svg, cx, rope_top, 146, rope_h, SMALL,
            [f"RoPE({name})", f"rotate by p · θ = {n(THETA)}"],
            "Llama 3 frequency scaling (factor 32); split-half pairing as in Hugging Face.",
        )

    cache_top, cache_h = 404, 58
    cache_cx, cache_w = 487.5, 311
    arrow(svg, (ck, rope_top + rope_h), (ck, cache_top))
    arrow(svg, (cv, proj_top + proj_h), (cv, cache_top))
    label(svg, cv + 8, 342, "v is not rotated")
    box(
        svg, cache_cx, cache_top, cache_w, cache_h, STATE,
        [
            "KV cache, this layer",
            f"K, V for positions 0…p · {KV_HEADS} KV heads × {HEAD_DIM}",
            "this token appends k, v at position p",
        ],
        "Persistent state: the only thing that carries information from earlier tokens.",
    )

    attn_top, attn_h = 506, 78
    attn_cx, attn_w = 405, 520
    arrow(svg, (cq, rope_top + rope_h), (cq, attn_top))
    label(svg, cq + 8, 480, f"q  [{Q_HEADS} × {HEAD_DIM}]")
    arrow(svg, (cache_cx, cache_top + cache_h), (cache_cx, attn_top))
    label(svg, cache_cx + 8, 488, "K, V")
    box(
        svg, attn_cx, attn_top, attn_w, attn_h, STATE,
        [
            f"Attention, per query head h  ({Q_HEADS} heads)",
            f"scores[t] = q_h · K[t] / √{HEAD_DIM},  t = 0…p     (causal: nothing after p exists yet)",
            "weights = softmax(scores);  out_h = Σ_t weights[t] · V[t]",
            f"query head h reads KV head ⌊h / {Q_HEADS // KV_HEADS}⌋  (grouped-query attention, "
            f"{Q_HEADS // KV_HEADS} : 1)",
        ],
        "Float32 scores and softmax over the visible prefix; cost grows with p.",
    )

    wo_top, wo_h = 616, 54
    arrow(svg, (cm, attn_top + attn_h), (cm, wo_top))
    label(svg, cm + 8, 606, f"concat {Q_HEADS} × {HEAD_DIM} = {n(Q_DIM)}")
    box(
        svg, cm, wo_top, 200, wo_h, WEIGHT,
        ["Wo", f"{n(Q_DIM)} → {n(HIDDEN)} · {megabytes(Q_DIM * HIDDEN)}"],
        "Output projection, bias-free.",
    )
    add1 = wo_top + wo_h / 2
    arrow(svg, (cm - 100, add1), (rail + 12, add1))
    operator(svg, rail, add1, "+", "Residual add: r = x + Wo · attention")
    label(svg, rail + 12, add1 + 22, f"r  [{n(HIDDEN)}]")

    # --- MLP ---
    norm2_top = 706
    box(
        svg, cm, norm2_top, 200, norm_h, SMALL,
        ["RMSNorm", "x / √(mean(x²) + ε) · w"],
        f"ffn_norm: scale [{n(HIDDEN)}], ε = 1e-5",
    )
    fork(norm2_top + norm_h / 2, cm - 100)
    cg, cu = 330, 480
    split2 = 768
    line(svg, (cm, norm2_top + norm_h), (cm, split2))
    line(svg, (cg, split2), (cu, split2))
    label(svg, cm + 8, split2 - 6, f"z  [{n(HIDDEN)}]")

    gate_top, gate_h = 788, 58
    for cx, name in ((cg, "Wgate"), (cu, "Wup")):
        arrow(svg, (cx, split2), (cx, gate_top))
        box(
            svg, cx, gate_top, 130, gate_h, WEIGHT,
            [name, f"{n(HIDDEN)} → {n(FFN)}", megabytes(HIDDEN * FFN)],
            f"{name}: bias-free, {megabytes(HIDDEN * FFN)} read per token",
        )
    silu_top, silu_h = 878, 34
    arrow(svg, (cg, gate_top + gate_h), (cg, silu_top))
    box(svg, cg, silu_top, 130, silu_h, SMALL, ["SiLU", "g · sigmoid(g)"])
    mul_y = 950
    line(svg, (cg, silu_top + silu_h), (cg, mul_y))
    arrow(svg, (cg, mul_y), (cm - 12, mul_y))
    line(svg, (cu, gate_top + gate_h), (cu, mul_y))
    arrow(svg, (cu, mul_y), (cm + 12, mul_y))
    operator(svg, cm, mul_y, "×", "Elementwise product SiLU(gate) · up")
    label(svg, cu + 12, mul_y - 8, f"[{n(FFN)}]")

    down_top, down_h = 984, 54
    arrow(svg, (cm, mul_y + 11), (cm, down_top))
    box(
        svg, cm, down_top, 200, down_h, WEIGHT,
        ["Wdown", f"{n(FFN)} → {n(HIDDEN)} · {megabytes(HIDDEN * FFN)}"],
        "Down projection, bias-free.",
    )
    add2 = down_top + down_h / 2
    arrow(svg, (cm - 100, add2), (rail + 12, add2))
    operator(svg, rail, add2, "+", "Residual add: out = r + Wdown · (SiLU(gate) · up)")
    svg.add(svg.text(rail + 12, 1056, f"x'  [{n(HIDDEN)}] to the next layer", size=11, color=MUTED))

    footer = (
        f"Per layer: {millions(LAYER_PARAMS)} weights ({megabytes(LAYER_PARAMS)} in BF16): "
        f"attention {millions(ATTN_PARAMS)} ({ATTN_PARAMS / LAYER_PARAMS:.0%}), "
        f"MLP {millions(MLP_PARAMS)} ({MLP_PARAMS / LAYER_PARAMS:.0%}), "
        f"plus two RMSNorm scales of {n(HIDDEN)}."
    )
    svg.add(svg.text(24, 1092, footer, size=11.5, color=MUTED))
    svg.write(out)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    overview(out / "architecture-model.svg")
    decoder_layer(out / "architecture-layer.svg")


if __name__ == "__main__":
    main()
