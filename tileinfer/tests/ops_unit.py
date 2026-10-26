"""Isolated CUDA operator tests using the decoder's actual device routines.

Each case runs one fused task (or, for attention, every chunk of every head
followed by the output projection) on a single block. Expected values come
from PyTorch with the same BF16 rounding boundaries as eager inference.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import torch.nn.functional as F

SENTINEL = -123.0


def fixtures() -> list[tuple[dict, str, torch.Tensor, dict | None]]:
    """Cases as (case, checked output buffer, expected values, optional extra check)."""
    torch.manual_seed(237)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    cases: list[tuple[dict, str, torch.Tensor, dict | None]] = []

    def random(*shape, dtype):
        return torch.randn(shape, device="cuda").to(dtype)

    def raw(value):
        return value.float().flatten().cpu().tolist()

    def norm(x, scale):
        """Eager RMSNorm: FP32 statistic, normalized value rounded before the scale."""
        normalized = x.float() * torch.rsqrt(x.float().square().mean() + 1e-5)
        return normalized.to(x.dtype) * scale

    def rope(head, position, frequencies):
        angle = position * frequencies
        c, s = angle.cos().to(head.dtype), angle.sin().to(head.dtype)
        first, second = head.chunk(2)
        return torch.cat((first * c - second * s, second * c + first * s))

    def owned(size, begin, end, values, dtype):
        """Values in [begin,end) and the sentinel elsewhere."""
        result = torch.full((size,), SENTINEL, device="cuda", dtype=dtype)
        result[begin:end] = values[begin:end]
        return result

    def add(case, buffer, expected, extra=None):
        cases.append((case, buffer, expected.float().flatten().cpu(), extra))

    for dtype in (torch.float32, torch.bfloat16):
        label = "bfloat16" if dtype == torch.bfloat16 else "float32"

        # Down projection: rows of W @ product plus the residual, with row and K
        # tails, through the FFN's split-K path (ops_unit.cu split_down). A
        # thread owns two adjacent rows, so the hidden size is even, as the
        # loader requires.
        for rows, width in (
            (2, 1),
            (8, 15),
            (18, 33),
            (66, 93),
            (42, 320),
            (36, 2048),
            (20, 8192),
        ):
            product, w, residual = (
                random(width, dtype=dtype),
                random(rows, width, dtype=dtype),
                random(rows, dtype=dtype),
            )
            # Two chunks keep the widest FP32 chunk within the shared vector.
            for with_residual, ffn_chunks in ((False, 2), (True, 3), (True, 20)):
                r = residual if with_residual else torch.zeros_like(residual)
                begin, end = (1, rows - 1) if rows > 2 else (0, rows)
                add(
                    {
                        "op": "down",
                        "dtype": label,
                        "hidden": rows,
                        "intermediate": width,
                        "ffn_chunks": ffn_chunks,
                        "product": raw(product),
                        "Wdown": raw(w),
                        "r": raw(r),
                        "begin": begin,
                        "end": end,
                    },
                    "y",
                    owned(rows, begin, end, F.linear(product, w) + r, dtype),
                )

        # Logits: whole and partial vocabulary ranges, then each range's
        # greedy candidate. Rows 7, 30 and 41 repeat row 40, so their logits
        # tie and the lowest ID among them wins if they hold the maximum.
        x, w = random(93, dtype=dtype), random(53, 93, dtype=dtype)
        w[40] = w[w.float() @ x.float() == (w.float() @ x.float()).max()][0]
        w[7] = w[30] = w[41] = w[40]
        for begin, end in ((0, 53), (1, 52), (31, 53)):
            expected = torch.full((53,), SENTINEL, device="cuda")
            expected[begin:end] = F.linear(x, w).float()[begin:end]
            add(
                {
                    "op": "logits",
                    "dtype": label,
                    "hidden": 93,
                    "rows": 53,
                    "x": raw(x),
                    "weights": raw(w),
                    "begin": begin,
                    "end": end,
                },
                "logits",
                expected,
            )
        # A non-finite logit makes the candidate kNonFinite (-1).
        for poison in ("nan", "inf"):
            add(
                {
                    "op": "logits",
                    "dtype": label,
                    "hidden": 93,
                    "rows": 53,
                    "x": raw(x),
                    "weights": raw(w),
                    "poison": poison,  # the harness sets row 20, column 5
                    "begin": 0,
                    "end": 53,
                    "non_finite": True,
                },
                "candidate",
                torch.tensor([0.0, -1.0]),
            )

        # Final norm and embedding rows, whole and partial.
        for width in (10, 64, 333, 2048):
            x, w = random(width, dtype=dtype), random(width, dtype=dtype)
            for begin, end in ((0, width), (1, width - 1)):
                base = {"dtype": label, "hidden": width, "begin": begin, "end": end}
                add(
                    {**base, "op": "final_norm", "x": raw(x), "weights": raw(w)},
                    "y",
                    owned(width, begin, end, norm(x, w), dtype),
                )
                add(
                    {**base, "op": "embedding", "weights": raw(w), "exact": True},
                    "y",
                    owned(width, begin, end, w, dtype),
                )

        # Rows of [Q; K; V]: normalized input and projection. Q and K stay
        # unrotated in scratch and V goes to the current cache row. Ranges
        # cross the Q/K and K/V boundaries and start inside a head.
        for nq, nkv, dh in ((4, 2, 10), (4, 1, 64), (2, 1, 160)):
            d = nq * dh
            kv_width = nkv * dh
            rows = d + 2 * kv_width
            for position in (0, 1, 63, 4095, 8192):
                x, scale = random(d, dtype=dtype), random(d, dtype=dtype)
                wq, wk, wv = (
                    random(d, d, dtype=dtype) / 8,
                    random(kv_width, d, dtype=dtype) / 8,
                    random(kv_width, d, dtype=dtype) / 8,
                )
                u = norm(x, scale)
                q, k, v = F.linear(u, wq), F.linear(u, wk), F.linear(u, wv)
                base = {
                    "op": "qkv",
                    "dtype": label,
                    "hidden": d,
                    "heads": nq,
                    "kv_heads": nkv,
                    "head_dim": dh,
                    "position": position,
                    "x": raw(x),
                    "attn_norm": raw(scale),
                    "Wq": raw(wq),
                    "Wk": raw(wk),
                    "Wv": raw(wv),
                }

                def part(begin, end, first, width):
                    """Rows [begin,end) of [Q; K; V] within the part at ``first``."""
                    low = min(max(begin - first, 0), width)
                    return low, max(min(end - first, width), low)

                for begin, end in (
                    (0, rows),
                    (d - 3, d + kv_width + 5),
                    (d + 1, rows - 1),
                ):
                    case = {**base, "begin": begin, "end": end}
                    add(case, "q", owned(d, *part(begin, end, 0, d), q, dtype))
                    add(
                        case,
                        "k",
                        owned(kv_width, *part(begin, end, d, kv_width), k, dtype),
                    )
                    first, last = part(begin, end, d + kv_width, kv_width)
                    row = position * kv_width
                    value = torch.full(
                        ((position + 1) * kv_width,),
                        SENTINEL,
                        device="cuda",
                        dtype=dtype,
                    )
                    value[row + first : row + last] = v[first:last]
                    add(case, "value_cache", value)

        # Attention over the visible prefix for every head, as one fused task
        # per K/V head up to fused_steps, otherwise split into at least two
        # chunks of at most chunk_steps: (length, head_dim, query heads, K/V
        # heads, chunk_steps, fused_steps). Uneven splits, chunks of one step,
        # both sides of the fused limit, groups of 1 to 8, and both the fixed
        # Llama shape (four heads of 64 per K/V head) and the general path.
        for length, dh, nq, nkv, chunk_steps, fused_steps in (
            (1, 8, 4, 2, 4, 4),
            (2, 64, 4, 4, 1, 1),
            (17, 64, 4, 1, 8, 4),
            (65, 128, 4, 2, 64, 64),
            (257, 16, 4, 2, 300, 300),
            (33, 32, 8, 1, 16, 8),
            (384, 64, 4, 1, 1024, 384),
            (385, 64, 4, 1, 1024, 384),
            (1025, 64, 4, 1, 512, 384),
        ):
            capacity = length + 3
            fused = length <= fused_steps
            d = nq * dh
            position = length - 1
            frequencies = 1.0 / (
                500000.0 ** (torch.arange(0, dh, 2, device="cuda").float() / dh)
            )
            for kind in ("random", "flat", "peaked"):
                q = random(nq, dh, dtype=dtype)
                k, v = (
                    random(capacity, nkv, dh, dtype=dtype),
                    random(capacity, nkv, dh, dtype=dtype),
                )
                current = random(nkv, dh, dtype=dtype)
                if kind == "flat":
                    q.zero_()
                if kind == "peaked":
                    q *= 16
                    k *= 16
                    current *= 16
                # Unused capacity must not participate in attention, and the
                # task takes the current K row from scratch, not the cache.
                k[length - 1 :] = 1000
                v[length:] = 1000
                rotated = torch.stack([rope(head, position, frequencies) for head in q])
                cache = k.clone()
                cache[position] = torch.stack(
                    [rope(head, position, frequencies) for head in current]
                )
                # Split chunks leave their BF16 scores in the workspace; a fused
                # task keeps them in shared memory.
                attended, scores = (
                    [],
                    torch.full((nq, capacity), SENTINEL, device="cuda", dtype=dtype),
                )
                for head in range(nq):
                    kv_head = head // (nq // nkv)
                    score = (rotated[head][None, :] @ cache[:length, kv_head].T) * (
                        dh**-0.5
                    )
                    probs = torch.softmax(score, dim=-1, dtype=torch.float32).to(dtype)
                    if not fused:
                        scores[head, :length] = score[0]
                    attended.append((probs @ v[:length, kv_head])[0])
                case = {
                    "op": "attention",
                    "dtype": label,
                    "hidden": d,
                    "heads": nq,
                    "kv_heads": nkv,
                    "head_dim": dh,
                    "position": position,
                    "capacity": capacity,
                    "chunk_steps": chunk_steps,
                    "fused_steps": fused_steps,
                    "q": raw(q),
                    "k": raw(current),
                    "key_cache": raw(k),
                    "value_cache": raw(v),
                    "frequencies": raw(frequencies),
                    "Wo": raw(torch.eye(d, device="cuda")),
                }
                add(
                    case,
                    "attention",
                    torch.cat(attended),
                    {"buffer": "scores", "expected": scores.flatten()},
                )
                if kind == "random":
                    # Chunk 0 stores rotated Q and the rotated current K row.
                    add(case, "q_rot", rotated)
                    add(case, "key_cache", cache)

        # Gate and up rows of the normalized input, combined as SiLU(gate) * up.
        for d, f in ((10, 7), (64, 93), (333, 41), (2048, 64)):
            r, scale = random(d, dtype=dtype), random(d, dtype=dtype)
            gate, up = random(f, d, dtype=dtype) / 8, random(f, d, dtype=dtype) / 8
            z = norm(r, scale)
            product = F.silu(F.linear(z, gate)) * F.linear(z, up)
            begin, end = (1, f - 1) if f > 2 else (0, f)
            add(
                {
                    "op": "gate_up",
                    "dtype": label,
                    "hidden": d,
                    "intermediate": f,
                    "r": raw(r),
                    "ffn_norm": raw(scale),
                    "Wgate": raw(gate),
                    "Wup": raw(up),
                    "begin": begin,
                    "end": end,
                },
                "product",
                owned(f, begin, end, product, dtype),
            )

        # Whole FFN tasks: each chunk's gate/up rows and down partial, then the
        # reduction plus the residual.
        for d, f, ffn_chunks in (
            (10, 7, 3),
            (64, 93, 5),
            (64, 2048, 20),
            (2048, 64, 20),
        ):
            r, scale = random(d, dtype=dtype), random(d, dtype=dtype)
            gate, up = random(f, d, dtype=dtype) / 8, random(f, d, dtype=dtype) / 8
            down = random(d, f, dtype=dtype) / 8
            z = norm(r, scale)
            product = F.silu(F.linear(z, gate)) * F.linear(z, up)
            begin, end = (1, d - 1) if d > 2 else (0, d)
            add(
                {
                    "op": "ffn",
                    "dtype": label,
                    "hidden": d,
                    "intermediate": f,
                    "ffn_chunks": ffn_chunks,
                    "r": raw(r),
                    "ffn_norm": raw(scale),
                    "Wgate": raw(gate),
                    "Wup": raw(up),
                    "Wdown": raw(down),
                    "begin": begin,
                    "end": end,
                },
                "y",
                owned(d, begin, end, F.linear(product, down) + r, dtype),
                # The product buffer holds max(d, f) elements; the rest keep the sentinel.
                {
                    "buffer": "product",
                    "expected": owned(
                        max(d, f), 0, f, F.pad(product, (0, max(d - f, 0))), dtype
                    ),
                },
            )

        if dtype == torch.bfloat16:
            # Sixteen 2048-wide norms must match PyTorch bit for bit.
            for _ in range(16):
                x, w = random(2048, dtype=dtype), random(2048, dtype=dtype)
                add(
                    {
                        "op": "final_norm",
                        "dtype": label,
                        "hidden": 2048,
                        "x": raw(x),
                        "weights": raw(w),
                        "exact": True,
                    },
                    "y",
                    norm(x, w),
                )
            # Projection then residual: both additions land on ties-to-even boundaries.
            for sign in (1.0, -1.0):
                add(
                    {
                        "op": "down",
                        "dtype": label,
                        "hidden": 2,
                        "intermediate": 2,
                        "product": [1.0, 1.0],
                        "Wdown": [sign, sign / 256] * 2,
                        "r": [sign / 256] * 2,
                        "exact": True,
                    },
                    "y",
                    torch.tensor([sign] * 2),
                )
        add(
            {
                "op": "final_norm",
                "dtype": label,
                "hidden": 64,
                "weights": [1.0] * 64,
            },
            "y",
            torch.zeros(64),
        )

        boundary = torch.tensor(
            [
                0.0,
                -0.0,
                2.0**-133,
                -(2.0**-133),
                2.0**-126,
                (2.0 - 2.0**-7) * 2.0**127,
                -(2.0 - 2.0**-7) * 2.0**127,
            ],
            device="cuda",
            dtype=dtype,
        )
        add(
            {
                "op": "embedding",
                "dtype": label,
                "hidden": boundary.numel(),
                "weights": raw(boundary),
                "exact_bits": True,
            },
            "y",
            boundary,
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--write-cases", type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        cases = fixtures()
    payload = json.dumps([case for case, _, _, _ in cases])
    if args.write_cases:
        args.write_cases.write_text(payload)
        return
    with tempfile.TemporaryDirectory(prefix="ops-unit-") as directory:
        path = Path(directory) / "cases.json"
        path.write_text(payload)
        completed = subprocess.run(
            [str(args.executor), str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    results = json.loads(completed.stdout)
    for index, ((case, buffer, reference, extra), result) in enumerate(
        zip(cases, results, strict=True)
    ):
        if case.get("non_finite", False):
            if result["candidate"] != [0.0, -1]:
                raise AssertionError(f"operator case {index}: {result['candidate']}")
            continue
        actual = torch.tensor(result[buffer][: reference.numel()])
        if case["op"] == "logits":
            # The candidate is the largest logit it stored, lowest ID first.
            begin, end = case["begin"], case["end"]
            owned_logits = actual[begin:end].tolist()
            best = max(range(len(owned_logits)), key=lambda i: (owned_logits[i], -i))
            if result["candidate"] != [owned_logits[best], begin + best]:
                raise AssertionError(f"operator case {index}: {result['candidate']}")
        bf16 = case["dtype"] == "bfloat16"
        atol, rtol = (1e-2, 1e-2) if bf16 else (1e-4, 5e-5)
        try:
            if bf16:
                assert torch.equal(actual, actual.to(torch.bfloat16).float())
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            # Rows a task does not own keep the sentinel exactly.
            assert torch.equal(
                actual[reference == SENTINEL], reference[reference == SENTINEL]
            )
            if case.get("exact", False):
                assert torch.equal(actual, reference)
            if case.get("exact_bits", False):
                assert torch.equal(
                    actual.view(torch.int32), reference.view(torch.int32)
                )
            if extra is not None and bf16:
                observed = torch.tensor(result[extra["buffer"]])
                expected = extra["expected"].float().cpu()
                torch.testing.assert_close(observed, expected, atol=5e-4, rtol=1e-2)
        except (AssertionError, RuntimeError) as error:
            raise AssertionError(
                f"operator case {index}: {case['op']} {case['dtype']} "
                f"hidden={case.get('hidden')} position={case.get('position', 0)} "
                f"chunk_steps={case.get('chunk_steps')} "
                f"fused_steps={case.get('fused_steps')}"
            ) from error
    print(f"{len(cases)} isolated CUDA operator cases passed (FP32, BF16).")


if __name__ == "__main__":
    main()
