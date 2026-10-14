"""Isolated CUDA operator tests using the decoder's actual device routines."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import torch.nn.functional as F


def fixtures() -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor | None]]:
    torch.manual_seed(237)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    cases, outputs, probabilities = [], [], []

    def random(*shape, dtype):
        return torch.randn(shape, device="cuda").to(dtype)

    def add(case, expected, probs=None):
        cases.append(case)
        outputs.append(expected.float().flatten().cpu())
        probabilities.append(None if probs is None else probs.float().flatten().cpu())

    def raw(value):
        return value.float().flatten().cpu().tolist()

    for dtype in (torch.float32, torch.bfloat16):
        label = "bfloat16" if dtype == torch.bfloat16 else "float32"
        for rows, width in (
            (1, 1),
            (7, 15),
            (17, 33),
            (65, 93),
            (41, 320),
            (35, 2048),
            (19, 8192),
        ):
            x, w, residual = (
                random(width, dtype=dtype),
                random(rows, width, dtype=dtype),
                random(rows, dtype=dtype),
            )
            for mma in (False, True) if dtype == torch.bfloat16 else (False,):
                for use_residual in (False, True):
                    projected = F.linear(x, w)
                    expected = projected + residual if use_residual else projected
                    begin, end = (1, rows - 1) if rows > 2 else (0, rows)
                    result = torch.full((rows,), -123.0, device="cuda", dtype=dtype)
                    result[begin:end] = expected[begin:end]
                    add(
                        {
                            "op": "gemv",
                            "dtype": label,
                            "mma": mma,
                            "width": width,
                            "rows": rows,
                            "input": raw(x),
                            "weights": raw(w),
                            "aux": raw(residual),
                            "residual": use_residual,
                            "begin": begin,
                            "end": end,
                        },
                        result,
                    )
        for mma in (False, True) if dtype == torch.bfloat16 else (False,):
            x, w = random(93, dtype=dtype), random(53, 93, dtype=dtype)
            add(
                {
                    "op": "logits",
                    "dtype": label,
                    "mma": mma,
                    "width": 93,
                    "rows": 53,
                    "input": raw(x),
                    "weights": raw(w),
                },
                F.linear(x, w),
            )
        if dtype == torch.bfloat16:
            for mma in (False, True):
                # Both additions land exactly on ties-to-even boundaries.
                add(
                    {
                        "op": "gemv",
                        "dtype": label,
                        "mma": mma,
                        "width": 2,
                        "rows": 1,
                        "input": [1.0, 1.0],
                        "weights": [1.0, 1 / 256],
                        "aux": [1 / 256],
                        "residual": True,
                        "exact": True,
                    },
                    torch.tensor([1.0]),
                )
        # Expose round-before-residual with a non-BF16 projection accumulator.
        for width in (10, 64, 333, 2048):
            x, w, up = (
                random(width, dtype=dtype),
                random(width, dtype=dtype),
                random(width, dtype=dtype),
            )
            for op in ("rms_norm", "final_norm"):
                normalized = x.float() * torch.rsqrt(x.float().square().mean() + 1e-5)
                expected = normalized.to(dtype) * w
                add(
                    {
                        "op": op,
                        "dtype": label,
                        "width": width,
                        "input": raw(x),
                        "weights": raw(w),
                    },
                    expected,
                )
            add(
                {
                    "op": "silu",
                    "dtype": label,
                    "width": width,
                    "input": raw(x),
                    "aux": raw(up),
                },
                F.silu(x) * up,
            )
            add(
                {"op": "embedding", "dtype": label, "width": width, "weights": raw(w)},
                w,
            )
        for dh in (10, 64, 160):
            for position in (0, 1, 63, 4095, 8192):
                q = random(3, dh, dtype=dtype)
                frequencies = 1.0 / (
                    500000.0 ** (torch.arange(0, dh, 2, device="cuda").float() / dh)
                )
                angle = position * frequencies
                c, s = angle.cos().to(dtype), angle.sin().to(dtype)
                first, second = q[1].chunk(2)
                result = torch.full_like(q, -123.0)
                result[1] = torch.cat((first * c - second * s, second * c + first * s))
                add(
                    {
                        "op": "rope",
                        "dtype": label,
                        "width": 3 * dh,
                        "heads": 3,
                        "head": 1,
                        "head_dim": dh,
                        "position": position,
                        "input": raw(q),
                        "frequencies": raw(frequencies),
                    },
                    result,
                )
        for length, dh in (
            (1, 10),
            (2, 64),
            (17, 64),
            (65, 160),
            (257, 10),
            (1025, 64),
        ):
            capacity, nq, nkv, head, kv_head = length + 3, 4, 2, 3, 1
            for kind in ("random", "flat", "peaked"):
                q = random(nq, dh, dtype=dtype)
                k, v = (
                    random(capacity, nkv, dh, dtype=dtype),
                    random(capacity, nkv, dh, dtype=dtype),
                )
                if kind == "flat":
                    q.zero_()
                if kind == "peaked":
                    q *= 16
                    k *= 16
                # Unused capacity must not participate in attention.
                k[length:] = 1000
                v[length:] = 1000
                score = (q[head][None, :] @ k[:length, kv_head].T) * (dh**-0.5)
                probs = torch.softmax(score, dim=-1, dtype=torch.float32).to(dtype)
                attended = probs @ v[:length, kv_head]
                expected = torch.full((nq, dh), -123.0, device="cuda", dtype=dtype)
                expected[head] = attended[0]
                add(
                    {
                        "op": "attention",
                        "dtype": label,
                        "width": nq * dh,
                        "heads": nq,
                        "kv_heads": nkv,
                        "head": head,
                        "kv_head": kv_head,
                        "head_dim": dh,
                        "position": length - 1,
                        "capacity": capacity,
                        "input": raw(q),
                        "kv": raw(torch.stack((k, v))),
                    },
                    expected,
                    probs if dtype == torch.bfloat16 else None,
                )
        if dtype == torch.bfloat16:
            for _ in range(16):
                x, w = random(2048, dtype=dtype), random(2048, dtype=dtype)
                normalized = x.float() * torch.rsqrt(x.float().square().mean() + 1e-5)
                add(
                    {
                        "op": "rms_norm",
                        "dtype": label,
                        "width": 2048,
                        "input": raw(x),
                        "weights": raw(w),
                        "exact": True,
                    },
                    normalized.to(dtype) * w,
                )
        x = torch.zeros(64, device="cuda", dtype=dtype)
        w = torch.ones_like(x)
        add(
            {
                "op": "rms_norm",
                "dtype": label,
                "width": 64,
                "input": raw(x),
                "weights": raw(w),
            },
            x,
        )
    for dtype in (torch.float32, torch.bfloat16):
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
                "dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
                "width": boundary.numel(),
                "weights": raw(boundary),
                "exact_bits": True,
            },
            boundary,
        )
    for mma in (False, True):
        add(
            {
                "op": "gemv",
                "dtype": "bfloat16",
                "mma": mma,
                "width": 2,
                "rows": 1,
                "input": [1.0, 1.0],
                "weights": [-1.0, -1 / 256],
                "aux": [-1 / 256],
                "residual": True,
                "exact": True,
            },
            torch.tensor([-1.0]),
        )
    return cases, outputs, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--write-cases", type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        cases, expected, probabilities = fixtures()
    if args.write_cases:
        args.write_cases.write_text(json.dumps(cases))
        return
    with tempfile.TemporaryDirectory(prefix="ops-unit-") as directory:
        path = Path(directory) / "cases.json"
        path.write_text(json.dumps(cases))
        completed = subprocess.run(
            [str(args.executor), str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    results = json.loads(completed.stdout)
    for index, (case, reference, result, probs) in enumerate(
        zip(cases, expected, results, probabilities, strict=True)
    ):
        actual = torch.tensor(result["output"])
        bf16 = case["dtype"] == "bfloat16"
        tolerance = (
            {"atol": 1e-2, "rtol": 1e-2} if bf16 else {"atol": 1e-4, "rtol": 5e-5}
        )
        try:
            if bf16:
                assert torch.equal(actual, actual.to(torch.bfloat16).float())
            torch.testing.assert_close(
                actual, reference, atol=tolerance["atol"], rtol=tolerance["rtol"]
            )
            # Inactive rows and unselected heads are an exact ownership check.
            assert torch.equal(actual[reference == -123], reference[reference == -123])
            if case["op"] in ("embedding", "silu") or case.get("exact", False):
                assert torch.equal(actual, reference)
            if case.get("exact_bits", False):
                assert torch.equal(
                    actual.view(torch.int32), reference.view(torch.int32)
                )
            if probs is not None:
                begin = case["head"] * case["capacity"]
                observed = torch.tensor(
                    result["probabilities"][begin : begin + probs.numel()]
                )
                torch.testing.assert_close(observed, probs, atol=5e-4, rtol=1e-2)
        except (AssertionError, RuntimeError) as error:
            raise AssertionError(
                f"operator case {index}: {case['op']} {case['dtype']} MMA={case.get('mma', False)} shape={case.get('rows', 1)}x{case['width']} position={case.get('position', 0)}"
            ) from error
    print(
        f"{len(cases)} isolated CUDA operator cases passed (FP32, BF16 SIMT, BF16 MMA)."
    )


if __name__ == "__main__":
    main()
