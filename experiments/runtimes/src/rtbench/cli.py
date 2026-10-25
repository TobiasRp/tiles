"""rtbench command line: convert, bench, parity, report."""

import argparse
import json
from pathlib import Path
from typing import Any

from . import gguf_tools, llamacpp, parity, report, tileinfer
from .paths import MODELS, RESULTS
from .schema import read_records, write_records


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--kv", choices=("bf16", "f16"), default="bf16")
    p.add_argument("--fa", choices=("on", "off", "auto"), default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-graphs", action="store_true", help="CUDA: disable CUDA graphs")


def _convert(args: argparse.Namespace) -> None:
    out = args.out or MODELS / f"{args.checkpoint.name}-bf16.gguf"
    gguf_tools.convert(args.checkpoint, out)
    print(
        json.dumps(
            {
                "gguf": str(out),
                "tensors": _jsonable(gguf_tools.check_bf16(out)),
                "sha256": gguf_tools.sha256(out),
            },
            indent=2,
        )
    )


def _jsonable(types: dict) -> dict[str, int]:
    return {
        f"{name}/rank{rank}": count for (name, rank), count in sorted(types.items())
    }


def _bench(args: argparse.Namespace) -> None:
    graphs = not args.no_graphs
    common: dict[str, Any] = dict(
        kv=args.kv, fa=args.fa, threads=args.threads, graphs=graphs
    )
    records = llamacpp.decode_records(
        args.backend,
        args.gguf,
        args.positions,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
        **common,
    )
    if args.llama_bench:
        records += llamacpp.bench_records(
            args.backend, args.gguf, args.positions, **common
        )
    ids = json.loads(args.inputs.read_text())["tokens"]
    data = llamacpp.generate(
        args.backend, args.gguf, ids, args.new_tokens, rounds=args.rounds, **common
    )
    records += llamacpp.generate_records(
        args.backend,
        args.gguf,
        ids,
        args.new_tokens,
        kv=args.kv,
        fa=args.fa,
        threads=args.threads,
        graphs=graphs,
        data=data,
    )
    write_records(args.out, records)
    print(report.report(records))
    print(args.out)


def _parity(args: argparse.Namespace) -> None:
    prompts_ids = parity.tokenize(args.checkpoint)
    reference = parity.reference_generate(
        args.checkpoint, prompts_ids, args.new_tokens, args.device
    )
    rows = []
    for index, ref in enumerate(reference):
        data = llamacpp.generate(
            args.backend,
            args.gguf,
            ref.ids,
            args.new_tokens,
            kv=args.kv,
            fa=args.fa,
            threads=args.threads,
            warmup=0,
            rounds=1,
            graphs=not args.no_graphs,
            forced=ref.tokens,
        )
        comparison = parity.compare(ref, data["tokens"], data["top_logits"])
        comparison.prompt_index = index
        rows.append(
            {
                **comparison.__dict__,
                "prompt_ids": ref.ids,
                "reference_tokens": ref.tokens,
                "llamacpp_tokens": data["tokens"],
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"backend": args.backend, "kv": args.kv, "fa": args.fa, "rows": rows},
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            f"prompt {row['prompt_index']}: {row['steps']} steps, "
            f"{len(row['real_mismatches'])} real / {len(row['tie_mismatches'])} tie argmax mismatches, "
            f"max |top logit err| {row['max_logit_error']:.3f}, "
            f"within tolerance {row['logits_within_tolerance']}"
        )
    print(args.out)


def _report(args: argparse.Namespace) -> None:
    records = []
    for path in args.records:
        records += read_records(path)
    for path in args.tileinfer_decode or []:
        records += tileinfer.decode_records(path, sampling=args.tileinfer_sampling)
    for path in args.tileinfer_generate or []:
        records += tileinfer.generate_records(path)
    print(report.report(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="HF checkpoint -> BF16 GGUF, with checks")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.set_defaults(run=_convert)

    p = sub.add_parser("bench", help="llama.cpp fixed-context decode and generation")
    _add_runtime_args(p)
    p.add_argument("--positions", type=int, nargs="+", default=[0, 63, 255, 1023])
    p.add_argument("--inputs", type=Path, required=True, help='JSON {"tokens": [...]}')
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--llama-bench", action="store_true", help="also run llama-bench")
    p.add_argument("--out", type=Path, default=RESULTS / "llamacpp.jsonl")
    p.set_defaults(run=_bench)

    p = sub.add_parser("parity", help="greedy tokens and top logit vs Transformers")
    _add_runtime_args(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument(
        "--device", default="cpu", help="device of the Transformers reference"
    )
    p.add_argument("--out", type=Path, default=RESULTS / "parity.json")
    p.set_defaults(run=_parity)

    p = sub.add_parser("report", help="merge records into markdown tables")
    p.add_argument("records", type=Path, nargs="*")
    p.add_argument(
        "--tileinfer-decode", type=Path, nargs="*", help="benchmark_decode.py reports"
    )
    p.add_argument(
        "--tileinfer-generate",
        type=Path,
        nargs="*",
        help="benchmark_generate.py reports",
    )
    p.add_argument("--tileinfer-sampling", choices=("none", "greedy"), default="none")
    p.set_defaults(run=_report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
