"""Fixed-context decode latency for TileInfer and Transformers eager/CUDA Graph."""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, StaticCache


def _timed(call, iterations: int, rounds: int) -> list[float]:
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return samples


def _baseline(
    checkpoint: Path,
    positions: list[int],
    token: int,
    warmup: int,
    iterations: int,
    rounds: int,
    dtype: str,
) -> dict:
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_tf32 = False
    scalar = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=scalar,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if next(model.parameters()).dtype != scalar:
        raise RuntimeError("reference weights differ from the package precision")
    rows = []
    device = torch.device("cuda:0")
    token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
    with torch.inference_mode():
        for p in positions:
            cache = StaticCache(model.config, max_cache_len=p + 1)
            if p:
                # Prefix setup is outside the timed region and matches the
                # executor's sequential one-token calls.
                for prefix_position in range(p):
                    model(
                        input_ids=token_tensor,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.tensor([prefix_position], device=device),
                        logits_to_keep=1,
                    )
                torch.cuda.synchronize()
            position = torch.tensor([p], dtype=torch.long, device=device)
            fixed_mask = torch.zeros((1, 1, 1, p + 1), device=device, dtype=scalar)

            def forward():
                return model(
                    input_ids=token_tensor,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=position,
                    attention_mask=fixed_mask,
                    logits_to_keep=1,
                ).logits

            for _ in range(warmup):
                eager_logits = forward()
            torch.cuda.synchronize()
            eager_samples = _timed(forward, iterations, rounds)
            eager_logits = forward().detach().clone()
            torch.cuda.synchronize()

            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    forward()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_logits = forward()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.allclose(graph_logits, eager_logits, atol=1e-3, rtol=1e-4):
                raise AssertionError(f"graph/eager logits differ at position {p}")
            graph_samples = _timed(graph.replay, iterations, rounds)
            rows.append(
                {
                    "position": p,
                    "context_length": p + 1,
                    "eager_samples_us": eager_samples,
                    "eager_mean_us": statistics.mean(eager_samples),
                    "graph_samples_us": graph_samples,
                    "graph_mean_us": statistics.mean(graph_samples),
                    "next_token": int(eager_logits.argmax()),
                    "top_logit": float(eager_logits.max()),
                }
            )
            del eager_logits, graph, graph_logits
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "torch_version": torch.__version__,
        "transformers_baseline": f"LlamaForCausalLM {dtype} eager attention, StaticCache, logits_to_keep=1",
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "tf32": False,
        "activation_dtype": dtype,
        "accumulation_dtype": "float32",
        "bf16_reduction": False,
        "split_k": False,
        "blas_library": "cublaslt",
        "rows": rows,
    }
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 63, 255, 1023])
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.positions or sorted(set(args.positions)) != args.positions:
        parser.error("positions must be strictly increasing")
    if min(args.positions) < 0 or min(args.warmup, args.iterations, args.rounds) < 0:
        parser.error("negative argument")
    if not args.iterations or not args.rounds:
        parser.error("iterations and rounds must be positive")
    package = json.loads((args.package / "plan.json").read_text())
    max_context = package["model"]["max_context"]
    if max(args.positions) >= max_context:
        parser.error(f"position must be below package max_context {max_context}")
    baseline_path = args.out.with_suffix(".baseline.json")
    if args.baseline_child:
        baseline = _baseline(
            args.checkpoint,
            args.positions,
            args.token_id,
            args.warmup,
            args.iterations,
            args.rounds,
            package["model"]["weight_dtype"],
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
        return
    # The reference exits before TileInfer allocates its own GPU weights.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--checkpoint",
            str(args.checkpoint),
            "--package",
            str(args.package),
            "--executor",
            str(args.executor),
            "--positions",
            *map(str, args.positions),
            "--token-id",
            str(args.token_id),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--rounds",
            str(args.rounds),
            "--out",
            str(args.out),
            "--baseline-child",
        ],
        check=True,
    )
    baseline = json.loads(baseline_path.read_text())
    runtime = []
    for p in args.positions:
        output = subprocess.check_output(
            [
                str(args.executor),
                "--benchmark-stack",
                str(args.package),
                str(p),
                str(args.token_id),
                str(args.warmup),
                str(args.iterations),
                str(args.rounds),
            ],
            text=True,
        )
        tile = json.loads(output)
        reference = baseline["rows"][len(runtime)]
        if tile["top_token"] != reference["next_token"]:
            raise AssertionError(f"next token mismatch at position {p}")
        atol, rtol = (
            (0.15, 0.03)
            if package["model"]["weight_dtype"] == "bfloat16"
            else (0.001, 0.0001)
        )
        limit = atol + rtol * abs(reference["top_logit"])
        if abs(tile["top_logit"] - reference["top_logit"]) > limit:
            raise AssertionError(f"top logit mismatch at position {p}")
        runtime.append(tile)
    report = {
        "positions": args.positions,
        "token_id": args.token_id,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "baseline": baseline,
        "runtime": runtime,
        "timing": "CUDA event batch average, GPU resident input/cache/weights; setup excluded",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for pytorch, tile in zip(baseline["rows"], runtime, strict=True):
        print(
            f"length={pytorch['context_length']:5d}  "
            f"eager={pytorch['eager_mean_us']:10.1f} us  "
            f"graph={pytorch['graph_mean_us']:10.1f} us  "
            f"tile={tile['mean_us']:10.1f} us"
        )
    print(args.out)


if __name__ == "__main__":
    main()
