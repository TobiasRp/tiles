"""Compare autoregressive generation with Transformers eager and CUDA graph replay."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, StaticCache


def reference(
    checkpoint: Path,
    prompt: list[int],
    count: int,
    warmup: int,
    rounds: int,
    weight_dtype: str,
) -> dict:
    torch.backends.cuda.preferred_blas_library("cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if weight_dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=dtype,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    device = torch.device("cuda:0")
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
    positions = [
        torch.tensor([len(prompt) + index], dtype=torch.long, device=device)
        for index in range(count)
    ]
    masks = []
    for index in range(count):
        mask = torch.full(
            (1, 1, 1, len(prompt) + count), float("-inf"), device=device, dtype=dtype
        )
        mask[..., : len(prompt) + index + 1] = 0
        masks.append(mask)
    cache = StaticCache(model.config, max_cache_len=len(prompt) + count)
    prefix_logits = torch.empty(
        (1, 1, model.config.vocab_size), device=device, dtype=dtype
    )

    def prefill() -> None:
        cache.reset()
        # Match the executor's one-token prefix shapes; setup is not timed.
        for position in range(len(prompt)):
            logits = model(
                input_ids=prompt_tensor[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=device),
                logits_to_keep=1,
            ).logits
        prefix_logits.copy_(logits)

    def generate() -> torch.Tensor:
        logits = prefix_logits
        selected = []
        for index in range(count):
            token = logits[:, -1, :].argmax(dim=-1).view(1, 1)
            selected.append(token)
            logits = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                cache_position=positions[index],
                attention_mask=masks[index],
                logits_to_keep=1,
            ).logits
        return torch.cat(selected, dim=1)

    def measure(call) -> dict:
        samples = []
        expected = None
        for round_index in range(-warmup, rounds):
            prefill()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = call()
            end.record()
            end.synchronize()
            tokens = output.cpu().flatten().tolist()
            if expected is None:
                expected = tokens
            elif tokens != expected:
                raise AssertionError(
                    "Transformers generated different IDs across rounds"
                )
            if round_index >= 0:
                samples.append(start.elapsed_time(end) * 1000)
        mean = statistics.mean(samples)
        return {
            "generated": expected,
            "samples_us": samples,
            "mean_us": mean,
            "us_per_token": mean / count,
            "tokens_per_second": count * 1_000_000 / mean,
        }

    with torch.inference_mode():
        eager = measure(generate)
        # Initialize cache storage and library workspaces before capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                prefill()
                generate()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        prefill()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = generate()

        def replay() -> torch.Tensor:
            graph.replay()
            return graph_output

        graphed = measure(replay)
    if eager["generated"] != graphed["generated"]:
        raise AssertionError("eager and graph replay generated different IDs")
    return {
        "eager": eager,
        "graph": graphed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "weight_dtype": weight_dtype,
        "activation_dtype": weight_dtype,
        "accumulation_dtype": "float32",
        "bf16_reduction": False,
        "split_k": False,
        "blas_library": "cublaslt",
        "tf32": False,
        "baseline": "Transformers eager attention, StaticCache, GPU argmax feedback",
        "graph_scope": "complete selection/decode loop for all requested tokens",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-dtype",
        choices=("float32", "bfloat16"),
        default=None,
        help="reference precision (defaults to the package policy)",
    )
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    prompt = json.loads(args.inputs.read_text())["tokens"]
    if not prompt or not all(isinstance(token, int) for token in prompt):
        parser.error("inputs must contain a nonempty integer token list")
    plan = json.loads((args.package / "plan.json").read_text())
    weight_dtype = plan["model"]["weight_dtype"]
    if args.reference_dtype is None:
        args.reference_dtype = weight_dtype
    if args.reference_dtype != weight_dtype:
        parser.error("reference precision must match the package")
    if weight_dtype not in {"float32", "bfloat16"}:
        parser.error("unsupported package weight dtype")
    if (
        args.new_tokens <= 0
        or len(prompt) + args.new_tokens > plan["model"]["max_context"]
    ):
        parser.error("generation exceeds package capacity")
    if args.warmup < 0 or args.rounds <= 0:
        parser.error("invalid warmup or rounds")
    reference_path = args.out.with_suffix(".reference.json")
    if args.reference_child:
        result = reference(
            args.checkpoint,
            prompt,
            args.new_tokens,
            args.warmup,
            args.rounds,
            args.reference_dtype,
        )
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(result, indent=2) + "\n")
        return
    child_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        str(args.checkpoint),
        "--package",
        str(args.package),
        "--executor",
        str(args.executor),
        "--inputs",
        str(args.inputs),
        "--new-tokens",
        str(args.new_tokens),
        "--warmup",
        str(args.warmup),
        "--rounds",
        str(args.rounds),
        "--out",
        str(args.out),
        "--reference-dtype",
        args.reference_dtype,
        "--reference-child",
    ]
    subprocess.run(child_command, check=True)
    baseline = json.loads(reference_path.read_text())
    runtime = json.loads(
        subprocess.check_output(
            [
                str(args.executor),
                "bench",
                str(args.package),
                str(args.inputs),
                "--new-tokens",
                str(args.new_tokens),
                "--warmup",
                str(args.warmup),
                "--rounds",
                str(args.rounds),
            ],
            text=True,
        )
    )
    matches = runtime["generated"] == baseline["eager"]["generated"]
    if not matches:
        raise AssertionError(
            "generated IDs differ from the matching-precision reference"
        )
    report = {
        "prompt_tokens": len(prompt),
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "reference": baseline,
        "runtime": runtime,
        "runtime_activation_dtype": weight_dtype,
        "runtime_accumulation_dtype": "float32",
        "matrix_kernel": plan["model"].get("matrix_kernel", "simt"),
        "generated_ids_match": matches,
        "timing": "CUDA events around GPU selection and decode; prefill and readback excluded",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for mode in ("eager", "graph"):
        print(
            f"PyTorch {args.reference_dtype} {mode}: "
            f"{baseline[mode]['us_per_token'] / 1000:.3f} ms/token"
        )
    print(
        f"TileInfer {weight_dtype} weights/activations, FP32 accumulation: "
        f"{runtime['us_per_token'] / 1000:.3f} ms/token; IDs match: {matches}"
    )
    print(args.out)


if __name__ == "__main__":
    main()
