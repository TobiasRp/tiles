"""Compare a packed Llama 3.2 1B package with float32 Transformers inference."""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "1, 2, 3,",
)


def _reference(checkpoint: Path) -> tuple[list[list[int]], list[dict], list[str]]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if next(model.parameters()).dtype != torch.float32:
        raise AssertionError(
            "Transformers did not convert checkpoint weights to float32"
        )
    layer_outputs: list[np.ndarray] = []

    def capture(_module: object, _inputs: object, output: object) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("decoder layer did not return a tensor")
        layer_outputs.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    handles = [layer.register_forward_hook(capture) for layer in model.model.layers]
    sequences: list[list[int]] = []
    records: list[dict] = []
    generated: list[str] = []
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
                past = None
                sequence = []
                logits = None
                for token in ids:
                    layer_outputs.clear()
                    outputs = model(
                        input_ids=torch.tensor([[token]], device="cuda"),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = outputs.past_key_values
                    logits = outputs.logits[0, -1].float().cpu().numpy().copy()
                    position = len(sequence)
                    records.append(
                        {
                            "token_id": token,
                            "position": position,
                            "logits": logits,
                            "blocks": np.stack(layer_outputs),
                            "k": np.stack(
                                [
                                    layer.keys[0, :, position, :].float().cpu().numpy()
                                    for layer in past.layers
                                ]
                            ),
                            "v": np.stack(
                                [
                                    layer.values[0, :, position, :]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    for layer in past.layers
                                ]
                            ),
                        }
                    )
                    sequence.append(token)
                    del outputs
                if logits is None:
                    raise AssertionError("empty prompt")
                new_tokens = []
                for _ in range(2):
                    token = int(np.argmax(logits))
                    new_tokens.append(token)
                    layer_outputs.clear()
                    outputs = model(
                        input_ids=torch.tensor([[token]], device="cuda"),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = outputs.past_key_values
                    logits = outputs.logits[0, -1].float().cpu().numpy().copy()
                    position = len(sequence)
                    records.append(
                        {
                            "token_id": token,
                            "position": position,
                            "logits": logits,
                            "blocks": np.stack(layer_outputs),
                            "k": np.stack(
                                [
                                    layer.keys[0, :, position, :].float().cpu().numpy()
                                    for layer in past.layers
                                ]
                            ),
                            "v": np.stack(
                                [
                                    layer.values[0, :, position, :]
                                    .float()
                                    .cpu()
                                    .numpy()
                                    for layer in past.layers
                                ]
                            ),
                        }
                    )
                    sequence.append(token)
                    del outputs
                sequences.append(sequence)
                generated.append(tokenizer.decode(new_tokens))
                del past
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return sequences, records, generated


def _compare(name: str, actual: np.ndarray, expected: np.ndarray) -> dict:
    delta = np.abs(actual - expected)
    limit = 0.001 + 0.0001 * np.abs(expected)
    return {
        "max_abs": float(delta.max()),
        "max_scaled_error": float((delta / limit).max()),
        "within_tolerance": bool(np.all(delta <= limit)),
        "shape": list(actual.shape),
        "name": name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--reuse-runtime", action="store_true")
    args = parser.parse_args()
    input_path = args.out / "inputs.json"
    reference_path = args.out / "reference.npz"
    if args.reference_child:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.mkdir(parents=True)
        sequences, reference, _generated = _reference(args.checkpoint)
        request: dict = {"trace_blocks": True, "tokens": []}
        for index, sequence in enumerate(sequences):
            if index:
                request["tokens"].append({"reset": True})
            request["tokens"].extend(sequence)
        input_path.write_text(json.dumps(request))
        np.savez(
            reference_path,
            token_ids=np.asarray(
                [step["token_id"] for step in reference], dtype=np.int32
            ),
            logits=np.stack([step["logits"] for step in reference]),
            blocks=np.stack([step["blocks"] for step in reference]),
            k=np.stack([step["k"] for step in reference]),
            v=np.stack([step["v"] for step in reference]),
        )
        return
    if not args.out.exists():
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
                "--out",
                str(args.out),
                "--reference-child",
            ],
            check=True,
        )
    request = json.loads(input_path.read_text())
    sequences = []
    sequence = []
    for item in request["tokens"]:
        if isinstance(item, dict):
            sequences.append(sequence)
            sequence = []
        else:
            sequence.append(item)
    sequences.append(sequence)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    generated = [tokenizer.decode(sequence[-2:]) for sequence in sequences]
    with np.load(reference_path) as source:
        reference = [
            {name: source[name][index] for name in source.files}
            for index in range(len(source["token_ids"]))
        ]
    for step in reference:
        step["token_id"] = int(step.pop("token_ids"))
    if [step["token_id"] for step in reference] != [
        token for sequence in sequences for token in sequence
    ]:
        raise AssertionError("saved reference token sequence differs")
    result_path = args.out / "runtime.json"
    if not args.reuse_runtime:
        with result_path.open("w") as stream:
            result = subprocess.run(
                [str(args.executor), "--run-stack", str(args.package), str(input_path)],
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"TileInfer failed: {result.stderr.strip()}")
    runtime = json.loads(result_path.read_text())
    actual = runtime["steps"]
    if len(actual) != len(reference):
        raise AssertionError("token count differs")
    checks = {}
    for name in ("blocks", "k", "v", "logits"):
        expected = np.stack([step[name] for step in reference])
        checks[name] = _compare(
            name,
            np.asarray([step[name] for step in actual], dtype=np.float32).reshape(
                expected.shape
            ),
            expected,
        )
    if [step["position"] for step in actual] != [
        position for sequence in sequences for position in range(len(sequence))
    ]:
        raise AssertionError("runtime position differs")
    top1 = [int(np.argmax(step["logits"])) for step in actual]
    expected_top1 = [int(np.argmax(step["logits"])) for step in reference]
    report = {
        "prompts": list(PROMPTS),
        "token_sequences": sequences,
        "generated": generated,
        "steps": len(actual),
        "checks": checks,
        "top1_matches": [a == b for a, b in zip(top1, expected_top1, strict=True)],
        "greedy_matches": all(a == b for a, b in zip(top1, expected_top1, strict=True)),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(check["within_tolerance"] for check in checks.values()):
        raise AssertionError("checkpoint comparison exceeded tolerance")
    if not report["greedy_matches"]:
        raise AssertionError("greedy token selection differs")


if __name__ == "__main__":
    main()
