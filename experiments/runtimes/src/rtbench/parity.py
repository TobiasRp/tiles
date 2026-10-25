"""Token and logit parity of a GGUF run against the Transformers reference.

The reference runs on whatever device is available (CPU on the Mac, CUDA on
tux) with BF16 weights and eager attention, one token at a time, like the
TileInfer checkpoint tests.
"""

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

# Frozen prompts from tileinfer/tests/llama32_checkpoint.py.
PROMPTS = (
    "Hello, my name is",
    "The capital of France is",
    "1, 2, 3,",
)
# Same BF16 rule as tileinfer/tests/benchmark_decode.py for the top logit.
LOGIT_ATOL = 0.15
LOGIT_RTOL = 0.03


@dataclass
class Reference:
    ids: list[int]
    tokens: list[int]
    top_logits: list[float]
    gaps: list[float]  # top-1 minus top-2 logit at each step


def tokenize(checkpoint: Path, prompts: tuple[str, ...] = PROMPTS) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return [tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]


def reference_generate(
    checkpoint: Path, prompts_ids: list[list[int]], new_tokens: int, device: str = "cpu"
) -> list[Reference]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    results = []
    with torch.inference_mode():
        for ids in prompts_ids:
            past = None
            logits = None
            for token in ids:
                out = model(
                    input_ids=torch.tensor([[token]], device=device),
                    past_key_values=past,
                    use_cache=True,
                )
                past, logits = out.past_key_values, out.logits[0, -1].float()
            if logits is None:
                raise ValueError("empty prompt")
            tokens, top, gaps = [], [], []
            for _ in range(new_tokens):
                best = logits.topk(2)
                index = best.indices[0]
                tokens.append(int(index))
                top.append(float(best.values[0]))
                gaps.append(float(best.values[0] - best.values[1]))
                out = model(
                    input_ids=torch.tensor([[int(index)]], device=device),
                    past_key_values=past,
                    use_cache=True,
                )
                past, logits = out.past_key_values, out.logits[0, -1].float()
            results.append(Reference(ids, tokens, top, gaps))
    return results


@dataclass
class Comparison:
    """Teacher-forced comparison: llama.cpp saw the reference tokens, so every
    step has the same context and can be compared on its own."""

    prompt_index: int
    steps: int
    # Steps where llama.cpp's argmax differs from the reference's, split by
    # whether the reference's top-2 logit gap is within the logit tolerance
    # (a BF16 near-tie that either side may break either way).
    tie_mismatches: list[int]
    real_mismatches: list[int]
    max_logit_error: float  # top logit, over all steps
    logits_within_tolerance: bool


def compare(
    reference: Reference, tokens: list[int], top_logits: list[float]
) -> Comparison:
    n = min(len(reference.tokens), len(tokens))
    mismatches = [i for i in range(n) if reference.tokens[i] != tokens[i]]
    errors = [abs(reference.top_logits[i] - top_logits[i]) for i in range(n)]
    return Comparison(
        prompt_index=-1,
        steps=n,
        tie_mismatches=[i for i in mismatches if reference.gaps[i] <= LOGIT_ATOL],
        real_mismatches=[i for i in mismatches if reference.gaps[i] > LOGIT_ATOL],
        max_logit_error=max(errors, default=0.0),
        logits_within_tolerance=all(
            err <= LOGIT_ATOL + LOGIT_RTOL * abs(reference.top_logits[i])
            for i, err in enumerate(errors)
        ),
    )
