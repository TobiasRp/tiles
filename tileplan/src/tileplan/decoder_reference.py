"""CPU PyTorch oracle for one synthetic Llama-style decoder block.

This module is deliberately separate from the planner and CUDA executor. The
incremental path writes a KV cache; recompute_prefix evaluates all visible
tokens with a full causal attention matrix for an independent state check.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecoderSpec:
    hidden_size: int = 32
    intermediate_size: int = 64
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    max_context: int = 6
    rms_norm_eps: float = 1e-5
    rope_type: str = "llama3"
    rope_pairing: str = "split_half"
    rope_theta: float = 500000.0
    rope_factor: float = 32.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192
    kv_cache_layout: str = "[layers,max_context,kv_heads,head_dim]"
    dtype: str = "float32"
    atol: float = 2e-5
    rtol: float = 2e-5

    def __post_init__(self) -> None:
        if (
            self.hidden_size != self.num_attention_heads * self.head_dim
            or self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
            or min(
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.max_context,
                self.rope_original_max_position_embeddings,
            )
            <= 0
            or self.rms_norm_eps <= 0
            or self.rope_theta <= 0
            or self.rope_factor <= 0
            or not (0 < self.rope_low_freq_factor < self.rope_high_freq_factor)
            or self.rope_type != "llama3"
            or self.rope_pairing != "split_half"
            or self.kv_cache_layout != "[layers,max_context,kv_heads,head_dim]"
            or self.dtype != "float32"
        ):
            raise ValueError("unsupported decoder reference specification")

    @property
    def head_to_kv(self) -> list[int]:
        group_size = self.num_attention_heads // self.num_key_value_heads
        return [head // group_size for head in range(self.num_attention_heads)]

    def as_dict(self) -> dict:
        return {**asdict(self), "head_to_kv": self.head_to_kv}


WEIGHT_SHAPES = {
    "attn_norm": lambda s: (s.hidden_size,),
    "Wq": lambda s: (s.num_attention_heads * s.head_dim, s.hidden_size),
    "Wk": lambda s: (s.num_key_value_heads * s.head_dim, s.hidden_size),
    "Wv": lambda s: (s.num_key_value_heads * s.head_dim, s.hidden_size),
    "Wo": lambda s: (s.hidden_size, s.hidden_size),
    "ffn_norm": lambda s: (s.hidden_size,),
    "Wgate": lambda s: (s.intermediate_size, s.hidden_size),
    "Wup": lambda s: (s.intermediate_size, s.hidden_size),
    "Wdown": lambda s: (s.hidden_size, s.intermediate_size),
}


def fixed_weights(spec: DecoderSpec, seed: int = 19) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = {}
    for name, shape_fn in WEIGHT_SHAPES.items():
        shape = shape_fn(spec)
        if name.endswith("_norm"):
            weights[name] = 1 + 0.05 * torch.randn(
                shape, generator=generator, dtype=torch.float32
            )
        else:
            weights[name] = torch.randn(
                shape, generator=generator, dtype=torch.float32
            ) / math.sqrt(
                spec.intermediate_size if name == "Wdown" else spec.hidden_size
            )
    return weights


def fixed_tokens(spec: DecoderSpec, seed: int = 19) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return torch.cat(
        (
            torch.zeros((1, spec.hidden_size), dtype=torch.float32),
            torch.randn(
                (4, spec.hidden_size), generator=generator, dtype=torch.float32
            ),
        ),
        dim=0,
    )


def _check_inputs(
    spec: DecoderSpec, weights: dict[str, torch.Tensor], tokens: torch.Tensor
) -> None:
    if (
        tokens.dtype != torch.float32
        or tokens.ndim != 2
        or tokens.shape[1] != spec.hidden_size
    ):
        raise ValueError(
            "tokens must have shape [positions, hidden_size] and float32 dtype"
        )
    if not 0 < tokens.shape[0] <= spec.max_context:
        raise ValueError("token count exceeds context capacity")
    if set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("decoder weight names do not match the specification")
    for name, shape_fn in WEIGHT_SHAPES.items():
        weight = weights[name]
        if weight.shape != shape_fn(spec) or weight.dtype != torch.float32:
            raise ValueError(f"invalid {name} shape or dtype")


def rope_inverse_frequencies(spec: DecoderSpec) -> torch.Tensor:
    indices = torch.arange(0, spec.head_dim, 2, dtype=torch.float32)
    inv = 1.0 / (spec.rope_theta ** (indices / spec.head_dim))
    wavelength = 2 * math.pi / inv
    low_wavelength = (
        spec.rope_original_max_position_embeddings / spec.rope_low_freq_factor
    )
    high_wavelength = (
        spec.rope_original_max_position_embeddings / spec.rope_high_freq_factor
    )
    smooth = (
        spec.rope_original_max_position_embeddings / wavelength
        - spec.rope_low_freq_factor
    ) / (spec.rope_high_freq_factor - spec.rope_low_freq_factor)
    scaled = torch.where(wavelength > low_wavelength, inv / spec.rope_factor, inv)
    medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
    blended = (1 - smooth) * scaled / spec.rope_factor + smooth * scaled
    return torch.where(medium, blended, scaled)


def apply_rope(
    values: torch.Tensor, positions: torch.Tensor, spec: DecoderSpec
) -> torch.Tensor:
    """Rotate [heads, head_dim] or [positions, heads, head_dim] in HF split halves."""
    angles = positions.to(torch.float32).unsqueeze(-1) * rope_inverse_frequencies(spec)
    cosine = angles.cos().unsqueeze(-2)
    sine = angles.sin().unsqueeze(-2)
    first, second = values.chunk(2, dim=-1)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    )


def _rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * scale


@dataclass
class ReferenceCache:
    k: torch.Tensor
    v: torch.Tensor
    position: int = 0

    @classmethod
    def empty(cls, spec: DecoderSpec) -> ReferenceCache:
        shape = (1, spec.max_context, spec.num_key_value_heads, spec.head_dim)
        return cls(
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    def reset(self) -> None:
        self.position = 0


def decode_one(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    spec: DecoderSpec,
    cache: ReferenceCache,
) -> dict[str, torch.Tensor]:
    """Append one token; return intermediate values for later GPU comparison."""
    if x.shape != (spec.hidden_size,) or x.dtype != torch.float32:
        raise ValueError("expected one float32 hidden vector")
    if cache.position >= spec.max_context:
        raise ValueError("context capacity reached")
    expected_shape = (1, spec.max_context, spec.num_key_value_heads, spec.head_dim)
    if cache.k.shape != expected_shape or cache.v.shape != expected_shape:
        raise ValueError("KV cache shape differs from the specification")
    p = cache.position
    u = _rms_norm(x, weights["attn_norm"], spec.rms_norm_eps)
    q = (weights["Wq"] @ u).reshape(spec.num_attention_heads, spec.head_dim)
    k = (weights["Wk"] @ u).reshape(spec.num_key_value_heads, spec.head_dim)
    v = (weights["Wv"] @ u).reshape(spec.num_key_value_heads, spec.head_dim)
    position = torch.tensor(p, dtype=torch.float32)
    q = apply_rope(q, position, spec)
    k = apply_rope(k, position, spec)
    cache.k[0, p] = k
    cache.v[0, p] = v
    head_to_kv = spec.head_to_kv
    keys = cache.k[0, : p + 1, head_to_kv]
    values = cache.v[0, : p + 1, head_to_kv]
    scores = torch.einsum("hd,thd->ht", q, keys) / math.sqrt(spec.head_dim)
    probabilities = torch.softmax(scores, dim=-1)
    attention = torch.einsum("ht,thd->hd", probabilities, values).reshape(
        spec.hidden_size
    )
    r = x + weights["Wo"] @ attention
    z = _rms_norm(r, weights["ffn_norm"], spec.rms_norm_eps)
    gate = weights["Wgate"] @ z
    up = weights["Wup"] @ z
    ffn = weights["Wdown"] @ (F.silu(gate) * up)
    y = r + ffn
    cache.position = p + 1
    return {
        "q": q,
        "k": k,
        "v": v,
        "attention": attention,
        "attn_residual": r,
        "ffn": ffn,
        "y": y,
    }


def recompute_prefix(
    tokens: torch.Tensor, weights: dict[str, torch.Tensor], spec: DecoderSpec
) -> dict[str, torch.Tensor]:
    """Full causal block evaluation; never reads or writes the incremental cache."""
    _check_inputs(spec, weights, tokens)
    length = tokens.shape[0]
    u = _rms_norm(tokens, weights["attn_norm"], spec.rms_norm_eps)
    q = (u @ weights["Wq"].T).reshape(length, spec.num_attention_heads, spec.head_dim)
    k = (u @ weights["Wk"].T).reshape(length, spec.num_key_value_heads, spec.head_dim)
    v = (u @ weights["Wv"].T).reshape(length, spec.num_key_value_heads, spec.head_dim)
    positions = torch.arange(length, dtype=torch.float32)
    q = apply_rope(q, positions, spec)
    k = apply_rope(k, positions, spec)
    keys = k[:, spec.head_to_kv]
    values = v[:, spec.head_to_kv]
    scores = torch.einsum("thd,shd->ths", q, keys) / math.sqrt(spec.head_dim)
    future = torch.triu(torch.ones((length, length), dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future[:, None, :], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    attention = torch.einsum("ths,shd->thd", probabilities, values).reshape(
        length, spec.hidden_size
    )
    r = tokens + attention @ weights["Wo"].T
    z = _rms_norm(r, weights["ffn_norm"], spec.rms_norm_eps)
    ffn = (F.silu(z @ weights["Wgate"].T) * (z @ weights["Wup"].T)) @ weights["Wdown"].T
    return {
        "q": q,
        "k": k,
        "v": v,
        "attention": attention,
        "attn_residual": r,
        "ffn": ffn,
        "y": r + ffn,
    }


def generate_fixture(
    out: Path, spec: DecoderSpec = DecoderSpec(), seed: int = 19
) -> None:
    """Write deterministic model, weights, tokens, traces, and cache snapshots."""
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    weights = fixed_weights(spec, seed)
    tokens = fixed_tokens(spec, seed)
    _check_inputs(spec, weights, tokens)
    cache = ReferenceCache.empty(spec)
    with torch.inference_mode():
        trace = [decode_one(x, weights, spec, cache) for x in tokens]
    (out / "model.json").write_text(
        json.dumps(
            {
                "kind": "llama_decoder_block_reference",
                "version": 1,
                "seed": seed,
                **spec.as_dict(),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    np.savez(
        out / "weights.npz", **{name: value.numpy() for name, value in weights.items()}
    )
    np.savez(
        out / "reference_cases.npz",
        x=tokens.numpy(),
        position=np.arange(len(tokens), dtype=np.int64),
        k_cache=cache.k.numpy(),
        v_cache=cache.v.numpy(),
        q=torch.stack([step["q"] for step in trace]).numpy(),
        k=torch.stack([step["k"] for step in trace]).numpy(),
        v=torch.stack([step["v"] for step in trace]).numpy(),
        attention=torch.stack([step["attention"] for step in trace]).numpy(),
        attn_residual=torch.stack([step["attn_residual"] for step in trace]).numpy(),
        ffn=torch.stack([step["ffn"] for step in trace]).numpy(),
        y=torch.stack([step["y"] for step in trace]).numpy(),
    )
