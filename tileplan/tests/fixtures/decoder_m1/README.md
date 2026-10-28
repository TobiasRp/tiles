# Decoder milestone 1 fixture

The files in this directory freeze the CPU PyTorch oracle for one synthetic,
batch-one Llama-style block. The model has d=32, f=64, four query heads, two KV
heads, head width eight, and a six-token context. The five-token test sequence
starts with a zero hidden vector. Weights and tokens use seed 19.

`model.json` records the RMSNorm epsilon, float32 dtype, GQA head mapping,
split-half Llama 3 RoPE parameters, cache layout, and comparison tolerances.
`weights.npz` stores the nine bias-free projection and norm tensors in
PyTorch `[out, in]` layout. `reference_cases.npz` stores the input sequence,
per-position Q/K/V after RoPE, attention result, attention residual, FFN
contribution, final hidden vector, and a final one-layer K/V cache snapshot.
The cache has shape `[1, 6, 2, 8]`; only the first five positions are valid.

Regenerate from `tileplan/` with:

```sh
uv run --locked python -m tileplan reference --out tests/fixtures/decoder_m1
```

Remove the existing directory first when intentionally changing the contract.
`test_decoder_reference.py` checks each incremental result and cache prefix
against a separate full-prefix causal recomputation, and checks that the
committed fixture is reproducible.
