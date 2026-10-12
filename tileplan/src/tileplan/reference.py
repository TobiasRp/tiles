"""Independent PyTorch reference fixture, created on the CPU."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Model


def generate(out: Path, model: Model, seed: int = 7) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.manual_seed(seed)
    with torch.inference_mode():
        w1 = torch.randn(model.h, model.d, dtype=torch.float32) / model.d**0.5
        b1 = torch.randn(model.h, dtype=torch.float32) / 4
        w2 = torch.randn(model.o, model.h, dtype=torch.float32) / model.h**0.5
        b2 = torch.randn(model.o, dtype=torch.float32) / 4
        x = torch.stack((torch.zeros(model.d), torch.randn(model.d), torch.randn(model.d))).to(torch.float32)
        hidden = F.gelu(x @ w1.T + b1, approximate="tanh")
        output = hidden @ w2.T + b2
    (out / "model.json").write_text(json.dumps(model.as_dict(), sort_keys=True, indent=2) + "\n")
    np.savez(out / "weights.npz", W1=w1.numpy(), b1=b1.numpy(), W2=w2.numpy(), b2=b2.numpy())
    np.savez(out / "reference_cases.npz", x=x.numpy(), h=hidden.numpy(), y=output.numpy(), seed=np.int64(seed))
