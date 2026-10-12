"""The fixed float32 MLP accepted by plan format version 1."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHT_NAMES = ("W1", "b1", "W2", "b2")


def weight_shapes(d: int, h: int, o: int) -> dict[str, tuple[int, ...]]:
    return {"W1": (h, d), "b1": (h,), "W2": (o, h), "b2": (o,)}


@dataclass(frozen=True)
class Model:
    d: int
    h: int
    o: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n <= 0 for n in (self.d, self.h, self.o)):
            raise ValueError("D, H, and O must be positive integers")

    @classmethod
    def read(cls, path: Path) -> Model:
        data = json.loads(path.read_text())
        expected = {"schema_version", "kind", "dtype", "activation", "D", "H", "O"}
        if set(data) != expected or (
            data["schema_version"],
            data["kind"],
            data["dtype"],
            data["activation"],
        ) != (1, "mlp_f32", "float32", "gelu_tanh"):
            raise ValueError("unsupported model schema or operation")
        return cls(data["D"], data["H"], data["O"])

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "mlp_f32",
            "dtype": "float32",
            "activation": "gelu_tanh",
            "D": self.d,
            "H": self.h,
            "O": self.o,
        }


def read_weights(path: Path, model: Model) -> dict[str, np.ndarray]:
    shapes = weight_shapes(model.d, model.h, model.o)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(WEIGHT_NAMES):
            raise ValueError("weights must contain exactly W1, b1, W2, b2")
        result = {}
        for name in WEIGHT_NAMES:
            value = archive[name]
            if (
                value.shape != shapes[name]
                or value.dtype != np.dtype("float32")
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"invalid {name}: expected finite float32 {shapes[name]}"
                )
            result[name] = np.ascontiguousarray(value, dtype="<f4")
    return result
