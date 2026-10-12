"""Numerical comparison shared by the eventual executor bridge and baselines."""

import numpy as np


ATOL = 2e-4
RTOL = 2e-4


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} versus {expected.shape}")
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = ATOL + RTOL * np.abs(expected.astype(np.float64))
    finite = bool(np.isfinite(actual).all())
    if not finite:
        return {"pass": False, "max_absolute_error": None, "max_normalized_error": None,
                "atol": ATOL, "rtol": RTOL}
    return {"pass": finite and bool(np.all(error <= allowed)),
            "max_absolute_error": float(error.max()),
            "max_normalized_error": float((error / allowed).max()),
            "atol": ATOL, "rtol": RTOL}
