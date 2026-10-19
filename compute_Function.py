from __future__ import annotations

import numpy as np


SUPPORTED_FUNCTIONS = {
    "O_mp",
    "O_p",
    "G_Mp",
    "G_p",
}


def _o_mp(u: np.ndarray, v: np.ndarray, p: float) -> np.ndarray:
    """Compute the parameterized minimum overlap function."""
    return np.minimum(np.power(u, p), np.power(v, p))


def _o_p(u: np.ndarray, v: np.ndarray, p: float) -> np.ndarray:
    """Compute the parameterized product overlap function."""
    return np.power(u, p) * np.power(v, p)


def _g_Mp(u: np.ndarray, v: np.ndarray, p: float) -> np.ndarray:
    """Compute the parameterized maximum grouping function."""
    return np.maximum(np.power(u, p), np.power(v, p))


def _g_p(u: np.ndarray, v: np.ndarray, p: float) -> np.ndarray:
    """Compute the parameterized probabilistic grouping function."""
    return 1 - np.power(1 - u, p) * np.power(1 - v, p)


def functions_matrix(function: str, p: float, obj: np.ndarray) -> np.ndarray:
    """Apply the selected overlap or grouping function to a relation matrix."""
    obj = np.asarray(obj)
    if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:
        raise ValueError("obj must be a square two-dimensional matrix")
    if function not in SUPPORTED_FUNCTIONS:
        raise ValueError(f"Unsupported function: {function!r}")

    u = np.maximum(obj, 0.0)
    v = np.maximum(obj.T, 0.0)

    if function == "O_mp":
        result = _o_mp(u, v, p)
    elif function == "O_p":
        result = _o_p(u, v, p)
    elif function == "G_Mp":
        result = _g_Mp(u, v, p)
    else:
        result = _g_p(u, v, p)

    return np.asarray(result, dtype="float32")


__all__ = ["SUPPORTED_FUNCTIONS", "functions_matrix"]
