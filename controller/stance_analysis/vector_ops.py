from __future__ import annotations

from typing import Iterable

import numpy as np


def to_np(vec: list[float] | np.ndarray, *, dtype=np.float32) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        return vec.astype(dtype, copy=False)
    return np.asarray(vec, dtype=dtype)


def mean_vector(vectors: Iterable[list[float] | np.ndarray]) -> list[float]:
    arr = np.asarray([to_np(v) for v in vectors], dtype=np.float32)
    if arr.size == 0:
        return []
    return arr.mean(axis=0).astype(np.float32, copy=False).tolist()


def l2_normalize(vec: list[float] | np.ndarray, *, eps: float = 1e-12) -> list[float]:
    v = to_np(vec)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.tolist()
    return (v / n).astype(np.float32, copy=False).tolist()


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray, *, eps: float = 1e-12) -> float:
    va = to_np(a)
    vb = to_np(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < eps:
        return 0.0
    return float(np.dot(va, vb) / denom)


def dot_similarity_normalized(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Fast similarity for vectors that are already L2-normalized.

    If both vectors are unit-length, cosine(a, b) == dot(a, b).
    """
    va = to_np(a)
    vb = to_np(b)
    return float(np.dot(va, vb))


def add_scaled(acc: list[float], vec: list[float], scale: float) -> list[float]:
    a = to_np(acc)
    v = to_np(vec)
    return (a + v * float(scale)).astype(np.float32, copy=False).tolist()


def sub_scaled(acc: list[float], vec: list[float], scale: float) -> list[float]:
    a = to_np(acc)
    v = to_np(vec)
    return (a - v * float(scale)).astype(np.float32, copy=False).tolist()


def add_scaled_np(acc: list[float] | np.ndarray, vec: list[float] | np.ndarray, scale: float) -> np.ndarray:
    a = to_np(acc)
    v = to_np(vec)
    return (a + v * float(scale)).astype(np.float32, copy=False)


def sub_scaled_np(acc: list[float] | np.ndarray, vec: list[float] | np.ndarray, scale: float) -> np.ndarray:
    a = to_np(acc)
    v = to_np(vec)
    return (a - v * float(scale)).astype(np.float32, copy=False)


def l2_normalize_np(vec: list[float] | np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    v = to_np(vec)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)
