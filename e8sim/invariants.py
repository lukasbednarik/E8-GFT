"""Symmetric-power invariant counting for the adjoint of E_8.

Provides integer-partition enumeration, SVD-based numerical rank, and
the expected Hilbert series from Bourbaki Plate VII (Casimir degrees
{2, 8, 12, 14, 18, 20, 24, 30}).
"""

from __future__ import annotations

import numpy as np


HILBERT_SERIES_E8: dict[int, int] = {
    2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2,
}
"""dim Inv(Sym^n adj E_8) for n = 2..8 (Bourbaki Plate VII)."""


def partitions(n: int, min_part: int = 2) -> list[tuple[int, ...]]:
    """All integer partitions of *n* with every part >= *min_part*.

    Default ``min_part=2`` matches the Newton–Girard basis relevant for
    Casimir invariant counting (Tr(ad^k) with k >= 2).
    """
    if n == 0:
        return [tuple()]
    if n < min_part:
        return []
    out: list[tuple[int, ...]] = []
    for first in range(min_part, n + 1):
        for rest in partitions(n - first, min_part=first):
            out.append((first,) + rest)
    return out


def numerical_rank(
    samples: np.ndarray,
    abs_threshold: float | None = None,
    rank_tol: float = 1e-9,
    *,
    return_singular_values: bool = False,
) -> int | tuple[int, np.ndarray]:
    """SVD-based numerical rank with optional absolute column filtering.

    Parameters
    ----------
    samples : (n_samples, n_candidates) 2-D array
    abs_threshold : float or None
        Columns whose max-abs is below this are treated as zero.
    rank_tol : float
        Relative singular-value cutoff (w.r.t. the largest SV).
    return_singular_values : bool
        If ``True`` return ``(rank, singular_values)`` instead of ``rank``.
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D")
    col_scale = np.max(np.abs(samples), axis=0)
    if abs_threshold is not None:
        keep = col_scale > abs_threshold
        if not keep.any():
            sv = np.zeros(samples.shape[1])
            return (0, sv) if return_singular_values else 0
        filtered = samples[:, keep]
    else:
        filtered = samples
    scale = np.maximum(np.max(np.abs(filtered), axis=0), 1e-30)
    norm = filtered / scale
    sv = np.linalg.svd(norm, compute_uv=False)
    rank = int(np.sum(sv > rank_tol * sv[0])) if sv.size > 0 else 0
    return (rank, sv) if return_singular_values else rank
