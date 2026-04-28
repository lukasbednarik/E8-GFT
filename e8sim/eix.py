"""EIX = E_8 / (E_7 x SU(2)) coset constants and primitives.

Single source of truth for the dual Coxeter numbers, coset dimensions,
Schur factor c_H, the document bilinear kappa, and the canonical vacuum
direction V_A used throughout the debug-plan scripts E2--E9.

Convention (theory-wip-p2.md §0.1, theory-wip-p5.md §D.2):
    kappa(X, Y) := -K(X, Y) / h^v = 2 * (X, Y)_E
where K is the Killing form and (.,.)_E is the Euclidean inner product
in the orthonormal e8sim 248-basis.
"""

from __future__ import annotations

import numpy as np


# ─── Dual Coxeter numbers ──────────────────────────────────────────
H_VEE_E8: int = 30
H_VEE_E7: int = 18
H_VEE_SU2: int = 2

# ─── Coset dimensions ──────────────────────────────────────────────
DIM_E8: int = 248
DIM_E7: int = 133
DIM_SU2: int = 3
DIM_H_EIX: int = DIM_E7 + DIM_SU2       # 136
DIM_M_EIX: int = DIM_E8 - DIM_H_EIX     # 112  =  (56, 2)
DIM_H_PERP: int = DIM_H_EIX - 1         # 135  =  h \ R*V_A

# ─── Schur factor (corrected Lemma D.2.3, 4/2026) ─────────────────
C_H_EIX: float = (H_VEE_E8 - H_VEE_SU2) / DIM_M_EIX   # = 28/112 = 1/4

# ─── Bilinear form convention ──────────────────────────────────────
KAPPA_OVER_EUCLID: float = 2.0
"""kappa(X, Y) = KAPPA_OVER_EUCLID * (X, Y)_E = 2 * (X, Y)_E."""


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    """Document bilinear form kappa(X, Y) = 2 * (X, Y)_E."""
    return KAPPA_OVER_EUCLID * float(x @ y)


def canonical_VA() -> tuple[np.ndarray, float]:
    """Return the canonical SU(2)-Cartan vacuum direction and r_*^2.

    V_A = alpha_su2 / ||alpha_su2||_E  with alpha_su2 = (1,1,0,...,0),
    so (V_A, V_A)_E = 1 and r_*^2 := kappa(V_A, V_A) = 2.
    """
    V_A = np.zeros(DIM_E8, dtype=np.float64)
    V_A[:8] = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0]) / np.sqrt(2.0)
    r_sq = KAPPA_OVER_EUCLID * float(V_A @ V_A)
    return V_A, r_sq


# ─── Triality gauge equivalence (theory-wip-p6-do1.md Q3 γ) ────────
#
# Per theory-wip-p6-do1.md §Q3 (rezoluce 4/2026, varianta γ): the three
# triality SU(2) factors {SU(2)_v, SU(2)_s, SU(2)_c} attached to the
# triality root reps {α_v, α_s, α_c} are Ad-conjugate to the canonical
# F4' SU(2) (attached to EIX_ALPHA_SU2 = (1,1,0,...,0)) via explicit
# gauge elements g_i ∈ E_8 constructed from short Weyl chains.  These
# gauge elements are exposed here primarily for theoretical reference
# and for future Spin(8)-spinor extensions; the runtime hedgehog seed
# in `e8sim/fields.py` constructs each sector's hedgehog directly via
# `su2_basis_for(α_i)` (= equivalent up to internal SU(2) rotation,
# Q3 §Důsledek pro implementaci).


def _weyl_reflect(v: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """s_γ(v) = v − (v·γ)·γ for |γ|² = 2."""
    return v - float(v @ gamma) * gamma


def _find_weyl_path(start: np.ndarray, target: np.ndarray,
                    all_roots: np.ndarray, max_depth: int = 3,
                    beam: int = 12, tol: float = 1e-9) -> list | None:
    """BFS-with-beam search for a chain of Weyl reflections start → target."""
    if np.linalg.norm(start - target) < tol:
        return []

    queue = [(start, [])]
    while queue:
        v, path = queue.pop(0)
        if len(path) >= max_depth:
            continue
        candidates = []
        for r in all_roots:
            if abs(v @ r) < 1e-9:
                continue  # this reflection has no effect on v
            v_new = _weyl_reflect(v, r)
            d = float(np.linalg.norm(v_new - target))
            candidates.append((d, r, v_new))
        candidates.sort(key=lambda x: x[0])
        for d, r, v_new in candidates[:beam]:
            if d < tol:
                return path + [r]
            queue.append((v_new, path + [r]))
    return None


def _realize_weyl_reflection(gamma: np.ndarray, pos_roots: np.ndarray,
                              f_idx: np.ndarray, f_val: np.ndarray) -> np.ndarray:
    """Return the 248×248 Ad-matrix n_γ ∈ E_8 realising the Weyl reflection s_γ.

    Construction: n_γ := exp((π/√2)·U_γ), where U_γ is the unit basis
    vector at slot ``8 + idx_γ`` (E_{+γ} root generator).  For negative
    γ, slot ``128 + idx_{|γ|}`` is used.  See
    ``debug_plan/scripts/eo1_q3_triality_gauge.py:realize_weyl_reflection``
    for the verification (T-Q3.3 PASS, 4/2026).
    """
    from .algebra import adjoint_exp
    g = np.asarray(gamma, dtype=np.float64)
    norm_sq = float(g @ g)
    if abs(norm_sq - 2.0) > 1e-9:
        raise ValueError(
            f"_realize_weyl_reflection requires |γ|² = 2; got {norm_sq}"
        )

    idx_pos = None
    sign = +1
    for k in range(pos_roots.shape[0]):
        if np.allclose(g, pos_roots[k], atol=1e-9):
            idx_pos = k
            sign = +1
            break
        if np.allclose(g, -pos_roots[k], atol=1e-9):
            idx_pos = k
            sign = -1
            break
    if idx_pos is None:
        raise ValueError(f"γ = {g} is not a long root of E_8 in pos_roots")

    U_g = np.zeros(DIM_E8, dtype=np.float64)
    if sign > 0:
        U_g[8 + idx_pos] = 1.0
    else:
        U_g[128 + idx_pos] = 1.0
    theta = np.pi / np.sqrt(2.0)
    return adjoint_exp(theta * U_g, f_idx, f_val)


_TRIALITY_GAUGE_CACHE: dict[tuple, np.ndarray] = {}


def triality_gauge_element(sector: int, f_idx: np.ndarray, f_val: np.ndarray,
                           pos_roots: np.ndarray) -> np.ndarray:
    """Return the 248×248 Ad-matrix g_i ∈ E_8 mapping α_F4' → α_sector.

    For ``sector ∈ {1, 2, 3}`` (= 8v, 8s, 8c respectively in the
    ``TRIALITY_SECTOR_MAP`` convention), returns the explicit gauge
    element such that ``Ad(g_i)·V_F4' = α_sector/√2`` (T-Q3.3 PASS,
    error ~ 10⁻¹⁵).  The construction proceeds through:

    1. find a Weyl chain of depth ≤ 2 mapping ``EIX_ALPHA_SU2`` to the
       sector's representative long root (BFS over the 240 long roots);
    2. realise each reflection as ``n_γ = exp((π/√2)·E_{+γ}) ∈ E_8``
       via the adjoint exponential;
    3. compose the chain.

    First-call cost: ~0.3 s per sector (matrix exp of 248×248).  Result
    is cached on (sector, ``id(f_idx)``, ``id(f_val)``).

    Notes
    -----
    The runtime hedgehog seed in ``e8sim/fields.py:eix_hedgehog_seed``
    uses ``su2_basis_for(α_sector, pos_roots)`` directly (= construct
    the SU(2)_sector frame from scratch); this is equivalent up to an
    internal SO(3) rotation inside SU(2)_sector and is the
    operationally simpler choice (no expensive 248×248 matrix exp).
    The explicit ``g_i`` returned here is exposed for theoretical
    reference — e.g., to verify gauge-equivalence claims of
    theory-wip-p6-do1.md §Q3 γ in scripts, or to apply Ad(g_i) on
    spinor representations in future Spin(8)-extension work.

    All three sectors require a non-trivial gauge element: the source
    root ``EIX_ALPHA_SU2 = (1, 1, 0, …, 0)`` is itself a D4xD4-type
    long root and is *not* in the 8v8v sector, so even ``g_1`` is a
    composition of two Weyl reflections.  Both endpoints lie in the
    single 240-element Weyl orbit of long roots of E_8 (Bourbaki
    *Lie 4-6* §VI.1.11).
    """
    from .roots import (
        EIX_ALPHA_SU2,
        TRIALITY_SECTOR_MAP,
        find_triality_roots,
    )

    if sector not in (1, 2, 3):
        raise ValueError(f"sector must be 1, 2, or 3; got {sector}")

    cache_key = (sector, id(f_idx), id(f_val))
    if cache_key in _TRIALITY_GAUGE_CACHE:
        return _TRIALITY_GAUGE_CACHE[cache_key]

    sector_name = TRIALITY_SECTOR_MAP[sector]
    triality_reps = find_triality_roots(pos_roots)
    if sector_name not in triality_reps:
        raise RuntimeError(
            f"triality sector '{sector_name}' not found in pos_roots; "
            f"check the constants file is the canonical e8_constants.pt."
        )
    _, alpha_target = triality_reps[sector_name]

    all_roots = np.concatenate([pos_roots, -pos_roots], axis=0)
    path = _find_weyl_path(EIX_ALPHA_SU2.astype(np.float64),
                           np.asarray(alpha_target, dtype=np.float64),
                           all_roots)
    if path is None:
        raise RuntimeError(
            f"no Weyl path of depth ≤ 3 found from α_F4' to α_{sector_name}; "
            f"check that the target is in the same Weyl orbit (it should be — "
            f"all 240 long roots of E_8 form a single Weyl orbit)."
        )

    g = np.eye(DIM_E8, dtype=np.float64)
    for gamma in path:
        n_gamma = _realize_weyl_reflection(gamma, pos_roots, f_idx, f_val)
        g = n_gamma @ g

    _TRIALITY_GAUGE_CACHE[cache_key] = g
    return g
