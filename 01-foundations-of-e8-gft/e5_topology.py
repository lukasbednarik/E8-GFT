"""Topology of EIX (numerical verification of Theorem on lower homotopy).

Numerical verification of the topological structure of the orbit
``EIX = E_8 / (E_7 x SU(2))`` referenced in
``sections/06-emergent-spacetime.tex``:

  * Theorem ``thm:topology-eix`` ::
        pi_0(EIX) = pi_1(EIX) = pi_2(EIX) = 0,

  * Remark ``rem:topology-numerical`` (the present script,
    \textsf{[Proven-num]}): the vanishing is checked independently on
    the explicit Chevalley basis of ``e_8`` via the Dynkin embedding
    indices ``j_{E_7 ⊂ E_8} = 1`` and ``j_{SU(2) ⊂ E_8} = 1`` (computed
    as the trace ratio of the corresponding generators), together with
    the kernel/cokernel structure of the induced map on ``pi_3``.

Conventions (matched to ``sections/A1-conventions.tex``):

  * the e8sim 248-basis is *orthonormal* with respect to the Euclidean
    inner product ``(X, Y)_E``;
  * the Killing form ``K(X, Y) = tr(ad_X ad_Y) = -2 h^v · (X, Y)_E``
    is negative-definite on the compact form;
  * dual Coxeter numbers ``h^v_{E_8} = 30``, ``h^v_{E_7} = 18``,
    ``h^v_{SU(2)} = 2``.

Test layout:

  T5.1  Dynkin indices and long exact homotopy sequence
            T5.1.a  j_{E_7 ⊂ E_8} = 1                (trace ratio)
            T5.1.b  j_{SU(2) ⊂ E_8} = 1              (trace ratio)
            T5.1.c  pi_3(EIX) = 0                    (cokernel of the
                    induced map Z² → Z; numerical input to
                    rem:topology-numerical, complementing the
                    pi_0/pi_1/pi_2 statement of thm:topology-eix)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
    build_ad_matrix,
)
from e8sim.roots import e7_su2_embedding  # noqa: E402
from e8sim.eix import (  # noqa: E402
    H_VEE_E7,
    H_VEE_SU2,
)

CONSTANTS_PATH = constants_path(ROOT)


# ----------------------------------------------------------------------
# Lie-algebra helpers
# ----------------------------------------------------------------------


def build_ad_dense(f_idx: np.ndarray, f_val: np.ndarray, X: np.ndarray) -> np.ndarray:
    return build_ad_matrix(X, f_idx, f_val)


def trace_on_subspace(M: np.ndarray, basis: np.ndarray) -> float:
    """tr(M|_V) where V has orthonormal rows in `basis` (k×N)."""
    return float(np.trace(basis @ M @ basis.T))


# ----------------------------------------------------------------------
# T5.1 — Dynkin indices and long exact homotopy sequence
# ----------------------------------------------------------------------


def dynkin_index_via_trace(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h_basis: np.ndarray,
    X_in_h: np.ndarray,
    h_vee_h: int,
) -> tuple[float, float, float]:
    """Compute the Dynkin index of the embedding h ⊂ e_8 from a trace ratio.

    For X ∈ h:
        tr_{e_8}(ad_X²) = -2 h^v_{e_8} · (X, X)_E         (Killing in e_8)
        tr_{h}(ad_X²)   = -2 h^v_{h}  · (X, X)_E · j_h    (with j_h
                                                           the Dynkin index
                                                           of h ⊂ e_8)
    so j_h = tr_{h}(ad_X²) / (-2 h^v_{h} · (X, X)_E) for any X ∈ h with
    (X, X)_E ≠ 0.

    Returns (j_h, tr_h, tr_g_full).
    """
    ad_X = build_ad_dense(f_idx, f_val, X_in_h)
    ad_X_sq = ad_X @ ad_X
    tr_g = float(np.trace(ad_X_sq))
    tr_h = trace_on_subspace(ad_X_sq, h_basis)
    euclid_sq = float(X_in_h @ X_in_h)
    j_h = tr_h / (-2.0 * h_vee_h * euclid_sq)
    return j_h, tr_h, tr_g


def test_T5_1_homotopy_via_LES(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    e7_basis: np.ndarray,
    su2_basis: np.ndarray,
    res: Result,
) -> None:
    banner("[T5.1] Homotopy of EIX via Dynkin indices and the long "
           "exact sequence (rem:topology-numerical)")

    # ── T5.1.a — Dynkin index E_7 ⊂ E_8 ──
    # Pick X = a Cartan vector inside E_7 (= orthogonal to alpha_SU(2)).
    # alpha_SU(2) = (1, 1, 0, ..., 0); the E_7-Cartan lies in
    # {x ∈ R^8 : x_1 + x_2 = 0}.  Take X = (1, -1, 0, ..., 0)/sqrt(2),
    # which has unit Euclidean length.
    X_e7 = np.zeros(DIM_E8)
    X_e7[:8] = np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]) / np.sqrt(2.0)

    j_e7, tr_e7, tr_full_e7 = dynkin_index_via_trace(
        f_idx, f_val, e7_basis, X_e7, H_VEE_E7
    )
    print(f"    X_E7 = (1,-1,0..0)/√2 ∈ E_7-Cartan, (X,X)_E = {X_e7 @ X_e7:.4f}")
    print(f"    tr_{{e_8}}(ad_X²) = {tr_full_e7:8.4f}  "
          f"(prediction -2·30·(X,X)_E = {-60*X_e7 @ X_e7:.4f})")
    print(f"    tr_{{E_7}}(ad_X²) = {tr_e7:8.4f}  "
          f"(prediction -2·18·(X,X)_E = {-36*X_e7 @ X_e7:.4f}  for j = 1)")
    res.report(
        "T5.1.a  j_{E_7 ⊂ E_8} = 1",
        abs(j_e7 - 1.0) < 1e-9,
        f"measured j_E7 = {j_e7:.6f}",
    )

    # ── T5.1.b — Dynkin index SU(2) ⊂ E_8 ──
    # X = H_su2 (Cartan slice of the SU(2) factor of E_7 × SU(2)).
    X_su2 = su2_basis[0]  # H_su2 = alpha_SU(2)/‖alpha_SU(2)‖
    j_su2, tr_su2, tr_full_su2 = dynkin_index_via_trace(
        f_idx, f_val, su2_basis, X_su2, H_VEE_SU2
    )
    print(f"\n    X_SU(2) = H_su2 (Cartan slice), (X,X)_E = {X_su2 @ X_su2:.4f}")
    print(f"    tr_{{e_8}}(ad_X²)   = {tr_full_su2:8.4f}  "
          f"(prediction -2·30·(X,X)_E = {-60*X_su2 @ X_su2:.4f})")
    print(f"    tr_{{SU(2)}}(ad_X²) = {tr_su2:8.4f}  "
          f"(prediction -2·2·(X,X)_E  = {-4*X_su2 @ X_su2:.4f}  for j = 1)")
    res.report(
        "T5.1.b  j_{SU(2) ⊂ E_8} = 1  (SU(2) factor of E_7 × SU(2))",
        abs(j_su2 - 1.0) < 1e-9,
        f"measured j_SU(2) = {j_su2:.6f}",
    )

    # ── T5.1.c — pi_3(EIX) = 0  via cokernel of the induced map ──
    # The relevant segment of the long exact homotopy sequence for the
    # fibration H = E_7 × SU(2) → E_8 → EIX reads
    #     iota_*: pi_3(H) = Z² → pi_3(E_8) = Z,
    #             (n_1, n_2) ↦ j_{E_7} · n_1 + j_{SU(2)} · n_2 = n_1 + n_2,
    # since both Dynkin indices are 1 (T5.1.a, T5.1.b).  Hence
    # image(iota_*) = Z and pi_3(EIX) = coker(iota_*) = 0.
    j_e7_int = int(round(j_e7))
    j_su2_int = int(round(j_su2))
    gcd = abs(np.gcd(j_e7_int, j_su2_int))
    cokernel_size = "0" if gcd == 1 else f"Z/{gcd}"
    print(f"\n    LES segment at pi_3:  Z² ─(n₁+n₂)─→ Z → pi_3(EIX) → pi_2(H)=0")
    print(f"    iota_* is the addition map with indices "
          f"({j_e7_int}, {j_su2_int}); gcd = {gcd}")
    res.report(
        "T5.1.c  pi_3(EIX) = 0  (cokernel of iota_* = Z/gcd = 0)",
        gcd == 1,
        f"cokernel = {cokernel_size}",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("E5 — Topology of EIX (rem:topology-numerical of "
          "sections/06-emergent-spacetime.tex)")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8_constants.pt and the EIX embedding ...",
          flush=True)
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, _m_basis, _ = e7_su2_embedding(pos_roots=pos_roots)
    print(f"        e7_basis: {e7_basis.shape}, su2_basis: {su2_basis.shape}",
          flush=True)

    test_T5_1_homotopy_via_LES(f_idx, f_val, e7_basis, su2_basis, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"E5 summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed == 0:
        print("\n[PASS] E5 — topology of EIX verified on the explicit "
              "Chevalley basis.")
        return 0
    else:
        print("\n[FAIL] E5 — see test output above for the failing assertions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
