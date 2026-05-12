"""Second Seeley-DeWitt coefficient a_2 on the E_8 Wolf space EIX.

TL;DR:  python3 a2_eix.py

Computes the heat-kernel coefficient a_2 of the scalar Laplacian on
EIX = E_8/(E_7 x SU(2)) via the Schur-lemma structure of the Riemann
tensor.  The result is a_2 = 1175384/15.

This script accompanies the paper "The second Seeley-DeWitt coefficient
of the scalar Laplacian on the E_8 Wolf space EIX".  It provides an
independent numerical cross-check of the analytical derivation using
explicit E_8 structure constants.

Method:
  1. Load E_8 structure constants in an orthonormal basis.
  2. Construct the E_7 x SU(2) reductive decomposition (133 + 3 + 112).
  3. Build the structure-constant tensor f^A_{ab} for A in h, a,b in m.
  4. Compute the Schur matrix K^{AB} = sum_{ab} f^A_{ab} f^B_{ab}.
  5. Verify K is block-diagonal with eigenvalues c_{E_7}=12, c_{SU(2)}=28
     (in kappa-orthonormal convention: c_{H_i} = h^v_G - h^v_{H_i}).
  6. Compute |Riem|^2 = ||K||_F^2 = 21504.
  7. Apply the Vassilevich formula: a_2 = (5R^2 - 2|Ric|^2 + 2|Riem|^2)/180.

References:
  - Vassilevich 2003, Phys. Rep. 388, 279, eq. (3.15).
  - Helgason 1978, Ch. IV.4 (curvature on symmetric spaces).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    build_dense_f,
    load_structure_constants_numpy,
    build_ad_matrix,
    extract_pos_roots_numpy,
)
from e8sim.roots import e7_su2_embedding  # noqa: E402
from e8sim.eix import (  # noqa: E402
    DIM_E7,
    DIM_SU2,
    DIM_H_EIX,
    DIM_M_EIX,
    H_VEE_E8,
    KAPPA_OVER_EUCLID,
)


CONSTANTS_PATH = constants_path(ROOT)

# Dual Coxeter numbers (standard Lie-algebra data).
H_VEE_E7 = 18
H_VEE_SU2 = 2

# Predicted Schur eigenvalues in kappa-orthonormal basis:
#   c_{H_i} = h^v_G - h^v_{H_i}   (for embedding index j = 1)
C_E7_PRED = H_VEE_E8 - H_VEE_E7   # = 12
C_SU2_PRED = H_VEE_E8 - H_VEE_SU2  # = 28

# In the e8sim basis (kappa(e_A, e_A) = 2, not 1), the diagonal of K
# is multiplied by 2 relative to the kappa-orthonormal basis:
C_E7_E8SIM = 2 * C_E7_PRED    # = 24
C_SU2_E8SIM = 2 * C_SU2_PRED  # = 56

# Geometric invariants (analytical predictions).
N = DIM_M_EIX          # 112
R_SCALAR = H_VEE_E8 * N / 2  # 1680
RIC_NORM_SQ = R_SCALAR**2 / N  # 25200
RIEM_NORM_SQ_PRED = (DIM_E7 * C_E7_PRED**2
                     + DIM_SU2 * C_SU2_PRED**2)  # 21504
A1 = R_SCALAR / 6  # 280
A2_NUM = 5 * int(R_SCALAR)**2 - 2 * int(RIC_NORM_SQ) + 2 * RIEM_NORM_SQ_PRED
A2_DENOM = 180
A2_GCD = math.gcd(A2_NUM, A2_DENOM)
A2_RED_NUM = A2_NUM // A2_GCD      # 1175384
A2_RED_DEN = A2_DENOM // A2_GCD    # 15

TOL = 1e-9


# ──────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────

def _setup():
    """Load E_8 structure constants and build the EIX reductive split."""
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    F_dense = build_dense_f(f_idx, f_val)
    pos_roots_arr = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots_arr)
    h_basis = np.vstack([e7_basis, su2_basis])
    return {
        "f_idx": f_idx, "f_val": f_val, "F_dense": F_dense,
        "e7_basis": e7_basis, "su2_basis": su2_basis,
        "h_basis": h_basis, "m_basis": m_basis,
    }


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def test_setup(g, res: Result) -> None:
    """T1: Verify dimensions and orthonormality of the EIX split."""
    banner("[T1] EIX reductive split: 133 + 3 + 112 = 248")

    e7 = g["e7_basis"]
    su2 = g["su2_basis"]
    m = g["m_basis"]

    res.report("dim(e_7) = 133", e7.shape[0] == DIM_E7)
    res.report("dim(su(2)) = 3", su2.shape[0] == DIM_SU2)
    res.report("dim(m) = 112", m.shape[0] == DIM_M_EIX)

    res.report("e7 orthonormal", np.allclose(e7 @ e7.T, np.eye(DIM_E7), atol=TOL))
    res.report("su2 orthonormal", np.allclose(su2 @ su2.T, np.eye(DIM_SU2), atol=TOL))
    res.report("m orthonormal", np.allclose(m @ m.T, np.eye(DIM_M_EIX), atol=TOL))
    res.report("e7 perp su2", np.allclose(e7 @ su2.T, 0, atol=TOL))
    res.report("e7 perp m", np.allclose(e7 @ m.T, 0, atol=TOL))
    res.report("su2 perp m", np.allclose(su2 @ m.T, 0, atol=TOL))


def test_ricci(g, res: Result) -> None:
    """T2: Cross-check R = 1680 via explicit Ricci on a tangent vector."""
    banner("[T2] Scalar curvature R = 1680")

    m = g["m_basis"]
    e_a = m[0]
    ad_e_a = build_ad_matrix(e_a, g["f_idx"], g["f_val"])
    ric_aa = -0.5 * float(np.trace(ad_e_a @ ad_e_a))
    kappa_aa = KAPPA_OVER_EUCLID * float(e_a @ e_a)
    ric_over_kappa = ric_aa / kappa_aa

    res.report(
        f"Ric/kappa = {ric_over_kappa:.6f} (expected h^v/2 = 15)",
        abs(ric_over_kappa - H_VEE_E8 / 2) < 1e-6,
    )

    g["R"] = R_SCALAR
    g["Ric_norm_sq"] = RIC_NORM_SQ


def test_schur_matrix(g, res: Result) -> None:
    """T3: Build K^{AB} and verify Schur block-diagonal structure."""
    banner("[T3] Schur matrix K^{AB}: block-diagonal with c_E7=24, c_SU2=56 (e8sim basis)")

    F = g["F_dense"]
    h_basis = g["h_basis"]
    m_basis = g["m_basis"]

    f_Aab = np.einsum("AR,PQR,aP,bQ->Aab", h_basis, F, m_basis, m_basis, optimize=True)
    K = np.einsum("Aab,Bab->AB", f_Aab, f_Aab, optimize=True)

    K_e7 = K[:DIM_E7, :DIM_E7]
    K_su2 = K[DIM_E7:, DIM_E7:]
    K_cross = K[:DIM_E7, DIM_E7:]

    diag_e7 = np.diag(K_e7)
    diag_su2 = np.diag(K_su2)
    offdiag_e7 = float(np.abs(K_e7 - np.diag(diag_e7)).max())
    offdiag_su2 = float(np.abs(K_su2 - np.diag(diag_su2)).max())
    cross_max = float(np.abs(K_cross).max())

    res.report(
        f"K_{{E7}} diagonal = {diag_e7.mean():.4f} (expected {C_E7_E8SIM})",
        abs(diag_e7.mean() - C_E7_E8SIM) / C_E7_E8SIM < 1e-6,
    )
    res.report(
        f"K_{{SU2}} diagonal = {diag_su2.mean():.4f} (expected {C_SU2_E8SIM})",
        abs(diag_su2.mean() - C_SU2_E8SIM) / C_SU2_E8SIM < 1e-6,
    )
    res.report(f"E7 off-diagonal max = {offdiag_e7:.2e} (expected 0)", offdiag_e7 < TOL)
    res.report(f"SU2 off-diagonal max = {offdiag_su2:.2e} (expected 0)", offdiag_su2 < TOL)
    res.report(f"Cross-block max = {cross_max:.2e} (expected 0)", cross_max < TOL)

    g["K"] = K


def test_riem_norm_sq(g, res: Result) -> None:
    """T4: |Riem|^2 = ||K||_F^2 / 4 = 21504."""
    banner("[T4] |Riem|^2 from Frobenius norm of K")

    K = g["K"]
    # In e8sim basis: |Riem|^2 = (1/4)||K||_F^2
    # (the factor 1/4 accounts for kappa(e_A,e_A)=2, see paper §3.4)
    K_F_sq = float(np.sum(K**2))
    riem_sq = K_F_sq / 4.0

    # Equivalently, in kappa-orthonormal basis: |Riem|^2 = ||K^{orth}||_F^2
    # = dim(e7)*c_E7^2 + dim(su2)*c_SU2^2 = 133*144 + 3*784 = 21504
    res.report(
        f"|Riem|^2 = {riem_sq:.4f} (expected {RIEM_NORM_SQ_PRED})",
        abs(riem_sq - RIEM_NORM_SQ_PRED) / RIEM_NORM_SQ_PRED < 1e-6,
    )

    g["Riem_norm_sq"] = riem_sq


def test_a2(g, res: Result) -> None:
    """T5: Vassilevich a_2 = (5R^2 - 2|Ric|^2 + 2|Riem|^2) / 180."""
    banner("[T5] Vassilevich a_2(EIX)")

    R = g["R"]
    ric_sq = g["Ric_norm_sq"]
    riem_sq = g["Riem_norm_sq"]

    a2 = (5 * R**2 - 2 * ric_sq + 2 * riem_sq) / 180.0
    a2_exact = A2_RED_NUM / A2_RED_DEN

    print(f"    a_2 = (5*{R:.0f}^2 - 2*{ric_sq:.0f} + 2*{riem_sq:.0f}) / 180")
    print(f"        = {5*R**2 - 2*ric_sq + 2*riem_sq:.0f} / 180")
    print(f"        = {A2_RED_NUM}/{A2_RED_DEN} = {a2_exact:.10f}")

    res.report(
        f"a_2(EIX) = {A2_RED_NUM}/{A2_RED_DEN} (approx {a2:.4f})",
        abs(a2 - a2_exact) < 1e-6,
    )


def test_consistency(g, res: Result) -> None:
    """T6: Consistency checks (Einstein, lower bound, Weyl positivity)."""
    banner("[T6] Consistency checks")

    riem_sq = g["Riem_norm_sq"]
    riem_min = 2 * R_SCALAR**2 / (N * (N - 1))

    res.report(
        f"|Riem|^2 > lower bound 2R^2/(n(n-1)) = {riem_min:.2f}",
        riem_sq > riem_min,
    )

    weyl_sq = riem_sq - riem_min
    res.report(
        f"|Weyl|^2 = {weyl_sq:.2f} > 0 (non-conformally-flat)",
        weyl_sq > 0,
    )

    weyl_frac = weyl_sq / riem_sq
    res.report(
        f"|Weyl|^2 / |Riem|^2 = {weyl_frac:.4f} (dominant Weyl contribution)",
        weyl_frac > 0.9,
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("a_2(EIX): Second Seeley-DeWitt coefficient on E_8/(E_7 x SU(2))")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading E_8 structure constants ...", flush=True)
    g = _setup()

    test_setup(g, res)
    test_ricci(g, res)
    test_schur_matrix(g, res)
    test_riem_norm_sq(g, res)
    test_a2(g, res)
    test_consistency(g, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Result: a_2(EIX) = {A2_RED_NUM}/{A2_RED_DEN} = {A2_RED_NUM/A2_RED_DEN:.6f}")
    print(f"Tests:  {res.summary()}")
    print(f"Time:   {elapsed:.2f} s")
    print("=" * 70)

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
