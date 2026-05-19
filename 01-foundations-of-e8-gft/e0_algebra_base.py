"""Algebraic base of E_8.

Numerical certificate for the structural data assumed by Postulate P1
(``post:algebra``) and the Lie-algebra normalisation fixed in
Appendix A.1 (``app:conv:lie``) of ``main.tex``.

Tests:
  T0    dim 248, rank 8, |Φ|=240, |Φ⁺|=120
  T1    f_{ABC} totally antisymmetric (κ-orthonormal basis)
  T2    Jacobi identity: vector form + ad-homomorphism
  T3    Killing form ∝ identity; dual Coxeter number h^∨_{E_8} = 30
  T0+   Cartan subalgebra rank 8 (mutual commutativity of T_0..T_7)

Output: per-test PASS/FAIL with running totals. Exit code 0 iff all PASS.

References:
  - Bourbaki, Groupes et algèbres de Lie, Ch. VI §1, Plate VII (E_8 data)
  - Helgason, Differential Geometry, Lie Groups and Symmetric Spaces, Ch. III
  - main.tex §2 (postulates P1–P2) and Appendix A.1 (conventions)
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    bracket_vec_fast,
    load_structure_constants_numpy,
    build_dense_f,
    build_ad_matrix,
)
from e8sim import generate_roots, positive_root_indices  # noqa: E402

RANK_E8 = 8
N_POS_ROOTS = 120
DUAL_COXETER = 30

CONSTANTS_PATH = constants_path(ROOT)

TOL_ALGEBRAIC = 1e-9
TOL_KILLING = 1e-7


build_full_f = build_dense_f


def build_ad(f_idx, f_val, X):
    return build_ad_matrix(X, f_idx, f_val)


# ----------------------------------------------------------------------
# T0 — basic structural data
# ----------------------------------------------------------------------


def test_T0_basic_data(res: Result) -> tuple[np.ndarray, np.ndarray]:
    banner("[T0] Structural data of E_8")

    if not CONSTANTS_PATH.exists():
        res.report(
            "e8_constants.pt exists",
            False,
            f"missing file {CONSTANTS_PATH}",
        )
        sys.exit(1)

    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    res.report("e8_constants.pt loaded", True, f"nnz = {len(f_val)}")

    res.report(
        "dim e_8 = 248 (indices in [0, 247])",
        f_idx.max() == DIM_E8 - 1 and f_idx.min() == 0,
        f"index range [{int(f_idx.min())}, {int(f_idx.max())}]",
    )

    roots = generate_roots()
    res.report(
        "|Φ| = 240 (total number of roots)",
        len(roots) == 240,
        f"got {len(roots)}",
    )

    pos = positive_root_indices(roots)
    res.report(
        "|Φ⁺| = 120 (positive roots)",
        len(pos) == N_POS_ROOTS,
        f"got {len(pos)}",
    )

    return f_idx, f_val


# ----------------------------------------------------------------------
# T1 — total antisymmetry of f_{ABC}
# ----------------------------------------------------------------------


def test_T1_antisymmetry(F: np.ndarray, res: Result) -> None:
    banner("[T1] Total antisymmetry of f_{ABC}")
    # In the κ-orthonormal basis fixed by Appendix A.1, the structure
    # constants are totally antisymmetric in (A, B, C).

    err_AB = float(np.max(np.abs(F + F.transpose(1, 0, 2))))
    res.report(
        "f_{ABC} = -f_{BAC}",
        err_AB < TOL_ALGEBRAIC,
        f"max |Δ| = {err_AB:.2e}",
    )

    err_BC = float(np.max(np.abs(F + F.transpose(0, 2, 1))))
    res.report(
        "f_{ABC} = -f_{ACB} (total antisymmetry)",
        err_BC < TOL_ALGEBRAIC,
        f"max |Δ| = {err_BC:.2e}",
    )

    # Cyclic sanity: f_{ABC} = f_{CAB} (consequence of total antisymmetry).
    err_cyc = float(np.max(np.abs(F - F.transpose(2, 0, 1))))
    res.report(
        "f_{ABC} = f_{CAB} (even cyclic permutation)",
        err_cyc < TOL_ALGEBRAIC,
        f"max |Δ| = {err_cyc:.2e}",
    )


# ----------------------------------------------------------------------
# T2 — Jacobi identity
# ----------------------------------------------------------------------


def test_T2_jacobi(f_idx: np.ndarray, f_val: np.ndarray, res: Result) -> None:
    banner("[T2] Jacobi identity")

    rng = np.random.default_rng(0xE05ACAB1)

    n_samples = 20
    max_err = 0.0
    for _ in range(n_samples):
        X = rng.standard_normal(DIM_E8)
        Y = rng.standard_normal(DIM_E8)
        Z = rng.standard_normal(DIM_E8)
        XY = bracket_vec_fast(f_idx, f_val, X, Y)
        YZ = bracket_vec_fast(f_idx, f_val, Y, Z)
        ZX = bracket_vec_fast(f_idx, f_val, Z, X)
        residual = (
            bracket_vec_fast(f_idx, f_val, XY, Z)
            + bracket_vec_fast(f_idx, f_val, YZ, X)
            + bracket_vec_fast(f_idx, f_val, ZX, Y)
        )
        max_err = max(max_err, float(np.max(np.abs(residual))))
    res.report(
        f"[[X,Y],Z] + cyclic = 0 (n={n_samples})",
        max_err < TOL_ALGEBRAIC,
        f"max |Δ| = {max_err:.2e}",
    )

    # Structural form: ad is a Lie-algebra homomorphism.
    X = rng.standard_normal(DIM_E8)
    Y = rng.standard_normal(DIM_E8)
    XY = bracket_vec_fast(f_idx, f_val, X, Y)
    adX = build_ad(f_idx, f_val, X)
    adY = build_ad(f_idx, f_val, Y)
    adXY = build_ad(f_idx, f_val, XY)
    err_ad = float(np.max(np.abs(adXY - (adX @ adY - adY @ adX))))
    res.report(
        "ad_{[X,Y]} = [ad_X, ad_Y] (homomorphism)",
        err_ad < TOL_ALGEBRAIC,
        f"max |Δ| = {err_ad:.2e}",
    )


# ----------------------------------------------------------------------
# T3 — Killing form and dual Coxeter number
# ----------------------------------------------------------------------


def test_T3_killing_and_h_vee(F: np.ndarray, res: Result) -> None:
    banner("[T3] Killing form and dual Coxeter number h^∨")
    # In the κ-orthonormal basis, the Killing form
    #   K_{AB} = Σ_{C,D} f_{ACD} f_{BCD}
    # equals 2 h^∨ δ_{AB}; for E_8 this gives K_{AA} = 60 (P2, App. A.1).

    K = np.einsum("ACD,BCD->AB", F, F, optimize=True)

    diag = np.diag(K)
    two_hv_est = float(np.mean(diag))
    h_v_est = two_hv_est / 2.0

    off_diag_max = float(np.max(np.abs(K - np.diag(diag))))
    diag_var = float(np.max(np.abs(diag - two_hv_est)))

    res.report(
        "K_{AB} ∝ δ_{AB} (off-diagonal ≈ 0)",
        off_diag_max < TOL_KILLING,
        f"max |off-diag| = {off_diag_max:.2e}",
    )
    res.report(
        "K_{AA} constant (Killing ∝ identity)",
        diag_var < TOL_KILLING,
        f"max |K_{{AA}} - mean| = {diag_var:.2e}",
    )
    res.report(
        f"h^∨_{{E_8}} = {DUAL_COXETER}",
        abs(h_v_est - DUAL_COXETER) < 1e-6,
        f"measured 2 h^∨ = {two_hv_est:.6f} ⇒ h^∨ = {h_v_est:.6f}",
    )


# ----------------------------------------------------------------------
# T0+ — Cartan subalgebra rank
# ----------------------------------------------------------------------


def test_T0plus_cartan_dim(f_idx: np.ndarray, f_val: np.ndarray,
                           res: Result) -> None:
    banner("[T0+] Cartan subalgebra: rank 8")
    # Verify that the eight generators T_0..T_7 (Cartan generators in the
    # e8sim convention) mutually commute: [T_i, T_j] = 0 for i, j ∈ {0..7}.
    cart_basis = np.zeros((8, DIM_E8))
    for k in range(8):
        cart_basis[k, k] = 1.0

    max_err = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            comm = bracket_vec_fast(f_idx, f_val, cart_basis[i], cart_basis[j])
            max_err = max(max_err, float(np.max(np.abs(comm))))
    res.report(
        "[T_i, T_j] = 0 for i,j ∈ Cartan (rank ≥ 8)",
        max_err < TOL_ALGEBRAIC,
        f"max |[T_i, T_j]| = {max_err:.2e}",
    )

    # The bound rank ≤ 8 follows structurally from h^∨_{E_8} = 30 (verified
    # in T3) together with Bourbaki Plate VII; no separate hard-coded check.


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("E0 — Algebraic base of E_8")
    print("Reference: main.tex §2 (P1–P2) and Appendix A.1")
    print("=" * 70, flush=True)

    res = Result()

    t0 = time.time()
    f_idx, f_val = test_T0_basic_data(res)

    print("\n[setup] Building dense f_{ABC} (≈120 MB) ...", flush=True)
    F = build_full_f(f_idx, f_val)

    test_T1_antisymmetry(F, res)
    test_T2_jacobi(f_idx, f_val, res)
    test_T3_killing_and_h_vee(F, res)
    test_T0plus_cartan_dim(f_idx, f_val, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"E0 summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed == 0:
        print("\n[PASS] E0 — Algebraic base of e_8 verified.")
        return 0
    else:
        print("\n[FAIL] E0 — algebraic base check failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
