"""Numerical cross-check identifying the
projection target pi_{su(2)} of definition ``def:B-charge`` with the
Sp(1)-factor of the Wolf-space holonomy of EIX = E_8/(E_7 x SU(2)),
referenced in section ``sec:topology:B-charge'' of

    "Effective sigma model of an E_8 group field theory: kinetic
     uniqueness, the Skyrme sector, and topological terms on the Wolf
     space EIX".

Let su(2) subset h = e_7 + su(2) be the stabiliser factor inside E_8,
spanned by an orthonormal basis (T_0, T_1, T_2) of su(2) in the
ambient e_8 (with T_0 propto V_A).  Define the three almost complex
structures on m = e_8 / h by

    J_a := (1/sqrt(lambda)) * ad_{T_a}|_m,   a = 0, 1, 2,

with the Schur eigenvalue lambda = (h^v_{E_8} - h^v_{SU(2)})/dim m * r_*^2
= (30 - 2)/112 * 2 = 1/2.  The script verifies the three structural
properties that single out the Sp(1)-factor of the holonomy
Sp(28).Sp(1) of the Wolf space EIX (Wolf 1965, Salamon 1982):

1.  Quaternionic algebra on m:

        J_a^2 = -I_m,
        J_a J_b - J_b J_a = 2 epsilon_{abc} J_c,        (sp(1) commutators)

    matching the structure constants of equation
    ``eq:appH4:trace-id`` (Appendix A2) and verifying that
    span(J_0, J_1, J_2) realises sp(1) inside so(m) = so(112).

2.  E_7 lies in the centraliser of su(2) on m:

        [ad(X)|_m, J_a] = 0   for all  X in basis(e_7),  a in {0, 1, 2},

    so ad(e_7)|_m subset Sp(28) and ad(su(2)_stab)|_m generates the
    complementary Sp(1) factor, identifying pi_{su(2)} of
    Definition ``def:B-charge'' with the Sp(1)-factor of the
    holonomy Sp(28).Sp(1).

3.  Quaternionic dimension of the tangent space:

        spec(J_a) = { +i with multiplicity 56,  -i with multiplicity 56 },

    consistent with m_C = C^56 + C^56 and m congruent H^28 = T_p EIX
    over the quaternions.

Conventions follow Appendix A of the article: the e8sim 248-basis is
orthonormal with respect to the Euclidean inner product (X, Y)_E; in
the compact real form every ad_X is antisymmetric in this basis.
The basis (T_0, T_1, T_2) is chosen as
    T_0 = alpha/sqrt(2),   T_1 = (E_alpha - E_{-alpha})/sqrt(2),
    T_2 = i (E_alpha + E_{-alpha})/sqrt(2)
with alpha the long root that defines the canonical SU(2) factor of
E_7 x SU(2); their commutators are [T_a, T_b] = sqrt(2) epsilon_{abc} T_c
in the orthonormal basis, which produces the rescaled J-commutators
above after dividing by sqrt(lambda).

Run:
    python3 eo1_q1_sp1_holonomy.py
    pytest -v eo1_q1_sp1_holonomy.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    build_ad_matrix,
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
)
from e8sim.eix import (  # noqa: E402
    DIM_E7,
    DIM_M_EIX,
    H_VEE_E8,
    H_VEE_SU2,
    KAPPA_OVER_EUCLID,
)
from e8sim.roots import EIX_ALPHA_SU2, e7_su2_embedding  # noqa: E402

CONSTANTS_PATH = constants_path(ROOT)

TOL_QUAT = 1e-8     # tolerance for the quaternionic relations
TOL_COMM = 1e-8     # tolerance for the centraliser commutators
TOL_SPEC = 1e-8     # tolerance for the J_a eigenvalue check


def _ad_on_m(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    X: np.ndarray,
    m_basis: np.ndarray,
) -> np.ndarray:
    """Return the 112 x 112 matrix of ad_X|_m in the orthonormal m-basis."""
    ad_X = build_ad_matrix(X, f_idx, f_val)
    return m_basis @ ad_X @ m_basis.T


def _build_J_triple(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    su2_basis: np.ndarray,
    m_basis: np.ndarray,
    sqrt_lambda: float,
) -> list[np.ndarray]:
    """Build J_a := ad_{T_a}|_m / sqrt(lambda) for T_a = su2_basis[a]."""
    return [
        _ad_on_m(f_idx, f_val, su2_basis[a], m_basis) / sqrt_lambda
        for a in range(3)
    ]


def test_sp1_holonomy_identification(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    e7_basis: np.ndarray,
    su2_basis: np.ndarray,
    m_basis: np.ndarray,
    res: Result,
) -> None:
    """Verify the three properties pinning down the Sp(1)-factor of the
    Wolf-space holonomy of EIX from the stabiliser su(2) subset h."""
    banner("Identification of pi_{su(2)} with the Sp(1)-factor of "
           "Hol(EIX) = Sp(28).Sp(1)")

    r_sq = KAPPA_OVER_EUCLID * float(su2_basis[0] @ su2_basis[0])
    lambda_val = (H_VEE_E8 - H_VEE_SU2) / DIM_M_EIX * r_sq
    sqrt_lambda = float(np.sqrt(lambda_val))
    print(f"    r_*^2 = kappa(T_0, T_0)            = {r_sq:.6f}")
    print(f"    lambda = (h^v_E8 - h^v_SU(2))/112 * r_*^2 = {lambda_val:.6f}  "
          f"(predicted: 1/2)")
    print(f"    sqrt(lambda)                       = {sqrt_lambda:.6f}  "
          f"(predicted: 1/sqrt(2))")

    # -- (1) Quaternionic algebra: J_a^2 = -I,  [J_a, J_b] = 2 eps_abc J_c.
    J_list = _build_J_triple(f_idx, f_val, su2_basis, m_basis, sqrt_lambda)
    I_m = np.eye(DIM_M_EIX)

    max_dev_sq = max(
        float(np.max(np.abs(J @ J + I_m))) for J in J_list
    )
    res.report(
        "J_a^2 = -I_m  for a in {0, 1, 2}  (almost complex structures)",
        max_dev_sq < TOL_QUAT,
        f"max ||J^2 + I_m||_inf = {max_dev_sq:.2e}",
    )

    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    max_dev_quat = 0.0
    for (a, b, c) in cyclic:
        comm = J_list[a] @ J_list[b] - J_list[b] @ J_list[a]
        dev = float(np.max(np.abs(comm - 2.0 * J_list[c])))
        max_dev_quat = max(max_dev_quat, dev)
    res.report(
        "[J_a, J_b] = 2 epsilon_{abc} J_c  (sp(1) commutators)",
        max_dev_quat < TOL_COMM,
        f"max ||[J_a, J_b] - 2 eps_abc J_c||_inf = {max_dev_quat:.2e}",
    )

    # -- (2) e_7 commutes with each J_a  (=> e_7 subset Sp(28) factor).
    n_e7 = e7_basis.shape[0]
    max_dev_central = 0.0
    for i in range(n_e7):
        ad_X_m = _ad_on_m(f_idx, f_val, e7_basis[i], m_basis)
        for J in J_list:
            dev = float(np.max(np.abs(ad_X_m @ J - J @ ad_X_m)))
            max_dev_central = max(max_dev_central, dev)
    print(f"\n    Probed {n_e7} basis generators of e_7 against 3 J_a "
          f"(= {3 * n_e7} commutators)")
    res.report(
        "[ad(X)|_m, J_a] = 0  for all  X in basis(e_7), a in {0, 1, 2}  "
        "(e_7 subset Sp(28))",
        max_dev_central < TOL_COMM,
        f"max ||[ad(X)|_m, J_a]||_inf = {max_dev_central:.2e}",
    )

    # -- (3) Spectrum of J_a = +/- i, multiplicity 56 each  (m congruent H^28).
    spectra_ok = True
    spec_diag = []
    for a, J in enumerate(J_list):
        eigvals = np.linalg.eigvals(J)
        n_pos_i = int(np.sum(eigvals.imag > 0.5))
        n_neg_i = int(np.sum(eigvals.imag < -0.5))
        max_re = float(np.max(np.abs(eigvals.real)))
        max_dev_im = float(np.max(np.abs(np.abs(eigvals.imag) - 1.0)))
        ok_a = (n_pos_i == 56 and n_neg_i == 56
                and max_re < TOL_SPEC and max_dev_im < TOL_SPEC)
        spectra_ok = spectra_ok and ok_a
        spec_diag.append((a, n_pos_i, n_neg_i, max_re, max_dev_im))
    print()
    for (a, n_pos_i, n_neg_i, max_re, max_dev_im) in spec_diag:
        print(f"    spec(J_{a}): #(+i) = {n_pos_i}, #(-i) = {n_neg_i},  "
              f"max |Re| = {max_re:.2e}, max ||Im| - 1| = {max_dev_im:.2e}")
    res.report(
        "spec(J_a) = +/- i with multiplicity 56 each  (m congruent H^28)",
        spectra_ok,
        "m_C = C^56 + C^56 over each J_a",
    )


def main() -> int:
    print("=" * 70)
    print("Identification of pi_{su(2)} with the Sp(1)-factor of the")
    print("Wolf-space holonomy Hol(EIX) = Sp(28).Sp(1) on EIX = E_8/(E_7 x SU(2))")
    print("Cross-check for sec:topology:B-charge of: 'Effective sigma model")
    print("of an E_8 group field theory: kinetic uniqueness, the Skyrme")
    print("sector, and topological terms on the Wolf space EIX'.")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8 structure constants and E_7 x SU(2) embedding ...",
          flush=True)
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _labels = e7_su2_embedding(
        pos_roots=pos_roots,
        alpha_su2=EIX_ALPHA_SU2,
    )
    print(f"        dim e_7 = {e7_basis.shape[0]} (predicted {DIM_E7}),  "
          f"dim su(2) = {su2_basis.shape[0]},  "
          f"dim m = {m_basis.shape[0]} (predicted {DIM_M_EIX})", flush=True)

    test_sp1_holonomy_identification(
        f_idx, f_val, e7_basis, su2_basis, m_basis, res,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Summary: {res.passed} PASS / {res.failed} FAIL  ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed > 0:
        print("\nFailed sub-tests:")
        for marker, name, detail in res.records:
            if marker == "FAIL":
                print(f"  - {name}")
                if detail:
                    print(f"      {detail}")
        return 1
    return 0


# ----------------------------------------------------------------------
# Pytest wrapper
# ----------------------------------------------------------------------


def test_sp1_holonomy_identification_pytest() -> None:
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _labels = e7_su2_embedding(
        pos_roots=pos_roots,
        alpha_su2=EIX_ALPHA_SU2,
    )
    res = Result()
    test_sp1_holonomy_identification(
        f_idx, f_val, e7_basis, su2_basis, m_basis, res,
    )
    assert res.failed == 0, f"{res.failed} sub-test(s) failed"


if __name__ == "__main__":
    sys.exit(main())
