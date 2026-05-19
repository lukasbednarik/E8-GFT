"""Numerical cross-check for the orbit-constancy of
the leading 1-loop Coleman--Weinberg potential on EIX.

This script verifies the claim of equation
``eq:potential:CW-orbit-constancy`` in section ``sec:potential:CW-form``
("One-loop orbit-constancy and the structural form of mu^4") of

    "Effective sigma model of an E_8 group field theory: kinetic
     uniqueness, the Skyrme sector, and topological terms on the Wolf
     space EIX".

For the slow configuration psi(x) in EIX = E_8 / (E_7 x SU(2)) and the
fluctuation matrix on the 248-dimensional fibre

    M^2(psi) = 16 c_4 psi psi^T  -  kappa_2 * ad_psi^2,            (*)

E_8-equivariance, Ad_g M^2(psi) Ad_g^{-1} = M^2(Ad_g psi), implies that
the spectrum of M^2(psi) is an Ad-invariant function on the orbit and
is therefore constant on EIX.  Each spectral moment Tr[M^2(psi)^k]
(k = 1, 2, 3) and the standard 1-loop CW spectral functional

    V_CW(psi) = (1/64 pi^2) * sum_i M_i^4(psi) [log M_i^2(psi) - 3/2]

are therefore likewise constant on EIX, so the leading CW contribution
adds only to a constant background vacuum-energy term and does *not*
generate the pseudo-Goldstone mass parameter mu^4.

The numerical check picks the canonical reference point V_A and four
random orbit points psi = Ad_{exp(X)} V_A with X drawn from a small
ball in e_8, and reports the relative deviations of (i) the quadratic
Casimir kappa(psi, psi), (ii) the moments Tr[M^2(psi)^k] for
k = 1, 2, 3, (iii) the sorted spectrum of M^2(psi), and (iv) the CW
functional V_CW(psi).  All deviations are required to lie below
``TOL_ORBIT_REL`` (= 1e-7).

Conventions follow Appendix A of the article: the e8sim 248-basis is
orthonormal with respect to the Euclidean inner product (X, Y)_E; the
Killing form is K(X, Y) = -2 h^v (X, Y)_E with dual Coxeter number
h^v = 30, and kappa(X, Y) := -K(X, Y)/h^v = 2 (X, Y)_E.  V_A is the
unit Cartan generator of the SU(2) factor of the E_7 x SU(2) maximal
subgroup, normalised to (V_A, V_A)_E = 1, hence
r_*^2 := kappa(V_A, V_A) = 2.  The dimensionless coefficients used
in (*) are c_4 = kappa_2 = 1; the orbit-constancy is independent of
this choice.

Run:
    python3 e2_cw_potential.py
    pytest -v e2_cw_potential.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    adjoint_exp,
    build_ad_matrix,
    load_structure_constants_numpy,
)
from e8sim.eix import (  # noqa: E402
    KAPPA_OVER_EUCLID,
    canonical_VA as build_VA,
)

CONSTANTS_PATH = constants_path(ROOT)

TOL_ORBIT_REL = 1e-7   # relative tolerance for Ad-invariance on the orbit
N_TRIALS = 4           # number of random orbit points sampled
RNG_SEED = 0xCB31      # deterministic seed for reproducibility
ORBIT_STEP_NORM = 0.3  # ||X|| in adjoint_exp; small enough for stable matrix exp


def _build_M2(psi: np.ndarray, f_idx: np.ndarray, f_val: np.ndarray,
              c_4: float = 1.0, kappa_2: float = 1.0) -> np.ndarray:
    """1-loop fluctuation matrix M^2(psi) on the 248-dimensional fibre.

    M^2_pot = 16 c_4 * psi psi^T   (rank-1 radial Higgs piece)
    M^2_kin = -kappa_2 * ad_psi^2  (Higgs mechanism via the kinetic term)
    """
    ad_psi = build_ad_matrix(psi, f_idx, f_val)
    return 16.0 * c_4 * np.outer(psi, psi) - kappa_2 * (ad_psi @ ad_psi)


def _cw_spectral_sum(eigvals: np.ndarray, mu_RG_sq: float = 1.0,
                     m_min_sq: float = 1e-9) -> float:
    """Standard 1-loop CW spectral sum sum_i M_i^4 [log(M_i^2/mu^2) - 3/2].

    Modes with M^2 < m_min_sq are treated as massless (their CW
    contribution vanishes in the IR limit).  Returns a dimensionless
    value in the convention M_* = 1.
    """
    eig = eigvals[eigvals > m_min_sq]
    return float(np.sum(eig * eig * (np.log(eig / mu_RG_sq) - 1.5)))


def test_orbit_constancy(V_A: np.ndarray, r_sq: float,
                         f_idx: np.ndarray, f_val: np.ndarray,
                         res: Result) -> None:
    """Spectrum of M^2(psi) and the leading 1-loop CW potential are
    constant along the E_8-orbit through V_A.
    """
    banner("Orbit-constancy of M^2 spectrum and 1-loop V_CW on EIX")

    M2_VA = _build_M2(V_A, f_idx, f_val)
    eigs_VA = np.sort(np.linalg.eigvalsh(M2_VA))
    cw_VA = _cw_spectral_sum(eigs_VA) / (64.0 * np.pi ** 2)

    moments_VA = (
        float(np.trace(M2_VA)),
        float(np.trace(M2_VA @ M2_VA)),
        float(np.trace(M2_VA @ M2_VA @ M2_VA)),
    )

    print(f"    Reference point V_A:")
    print(f"      Tr[M^2(V_A)]         = {moments_VA[0]:+.6e}")
    print(f"      Tr[M^2(V_A)^2]       = {moments_VA[1]:+.6e}")
    print(f"      Tr[M^2(V_A)^3]       = {moments_VA[2]:+.6e}")
    print(f"      V_CW(V_A)            = {cw_VA:+.6e}    (mu_RG = 1)")

    rng = np.random.default_rng(RNG_SEED)
    max_rel_moments = [0.0, 0.0, 0.0]
    max_rel_eig = 0.0
    max_rel_cw = 0.0
    max_casimir_err = 0.0

    print(f"\n    Sampling {N_TRIALS} random orbit points "
          f"psi = Ad_exp(X) V_A,  ||X|| = {ORBIT_STEP_NORM:.2f} ...",
          flush=True)
    for k in range(N_TRIALS):
        coeffs = rng.standard_normal(DIM_E8)
        coeffs *= ORBIT_STEP_NORM / np.linalg.norm(coeffs)
        Ad_g = adjoint_exp(coeffs, f_idx, f_val)
        psi = Ad_g @ V_A

        psi_kappa = KAPPA_OVER_EUCLID * float(psi @ psi)
        casimir_err = abs(psi_kappa - r_sq) / r_sq
        max_casimir_err = max(max_casimir_err, casimir_err)

        M2_psi = _build_M2(psi, f_idx, f_val)
        moments_psi = (
            float(np.trace(M2_psi)),
            float(np.trace(M2_psi @ M2_psi)),
            float(np.trace(M2_psi @ M2_psi @ M2_psi)),
        )
        eigs_psi = np.sort(np.linalg.eigvalsh(M2_psi))
        cw_psi = _cw_spectral_sum(eigs_psi) / (64.0 * np.pi ** 2)

        rel_mom = tuple(
            abs(moments_psi[i] - moments_VA[i]) / max(abs(moments_VA[i]), 1e-30)
            for i in range(3)
        )
        rel_eig = float(np.max(np.abs(eigs_psi - eigs_VA))) / max(
            float(np.max(np.abs(eigs_VA))), 1e-30
        )
        rel_cw = abs(cw_psi - cw_VA) / max(abs(cw_VA), 1e-30)

        for i in range(3):
            max_rel_moments[i] = max(max_rel_moments[i], rel_mom[i])
        max_rel_eig = max(max_rel_eig, rel_eig)
        max_rel_cw = max(max_rel_cw, rel_cw)

        print(f"      trial {k+1}: |kappa(psi,psi) - r_*^2|/r_*^2 = "
              f"{casimir_err:.2e};  Delta_k Tr[M^{{2k}}] (k=1,2,3) = "
              f"({rel_mom[0]:.1e}, {rel_mom[1]:.1e}, {rel_mom[2]:.1e})",
              flush=True)

    res.report(
        "kappa(psi, psi) = r_*^2 along the orbit (Ad-invariance of C_2)",
        max_casimir_err < TOL_ORBIT_REL,
        f"max relative drift = {max_casimir_err:.2e}",
    )
    res.report(
        "Tr[M^2(psi)] is constant on the orbit",
        max_rel_moments[0] < TOL_ORBIT_REL,
        f"max relative deviation = {max_rel_moments[0]:.2e}  "
        f"({N_TRIALS} trials)",
    )
    res.report(
        "Tr[M^2(psi)^2] is constant on the orbit",
        max_rel_moments[1] < TOL_ORBIT_REL,
        f"max relative deviation = {max_rel_moments[1]:.2e}",
    )
    res.report(
        "Tr[M^2(psi)^3] is constant on the orbit",
        max_rel_moments[2] < TOL_ORBIT_REL,
        f"max relative deviation = {max_rel_moments[2]:.2e}",
    )
    res.report(
        "Sorted spectrum of M^2(psi) is constant on the orbit",
        max_rel_eig < TOL_ORBIT_REL,
        f"max ||Delta lambda||_inf / ||lambda||_inf = {max_rel_eig:.2e}",
    )
    res.report(
        "Leading 1-loop V_CW(psi) is constant on the orbit",
        max_rel_cw < TOL_ORBIT_REL,
        f"max relative deviation = {max_rel_cw:.2e}",
    )


def main() -> int:
    print("=" * 70)
    print("Orbit-constancy of the leading 1-loop Coleman-Weinberg potential")
    print("on the E_8 orbit through V_A (Wolf space EIX = E_8/(E_7 x SU(2)))")
    print("Cross-check for sec:potential:CW-form, eq:potential:CW-orbit-constancy")
    print("of: 'Effective sigma model of an E_8 group field theory: kinetic")
    print("uniqueness, the Skyrme sector, and topological terms on the Wolf")
    print("space EIX'.")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8 structure constants and EIX vacuum V_A ...",
          flush=True)
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    V_A, r_sq = build_VA()
    print(f"        (V_A, V_A)_E = {float(V_A @ V_A):.4f}, "
          f"r_*^2 := kappa(V_A, V_A) = {r_sq:.4f}", flush=True)

    test_orbit_constancy(V_A, r_sq, f_idx, f_val, res)

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


def test_cw_orbit_constancy() -> None:
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    V_A, r_sq = build_VA()
    res = Result()
    test_orbit_constancy(V_A, r_sq, f_idx, f_val, res)
    assert res.failed == 0, f"{res.failed} sub-test(s) failed"


if __name__ == "__main__":
    sys.exit(main())
