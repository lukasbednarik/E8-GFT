"""Symmetry-breaking vacuum (BEC analogue).

Numerical certificates for the tree-level content of the condensate
section ``sections/04-condensate.tex``. The Cas_2 truncated potential
is V_eff(r^2) = c_2 r^2 + c_4 r^4 with r^2 = C_2(Phi_0) = kappa(Phi_0,Phi_0),
and the three targets verified here are:

  T1  Existence of the condensate
        (Theorem 4.x, ``thm:condensate:existence``,
         ``eq:condensate:r-star``):
        r_*^2 = -c_2/(2 c_4) > 0,  V_eff(r_*^2) = -c_2^2/(4 c_4) < 0.

  T3  Bare radial mass
        (Proposition ``prop:condensate:rad-higgs``,
         ``eq:condensate:bare-rad-mass``):
        in the no-1/2 convention, M_rad^2 = (1/2) V''_eff(r_*) = -2 c_2,
        verified as the Hessian eigenvalue of V on R^248 along Phi_0.

  T4  Goldstone count for the generic semisimple stratum
        (Lemma ``lem:condensate:goldstone-count``,
         Table ``tab:condensate:stabilizers``, row T^8):
        for a generic Phi_0 in the level sphere,
        dim Orb(Phi_0) = rank(ad_{Phi_0}) = 240,
        i.e. dim Stab(Phi_0) = 8 (Cartan torus, no extra root nullings).

Reference: ``sections/04-condensate.tex``.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    load_structure_constants_numpy,
    build_ad_matrix,
)

CONSTANTS_PATH = constants_path(ROOT)

TOL_SCALAR = 1e-12
TOL_HESS = 1e-9
RANK_TOL = 1e-9


def random_orbit_point(rng: np.random.Generator, r_target: float) -> np.ndarray:
    """Uniform sample on S^{247}_{r_target} subset R^{248}."""
    v = rng.standard_normal(DIM_E8)
    v *= r_target / np.linalg.norm(v)
    return v


# ----------------------------------------------------------------------
# T1 — existence of the BEC vacuum (scalar identity)
# ----------------------------------------------------------------------


def test_T1_existence(res: Result, c2: float, c4: float) -> None:
    banner("[T1] Existence: r_*^2 = -c_2/(2 c_4), V_eff = -c_2^2/(4 c_4)")

    r_sq = -c2 / (2 * c4)
    V_min = -(c2 ** 2) / (4 * c4)

    # Stationarity:  d/dt [c2 t + c4 t^2] = c2 + 2 c4 t = 0 at t = r_*^2.
    grad = c2 + 2 * c4 * r_sq
    # Value:  V_eff(r_*^2) = c2 r_*^2 + c4 r_*^4.
    V_at_min = c2 * r_sq + c4 * r_sq ** 2

    ok = (
        r_sq > 0
        and V_min < 0
        and abs(grad) < TOL_SCALAR
        and abs(V_at_min - V_min) < TOL_SCALAR
    )
    res.report(
        "stationarity and depth match closed form",
        ok,
        f"r_*^2 = {r_sq:.6f}, V_eff(r_*^2) = {V_at_min:.6e}, "
        f"|grad| = {abs(grad):.2e}",
    )


# ----------------------------------------------------------------------
# T3 — bare radial mass from Hessian
# ----------------------------------------------------------------------


def test_T3_radial_mass(res: Result, c2: float, c4: float,
                         r_target: float) -> None:
    banner("[T3] Bare radial mass M_rad^2 = -2 c_2 (Hessian eigenvalue)")

    # V(Phi) = c2 kappa(Phi,Phi) + c4 kappa(Phi,Phi)^2, with kappa = delta
    # in the orthonormal Chevalley basis. Hence
    #     H_AB = (2 c2 + 4 c4 |Phi|^2) delta_AB + 8 c4 Phi^A Phi^B.
    # At |Phi|^2 = r_*^2 = -c2/(2 c4) the diagonal piece vanishes and
    # H = 8 c4 Phi_0 Phi_0^T has rank 1, eigenvalue 8 c4 r_*^2 = -4 c2 along
    # the radial direction Phi_0, and zero elsewhere. The bare radial mass
    # in the no-1/2 convention is M_rad^2 = (1/2) V''_eff(r_*) = -2 c_2.
    rng = np.random.default_rng(0xE24)
    Phi0 = random_orbit_point(rng, r_target)

    n2 = float(Phi0 @ Phi0)
    H = (2 * c2 + 4 * c4 * n2) * np.eye(DIM_E8) + 8 * c4 * np.outer(Phi0, Phi0)

    n_hat = Phi0 / np.linalg.norm(Phi0)
    lam_radial = float(n_hat @ H @ n_hat)
    M2_rad = 0.5 * lam_radial  # no-1/2 convention of the article

    expected = -2 * c2
    rel = abs(M2_rad - expected) / max(abs(expected), 1e-30)

    # The 247 directions orthogonal to Phi_0 must give Hessian eigenvalue 0
    # (Goldstone + spectator flat directions, Lemma 4.y "spectator-flat").
    #
    # Build an orthonormal basis of n_hat^perp via QR on a random matrix,
    # then verify max |H v| = 0.
    A = rng.standard_normal((DIM_E8, DIM_E8 - 1))
    A -= np.outer(n_hat, n_hat @ A)
    Q, _ = np.linalg.qr(A)  # 248 x 247, columns orthonormal, all in n_hat^perp
    HQ = H @ Q
    max_perp = float(np.max(np.linalg.norm(HQ, axis=0)))

    ok = rel < TOL_HESS and max_perp < TOL_HESS * max(abs(c4) * n2, 1.0)
    res.report(
        "M_rad^2 = -2 c_2  and  H |_{Phi_0^perp} = 0",
        ok,
        f"M_rad^2 = {M2_rad:.6f} (expected {expected:.6f}, "
        f"rel. err = {rel:.2e}); ‖H v‖_max on Phi_0^perp = {max_perp:.2e}",
    )


# ----------------------------------------------------------------------
# T4 — Goldstone count for the generic semisimple stratum
# ----------------------------------------------------------------------


def test_T4_goldstone_count(res: Result, r_target: float,
                             f_idx: np.ndarray, f_val: np.ndarray) -> None:
    banner("[T4] Goldstone count: rank(ad_{Phi_0}) = 240 (T^8 stratum)")

    # For Phi_0 in the generic semisimple stratum (T^8 row of
    # tab:condensate:stabilizers), Stab(Phi_0) = T^8 has dim 8 and the
    # adjoint orbit has dim 240. Equivalently,
    #     ker(ad_{Phi_0}) = Lie(Stab(Phi_0)),    rank(ad_{Phi_0}) = 240.
    # A uniformly random Phi_0 on S^{247} hits the generic stratum almost
    # surely; we sample a handful and report the modal rank.
    rng = np.random.default_rng(0xE25)
    n_samples = 5
    ranks = []
    for _ in range(n_samples):
        Phi0 = random_orbit_point(rng, r_target)
        ad = build_ad_matrix(Phi0, f_idx, f_val)
        # Use SVD with absolute tolerance scaled by the operator norm.
        s = np.linalg.svd(ad, compute_uv=False)
        thresh = RANK_TOL * s[0]
        ranks.append(int(np.sum(s > thresh)))

    rank_mode = max(set(ranks), key=ranks.count)
    expected = 240  # = dim(E_8) - dim(T^8) = 248 - 8
    ok = rank_mode == expected and all(r == expected for r in ranks)
    res.report(
        f"rank(ad_{{Phi_0}}) = {expected} on the generic stratum (n={n_samples})",
        ok,
        f"observed ranks = {ranks}",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("E2 — symmetry-breaking vacuum (BEC analogue)")
    print("Reference: sections/04-condensate.tex "
          "(thm:condensate:existence,")
    print("           prop:condensate:rad-higgs, lem:condensate:goldstone-count)")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    # Working choice c_2 < 0, c_4 > 0 (Convention A of A1-conventions.tex):
    # the absolute scale drops out of every test below and only fixes r_*.
    c2 = -1.0
    c4 = 0.5
    r_sq = -c2 / (2 * c4)
    r_target = float(np.sqrt(r_sq))
    print(f"\n[setup] c_2 = {c2}, c_4 = {c4}  =>  r_*^2 = {r_sq:.4f}, "
          f"r_* = {r_target:.6f}")

    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))

    test_T1_existence(res, c2, c4)
    test_T3_radial_mass(res, c2, c4, r_target)
    test_T4_goldstone_count(res, r_target, f_idx, f_val)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"E2 summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed == 0:
        print("\n[PASS] E2 — BEC vacuum content verified "
              "(existence, radial mass, Goldstone count).")
        return 0
    else:
        print("\n[FAIL] E2 — see sections/04-condensate.tex for the predicted values.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
