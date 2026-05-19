"""Numerical cross-check of the tree-level Schur
factor for the Skyrme term on EIX.

This script verifies the identity ``eq:quartic:cH-prime`` of
Theorem ``thm:c-H-prime`` (section ``sec:quartic:cH-prime``,
"Tree-level Schur factor c_H' = 1/16") of

    "Effective sigma model of an E_8 group field theory: kinetic
     uniqueness, the Skyrme sector, and topological terms on the Wolf
     space EIX".

For Theta = ad_{V_A}|_m and arbitrary X, Y, X', Y' in m, the theorem
states

    kappa([Theta X, Theta Y], [Theta X', Theta Y'])
        = (c_H r_*^2)^2 * kappa([X, Y], [X', Y']),                  (*)

with c_H = 1/4 and, in the Convention-A normalisation r_*^2 = 1,
c_H'^{(EIX, tree)} = c_H^2 = 1/16.

The numerical check evaluates both sides of (*) in the contracted
sigma-tuple form

    S_phys[X] := sum_{mu nu} kappa([X_mu, X_nu], [X^mu, X^nu]),
    S_GFT [j] := sum_{mu nu} kappa([j_mu, j_nu], [j^mu, j^nu]),

with X_mu = -ad_{V_A}(j_mu) = -Theta(j_mu) on N = 200 random Gaussian
4-tuples (j_0, ..., j_3) in m^4.  Two assertions are reported:

  (a) the global ratio S_phys / S_GFT equals (c_H r_*^2)^2 to standard
      deviation ~ machine epsilon (and hence c_H'^{(EIX, tree)} = 1/16
      in Convention A);
  (b) the same ratio holds separately for the e_7- and the
      su(2)-projection of the bracket [X_mu, X_nu], which is the
      empirical content of the identity D^2 = 1 on Sym^2(2) used in
      Step (iv) of the proof of Theorem ``thm:c-H-prime``.

Conventions follow Appendix A of the article: the e8sim 248-basis is
orthonormal with respect to the Euclidean inner product (X, Y)_E; the
Killing form is K(X, Y) = -2 h^v (X, Y)_E with dual Coxeter number
h^v = 30, and kappa(X, Y) := -K(X, Y)/h^v = 2 (X, Y)_E.  V_A is the
unit Cartan generator of the SU(2) factor of the E_7 x SU(2) maximal
subgroup, normalised to (V_A, V_A)_E = 1, hence
r_*^2 := kappa(V_A, V_A) = 2.  In this normalisation (*) yields a
ratio of (c_H r_*^2)^2 = 1/4; the dimensionless tree-level Schur
factor c_H'^{(EIX, tree)} = ratio / r_*^4 = 1/16 in Convention A is
reported as a derived quantity.

Run:
    python3 eo5_skyrme_schur.py
    pytest -v eo5_skyrme_schur.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    bracket_vec_fast,
    build_ad_matrix,
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
)
from e8sim.eix import (  # noqa: E402
    C_H_EIX,
    canonical_VA,
    kappa,
)
from e8sim.roots import e7_su2_embedding  # noqa: E402

CONSTANTS_PATH = constants_path(ROOT)

TOL_ALG = 1e-9        # algebraic tolerance (machine epsilon scale)
N_TRIALS = 200        # random sigma-tuples for the global identity
N_TRIALS_PROJ = 100   # random sigma-tuples for the per-component identity
RNG_SEED_GLOBAL = 0xD050
RNG_SEED_PROJ = 0xD054


def _random_m_vector(rng: np.random.Generator,
                     m_basis: np.ndarray) -> np.ndarray:
    """Random Gaussian vector in m = span(m_basis), expressed in the
    248-component e8sim basis."""
    return rng.standard_normal(m_basis.shape[0]) @ m_basis


def _skyrme_invariant(f_idx: np.ndarray, f_val: np.ndarray,
                      X_tuple: list[np.ndarray]) -> float:
    """Sum_{mu, nu} kappa([X_mu, X_nu], [X^mu, X^nu]) for a 4-tuple X.

    The two index ranges run independently over (0, 1, 2, 3); in the
    Euclidean signature used here this is the contraction relevant to
    the Skyrme four-derivative invariant on EIX.
    """
    total = 0.0
    n = len(X_tuple)
    for mu in range(n):
        for nu in range(n):
            B = bracket_vec_fast(f_idx, f_val, X_tuple[mu], X_tuple[nu])
            total += kappa(B, B)
    return float(total)


def test_skyrme_schur_identity(m_basis: np.ndarray,
                               e7_basis: np.ndarray,
                               su2_basis: np.ndarray,
                               f_idx: np.ndarray,
                               f_val: np.ndarray,
                               res: Result) -> None:
    """Numerical content of Theorem ``thm:c-H-prime``: the four-tensor
    Schur factor on the EIX isotropy module m = (56, 2)."""
    banner("Tree-level Schur factor on EIX:  c_H'^{(EIX, tree)} = c_H^2 = 1/16")

    V_A, r_sq = canonical_VA()
    ad_VA = build_ad_matrix(V_A, f_idx, f_val)
    pred_ratio = (C_H_EIX * r_sq) ** 2          # = 1/4 in r_*^2 = 2 normalisation
    pred_cH_prime = C_H_EIX ** 2                # = 1/16 in Convention A (r_*^2 = 1)

    rng = np.random.default_rng(RNG_SEED_GLOBAL)
    ratios = np.empty(N_TRIALS, dtype=float)
    for k in range(N_TRIALS):
        j_tuple = [_random_m_vector(rng, m_basis) for _ in range(4)]
        X_tuple = [-(ad_VA @ j) for j in j_tuple]
        S_GFT = _skyrme_invariant(f_idx, f_val, j_tuple)
        S_phys = _skyrme_invariant(f_idx, f_val, X_tuple)
        ratios[k] = S_phys / S_GFT

    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    cH_prime_meas = mean_ratio / r_sq ** 2

    res.report(
        "Skyrme Schur identity:  S_phys / S_GFT = (c_H r_*^2)^2",
        abs(mean_ratio - pred_ratio) < TOL_ALG and std_ratio < TOL_ALG,
        f"mean = {mean_ratio:.12f}, std = {std_ratio:.2e}  "
        f"(predicted {pred_ratio:.6f}; N = {N_TRIALS}; seed 0x{RNG_SEED_GLOBAL:X})",
    )
    res.report(
        "Tree-level Schur factor:  c_H'^{(EIX, tree)} = c_H^2 = 1/16",
        abs(cH_prime_meas - pred_cH_prime) < TOL_ALG,
        f"measured c_H'^{{(EIX, tree)}} = {cH_prime_meas:.12f} "
        f"(predicted {pred_cH_prime:.6f}; Convention A r_*^2 = 1)",
    )

    rng = np.random.default_rng(RNG_SEED_PROJ)
    e7_ratios = np.empty(N_TRIALS_PROJ, dtype=float)
    su2_ratios = np.empty(N_TRIALS_PROJ, dtype=float)
    for k in range(N_TRIALS_PROJ):
        j_tuple = [_random_m_vector(rng, m_basis) for _ in range(4)]
        X_tuple = [-(ad_VA @ j) for j in j_tuple]

        S_e7_GFT = S_su2_GFT = 0.0
        S_e7_phys = S_su2_phys = 0.0
        for mu in range(4):
            for nu in range(4):
                B_GFT = bracket_vec_fast(f_idx, f_val,
                                         j_tuple[mu], j_tuple[nu])
                B_phys = bracket_vec_fast(f_idx, f_val,
                                          X_tuple[mu], X_tuple[nu])

                B_GFT_e7 = e7_basis.T @ (e7_basis @ B_GFT)
                B_GFT_su2 = su2_basis.T @ (su2_basis @ B_GFT)
                B_phys_e7 = e7_basis.T @ (e7_basis @ B_phys)
                B_phys_su2 = su2_basis.T @ (su2_basis @ B_phys)

                S_e7_GFT += kappa(B_GFT_e7, B_GFT_e7)
                S_su2_GFT += kappa(B_GFT_su2, B_GFT_su2)
                S_e7_phys += kappa(B_phys_e7, B_phys_e7)
                S_su2_phys += kappa(B_phys_su2, B_phys_su2)

        e7_ratios[k] = S_e7_phys / S_e7_GFT
        su2_ratios[k] = S_su2_phys / S_su2_GFT

    mean_e7 = float(np.mean(e7_ratios))
    mean_su2 = float(np.mean(su2_ratios))
    std_e7 = float(np.std(e7_ratios))
    std_su2 = float(np.std(su2_ratios))

    res.report(
        "e_7-projection of [X, Y] yields the same Schur factor (c_H r_*^2)^2",
        abs(mean_e7 - pred_ratio) < TOL_ALG and std_e7 < TOL_ALG,
        f"mean = {mean_e7:.12f}, std = {std_e7:.2e}",
    )
    res.report(
        "su(2)-projection of [X, Y] yields the same Schur factor (c_H r_*^2)^2",
        abs(mean_su2 - pred_ratio) < TOL_ALG and std_su2 < TOL_ALG,
        f"mean = {mean_su2:.12f}, std = {std_su2:.2e}",
    )
    res.report(
        "Equality of the two projections (empirical content of D^2 = 1 on Sym^2(2))",
        abs(mean_e7 - mean_su2) < TOL_ALG,
        f"|<e_7> - <su(2)>| = {abs(mean_e7 - mean_su2):.2e}  "
        f"(N = {N_TRIALS_PROJ}; seed 0x{RNG_SEED_PROJ:X})",
    )


def main() -> int:
    print("=" * 70)
    print("Tree-level Schur factor for the Skyrme term on EIX")
    print("Cross-check for Theorem thm:c-H-prime, eq:quartic:cH-prime of:")
    print("'Effective sigma model of an E_8 group field theory: kinetic")
    print(" uniqueness, the Skyrme sector, and topological terms on the")
    print(" Wolf space EIX'.")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8 structure constants and EIX basis ...",
          flush=True)
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots)

    test_skyrme_schur_identity(m_basis, e7_basis, su2_basis,
                               f_idx, f_val, res)

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


def test_cH_prime_tree() -> None:
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots)
    res = Result()
    test_skyrme_schur_identity(m_basis, e7_basis, su2_basis,
                               f_idx, f_val, res)
    assert res.failed == 0, f"{res.failed} sub-test(s) failed"


if __name__ == "__main__":
    sys.exit(main())
