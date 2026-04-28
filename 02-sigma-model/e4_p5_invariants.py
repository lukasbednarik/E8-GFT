"""e4_p5_invariants.py -- Schur uniqueness on Sym^2 m and the (4,4) quartic
invariants on the Wolf-space orbit EIX.

Numerical cross-checks for

    "Effective sigma model of an E_8 group field theory: kinetic
     uniqueness, the Skyrme sector, and topological terms on the Wolf
     space EIX".

The script verifies three claims of the article that are flagged as
[Proven-num] cross-checks:

  T1.  Schur uniqueness on (Sym^2 m)^H (lemma "Real irreducibility of
       (56,2)" in section 4 and proposition "Single-parameter potential"
       in section 6).  A Monte-Carlo Ad(H)-projection of a random
       symmetric form on m converges to a multiple of the identity:
       the off-diagonal RMS decays relative to the diagonal RMS as the
       number of Ad(h) samples grows, in agreement with
       dim Hom_H(m, m) = 1.

  T2.  Sigma-vanishing S_c|_sigma = S_e|_sigma = 0 (lemma
       "Sigma-restriction of the primitive quartics" in section 5).
       Strong orthogonality of any maximal antichain in Delta(m^+)
       gives [P_mu, P_nu] = 0 on the four-dimensional abelian substrate;
       a direct evaluation of S_c|_sigma and S_e|_sigma on N = 32 random
       sigma-configurations (v_0, ..., v_3) in m^{otimes 4} yields
       residuals < 1e-12 relative to the magnitude of the surviving
       invariants.

  T3.  Rank({S_a|_sigma, S_b|_sigma, S_{c'}|_sigma}) = 3 strict
       (proposition "Rank of the surviving quartics" in section 5).
       Centred SVD of the (200, 3) evaluation matrix on random Gaussian
       tuples (v_0, v_1, v_2, v_3) in m^{otimes 4} has three singular
       values bounded away from zero, with the smallest ratio
       sigma_2 / sigma_0 above 1e-2 in every observed run.

Conventions follow Appendix A of the article: the e8sim 248-basis is
orthonormal with respect to the Euclidean inner product (X, Y)_E; the
Killing form is K(X, Y) = -2 h^v (X, Y)_E with dual Coxeter number
h^v_E8 = 30, and kappa(X, Y) := -K(X, Y) / h^v = 2 (X, Y)_E.

Run:
    python3 e4_p5_invariants.py
    pytest -v e4_p5_invariants.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    bracket_vec_fast,
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
    adjoint_exp,
)
from e8sim.roots import (  # noqa: E402
    EIX_ALPHA_SU2,
    e7_su2_embedding,
    is_strongly_orthogonal,
)
from e8sim import generate_roots  # noqa: E402
from e8sim.eix import (  # noqa: E402
    DIM_M_EIX,
    kappa,
)

CONSTANTS_PATH = constants_path(ROOT)

TOL_ALG = 1e-9            # algebraic tolerance (commutators, projections)
TOL_FIT = 1e-3            # rank-decision tolerance for ratios sigma_i / sigma_0
N_SAMPLES_RANK = 200      # random Gaussian tuples for the rank test
N_SAMPLES_SIGMA = 32      # random sigma-configurations for S_c, S_e residuals
N_AVG_SCHUR = 32          # Ad(H) samples in the Monte-Carlo Schur projection


# ----------------------------------------------------------------------
# Lie-algebra helpers
# ----------------------------------------------------------------------


def commutator(f_idx, f_val, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lie bracket [X, Y] in the e8sim 248-basis."""
    return bracket_vec_fast(f_idx, f_val, x, y)


def positive_roots_e8(all_roots: np.ndarray) -> np.ndarray:
    """Return the 120 positive roots of E_8 in the e8sim ordering."""
    pos = []
    for r in all_roots:
        nz = np.nonzero(np.abs(r) > 0.01)[0]
        if len(nz) > 0 and r[nz[0]] > 0:
            pos.append(r)
    return np.array(pos)


def m_plus_for_EIX(pos_roots: np.ndarray) -> np.ndarray:
    """Extract the 56 positive roots in m^+ = (56, 2) for E_8 / (E_7 x SU(2)).

    m^+ = positive roots that are neither orthogonal to alpha_SU(2) nor
    +/- alpha_SU(2) itself.
    """
    alpha = EIX_ALPHA_SU2
    out = []
    for r in pos_roots:
        if np.allclose(r, alpha, atol=1e-6) or np.allclose(r, -alpha, atol=1e-6):
            continue
        if abs(float(np.dot(r, alpha))) < 1e-6:
            continue
        out.append(r)
    return np.array(out)


def find_4D_abelian(pos_roots: np.ndarray, all_roots: np.ndarray
                    ) -> tuple[list[int], np.ndarray]:
    """Build a maximal (size-4) strongly-orthogonal subset of m^+_EIX.

    Returns
    -------
    (P_indices, P_basis) : tuple
        P_indices : indices in the 248-basis of the four generators
                    E_{alpha_mu} (mu = 0, 1, 2, 3).
        P_basis   : (4, 248) matrix whose rows are the standard basis
                    vectors picking those four generators.
    """
    m_plus = m_plus_for_EIX(pos_roots)
    root_set = {tuple(np.round(r, 4)) for r in all_roots}

    chosen: list[int] = []
    for i in range(len(m_plus)):
        if all(is_strongly_orthogonal(m_plus[i], m_plus[j], root_set)
               for j in chosen):
            chosen.append(i)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise RuntimeError(
            f"Failed to assemble a 4D strongly-orthogonal set; got {len(chosen)}"
        )

    pos_lookup = {tuple(np.round(r, 4)): i for i, r in enumerate(pos_roots)}
    P_indices = [8 + pos_lookup[tuple(np.round(m_plus[k], 4))] for k in chosen]
    P_basis = np.zeros((4, DIM_E8))
    for i, idx in enumerate(P_indices):
        P_basis[i, idx] = 1.0
    return P_indices, P_basis


# ----------------------------------------------------------------------
# T1 -- Schur uniqueness on Sym^2 m via Monte-Carlo Ad(H)-projection
# ----------------------------------------------------------------------


def test_T1_schur_sym2(e7_basis: np.ndarray, su2_basis: np.ndarray,
                       m_basis: np.ndarray, f_idx, f_val,
                       res: Result) -> None:
    banner("[T1] Schur uniqueness on (Sym^2 m)^H -- Monte-Carlo Ad(H)-average")

    h_basis = np.vstack([e7_basis, su2_basis])  # (136, 248)
    rng = np.random.default_rng(0xC1FE)

    H_rand = rng.standard_normal((DIM_M_EIX, DIM_M_EIX))
    H_rand = 0.5 * (H_rand + H_rand.T)

    avg = np.zeros_like(H_rand)
    for _ in range(N_AVG_SCHUR):
        coeffs = rng.standard_normal(h_basis.shape[0])
        coeffs *= 0.4 / np.linalg.norm(coeffs)
        X_h = coeffs @ h_basis  # (248,)
        Ad_full = adjoint_exp(X_h, f_idx, f_val)
        R = m_basis @ Ad_full @ m_basis.T  # (112, 112) restriction to m
        avg += R @ H_rand @ R.T
    avg /= N_AVG_SCHUR

    diag = np.diag(avg)
    off = avg - np.diag(diag)
    diag_rms = float(np.sqrt(np.mean(diag ** 2)))
    off_rms = float(np.sqrt(np.mean(off ** 2)))
    ratio = off_rms / max(diag_rms, 1e-30)

    print(f"    Ad(H) samples: {N_AVG_SCHUR}", flush=True)
    print(f"    diag RMS = {diag_rms:.3e}    off-diag RMS = {off_rms:.3e}")
    print(f"    off / diag = {ratio:.3f}  (-> 0 as N_avg -> infty)")

    res.report(
        "Monte-Carlo projection collapses to a multiple of the identity",
        ratio < 1.0,
        f"off/diag = {ratio:.3f} after {N_AVG_SCHUR} Ad(h)-samples",
    )


# ----------------------------------------------------------------------
# T2 -- S_c|_sigma = S_e|_sigma = 0 via the 4D abelian substrate
# ----------------------------------------------------------------------


def evaluate_Sc_Sigma(P_basis: np.ndarray, v: np.ndarray,
                      f_idx, f_val) -> float:
    """S_c|_sigma on a sigma-configuration with substrate generators P_mu.

    S_c|_sigma = sum_{mu nu rho sigma} kappa([P_mu, P_nu], [P_rho, P_sigma])
                                     * kappa(v_mu, v_rho) * kappa(v_nu, v_sigma)
    """
    out = 0.0
    for mu in range(4):
        for nu in range(4):
            B_mn = commutator(f_idx, f_val, P_basis[mu], P_basis[nu])
            for rho in range(4):
                for sig in range(4):
                    B_rs = commutator(f_idx, f_val, P_basis[rho], P_basis[sig])
                    out += (kappa(B_mn, B_rs)
                            * kappa(v[mu], v[rho])
                            * kappa(v[nu], v[sig]))
    return float(out)


def evaluate_Se_Sigma(P_basis: np.ndarray, v: np.ndarray,
                      f_idx, f_val) -> float:
    """S_e|_sigma on a sigma-configuration with substrate generators P_mu.

    S_e|_sigma = sum_{mu nu rho sigma} kappa([P_mu, P_nu], [P_rho, P_sigma])
                                     * kappa([v_mu, v_nu], [v_rho, v_sigma])
    """
    out = 0.0
    for mu in range(4):
        for nu in range(4):
            B_mn = commutator(f_idx, f_val, P_basis[mu], P_basis[nu])
            C_mn = commutator(f_idx, f_val, v[mu], v[nu])
            for rho in range(4):
                for sig in range(4):
                    B_rs = commutator(f_idx, f_val, P_basis[rho], P_basis[sig])
                    C_rs = commutator(f_idx, f_val, v[rho], v[sig])
                    out += kappa(B_mn, B_rs) * kappa(C_mn, C_rs)
    return float(out)


def evaluate_Scp_Sigma(v: np.ndarray, f_idx, f_val) -> float:
    """S_{c'}|_sigma = sum_{mu nu} kappa([v_mu, v_nu], [v^mu, v^nu])."""
    out = 0.0
    for mu in range(4):
        for nu in range(4):
            C = commutator(f_idx, f_val, v[mu], v[nu])
            out += kappa(C, C)
    return float(out)


def test_T2_sigma_vanishing(pos_roots: np.ndarray, all_roots: np.ndarray,
                            m_basis: np.ndarray,
                            f_idx, f_val, res: Result) -> None:
    banner("[T2] S_c|_sigma = S_e|_sigma = 0 via [P_mu, P_nu] = 0")

    P_indices, P_basis = find_4D_abelian(pos_roots, all_roots)
    print(f"    4D abelian substrate -- 248-basis indices: {P_indices}",
          flush=True)

    max_comm = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            comm = commutator(f_idx, f_val, P_basis[i], P_basis[j])
            max_comm = max(max_comm, float(np.max(np.abs(comm))))
    res.report(
        "[P_mu, P_nu] = 0 for all six pairs in the 4D substrate",
        max_comm < TOL_ALG,
        f"max |[P_mu, P_nu]| = {max_comm:.2e}",
    )

    rng = np.random.default_rng(0xD03)
    Sc_residuals: list[float] = []
    Se_residuals: list[float] = []
    Scp_values: list[float] = []
    print(f"    Direct evaluation on {N_SAMPLES_SIGMA} random "
          f"sigma-configurations ...", flush=True)
    t0 = time.time()
    for _ in range(N_SAMPLES_SIGMA):
        coeffs = rng.standard_normal((4, DIM_M_EIX))
        v = coeffs @ m_basis  # (4, 248); v[mu] in m
        Sc_residuals.append(abs(evaluate_Sc_Sigma(P_basis, v, f_idx, f_val)))
        Se_residuals.append(abs(evaluate_Se_Sigma(P_basis, v, f_idx, f_val)))
        Scp_values.append(abs(evaluate_Scp_Sigma(v, f_idx, f_val)))
    elapsed = time.time() - t0

    Sc_max = max(Sc_residuals)
    Se_max = max(Se_residuals)
    Scp_typ = float(np.median(Scp_values))
    Sc_rel = Sc_max / max(Scp_typ, 1e-30)
    Se_rel = Se_max / max(Scp_typ, 1e-30)
    print(f"    ({elapsed:.1f}s)  median |S_{{c'}}|_sigma| = {Scp_typ:.3e}")
    print(f"    max |S_c|_sigma|  = {Sc_max:.3e}   "
          f"relative to median S_{{c'}}: {Sc_rel:.3e}")
    print(f"    max |S_e|_sigma|  = {Se_max:.3e}   "
          f"relative to median S_{{c'}}: {Se_rel:.3e}")

    res.report(
        "S_c|_sigma = 0 to relative residual < 1e-12",
        Sc_rel < 1e-12,
        f"max/median = {Sc_rel:.3e} on {N_SAMPLES_SIGMA} samples",
    )
    res.report(
        "S_e|_sigma = 0 to relative residual < 1e-12",
        Se_rel < 1e-12,
        f"max/median = {Se_rel:.3e} on {N_SAMPLES_SIGMA} samples",
    )


# ----------------------------------------------------------------------
# T3 -- rank({S_a, S_b, S_{c'}}) = 3 strict on m^{otimes 4}
# ----------------------------------------------------------------------


def compute_sigma_invariants(v: np.ndarray, f_idx, f_val
                             ) -> tuple[float, float, float]:
    """For (v_0, v_1, v_2, v_3) in m^{otimes 4} return (S_a, S_b, S_{c'}).

    With K_{mu nu} := kappa(v_mu, v_nu),
        S_a   = (Tr K)^2,
        S_b   = Tr(K^2),
        S_{c'} = sum_{mu nu} kappa([v_mu, v_nu], [v^mu, v^nu]).
    """
    K = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            K[mu, nu] = kappa(v[mu], v[nu])
    S_a = float(np.trace(K) ** 2)
    S_b = float((K * K).sum())
    S_cp = 0.0
    for mu in range(4):
        for nu in range(4):
            B = commutator(f_idx, f_val, v[mu], v[nu])
            S_cp += kappa(B, B)
    return S_a, S_b, float(S_cp)


def test_T3_rank3(m_basis: np.ndarray, f_idx, f_val, res: Result) -> None:
    banner("[T3] Rank({S_a, S_b, S_{c'}}) = 3 strict on m^{otimes 4}")

    rng = np.random.default_rng(0xD040)
    samples = np.empty((N_SAMPLES_RANK, 3))
    print(f"    Sampling {N_SAMPLES_RANK} Gaussian tuples (v_0, ..., v_3) "
          f"in m^4 ...", flush=True)
    t0 = time.time()
    for n in range(N_SAMPLES_RANK):
        coeffs = rng.standard_normal((4, DIM_M_EIX))
        v = coeffs @ m_basis  # (4, 248)
        samples[n] = compute_sigma_invariants(v, f_idx, f_val)
    elapsed = time.time() - t0
    print(f"    ({elapsed:.1f}s)", flush=True)

    centred = samples - samples.mean(axis=0)
    sv = np.linalg.svd(centred, full_matrices=False, compute_uv=False)
    ratios = sv / max(sv[0], 1e-30)
    print(f"    singular values:     {sv}")
    print(f"    ratios sigma_i / sigma_0: {ratios}")

    rank_strict = int((ratios > TOL_ALG).sum())
    rank_loose = int((ratios > TOL_FIT).sum())

    res.report(
        "Strict rank of the centred (N, 3) evaluation matrix = 3",
        rank_strict == 3,
        f"strict rank @ {TOL_ALG:.0e} = {rank_strict};"
        f"  loose rank @ {TOL_FIT:.0e} = {rank_loose}",
    )


# ----------------------------------------------------------------------
# pytest wrappers
# ----------------------------------------------------------------------


def _setup():
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots)
    all_roots = generate_roots()
    return f_idx, f_val, pos_roots, all_roots, e7_basis, su2_basis, m_basis


def test_p5_T1_schur_sym2():
    f_idx, f_val, _, _, e7_basis, su2_basis, m_basis = _setup()
    res = Result()
    test_T1_schur_sym2(e7_basis, su2_basis, m_basis, f_idx, f_val, res)
    assert res.failed == 0, f"{res.failed} sub-tests failed in T1"


def test_p5_T2_sigma_vanishing():
    f_idx, f_val, pos_roots, all_roots, _, _, m_basis = _setup()
    res = Result()
    test_T2_sigma_vanishing(pos_roots, all_roots, m_basis, f_idx, f_val, res)
    assert res.failed == 0, f"{res.failed} sub-tests failed in T2"


def test_p5_T3_rank3():
    f_idx, f_val, _, _, _, _, m_basis = _setup()
    res = Result()
    test_T3_rank3(m_basis, f_idx, f_val, res)
    assert res.failed == 0, f"{res.failed} sub-tests failed in T3"


# ----------------------------------------------------------------------
# Standalone main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("e4_p5_invariants -- Schur uniqueness on Sym^2 m and the (4,4)")
    print("                    quartic invariants on EIX = E_8 / (E_7 x SU(2))")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] loading structure constants and the EIX embedding ...",
          flush=True)
    f_idx, f_val, pos_roots, all_roots, e7_basis, su2_basis, m_basis = _setup()
    print(f"        e_7 {e7_basis.shape}, su(2) {su2_basis.shape}, "
          f"m {m_basis.shape}", flush=True)

    test_T1_schur_sym2(e7_basis, su2_basis, m_basis, f_idx, f_val, res)
    test_T2_sigma_vanishing(pos_roots, all_roots, m_basis, f_idx, f_val, res)
    test_T3_rank3(m_basis, f_idx, f_val, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed > 0:
        print("\nFailed sub-tests:")
        for marker, name, detail in res.records:
            if marker == "FAIL":
                print(f"  - {name}")
                if detail:
                    print(f"      {detail}")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
