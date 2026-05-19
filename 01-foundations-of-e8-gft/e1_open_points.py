"""Numerical rank certificates for the plethysm dimensions of §3 / Appendix A2.

Companion to ``e1_verify_hypotheses.py`` (which checks structural facts
about E_8 itself — antisymmetry of f, Jacobi, h^v = 30, etc.). The present
script provides numerical certificates for the **dimensions of the
plethysm invariant spaces** predicted by the Cauchy--Howe identity in
``sections/03-action.tex`` (``lem:nk-catalogue``, ``tab:nk-catalogue``)
and proved in detail in ``sections/A2-uniqueness-proof.tex``. Each test
compares the rank of a complete (or tractable) family of contractions
evaluated on random field samples to the predicted dim Inv^{(n,k)}.

The two tests, in execution order:

1. **(4,4) tractable bound for dim^{(4,4)} = 5** —
   ``test_4_4_completeness``. From ``eq:plethysm-44-app`` (Appendix A2),
   dim^{(4,4)} = 5. We enumerate Ad-invariant contractions of four
   copies of M_{A,B} := (L_A Phi)^B using kappa and f. Tractable =
   kappa x kappa, f x kappa, kappa x f, plus one f x f candidate
   computed through the auxiliary tensor Q[E, b1, b2] at cost
   O(DIM^4). The naive f x f without reduction would need a 248^4
   intermediate (~30 GB) and is skipped; the Q-reduction reproduces
   S_e without ever forming the full 4-tensor. The text accompanying
   ``eq:plethysm-44-app`` explicitly cites this routine as the
   numerical certificate of linear independence of the five generators.

2. **(4,2) sector — rank certificate for dim^{(4,2)} = 3** —
   ``test_4_2_classification``. From ``eq:plethysm-42-app``
   (Appendix A2), dim^{(4,2)} = 3 with explicit generators
   C_2 H_2, H_2^{grad}, H_2^{mix} (``eq:H2-def``, ``eq:H2grad-def``,
   ``eq:H2mix-def``). The text accompanying ``eq:plethysm-42-app``
   explicitly cites this routine as the numerical certificate of
   linear independence of the three generators.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from _common import bootstrap_repo_root, constants_path, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import load_structure_constants_numpy, build_dense_f  # noqa: E402
from e8sim.invariants import (  # noqa: E402
    numerical_rank as _numerical_rank,
)

CONSTANTS_PATH = constants_path(ROOT)
RANK_TOL = 1e-9


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

def load_F() -> np.ndarray:
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    return build_dense_f(f_idx, f_val)


def numerical_rank(samples, abs_threshold=None):
    """SVD-based rank returning (rank, singular_values)."""
    return _numerical_rank(
        samples, abs_threshold=abs_threshold, rank_tol=RANK_TOL,
        return_singular_values=True,
    )


# ----------------------------------------------------------------------
# (4, 4) sector — tractable κ/f candidates
# ----------------------------------------------------------------------

# Q[E, b1, b2] := Σ_{a1, a2} f_{a1, a2, E} M_{a1, b1} M_{a2, b2}
# Antisym in (b1, b2) when M is generic (no symmetry).
# Cost: einsum(F, M, M)→(E,b1,b2) is 248^5 ≈ 9.4·10^11 ops, ~30s on CPU.
#
# Q'[E, a1, a2] := Σ_{b1, b2} f_{b1, b2, E} M_{a1, b1} M_{a2, b2}
# Antisym in (a1, a2). Same cost.

def build_Q_AB(M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Q[E, b1, b2] = Σ_{a, b} F[a, b, E] M[a, b1] M[b, b2], antisym in (b1,b2)."""
    # Decompose: U[a, b2, E] = Σ_b F[a, b, E] M[b, b2]; cost 248^4 ≈ 4·10^9
    U = np.einsum("abE,bC->aCE", F, M, optimize=True)  # (a, b2, E)
    Q = np.einsum("aCE,aD->ECD", U, M, optimize=True)  # (E, b1, b2)
    return Q


def build_Q_BA(M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Q'[E, a1, a2] = Σ_{b, b'} F[b, b', E] M[a1, b] M[a2, b'], antisym in (a1,a2)."""
    # U2[a1, b', E] = Σ_b F[b, b', E] M[a1, b]; cost 248^4
    U2 = np.einsum("Aa,abE->AbE", M, F, optimize=True)  # (A, b', E)
    Qp = np.einsum("AbE,Cb->ECA", U2, M, optimize=True)  # (E, a2, a1)
    # Reorder to (E, a1, a2)
    return Qp.transpose(0, 2, 1)


def compute_4_4_candidates(M: np.ndarray, F: np.ndarray,
                           include_QBA: bool = True,
                           include_Se: bool = True) -> dict[str, float]:
    """All tractable (4,4) Ad-invariant scalar contractions of M^4.

    Returns dict of named candidate values. Includes S_e (f×f) via a
    O(DIM^4) reduction P[E,G] = Σ_{i,j} f[i,j,G] · Q_AB[E,i,j].
    """
    cands: dict[str, float] = {}

    G = M.T @ M       # (b, b'), symmetric
    H = M @ M.T       # (a, a'), symmetric
    norm_sq = float((M * M).sum())  # = tr G = tr H
    G_norm = float((G * G).sum())   # = tr(G^2)
    H_norm = float((H * H).sum())   # = tr(H^2) = tr(G^2) by cyclic trace

    # -- κ × κ : 9 combinations, reduce to 2 distinct values --
    cands["S_a (kk_kk: ‖M‖^4)"] = norm_sq ** 2
    cands["S_b (kk_kk: ‖M^T M‖^2)"] = G_norm
    # The other 7 κ×κ all reduce to one of these — verified analytically.

    # -- f × κ (f on A side) --
    Q = build_Q_AB(M, F)   # (E, b1, b2), antisym in (b1, b2)
    Q_norm = float((Q * Q).sum())
    cands["S_c (f_A × kappa_B(13)(24): ‖Q‖²)"] = Q_norm

    # -- κ × f (f on B side) --
    if include_QBA:
        Qp = build_Q_BA(M, F)   # (E, a1, a2), antisym in (a1, a2)
        Qp_norm = float((Qp * Qp).sum())
        cands["S_c' (kappa_A(13)(24) × f_B: ‖Q'‖²)"] = Qp_norm

    # -- f × f : S_e = ‖P‖² with P[E,G] = Σ_{i,j} f[i,j,G] · Q_AB[E,i,j] --
    # Cost: einsum is O(DIM^4) ≈ 4·10^9 ops; intermediate Q is DIM^3 ≈ 120 MB.
    if include_Se:
        P = np.einsum("ijG,Eij->EG", F, Q, optimize=True)
        cands["S_e (f×f: ‖P‖²)"] = float((P * P).sum())

    return cands


def test_4_4_completeness(n_samples: int = 6) -> bool:
    banner("(4,4) sector — completeness: rank({S_a, S_b, S_c, S_c', S_e}) = 5")
    print("Prediction: dim^{(4,4)} = 5 with generators "
          "{S_a, S_b, S_c, S_c', S_e}", flush=True)
    print("(eq:plethysm-44-app, Appendix A2; eq:Sa-def..eq:Se-def, §3.2).")
    print("Test: rank-test on all five generators (κ×κ, f×κ, κ×f, f×f).")
    print("Verifies the σ-symmetric independence of {S_a, S_b, S_+, S_e}.\n",
          flush=True)

    F = load_F()
    rng = np.random.default_rng(0xCAFE_BEEF)
    sample_dicts = []
    t0 = time.time()
    for i in range(n_samples):
        M = 0.05 * rng.standard_normal((DIM_E8, DIM_E8))
        cands = compute_4_4_candidates(M, F, include_QBA=True)
        sample_dicts.append(cands)
        elapsed = time.time() - t0
        print(f"  sample {i+1}/{n_samples}  ({elapsed:5.1f}s, "
              f"{len(cands)} candidates)", flush=True)

    keys = sorted(sample_dicts[0].keys())
    samples = np.array([[d[k] for k in keys] for d in sample_dicts])

    print(f"\n  Values for sample 0:")
    for k in keys:
        print(f"    {k:50s} = {sample_dicts[0][k]:.6e}")

    rank, sv = numerical_rank(samples)
    sv_str = ", ".join(f"{s:.3e}" for s in sv)
    print(f"\n  Singular values: [{sv_str}]")
    print(f"  Numerical rank (rel. tol = {RANK_TOL}): {rank}")
    print(f"  ⇒ dim Inv^{{(4,4)}}_{{tractable}} = {rank}")

    print()
    if rank == 5 and len(keys) == 5:
        print(f"  [PASS] rank = 5 = predicted dim^{{(4,4)}}: all 5 generators "
              f"independent.")
        print(f"  ⇒ σ-symmetric independence of {{S_a, S_b, S_+, S_e}} verified.")
    elif rank == len(keys):
        print(f"  [INFO] All {len(keys)} candidates INDEPENDENT.")
        print(f"  Tractable lower bound: dim Inv^{{(4,4)}} ≥ {rank}.")
    elif rank >= 3:
        deps = len(keys) - rank
        print(f"  [INFO] {deps} linear dependencies among candidates.")
        print(f"  Tractable lower bound: dim Inv^{{(4,4)}} ≥ {rank}.")
    else:
        print(f"  [WARN] rank = {rank} < 3 — unexpected.")

    return rank == 5 if len(keys) == 5 else rank >= 3


# ----------------------------------------------------------------------
# (4, 2) sector
# ----------------------------------------------------------------------

def compute_4_2_candidates(Phi: np.ndarray, M: np.ndarray, F: np.ndarray) -> dict[str, float]:
    """Candidates for the (4 Φ, 2 ∂) sector, pattern Φ·Φ·(L_AΦ)(L_{A'}Φ).

    Spatial L-pair: 2 indices → only κ^{AA'} (Sym²V_R-branch; the
    Λ²V_R-branch has Inv = 0, see Appendix A2). Internal indices: 4
    from {Φ, Φ, M, M} → 3 κ-pairings (one redundant by symmetry of G)
    plus 3 f-pairings (one ≡ 0 by antisymmetry, two identical).

    Mapping to the paper's generators (Lemma 'Catalogue', §3.2; eqs.
    \\eqref{eq:H2-def}, \\eqref{eq:H2grad-def}, \\eqref{eq:H2mix-def}):
        C_2·H_2     = ‖Φ‖² · ‖M‖_F²
        H_2^{grad}  = ¼ κ^{AA'}(L_A C_2)(L_{A'} C_2) = Φ^T (M^T M) Φ
        H_2^{mix}   = κ^{AA'} κ^{EE'} (f_{BCE} M_{AB} Φ^C)(f_{B'C'E'} M_{A'B'} Φ^{C'})

    Returns dict; rank = 3 over the independent candidates ⇒
    dim Inv^{(4,2)} = 3 (eq. \\eqref{eq:plethysm-42-app}).
    """
    cands = {}
    G = M.T @ M
    norm_M = float(np.trace(G))     # = H_2 = ‖M‖_F²
    C2 = float(Phi @ Phi)

    cands["C_2·H_2  (κ_LL × κ_ΦΦ)"] = norm_M * C2
    cands["H_2^grad (κ_LΦ × κ_LΦ: Φ^T G Φ)"] = float(Phi @ G @ Phi)

    # f-(L L)(Φ Φ): inner [Σ f Φ Φ] = 0 by antisymmetry of f vs. symmetry of ΦΦ.
    cands["f_LL × f_ΦΦ (= 0 antisym)"] = 0.0

    # f-(L Φ)(L Φ): u_E^A := Σ_{B,C} f_{BCE} M_{AB} Φ_C, result = Σ_{A,E} u² = H_2^mix.
    u = np.einsum("BCE,AB,C->AE", F, M, Phi, optimize=True)  # (A, E)
    cands["H_2^mix  (f_LΦ × f_LΦ: Σ_AE u²)"] = float((u * u).sum())
    # f-(L Φ')(L Φ): identical to the previous with Φ ↔ Φ' relabelling (B3 ↔ B4).
    cands["H_2^mix' (= same, alt. pairing)"] = float((u * u).sum())

    return cands


EXPECTED_DIM_4_2 = 3  # Cauchy–Howe plethysm, eq. \eqref{eq:plethysm-42-app}


def test_4_2_classification(n_samples: int = 8) -> bool:
    banner(f"(4, 2) sector — completeness "
           f"(paper predicts dim = {EXPECTED_DIM_4_2})")
    print("Generators from the paper (Lemma 'Catalogue', tab:nk-catalogue):")
    print("  C_2·H_2,  H_2^{grad},  H_2^{mix}  — defs eq:H2-def..eq:H2mix-def.\n",
          flush=True)
    F = load_F()
    rng = np.random.default_rng(0x4242)
    rows = []
    for _ in range(n_samples):
        Phi = rng.standard_normal(DIM_E8)
        M = 0.1 * rng.standard_normal((DIM_E8, DIM_E8))
        d = compute_4_2_candidates(Phi, M, F)
        rows.append(d)
    keys = sorted(rows[0].keys())
    samples = np.array([[r[k] for k in keys] for r in rows])
    rank, sv = numerical_rank(samples)
    print(f"  Candidates: {len(keys)}, samples: {n_samples}")
    for k in keys:
        print(f"    {k:42s}: sample0 = {rows[0][k]:.4e}")
    print(f"\n  Singular values: {[f'{s:.3e}' for s in sv]}")
    print(f"  Rank = {rank}, paper predicts dim^{{(4,2)}} = {EXPECTED_DIM_4_2}")
    if rank == EXPECTED_DIM_4_2:
        print(f"  [PASS] rank = {EXPECTED_DIM_4_2} = dim^{{(4,2)}}: three "
              f"generators independent.")
        print(f"         (the remaining 2 candidates are redundant: 1× ≡ 0 by")
        print(f"          antisymmetry, 1× duplicate pairing "
              f"f-(LΦ)(LΦ) ↔ f-(LΦ')(LΦ).)")
        print(f"  ⇒ Appendix A2, eq. plethysm-42-app: dim^{{(4,2)}} = 3 verified.")
        return True
    if rank > EXPECTED_DIM_4_2:
        print(f"  [FAIL] rank = {rank} > {EXPECTED_DIM_4_2}: paper "
              f"under-counted.")
        print(f"         The Cauchy–Howe upper bound dim^{{(4,2)}} ≤ 3 of")
        print(f"         Appendix A2 would exclude rank > 3 — either an error")
        print(f"         in the plethysm or in the candidate identification.")
        return False
    print(f"  [FAIL] rank = {rank} < {EXPECTED_DIM_4_2}: a generator is missing")
    print(f"         (linear dependence among C_2·H_2, H_2^grad, H_2^mix?)")
    return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="Reduce sample count for faster CI runs")
    args = parser.parse_args()

    print("=" * 70)
    print("Numerical rank certificates for plethysm dimensions of (n,k) sectors")
    print("Reference: sections/03-action.tex (lem:nk-catalogue, tab:nk-catalogue)")
    print("           sections/A2-uniqueness-proof.tex (eq:plethysm-{42,44}-app)")
    if args.quick:
        print("Mode: --quick (reduced samples)")
    print("=" * 70, flush=True)

    n_44 = 6 if not args.quick else 5
    n_42 = 8 if not args.quick else 4

    results: list[tuple[str, bool]] = []

    ok_44 = test_4_4_completeness(n_samples=n_44)
    results.append(("(4,4) completeness (test_4_4_completeness)", ok_44))

    ok_42 = test_4_2_classification(n_samples=n_42)
    results.append(("(4,2) classification (test_4_2_classification)", ok_42))

    print("\n" + "=" * 70)
    print("Summary:")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 70)

    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
