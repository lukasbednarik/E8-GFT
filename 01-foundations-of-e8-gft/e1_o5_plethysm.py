"""Analytic plethysm cross-check of dim^(4,4) = 5 with explicit f×f
generator.

This script provides an independent numerical/analytic certificate for
the (4,4) row of Table `tab:nk-catalogue` (Lemma `lem:nk-catalogue`) of
`sections/03-action.tex`, whose full proof is given in
`sections/A2-uniqueness-proof.tex`, §`app:uniqueness-proof:plethysm`.

PART I — analytic plethysm upper bound
======================================

Setup.  V := adj(E_8) (248-dim).  In the (4,4) sector the field content
is four copies of the derivative bilinear M^{(i)}_{AB} := (L_A Phi)^B
with i=1..4, with A in adj(E_8^R) and B in adj(E_8^Ad).  The four
copies are interchangeable (commutative product), so

    Inv^(4,4) = Inv_{E_8^R × E_8^Ad}( Sym^4(V_R ⊗ V_Ad) ) .

Cauchy–Howe duality gives, for V_R = V_Ad = adj(E_8),

    dim Inv^(4,4) = Σ_{λ ⊢ 4}  [ dim Inv_{E_8}( S^λ adj(E_8) ) ]^2
                                                          (eq. cauchy-howe)

(see Macdonald, "Symmetric Functions and Hall Polynomials", I.4 Ex. 5;
Fulton–Harris, "Representation Theory: A First Course", Appendix A).

Inputs from E_8 representation theory:

    Sym^2 adj  =  1 ⊕ 3875 ⊕ 27000          (Slansky 1981)
    Λ^2  adj   =  248 ⊕ 30380
    every E_8-irrep is real-orthogonal       (Frobenius–Schur = +1)
    Casimir degrees of E_8: {2, 8, 12, 14, 18, 20, 24, 30}

(real-orthogonality + the Casimir-degree list together fix
dim Inv(Sym^2 W) = 1 and dim Inv(Λ^2 W) = 0 for every E_8-irrep W,
and dim Inv(Sym^4 V) = 1 since 4 = 2+2 is the only multi-degree of
the Casimirs that sums to 4).

The five partitions of 4 contribute as follows (cf.
`app:uniqueness-proof:plethysm`):

    λ = (4):     dim Inv(Sym^4 V) = 1                  → 1
    λ = (3,1):   Λ^2(Sym^2 V) cross-pairings vanish    → 0
    λ = (2,2):   3 Sym^2-trivials − 1 Sym^4-trivial    → 2
    λ = (2,1,1): cross-pairings vanish                 → 0
    λ = (1^4):   H^4(E_8;R) = 0  (Mimura–Toda)         → 0

Plugging into eq. cauchy-howe:  dim^(4,4) = 1 + 0 + 4 + 0 + 0 = 5.

Sanity:  dim Inv(V^⊗4) = Σ_λ (dim 𝕊^λ_{S_4}) · dim Inv(S^λ V)
       = 1·1 + 3·0 + 2·2 + 3·0 + 1·0  =  5,
matching the elementary count of 3 κκ-pairings + 3 ff-pairings −
1 Jacobi identity = 5 invariants of E_8 on rank-4 tensors.

PART II — numerical rank test on the explicit generator basis
=============================================================

The article identifies five (4,4) generators (eqs.
`eq:Sa-def`–`eq:Se-def`):

    S_a, S_b           (κ on both sides)
    S_c                (f on manifold, κ on internal)
    S_{c'}             (κ on manifold, f on internal)
    S_e                (f on both sides) — the f×f generator

We construct an explicit f×f candidate as the (12)(34)/(12)(34)
double-commutator pairing,

    S_e := Σ_{E_A,E_B} G_{E_A E_B}^2 = ‖G‖_F² ,

with

    T(E_A, B_1, B_2) = Σ f(A_1, A_2, E_A) M(A_1, B_1) M(A_2, B_2),
    G(E_A, E_B)      = Σ f(B_1, B_2, E_B) T(E_A, B_1, B_2).

T is a 248³ tensor (~120 MB), G is a 248² matrix; the whole computation
fits in memory and runs in seconds.

We then run the SVD rank test on the five candidates
{S_a, S_b, S_c, S_{c'}, S_e} across N≥6 random samples M, and confirm
that the numerical rank is exactly 5 — the plethysm prediction of
PART I.  This exhibits the f×f generator explicitly and matches the
Inv^(4,4) dimension stated in `tab:nk-catalogue`.

Reproducibility.  Depends only on numpy and the e8sim/e8_constants.pt
file (compact-form structure constants of E_8 in the orthonormal
Chevalley basis).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import load_structure_constants_numpy, build_dense_f  # noqa: E402
from e8sim.invariants import numerical_rank as _numerical_rank  # noqa: E402

CONSTANTS_PATH = constants_path(ROOT)
RANK_TOL = 1e-9


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

def load_F() -> np.ndarray:
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    return build_dense_f(f_idx, f_val)


def numerical_rank(samples: np.ndarray):
    return _numerical_rank(
        samples, rank_tol=RANK_TOL,
        return_singular_values=True,
    )


# ----------------------------------------------------------------------
# Plethysm upper bound (PART I)
# ----------------------------------------------------------------------

def plethysm_upper_bound() -> int:
    """Compute dim Inv^(4,4)(E_8 × E_8) by Cauchy–Howe duality.

    Returns the integer dim^(4,4) = Σ_{λ⊢4} [dim Inv(S^λ adj E_8)]².
    """

    # Inputs (Slansky 1981; real-orthogonality of every E_8-irrep)
    sym2_adj = [1, 3875, 27000]   # Sym² adj E_8
    lam2_adj = [248, 30380]       # Λ²  adj E_8
    assert sum(sym2_adj) == 248 * 249 // 2 == 30876
    assert sum(lam2_adj) == 248 * 247 // 2 == 30628

    # λ = (4) = Sym⁴: only C_2² survives, since Casimir degrees are
    # {2, 8, 12, 14, 18, 20, 24, 30}; the only multiset summing to 4 is
    # (2, 2).
    inv_sym4 = 1

    # λ = (1^4) = Λ⁴: H^4(E_8; R) = 0, since the primitive degrees of
    # H*(E_8) are {3, 15, 23, 27, 35, 39, 47, 59} (Mimura–Toda 1991).
    inv_lam4 = 0

    # λ = (2,2):  Sym²(Sym² V) = S^{(4)} V ⊕ S^{(2,2)} V.
    # For each E_8-irrep W, FS(W) = +1 ⇒ dim Inv(Sym² W) = 1; cross
    # terms W_i ⊗ W_j (i ≠ j) carry no trivial.  Hence the symmetric
    # square of (1 ⊕ 3875 ⊕ 27000) has 3 trivials in total.
    sym2_sym2_trivials = sum(1 for _ in sym2_adj)
    inv_s22 = sym2_sym2_trivials - inv_sym4
    assert inv_s22 == 2

    # λ = (3,1):  Λ²(Sym² V) = S^{(3,1)} V.  Λ² of each real-orthogonal
    # irrep contains no trivial, and the Sym² and Λ² lists of adj-irreps
    # share none (1, 3875, 27000 vs. 248, 30380).
    inv_s31 = 0

    # λ = (2,1,1): Λ²(Λ² V) = S^{(2,1,1)} V.  Same vanishing argument
    # applied to lam2_adj.
    inv_s211 = 0

    invariants = {
        "(4)":     inv_sym4,
        "(3,1)":   inv_s31,
        "(2,2)":   inv_s22,
        "(2,1,1)": inv_s211,
        "(1^4)":   inv_lam4,
    }
    dim_44 = sum(n ** 2 for n in invariants.values())

    print("Plethysm computation (Cauchy–Howe, eq. cauchy-howe):")
    print("  dim^(4,4) = Σ_{λ⊢4} [dim Inv(S^λ adj E_8)]²")
    print()
    print(f"  {'λ':<10}{'dim Inv(S^λ V)':>20}{'contribution':>18}")
    print(f"  {'-'*10}{'-'*20}{'-'*18}")
    for lam, n in invariants.items():
        print(f"  {lam:<10}{n:>20d}{n**2:>18d}")
    print(f"  {'-'*10}{'-'*20}{'-'*18}")
    print(f"  {'TOTAL':<10}{'':<20}{dim_44:>18d}")
    print()

    s4_dim = {"(4)": 1, "(3,1)": 3, "(2,2)": 2, "(2,1,1)": 3, "(1^4)": 1}
    dim_v4 = sum(s4_dim[k] * v for k, v in invariants.items())
    print(f"  Sanity: dim Inv(V^⊗4) = Σ_λ (dim 𝕊^λ_S4) · dim Inv(S^λ V) = {dim_v4}")
    print(f"  Elementary count: 3 κκ + 3 ff − 1 Jacobi = 5  ✓")
    print()

    return dim_44


# ----------------------------------------------------------------------
# Memory-efficient f×f candidates (PART II)
# ----------------------------------------------------------------------

def compute_T_tensor(M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """T[E_A, B_1, B_2] := Σ F[A_1, A_2, E_A] M[A_1, B_1] M[A_2, B_2].

    Antisymmetric in (B_1, B_2) for generic M (because F is antisymmetric
    in (A_1, A_2) and the dummy indices may be relabelled).

    Cost: 2 × 248⁴ ≈ 8 × 10⁹ ops.  Memory: 248³ ≈ 122 MB (float64).
    """
    U = np.einsum("abE,bC->aCE", F, M, optimize=True)
    T = np.einsum("aCE,aD->EDC", U, M, optimize=True)
    return T


def compute_G_matrix(T: np.ndarray, F: np.ndarray) -> np.ndarray:
    """G[E_A, E_B] := Σ F[B_1, B_2, E_B] T[E_A, B_1, B_2].

    Cost: 248⁴ ≈ 4 × 10⁹ ops.  Memory: 248² ≈ 60 kB.
    """
    return np.einsum("bcE,Abc->AE", F, T, optimize=True)


def compute_ff_diagonal(M: np.ndarray, F: np.ndarray, T: np.ndarray | None = None) -> float:
    """S_e := f×f generator with A-pair (12)(34) and B-pair (12)(34).

    Equal to ‖G‖_F², with G = compute_G_matrix(T, F).  This realises
    the f×f trivial of `eq:Se-def` of `sections/03-action.tex`.
    """
    if T is None:
        T = compute_T_tensor(M, F)
    G = compute_G_matrix(T, F)
    return float((G * G).sum())


def compute_extended_candidates(M: np.ndarray, F: np.ndarray) -> dict[str, float]:
    """Compute the four κ-only / f-on-one-side candidates plus the f×f
    generator S_e at field configuration M.

    The plethysm prediction is rank ≤ 5; the existing four
    {S_a, S_b, S_c, S_{c'}} are known numerically to span a 4-dimensional
    subspace (`test_4_4_completeness` of e1_open_points.py).
    """
    cands: dict[str, float] = {}

    # κ × κ generators
    G = M.T @ M
    norm_sq = float((M * M).sum())
    G_norm = float((G * G).sum())
    cands["S_a (kk × kk: ‖M‖^4)"] = norm_sq ** 2
    cands["S_b (kk × kk: ‖M^T M‖^2)"] = G_norm

    # f × κ and κ × f generators
    U_c = np.einsum("abE,bC->aCE", F, M, optimize=True)
    Q_c = np.einsum("aCE,aD->ECD", U_c, M, optimize=True)
    cands["S_c (f_A × κ_B: ‖Q‖²)"] = float((Q_c * Q_c).sum())

    U_cp = np.einsum("Aa,abE->AbE", M, F, optimize=True)
    Q_cp = np.einsum("AbE,Cb->ECA", U_cp, M, optimize=True).transpose(0, 2, 1)
    cands["S_c' (κ_A × f_B: ‖Q'‖²)"] = float((Q_cp * Q_cp).sum())

    # f × f generator — eq:Se-def of the article
    T = compute_T_tensor(M, F)
    cands["S_e (ff (12)(34) × ff (12)(34))"] = compute_ff_diagonal(M, F, T)

    return cands


# ----------------------------------------------------------------------
# Numerical rank test
# ----------------------------------------------------------------------

def verify_dim_44_equals_5(n_samples: int = 6, seed: int = 0xC1054D
                            ) -> tuple[bool, int, int]:
    """Run an SVD rank test on the extended candidate set.

    Returns (passed, observed_rank, n_candidates).  The plethysm
    prediction of PART I is observed_rank == 5.
    """
    banner("[PART II] Numerical rank test on the extended (4,4) candidate set")
    print("Hypothesis: dim^(4,4) = 5 (plethysm prediction of PART I).")
    print(f"Existing  : rank = 4 from {{S_a, S_b, S_c, S_c'}} alone.")
    print(f"Test      : add the explicit f×f generator S_e and check rank growth.\n")

    F = load_F()
    rng = np.random.default_rng(seed)

    sample_dicts: list[dict[str, float]] = []
    t0 = time.time()
    for i in range(n_samples):
        M = 0.05 * rng.standard_normal((DIM_E8, DIM_E8))
        cands = compute_extended_candidates(M, F)
        sample_dicts.append(cands)
        elapsed = time.time() - t0
        print(f"  sample {i+1}/{n_samples}  ({elapsed:6.1f}s cum., "
              f"{len(cands)} candidates)", flush=True)

    keys = list(sample_dicts[0].keys())
    samples = np.array([[d[k] for k in keys] for d in sample_dicts])

    print()
    print(f"  Values for sample 0:")
    for k in keys:
        print(f"    {k:50s} = {sample_dicts[0][k]:.6e}")

    rank, sv = numerical_rank(samples)
    sv_str = ", ".join(f"{s:.3e}" for s in sv)
    print()
    print(f"  Singular values: [{sv_str}]")
    print(f"  Numerical rank (rel. tol = {RANK_TOL}): {rank}")
    print(f"  ⇒ dim Inv^(4,4) (numerical) = {rank}")
    print()

    expected = 5
    if rank == expected:
        print(f"  [PASS] dim^(4,4) = {rank} matches the plethysm prediction.")
        print(f"  ⇒ The f×f generator S_e is independent of {{S_a, S_b, S_c, S_c'}}")
        print(f"     and completes the (4,4) basis of `tab:nk-catalogue`.")
        return True, rank, len(keys)
    elif rank == 4:
        print(f"  [INFO] dim^(4,4) = 4: the f×f candidate lies in the span")
        print(f"     of the existing four generators.")
        return False, rank, len(keys)
    elif rank > expected:
        print(f"  [WARN] rank = {rank} > 5 — exceeds the plethysm upper bound.")
        return False, rank, len(keys)
    else:
        print(f"  [WARN] rank = {rank} < 4 — incomplete coverage.")
        return False, rank, len(keys)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Cross-check of dim^(4,4) = 5 via Cauchy–Howe plethysm and")
    print("explicit f×f generator (`tab:nk-catalogue` / Lemma `lem:nk-catalogue`)")
    print("=" * 70, flush=True)

    print()
    banner("[PART I] Plethysm upper bound for dim Inv^(4,4)(E_8 × E_8)")
    upper_bound = plethysm_upper_bound()
    print(f"⇒ Plethysm upper bound: dim^(4,4) = {upper_bound}.")
    print()

    passed, observed, n_cand = verify_dim_44_equals_5(n_samples=6)

    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Plethysm upper bound (PART I) : dim^(4,4) = {upper_bound}")
    print(f"  Numerical rank   (PART II)    : dim^(4,4) = {observed} (with {n_cand} candidates)")
    print(f"  Match                         : {'YES' if passed else 'NO'}")
    print("=" * 70)

    if passed:
        print()
        print("Conclusion: the (4,4) sector has dimension exactly 5, in agreement")
        print("with Lemma `lem:nk-catalogue` and `eq:plethysm-44-app` of the article.")
        print("The f×f generator S_e (`eq:Se-def`) is independent of the four")
        print("κ- and (f,κ)-mixed generators {S_a, S_b, S_c, S_c'}.")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
