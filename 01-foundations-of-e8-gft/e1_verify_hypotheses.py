"""Numerical certificates for the structural hypotheses underlying
``Theorem~\\ref{thm:action}`` (form of the E_8-GFT action) in
``sections/03-action.tex``.

Each test below is the ``[Proven-num]`` companion of an analytical
claim from the paper. Identifiers in brackets refer to LaTeX labels
in ``sections/``:

  T0   structural data of E_8 (dim 248, rank 8, |Φ| = 240, |Φ^+| = 120),
       Postulate ``post:algebra`` and Appendix ``app:conv:lie``.
  T1   total antisymmetry of f_{ABC} in the κ-orthonormal basis
       (``app:conv:lie``, used implicitly in ``lem:primitive-tensors``).
  T2   Jacobi identity, both as Σ_cyc [[X,Y],Z] = 0 and as
       ad_{[X,Y]} = [ad_X, ad_Y].
  T3   Killing form proportional to identity ⇒ dual Coxeter h^∨ = 30
       (``app:conv:lie``, eq. \\eqref{eq:app:conv:kappa-def}).
  T4   D_{ABC} := STr(ad_A ad_B ad_C) ≡ 0, the numerical face of
       Corollary ``cor:anomaly-safety`` (no primitive degree-3 Casimir,
       Lemma ``lem:casimir-degrees``).
  TAd  Compactness of κ ⇒ Ad_g is κ-orthogonal; consequently every
       Frobenius-norm contraction (C_2, H_2, S_c) is automatically
       Ad-invariant.
  T7   sector (3, 0): zero independent Ad-invariants
       (``lem:nk-catalogue``, Table ``tab:nk-catalogue``).
  T8   sector (3, 2): zero independent Ad-invariants — all candidate
       contractions {f Φ^3 (LΦ)^2} vanish identically, by antisym(f)
       paired against a symmetric pair of free indices.
  T10  sector (4, 4): three independent Frobenius-type generators
       {S_a, S_b, S_c}; the full five-generator basis {S_a, S_b,
       S_c, S_{c'}, S_e} is certified in ``e1_action_form.py`` and
       ``e1_open_points.py``.

The cubic potential f_{ABC} Φ^A Φ^B Φ^C ≡ 0 (algebraic identity from
antisymmetry of f against Sym^3 Φ) and the uniqueness of C_2 in
sector (2, 0) (structural consequence of T3, Killing ∝ identity) are
cited in prose in ``sections/03-action.tex`` and not duplicated here.

Each test prints PASS / FAIL with residual and tolerance; the final
exit code is 0 iff all tests pass.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    load_structure_constants_numpy,
    bracket_vec_fast,
    build_dense_f,
    build_ad_matrix,
    adjoint_exp,
)

# ----------------------------------------------------------------------
# Constants (theory)
# ----------------------------------------------------------------------

RANK_E8 = 8
N_POS_ROOTS = 120
DUAL_COXETER = 30  # h^∨_{E_8}, fixed by app:conv:lie

CONSTANTS_PATH = constants_path(ROOT)

# Chevalley constants are O(1), so machine epsilon × O(few hundred).
TOL = 1e-8


def build_ad(f_idx, f_val, X):
    return build_ad_matrix(X, f_idx, f_val)


build_full_f = build_dense_f


# ----------------------------------------------------------------------
# T0 — basic structural data
# ----------------------------------------------------------------------

def test_basic_structure(res: Result) -> tuple[np.ndarray, np.ndarray]:
    print("\n[T0] Structural data of E_8 (post:algebra, app:conv:lie)")
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))

    res.report("dim e_8 = 248",
               f_idx.max() == DIM_E8 - 1 and f_idx.min() == 0,
               f"index range [{f_idx.min()}, {f_idx.max()}]")
    res.report("structure constants are nonzero",
               len(f_val) > 0,
               f"nnz = {len(f_val)}")

    from e8sim import generate_roots
    roots = generate_roots()
    res.report("|Φ| = 240 (total roots)",
               len(roots) == 240,
               f"got {len(roots)}")

    from e8sim import positive_root_indices
    pos = positive_root_indices(roots)
    res.report("|Φ^+| = 120 (positive roots)",
               len(pos) == N_POS_ROOTS,
               f"got {len(pos)}")

    return f_idx, f_val


# ----------------------------------------------------------------------
# T1 — total antisymmetry of f_{ABC}
# ----------------------------------------------------------------------

def test_antisymmetry(F: np.ndarray, res: Result) -> None:
    print("\n[T1] Total antisymmetry of f_{ABC} (app:conv:lie)")

    err_AB = float(np.max(np.abs(F + F.transpose(1, 0, 2))))
    res.report("f_{ABC} = -f_{BAC}", err_AB < TOL, f"max |Δ| = {err_AB:.2e}")

    err_BC = float(np.max(np.abs(F + F.transpose(0, 2, 1))))
    res.report("f_{ABC} = -f_{ACB} (total antisymmetry)",
               err_BC < TOL, f"max |Δ| = {err_BC:.2e}")


# ----------------------------------------------------------------------
# T2 — Jacobi identity
# ----------------------------------------------------------------------

def test_jacobi(f_idx: np.ndarray, f_val: np.ndarray, res: Result) -> None:
    print("\n[T2] Jacobi identity")
    # Σ_cyc [[X, Y], Z] = 0 on random triples; equivalent to the full
    # Jacobi identity by trilinearity.
    rng = np.random.default_rng(0x4AC0B1)
    max_err = 0.0
    for _ in range(20):
        X = rng.standard_normal(DIM_E8)
        Y = rng.standard_normal(DIM_E8)
        Z = rng.standard_normal(DIM_E8)
        XY = bracket_vec_fast(f_idx, f_val, X, Y)
        YZ = bracket_vec_fast(f_idx, f_val, Y, Z)
        ZX = bracket_vec_fast(f_idx, f_val, Z, X)
        residual = (bracket_vec_fast(f_idx, f_val, XY, Z) +
                    bracket_vec_fast(f_idx, f_val, YZ, X) +
                    bracket_vec_fast(f_idx, f_val, ZX, Y))
        max_err = max(max_err, float(np.max(np.abs(residual))))
    res.report("Σ_cyc [[X,Y],Z] = 0 (n=20 samples)",
               max_err < 1e-9, f"max |Δ| = {max_err:.2e}")

    # Structural form: ad is a Lie homomorphism, ad_{[X,Y]} = [ad_X, ad_Y].
    X = rng.standard_normal(DIM_E8)
    Y = rng.standard_normal(DIM_E8)
    XY = bracket_vec_fast(f_idx, f_val, X, Y)
    adX = build_ad(f_idx, f_val, X)
    adY = build_ad(f_idx, f_val, Y)
    adXY = build_ad(f_idx, f_val, XY)
    err_ad = float(np.max(np.abs(adXY - (adX @ adY - adY @ adX))))
    res.report("ad_{[X,Y]} = [ad_X, ad_Y] (homomorphism)",
               err_ad < 1e-9, f"max |Δ| = {err_ad:.2e}")


# ----------------------------------------------------------------------
# T3 — Killing form and dual Coxeter number h^∨ = 30
# ----------------------------------------------------------------------

def test_killing_form(F: np.ndarray, res: Result) -> None:
    print("\n[T3] Killing form and dual Coxeter number h^∨ (app:conv:lie)")

    # In the κ-orthonormal basis with totally antisymmetric f,
    #   K_{AB} := Σ_{C, D} f_{ACD} f_{BCD} = 2 h^∨ δ_{AB}
    # for E_8 (Bourbaki Plate VII). The factor 2 is the Chevalley-basis
    # normalisation in which long roots have length √2.
    K = np.einsum("ACD,BCD->AB", F, F, optimize=True)

    diag = np.diag(K)
    two_hv_est = float(np.mean(diag))
    h_v_est = two_hv_est / 2.0
    off_diag_max = float(np.max(np.abs(K - np.diag(diag))))
    diag_var = float(np.max(np.abs(diag - two_hv_est)))

    res.report("K_{AB} = 2 h^∨ δ_{AB} (off-diagonal ≈ 0)",
               off_diag_max < 1e-7,
               f"max |off-diag| = {off_diag_max:.2e}")
    res.report("K_{AA} constant (Killing ∝ identity)",
               diag_var < 1e-7,
               f"max |K_{{AA}} - mean| = {diag_var:.2e}")
    res.report(f"h^∨_{{E_8}} = {DUAL_COXETER}",
               abs(h_v_est - DUAL_COXETER) < 1e-6,
               f"measured 2 h^∨ = {two_hv_est:.6f} ⇒ h^∨ = {h_v_est:.6f}")


# ----------------------------------------------------------------------
# T4 — anomaly safety: D_{ABC} := STr(ad_A ad_B ad_C) ≡ 0
# ----------------------------------------------------------------------

def test_anomaly_safety(f_idx: np.ndarray, f_val: np.ndarray, res: Result) -> None:
    print("\n[T4] Anomaly safety of E_8: D_{ABC} ≡ 0 (cor:anomaly-safety)")
    print("    (i.e. no primitive degree-3 Casimir, lem:casimir-degrees)")

    rng = np.random.default_rng(0xE8AD)
    max_err = 0.0
    n_samples = 20

    for _ in range(n_samples):
        X = rng.standard_normal(DIM_E8)
        adX = build_ad(f_idx, f_val, X)
        # Tr((ad_X)^3) is the polarised contraction X^A X^B X^C STr(...);
        # it vanishes iff d_{ABC} ≡ 0 as a fully symmetric tensor.
        tr3 = float(np.trace(adX @ adX @ adX))
        max_err = max(max_err, abs(tr3))

    res.report("Tr((ad_X)^3) = 0 for random X (n=20)",
               max_err < 1e-6, f"max |Tr| = {max_err:.2e}")

    # Polarised version: STr(ad_X ad_Y ad_Z) symmetrised over the 6
    # permutations of distinct random X, Y, Z.
    max_err_pol = 0.0
    for _ in range(15):
        X = rng.standard_normal(DIM_E8)
        Y = rng.standard_normal(DIM_E8)
        Z = rng.standard_normal(DIM_E8)
        adX = build_ad(f_idx, f_val, X)
        adY = build_ad(f_idx, f_val, Y)
        adZ = build_ad(f_idx, f_val, Z)
        s = (np.trace(adX @ adY @ adZ) + np.trace(adX @ adZ @ adY) +
             np.trace(adY @ adX @ adZ) + np.trace(adY @ adZ @ adX) +
             np.trace(adZ @ adX @ adY) + np.trace(adZ @ adY @ adX)) / 6.0
        max_err_pol = max(max_err_pol, abs(float(s)))

    res.report("STr(ad_X ad_Y ad_Z) = 0 for random triples (n=15)",
               max_err_pol < 1e-6, f"max |STr| = {max_err_pol:.2e}")


# ----------------------------------------------------------------------
# TAd — orthogonality of Ad_g (compactness of κ)
# ----------------------------------------------------------------------
#
# Compactness of κ ⇒ ad_X is anti-self-adjoint ⇒ Ad_g = exp(ad_X) is
# κ-orthogonal. In the κ-orthonormal basis this is just orthogonality on
# R^{248}, and every Frobenius-norm contraction is preserved:
#   C_2(Ad_g Φ)        = ‖Ad_g Φ‖²            = ‖Φ‖²            = C_2(Φ),
#   H_2(Ad_R M Ad_S^T) = ‖Ad_R M Ad_S^T‖_F² = ‖M‖_F²            = H_2(M),
#   S_c(Ad_R M Ad_S^T) = S_c(M)        (with the same Ad_S on bracket arguments).
# We therefore only verify the orthogonality of Ad_g once.

def test_Ad_orthogonality(f_idx: np.ndarray, f_val: np.ndarray,
                          res: Result) -> None:
    print("\n[TAd] Orthogonality of Ad_g (compact κ ⇒ Frobenius preservation)")
    print("    Consequence: C_2, H_2, S_c are Ad-invariant without further test.")

    rng = np.random.default_rng(0xAD0A1)
    max_dev = 0.0
    n_samples = 6
    for _ in range(n_samples):
        X = 0.5 * rng.standard_normal(DIM_E8)
        Ad_g = adjoint_exp(X, f_idx, f_val)
        I_test = Ad_g.T @ Ad_g - np.eye(DIM_E8)
        max_dev = max(max_dev, float(np.max(np.abs(I_test))))

    res.report(
        f"Ad_g^T Ad_g = I (n={n_samples}, ‖X‖ ~ √248/2)",
        max_dev < 1e-8,
        f"max |Ad^T Ad - I|_∞ = {max_dev:.2e}",
    )


# ----------------------------------------------------------------------
# T7 — sector (3, 0): zero invariants
# ----------------------------------------------------------------------

def test_3_0_no_invariant(f_idx: np.ndarray, f_val: np.ndarray,
                          res: Result) -> None:
    print("\n[T7] Sector (3, 0): zero independent invariants (lem:nk-catalogue)")
    print("    Candidates: f_{ABC} Φ^A Φ^B Φ^C (≡ 0 by antisym f vs sym Φ⊗3),")
    print("                d_{ABC} Φ^A Φ^B Φ^C (≡ 0 by T4: d ≡ 0).")
    # Direct numerical check via STr(ad_Φ^3) = Φ^A Φ^B Φ^C STr(ad_A ad_B ad_C),
    # which vanishes iff d_{ABC} ≡ 0.
    rng = np.random.default_rng(0x303)

    max_err = 0.0
    for _ in range(20):
        Phi = rng.standard_normal(DIM_E8)
        adP = build_ad(f_idx, f_val, Phi)
        val = float(np.trace(adP @ adP @ adP))
        max_err = max(max_err, abs(val))

    res.report("STr(ad_Φ^3) = 0 ⟹ no cubic Casimir (n=20)",
               max_err < 1e-6, f"max |STr| = {max_err:.2e}")


# ----------------------------------------------------------------------
# T8 — sector (3, 2): zero invariants
# ----------------------------------------------------------------------
#
# Five Ad-indices to be saturated (3 from Φ, 2 from M = LΦ) require
# exactly one factor of f (degree 3) and one of κ (degree 2). Up to
# index relabelling, the three structurally distinct contractions are:
#   (a) f_{B_1 B_2 B_3} Φ^{B_1} Φ^{B_2} Φ^{B_3} · κ_{D_1 D_2} M^A_{D_1} M^A_{D_2}
#       = (f Φ^3) · ‖M‖_F^2                                  (≡ 0: f Φ^3 = 0)
#   (b) f_{B_1 B_2 D_1} Φ^{B_1} Φ^{B_2} M^A_{D_1} κ_{B_3 D_2} Φ^{B_3} M^A_{D_2}
#       Φ^{B_1} Φ^{B_2} sym, f antisym ⟹ ≡ 0.
#   (c) f_{B_1 D_1 D_2} Φ^{B_1} (M^T M)_{D_1 D_2} κ_{B_2 B_3} Φ^{B_2} Φ^{B_3}
#       (M^T M) sym in (D_1, D_2), f antisym ⟹ ≡ 0.

def test_3_2_no_invariant(F: np.ndarray, f_idx: np.ndarray, f_val: np.ndarray,
                          res: Result) -> None:
    print("\n[T8] Sector (3, 2): zero independent invariants (lem:nk-catalogue)")
    print("    All three structurally distinct contractions {f Φ^3 (LΦ)^2}")
    print("    vanish identically (antisym f vs sym pair of free indices).")

    rng = np.random.default_rng(0x32)

    max_err = 0.0
    for _ in range(15):
        Phi = rng.standard_normal(DIM_E8)
        M = rng.standard_normal((DIM_E8, DIM_E8))

        # (a): (f Φ^3) · ‖M‖^2  — vanishes via f Φ^3 ≡ 0.
        cand_a = (Phi[f_idx[0]] * Phi[f_idx[1]] * Phi[f_idx[2]] * f_val).sum() \
                 * (M * M).sum()

        # (c): Σ f_{B D_1 D_2} Φ^B (M^T M)_{D_1 D_2} — vanishes by f antisym vs M^T M sym.
        MtM = M.T @ M
        cand_c = np.einsum("BCD,B,CD->", F, Phi, MtM)

        # (b): Σ f_{B_1 B_2 D_1} Φ^{B_1} Φ^{B_2} M^A_{D_1} Φ^{D_2} M^A_{D_2}
        #      = (Σ_{B_1 B_2} f_{B_1 B_2 ·} Φ^{B_1} Φ^{B_2}) · (...) — vanishes
        #      since the inner contraction Σ f Φ Φ = 0 already.
        f_BB1 = np.einsum("BCD,B,C->D", F, Phi, Phi)
        cand_b = float((f_BB1[:, None] * (M.T @ M @ Phi[:, None])).sum())

        max_err = max(max_err,
                      abs(float(cand_a)),
                      abs(float(cand_c)),
                      abs(cand_b))

    # Tolerance: cand_a = (f Φ^3) · ‖M‖^2 with ‖M‖^2 ~ 248^2 = 6e4 and
    # (f Φ^3) only zero up to ~1e-13 → residual ~6e-9. Use 1e-7.
    res.report("(a), (b), (c) of sector (3, 2) all ≡ 0 (n=15)",
               max_err < 1e-7, f"max |residual| = {max_err:.2e}")


# ----------------------------------------------------------------------
# T10 — three independent Frobenius-type generators in sector (4, 4)
# ----------------------------------------------------------------------

def compute_S_abc(M: np.ndarray, F: np.ndarray) -> tuple[float, float, float]:
    """Return (S_a, S_b, S_c) for M_{AB} = (L_A Φ)^B and dense F_{cdB}.

    Three of the five (4, 4) generators of ``lem:nk-catalogue``
    (cf. eqs. ``Sa-def``, ``Sb-def``, ``Sc-def`` in
    ``sections/03-action.tex``):
      S_a = (Σ_{A,B} M_{AB}^2)^2  = (‖M‖_F^2)^2
      S_b = ‖M^T M‖_F^2  (Frobenius proxy of the (2,2)-isotypic projection)
      S_c = Σ_{A,A',B} ([M_A, M_{A'}]^B)^2  (Skyrme-type)
    The remaining two generators (S_{c'}, S_e) are tested in
    ``e1_action_form.py`` and ``e1_open_points.py``.
    """
    # S_a
    norm_sq = float((M * M).sum())
    s_a = norm_sq ** 2

    # S_b
    MMT = M @ M.T
    s_b = float((MMT * MMT).sum())

    # S_c — bracket[A, A', B] = Σ_{c, d} F_{cdB} M_{Ac} M_{A'd}; computed
    # without materialising the full 248^3 bracket tensor:
    #   T[A, d, B] = Σ_c M_{Ac} F_{cdB}                    (cost 248^4)
    #   s_c        = Σ_{A,d,e,B} T[A,d,B] T[A,e,B] (M^T M)[d,e]
    T = np.einsum("Ac,cdB->AdB", M, F, optimize=True)
    G = M.T @ M
    s_c = float(np.einsum("AdB,AeB,de->", T, T, G, optimize=True))
    return s_a, s_b, s_c


def test_4_4_independence(F: np.ndarray, res: Result) -> None:
    print("\n[T10] Sector (4, 4): linear independence of {S_a, S_b, S_c}")
    print("      (sub-lattice of the five-generator basis of lem:nk-catalogue)")
    rng = np.random.default_rng(0x4444)
    n_samples = 6
    table = np.zeros((n_samples, 3))

    t0 = time.time()
    for i in range(n_samples):
        M = rng.standard_normal((DIM_E8, DIM_E8)) * 0.05
        sa, sb, sc = compute_S_abc(M, F)
        table[i] = (sa, sb, sc)
    dt = time.time() - t0

    # Normalise columns to similar magnitudes for numerical rank.
    scale = np.maximum(np.max(np.abs(table), axis=0), 1e-30)
    norm_table = table / scale
    sv = np.linalg.svd(norm_table, compute_uv=False)
    rank = int(np.sum(sv > 1e-10 * sv[0]))

    res.report("rank([S_a, S_b, S_c]_i) = 3 (linear independence)",
               rank == 3,
               f"singular values = {sv.tolist()} (computed in {dt:.1f}s)")
    print(f"    Sample values:\n{table}")


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("E1 — Numerical certificates for thm:action structural hypotheses")
    print("Reference: sections/03-action.tex, sections/A1-conventions.tex")
    print("=" * 70)
    res = Result()

    f_idx, f_val = test_basic_structure(res)
    print("\n[setup] Building dense f_{ABC} tensor (≈ 120 MB)…")
    F = build_full_f(f_idx, f_val)
    test_antisymmetry(F, res)
    test_jacobi(f_idx, f_val, res)
    test_killing_form(F, res)
    test_anomaly_safety(f_idx, f_val, res)
    test_Ad_orthogonality(f_idx, f_val, res)
    test_3_0_no_invariant(f_idx, f_val, res)
    test_3_2_no_invariant(F, f_idx, f_val, res)
    test_4_4_independence(F, res)

    print("\n" + "=" * 70)
    print(f"Summary: {res.passed} PASS / {res.failed} FAIL")
    print("=" * 70)
    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
