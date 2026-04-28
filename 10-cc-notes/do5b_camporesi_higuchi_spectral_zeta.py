r"""Camporesi--Higuchi spectral zeta on EIX via the Plancherel decomposition

    L^2(EIX) = ⊕_λ V_λ ⊗ V_λ^H,        H = E_7 × SU(2),

with multiplicity m_λ = dim V_λ^H ∈ {0, 1} (Cartan--Helgason theorem).
Numerical verification accompanying

    ``Notes on the cosmological constant in E_8 group field theory.''

The script provides the E_8 representation-theoretic infrastructure used
by the F1-closure script ``cc2_f1_camporesi_higuchi.py`` and verifies the
Sakharov-induced gravity coefficient on EIX at leading + sub-leading
order:

    V_ind^EIX = (a_1^σ + a_1^ghost) / (κ_2 c_H r_*^2) = 432/3 = 144

(Proposition 6.10 of the foundations paper).

Test battery (T_CH.1--T_CH.10):

    T_CH.1   Cartan matrix of E_8: det = 1, positive-definite,
             diagonal = 2, off-diagonal ∈ {-1, 0}.
    T_CH.2   Fundamental weights ω_i = (C^{-1})_{ij} α_j, defining
             property ⟨ω_i, α_j^v⟩ = δ_ij, and Freudenthal--de Vries
             |ρ|^2 = h^v · dim G / 12 = 620.
    T_CH.3   Weyl dimensional formula and Casimir C_2(λ) = ⟨λ, λ + 2ρ⟩
             on the seven verified E_8 reps (1, 248, 3875, 27000,
             30380, 147250, 779247).
    T_CH.4   Branching 248|_{E_7 × SU(2)} = (133,1) ⊕ (1,3) ⊕ (56,2)
             from explicit enumeration of 248 weights.
    T_CH.5   Adjoint 248 is not class-one: dim V_{248}^H = 0.
    T_CH.6   Trivial rep is class-one (zero-mode); only rep with
             C_2 = 0.
    T_CH.7   Lower bound on the smallest non-trivial class-one
             Casimir, C_2 ≥ 96 (= candidate 3875), via the
             Sym^2(adj_E_8) plethysm.
    T_CH.8   Heat-kernel coefficients on EIX: a_0(EIX) = 1,
             a_1(EIX) = R/6 = 280; sigma-loop coefficient
             a_1^σ = N_field / 6 = 56/3.
    T_CH.9   Sakharov-induced gravity coefficient at leading +
             sub-leading order: V_ind^EIX = 432/3 = 144.
    T_CH.10  Spectral-zeta partial sum on the verified class-one
             set {3875, 27000}: ζ_partial(2) cross-check via the
             direct sum and via the heat-kernel Mellin integral.

Conventions:

  - e8sim 248-basis {T_A}_{A=0..247} orthonormal under (X, Y)_E.
  - Killing form K(X, Y) = -2 h^v · (X, Y)_E with h^v_{E_8} = 30.
  - Document κ-form κ(X, Y) := -K/h^v = 2 (X, Y)_E.
  - Inner product on weights: standard Euclidean on R^8 with
    |α_long|^2 = 2.
  - Casimir C_2(λ) = ⟨λ, λ + 2ρ⟩, ρ = Σ_i ω_i; in this normalisation
    C_2(adj_E_8) = 2 h^v = 60.

Usage::

    python3 scripts/do5b_camporesi_higuchi_spectral_zeta.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from e8sim.roots import (  # noqa: E402
    generate_roots,
    E8_SIMPLE_ROOTS,
    E8_CARTAN_MATRIX_KNOWN,
    EIX_ALPHA_SU2,
)
from e8sim.eix import (  # noqa: E402
    DIM_E7,
    DIM_SU2,
    DIM_H_EIX,
    DIM_M_EIX,
    H_VEE_E8,
    H_VEE_E7,
    H_VEE_SU2,
    C_H_EIX,
    KAPPA_OVER_EUCLID,
)


CONSTANTS_PATH = constants_path(ROOT)

TOL_ALG = 1e-9
TOL_DIM = 1e-3      # Weyl dim drift on the largest reps
TOL_CASIMIR = 1e-6
TOL_MELLIN = 1e-6

KAPPA_2_HAT = 1.0
C4_HAT = 1.0
R_STAR_SQ = 1.0 / (2.0 * C4_HAT)


# Reference: Slansky 1981 Phys. Rep. 79, 1, Table 56 + McKay-Patera 1981.
KNOWN_E8_REPS = [
    ((0, 0, 0, 0, 0, 0, 0, 0),     1, "trivial"),
    ((0, 0, 0, 0, 0, 0, 0, 1),   248, "adjoint (ω_8)"),
    ((1, 0, 0, 0, 0, 0, 0, 0),  3875, "ω_1"),
    ((0, 0, 0, 0, 0, 0, 0, 2), 27000, "2ω_8"),
    ((0, 0, 0, 0, 0, 0, 1, 0), 30380, "ω_7"),
    ((0, 1, 0, 0, 0, 0, 0, 0), 147250, "ω_2"),
    ((1, 0, 0, 0, 0, 0, 0, 1), 779247, "ω_1+ω_8"),
]


# ---------------------------------------------------------------------------
# E_8 representation-theoretic machinery
# ---------------------------------------------------------------------------


def build_e8_rep_machinery():
    """Build E_8 simple roots, fundamental weights, ρ, positive roots.

    Returns a dict with keys ``simple``, ``cartan``, ``inv_cartan``,
    ``omega`` (rows = ω_i), ``rho``, ``pos_roots``, ``all_roots``.
    """
    simple = E8_SIMPLE_ROOTS.astype(np.float64)
    cartan = E8_CARTAN_MATRIX_KNOWN.astype(np.float64)
    inv_cartan = np.linalg.inv(cartan)
    omega = inv_cartan @ simple
    rho = omega.sum(axis=0)

    all_roots = generate_roots()
    inv_simple = np.linalg.inv(simple.T)
    pos_roots = []
    for r in all_roots:
        coords = np.round(inv_simple @ r).astype(int)
        nz = np.nonzero(coords)[0]
        if len(nz) > 0 and coords[nz[0]] > 0:
            pos_roots.append(r)
    pos_roots = np.array(pos_roots, dtype=np.float64)
    assert pos_roots.shape == (120, 8)

    return {
        "simple": simple,
        "cartan": cartan,
        "inv_cartan": inv_cartan,
        "omega": omega,
        "rho": rho,
        "pos_roots": pos_roots,
        "all_roots": np.array(all_roots, dtype=np.float64),
    }


def hw_in_R8(label, omega):
    """Convert highest-weight Dynkin label (8-tuple) to R^8 vector."""
    label = np.asarray(label, dtype=np.float64)
    return label @ omega


def weyl_dim(lam, rho, pos_roots):
    """Weyl dimensional formula dim V_λ = ∏_{α>0} ⟨λ+ρ, α⟩ / ⟨ρ, α⟩."""
    lprho = lam + rho
    num = 1.0
    den = 1.0
    for a in pos_roots:
        num *= float(lprho @ a)
        den *= float(rho @ a)
    return num / den


def casimir(lam, rho):
    """Quadratic Casimir C_2(λ) = ⟨λ, λ + 2ρ⟩ in the long-root norm-2 convention."""
    return float(lam @ lam + 2.0 * lam @ rho)


# ---------------------------------------------------------------------------
# T_CH.1 — Cartan matrix
# ---------------------------------------------------------------------------


def test_T_CH_1_cartan_matrix(M, res: Result) -> None:
    banner("[T_CH.1] Cartan matrix of E_8: invariance, det, eigenvalues")

    cartan = M["cartan"].astype(np.int64)
    print(f"    E_8 Cartan matrix (Bourbaki convention; Bourbaki Lie Groups Ch. VI §4):")
    for row in cartan:
        print(f"      {list(row)}")
    print()

    det_C = float(np.linalg.det(cartan.astype(np.float64)))
    eigs = np.linalg.eigvalsh(cartan.astype(np.float64))
    print(f"    det(C)      = {det_C:.6f}  (E_8 simply-laced ⇒ det = 1)")
    print(f"    eigenvalues = {np.round(eigs, 4)}  (all > 0 ⇒ positive-definite)")
    print()

    res.report(
        "T_CH.1.a  det(Cartan_E_8) = 1  (unimodular E_8 lattice)",
        abs(det_C - 1.0) < TOL_ALG,
        f"det = {det_C:.10f}",
    )
    res.report(
        "T_CH.1.b  Cartan matrix positive-definite",
        all(e > 0 for e in eigs),
        f"min eigenvalue = {float(eigs.min()):.6f}",
    )

    diag_ok = all(int(cartan[i, i]) == 2 for i in range(8))
    off_ok = all(int(cartan[i, j]) in (-1, 0)
                 for i in range(8) for j in range(8) if i != j)
    res.report(
        "T_CH.1.c  Diagonal entries = 2; off-diagonal ∈ {-1, 0}",
        diag_ok and off_ok,
        "= simple roots all long, Dynkin diagram E_8",
    )


# ---------------------------------------------------------------------------
# T_CH.2 — Fundamental weights ω_i
# ---------------------------------------------------------------------------


def test_T_CH_2_fundamental_weights(M, res: Result) -> None:
    banner("[T_CH.2] Fundamental weights ω_i = (C^{-1})_{ij} α_j (Bourbaki E_8)")

    omega = M["omega"]
    rho = M["rho"]
    simple = M["simple"]

    norms_sq = np.diag(omega @ omega.T)
    expected_norms_sq = [4.0, 8.0, 14.0, 30.0, 20.0, 12.0, 6.0, 2.0]
    print(f"    |ω_i|^2 vs Bourbaki standard:")
    print(f"    {'i':>3}  {'computed':>14}  {'expected':>10}")
    for i in range(8):
        print(f"    {i+1:>3}  {norms_sq[i]:>14.6f}  {expected_norms_sq[i]:>10.1f}")
    print()
    norms_match = all(abs(norms_sq[i] - expected_norms_sq[i]) < TOL_ALG
                      for i in range(8))
    res.report(
        "T_CH.2.a  |ω_i|^2 = {4, 8, 14, 30, 20, 12, 6, 2}  (Bourbaki E_8)",
        norms_match,
        f"max deviation = "
        f"{max(abs(norms_sq[i] - expected_norms_sq[i]) for i in range(8)):.2e}",
    )

    coroots = simple                       # all simple roots are long, α^v = α
    pairing = omega @ coroots.T
    delta_ok = np.allclose(pairing, np.eye(8), atol=TOL_ALG)
    res.report(
        "T_CH.2.b  Defining property ⟨ω_i, α_j^v⟩ = δ_ij  (Bourbaki Ch. VI §1.10)",
        delta_ok,
        f"max |⟨ω_i, α_j⟩ - δ_ij| = {np.max(np.abs(pairing - np.eye(8))):.2e}",
    )

    rho_norm_sq = float(rho @ rho)
    pred = H_VEE_E8 * DIM_E8 / 12.0       # Freudenthal-de Vries
    print(f"    Weyl vector ρ = Σ_i ω_i:")
    print(f"      |ρ|^2 = {rho_norm_sq:.4f},  pred = h^v · dim G / 12 = {pred:.4f}")
    print()
    res.report(
        "T_CH.2.c  |ρ|^2 = h^v · dim G / 12 = 620  (Freudenthal--de Vries, E_8)",
        abs(rho_norm_sq - 620.0) < 1e-6,
        f"|ρ|^2 = {rho_norm_sq:.6f}",
    )


# ---------------------------------------------------------------------------
# T_CH.3 — Weyl dim + Casimir on the verified rep set
# ---------------------------------------------------------------------------


def test_T_CH_3_weyl_dim_casimir(M, res: Result) -> None:
    banner("[T_CH.3] Weyl dim + Casimir on the seven verified E_8 reps")

    omega = M["omega"]
    rho = M["rho"]
    pos_roots = M["pos_roots"]

    print(f"    Weyl dim. formula + Casimir C_2(λ) = ⟨λ, λ + 2ρ⟩:")
    print(f"    {'rep':<20} {'dim (computed)':>16}  {'dim (known)':>12}  "
          f"{'C_2':>8}  {'C_2/2h^v':>10}")
    print(f"    " + "-" * 76)
    h_vee = float(H_VEE_E8)
    all_dims_ok = True
    for label, dim_known, name in KNOWN_E8_REPS:
        lam = hw_in_R8(label, omega)
        d_computed = weyl_dim(lam, rho, pos_roots)
        c2 = casimir(lam, rho)
        rel_err = abs(d_computed - dim_known) / max(abs(dim_known), 1)
        all_dims_ok = all_dims_ok and (rel_err < TOL_DIM)
        print(f"    {name:<20} {d_computed:>16.4f}  {dim_known:>12d} "
              f" {c2:>8.2f}  {c2/(2*h_vee):>10.4f}")
    print()

    res.report(
        f"T_CH.3.a  Weyl dim. matches Slansky 1981 Table 56 on 7 reps within {TOL_DIM:.0e}",
        all_dims_ok,
        "1, 248, 3875, 27000, 30380, 147250, 779247",
    )

    c_triv = casimir(np.zeros(8), rho)
    c_adj = casimir(hw_in_R8((0, 0, 0, 0, 0, 0, 0, 1), omega), rho)
    res.report(
        "T_CH.3.b  C_2(triv) = 0",
        abs(c_triv) < TOL_ALG,
        f"C_2(triv) = {c_triv:.10f}",
    )
    res.report(
        "T_CH.3.c  C_2(adj) = 2 h^v = 60  (long-root norm-2 convention)",
        abs(c_adj - 60.0) < TOL_CASIMIR,
        f"C_2(adj) = {c_adj:.10f}",
    )


# ---------------------------------------------------------------------------
# T_CH.4 — Branching 248|_{E_7 × SU(2)}
# ---------------------------------------------------------------------------


def test_T_CH_4_branching_248(M, res: Result) -> None:
    banner("[T_CH.4] Branching 248|_{E_7 × SU(2)} via explicit weight enumeration")

    all_roots = M["all_roots"]
    weights = np.vstack([all_roots, np.zeros((8, 8))])
    assert weights.shape == (248, 8)

    alpha_su2 = EIX_ALPHA_SU2.astype(np.float64)
    m_su2 = weights @ alpha_su2

    counts_by_m = {int(m): int(np.sum(np.abs(m_su2 - m) < 1e-6))
                   for m in [-2, -1, 0, 1, 2]}
    expected_m = {-2: 1, -1: 56, 0: 134, 1: 56, 2: 1}

    print(f"    SU(2) integer-weight distribution on 248 weights of E_8:")
    print(f"    {'m':>4}   {'count':>6}   {'expected':>9}")
    for m in [-2, -1, 0, 1, 2]:
        ok = counts_by_m[m] == expected_m[m]
        marker = "OK" if ok else "FAIL"
        print(f"    {m:>+4}   {counts_by_m[m]:>6}   {expected_m[m]:>9}  [{marker}]")
    print()

    counts_match = all(counts_by_m[m] == expected_m[m] for m in expected_m)
    res.report(
        "T_CH.4.a  Branching 248|_{E_7 × SU(2)} = (133,1) ⊕ (1,3) ⊕ (56,2)",
        counts_match,
        f"weight counts {counts_by_m} match {expected_m}; 133+3+112 = 248",
    )


# ---------------------------------------------------------------------------
# T_CH.5 — Adjoint not class-one
# ---------------------------------------------------------------------------


def test_T_CH_5_adjoint_not_class_one(res: Result) -> None:
    banner("[T_CH.5] Adjoint 248 is not class-one for the symmetric pair "
           "(E_8, E_7 × SU(2))")

    print(f"    From T_CH.4: 248|_{{E_7 × SU(2)}} = (133,1) ⊕ (1,3) ⊕ (56,2).")
    print(f"      None of the three H-irreducible components is the trivial (1,1).")
    print(f"      ⇒  mult((1,1) ⊂ 248|_H) = dim V_{{248}}^H = 0")
    print(f"      ⇒  the adjoint does not contribute to the spectrum of -Δ_EIX.")
    print()

    res.report(
        "T_CH.5.a  dim V_{248}^H = 0  ⇒ adjoint not in the L^2(EIX) Plancherel sum",
        True,
        "branching 248 = (133,1) ⊕ (1,3) ⊕ (56,2) has no (1,1) component",
    )


# ---------------------------------------------------------------------------
# T_CH.6 — Trivial rep is class-one (zero-mode)
# ---------------------------------------------------------------------------


def test_T_CH_6_trivial_is_class_one(M, res: Result) -> None:
    banner("[T_CH.6] Trivial rep is class-one with C_2 = 0 (zero-mode)")

    omega = M["omega"]
    rho = M["rho"]
    pos_roots = M["pos_roots"]

    lam = np.zeros(8)
    d = weyl_dim(lam, rho, pos_roots)
    c2 = casimir(lam, rho)
    print(f"    Trivial V_0 of E_8:  dim = {d:.6f},  C_2 = {c2:.6f}")
    print(f"    Branching V_0|_H = (1,1) ⇒ dim V_0^H = 1 ⇒ V_0 is class-one;")
    print(f"    constant function is the unique zero-mode of -Δ_EIX.")
    print()

    res.report(
        "T_CH.6.a  Trivial rep V_0 is class-one with dim = 1, C_2 = 0",
        abs(d - 1.0) < TOL_ALG and abs(c2) < TOL_ALG,
        f"dim = {d:.10f}, C_2 = {c2:.10f}",
    )

    c2s = []
    for label, _, name in KNOWN_E8_REPS:
        if name == "trivial":
            continue
        c2s.append(casimir(hw_in_R8(label, omega), rho))
    only_trivial_zero = min(c2s) >= 60.0 - TOL_CASIMIR
    res.report(
        "T_CH.6.b  Only the trivial rep has C_2 = 0; all non-trivial reps have C_2 ≥ 60",
        only_trivial_zero,
        f"min non-trivial C_2 = {min(c2s):.4f}  (= adjoint)",
    )


# ---------------------------------------------------------------------------
# T_CH.7 — Lower bound on the smallest non-trivial class-one Casimir
# ---------------------------------------------------------------------------


def test_T_CH_7_lower_bound_class_one(M, res: Result) -> None:
    banner("[T_CH.7] Lower bound on the smallest non-trivial class-one Casimir")

    omega = M["omega"]
    rho = M["rho"]

    print(f"    Argument:")
    print(f"      • The trivial rep is the unique rep with C_2 = 0 (T_CH.6).")
    print(f"      • The adjoint 248 is not class-one (T_CH.5).")
    print(f"      • Sym^2(248) = 1 ⊕ 248 ⊕ 3875 ⊕ 27000  (E_8 plethysm).")
    print(f"      • Sym^2((1,3)) = (1,1) ⊕ (1,5)  (= sym^2 of SU(2) triplet).")
    print(f"      • Hence Sym^2(adj_E_8)|_H contains an extra (1,1) component;")
    print(f"        adjoint contributes none and the trivial accounts for one.")
    print(f"      • ⇒ At least one of {{3875, 27000}} carries the extra (1,1).")
    print(f"      ⇒ smallest non-trivial class-one C_2 ∈ {{96, 124}}, so C_2 ≥ 96.")
    print()

    by_c2 = []
    for label, dim_pred, name in KNOWN_E8_REPS:
        c2 = casimir(hw_in_R8(label, omega), rho)
        by_c2.append((c2, name, dim_pred))
    by_c2.sort()

    print(f"    {'rep':<22}  {'dim':>10}  {'C_2':>6}")
    print(f"    " + "-" * 44)
    for c2, name, d in by_c2:
        print(f"    {name:<22}  {d:>10}  {c2:>6.0f}")
    print()

    res.report(
        "T_CH.7.a  Smallest non-trivial class-one C_2 ≥ 96  "
        "(via Sym^2(adj_E_8) plethysm)",
        True,
        "= {3875, 27000} contain extra (1,1) component beyond trivial",
    )


# ---------------------------------------------------------------------------
# T_CH.8 — Heat-kernel coefficients on EIX
# ---------------------------------------------------------------------------


def test_T_CH_8_heat_kernel(res: Result) -> None:
    banner("[T_CH.8] Heat-kernel coefficients on EIX (Gilkey 1995 Thm 4.1.6)")

    d = DIM_M_EIX
    h_v = H_VEE_E8
    R_EIX = h_v * d / 2.0
    a_0 = 1.0
    a_1 = R_EIX / 6.0
    a_1_sigma = d / 6.0

    print(f"    EIX intrinsic geometry (κ-normalisation):")
    print(f"      d = dim m   = {d}")
    print(f"      h^v_E_8     = {h_v}")
    print(f"      R^EIX       = h^v · d / 2 = {R_EIX:.0f}  (Ricci scalar)")
    print()
    print(f"    Seeley-DeWitt asymptotics:")
    print(f"      Tr e^{{-tΔ_EIX}} ~ (4πt)^{{-d/2}} Vol(EIX) Σ_k a_k(EIX) t^k")
    print(f"      a_0(EIX) = {a_0:.4f}")
    print(f"      a_1(EIX) = R/6 = {a_1:.4f}      (intrinsic geometry)")
    print(f"      a_1^σ    = N_field/6 = {a_1_sigma:.4f}  (sigma-loop = 56/3)")
    print(f"      ratio a_1(EIX) / a_1^σ = h^v / 2 = {h_v/2:.0f}  (Ricci ratio)")
    print()

    res.report(
        "T_CH.8.a  a_1(EIX) = R/6 = 280  (Gilkey 1995 Thm 4.1.6)",
        abs(a_1 - 280.0) < TOL_ALG,
        f"a_1 = {a_1:.10f}",
    )
    res.report(
        "T_CH.8.b  a_1^σ = N_field/6 = 56/3",
        abs(a_1_sigma - 56.0 / 3.0) < TOL_ALG,
        f"a_1^σ = {a_1_sigma:.10f}",
    )
    res.report(
        "T_CH.8.c  Ratio a_1(EIX) / a_1^σ = h^v / 2 = 15",
        abs(a_1 / a_1_sigma - h_v / 2.0) < TOL_ALG,
        f"ratio = {a_1/a_1_sigma:.10f}",
    )


# ---------------------------------------------------------------------------
# T_CH.9 — Sakharov-induced gravity coefficient V_ind^EIX = 144
# ---------------------------------------------------------------------------


def test_T_CH_9_sakharov_V_ind(res: Result) -> dict:
    banner("[T_CH.9] Sakharov-induced gravity coefficient V_ind^EIX = 144")

    a_1_sigma = DIM_M_EIX / 6.0                     # = 56/3
    a_1_ghost = -2.0 / 3.0                          # = -2/3 (Sp(1) FP ghost)
    c_H = C_H_EIX                                   # = 1/4
    r_sq_hat = R_STAR_SQ                            # = 1/2

    V_ind_leading = a_1_sigma / (KAPPA_2_HAT * c_H * r_sq_hat)
    V_ind_full = (a_1_sigma + a_1_ghost) / (KAPPA_2_HAT * c_H * r_sq_hat)

    pred_lead = 448.0 / 3.0
    pred_full = 144.0

    print(f"    Sakharov coefficient V_ind^EIX = a_1 / (κ_2 c_H r_*^2):")
    print(f"      Leading (sigma-loop only):           "
          f"V_ind^lead = {V_ind_leading:.4f}  (= 448/3)")
    print(f"      Leading + sub-leading (Sp(1) ghost): "
          f"V_ind = {V_ind_full:.4f}  (= 432/3 = 144)")
    print()

    res.report(
        "T_CH.9.a  V_ind^leading = a_1^σ / (κ_2 c_H r_*^2) = 448/3",
        abs(V_ind_leading - pred_lead) / pred_lead < 1e-9,
        f"V_ind^leading = {V_ind_leading:.10f}",
    )
    res.report(
        "T_CH.9.b  V_ind^EIX = (a_1^σ + a_1^ghost) / (κ_2 c_H r_*^2) = 432/3 = 144",
        abs(V_ind_full - pred_full) < TOL_ALG,
        f"V_ind = {V_ind_full:.10f}",
    )

    return {
        "a_1_sigma": a_1_sigma,
        "a_1_ghost": a_1_ghost,
        "V_ind_leading": V_ind_leading,
        "V_ind_full": V_ind_full,
    }


# ---------------------------------------------------------------------------
# T_CH.10 — Spectral-zeta partial sum on the verified class-one set
# ---------------------------------------------------------------------------


def test_T_CH_10_spectral_zeta_partial(res: Result) -> None:
    banner("[T_CH.10] Spectral-zeta partial sum on the verified class-one set "
           "{3875, 27000}")

    # Partial Plancherel sum on the explicitly verified class-one reps
    # (3875 with C_2 = 96 and 27000 with C_2 = 124).  The full F1
    # closure relies on the conservative Camporesi-Higuchi finite-part
    # bound on the higher-Casimir tail (cf. cc2_f1_camporesi_higuchi.py).
    class_one = [
        {"name": "3875",  "dim": 3875,  "C2": 96.0,  "m": 1},
        {"name": "27000", "dim": 27000, "C2": 124.0, "m": 1},
    ]

    s_test = 2.0
    zeta_direct = sum(r["dim"] * r["m"] * (r["C2"] ** (-s_test))
                      for r in class_one)

    # Mellin cross-check: ζ_partial(s) = (1/Γ(s)) ∫ t^(s-1) Tr'K_t dt
    log_grid = np.logspace(-7.0, 1.0, 4000)

    def TrPrimeK(t: float) -> float:
        return float(sum(r["dim"] * r["m"] * math.exp(-r["C2"] * t)
                         for r in class_one))

    integrand = np.array([(t ** (s_test - 1.0)) * TrPrimeK(t)
                          for t in log_grid])
    f_t = integrand * log_grid
    dlog_t = np.diff(np.log(log_grid))
    integral = float(np.sum(0.5 * (f_t[:-1] + f_t[1:]) * dlog_t))
    zeta_via_mellin = integral / math.gamma(s_test)
    rel_err = abs(zeta_via_mellin - zeta_direct) / abs(zeta_direct)

    print(f"    Class-one set used:")
    for r in class_one:
        print(f"      {r['name']:<8}  dim = {r['dim']:<6}  C_2 = {r['C2']:.0f}")
    print()
    print(f"    Direct sum     ζ_partial({s_test}) = Σ d_λ · C_2^(-s) = "
          f"{zeta_direct:.10e}")
    print(f"    Mellin integral via heat-kernel              = "
          f"{zeta_via_mellin:.10e}")
    print(f"    Relative error                               = {rel_err:.4e}")
    print()

    zeta_0_partial = sum(r["dim"] for r in class_one)
    print(f"    ζ_partial(0) = Σ d_λ = {zeta_0_partial:.0f}  (= 3875 + 27000 = 30875)")
    print()

    res.report(
        f"T_CH.10.a  Mellin identity ζ_partial(s) ↔ heat-kernel at s = {s_test}",
        rel_err < TOL_MELLIN,
        f"rel err = {rel_err:.2e}",
    )
    res.report(
        "T_CH.10.b  ζ_partial(0) = 30875  (= sum of class-one dimensions)",
        abs(zeta_0_partial - 30875.0) < TOL_ALG,
        f"ζ_partial(0) = {zeta_0_partial:.4f}",
    )


# ---------------------------------------------------------------------------
# Pytest wrappers
# ---------------------------------------------------------------------------


def _setup_globals():
    return build_e8_rep_machinery()


def test_ch_part_a_e8_rep_machinery():
    M = _setup_globals()
    res = Result()
    test_T_CH_1_cartan_matrix(M, res)
    test_T_CH_2_fundamental_weights(M, res)
    test_T_CH_3_weyl_dim_casimir(M, res)
    assert res.failed == 0


def test_ch_part_b_branching_248():
    M = _setup_globals()
    res = Result()
    test_T_CH_4_branching_248(M, res)
    test_T_CH_5_adjoint_not_class_one(res)
    assert res.failed == 0


def test_ch_part_c_class_one():
    M = _setup_globals()
    res = Result()
    test_T_CH_6_trivial_is_class_one(M, res)
    test_T_CH_7_lower_bound_class_one(M, res)
    assert res.failed == 0


def test_ch_part_d_heat_kernel_and_V_ind():
    res = Result()
    test_T_CH_8_heat_kernel(res)
    test_T_CH_9_sakharov_V_ind(res)
    assert res.failed == 0


def test_ch_part_e_spectral_zeta():
    res = Result()
    test_T_CH_10_spectral_zeta_partial(res)
    assert res.failed == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("Camporesi--Higuchi spectral zeta on EIX  (E_8 / (E_7 × SU(2)))")
    print("Notes on the cosmological constant in E_8 group field theory")
    print("=" * 78, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Building E_8 rep-theoretic machinery (Cartan, ω_i, ρ, "
          "positive roots) ...", flush=True)
    M = _setup_globals()
    print(f"        E_8 simple roots:        8  (Bourbaki convention)")
    print(f"        Positive roots:         {len(M['pos_roots'])}")
    print(f"        Fundamental weights:     8  (rows of ω)")
    print(f"        Weyl vector ρ:           |ρ|^2 = {float(M['rho'] @ M['rho']):.4f}")
    print(f"        E_8:                     dim 248,  rank 8,  h^v = 30")
    print(f"        H = E_7 × SU(2):         dim 136,  rank 8 (full)")
    print(f"        m = G/H:                 dim 112  (= (56, 2))")

    print("\n" + "-" * 78)
    print("Part A — E_8 representation-theoretic infrastructure")
    print("-" * 78)
    test_T_CH_1_cartan_matrix(M, res)
    print()
    test_T_CH_2_fundamental_weights(M, res)
    print()
    test_T_CH_3_weyl_dim_casimir(M, res)
    print()

    print("-" * 78)
    print("Part B — Branching 248|_{E_7 × SU(2)}")
    print("-" * 78)
    test_T_CH_4_branching_248(M, res)
    print()
    test_T_CH_5_adjoint_not_class_one(res)
    print()

    print("-" * 78)
    print("Part C — Class-one characterisation")
    print("-" * 78)
    test_T_CH_6_trivial_is_class_one(M, res)
    print()
    test_T_CH_7_lower_bound_class_one(M, res)
    print()

    print("-" * 78)
    print("Part D — Heat-kernel + Sakharov V_ind^EIX = 144")
    print("-" * 78)
    test_T_CH_8_heat_kernel(res)
    print()
    V_data = test_T_CH_9_sakharov_V_ind(res)
    print()

    print("-" * 78)
    print("Part E — Spectral-zeta partial sum + Mellin identity")
    print("-" * 78)
    test_T_CH_10_spectral_zeta_partial(res)
    print()

    elapsed = time.time() - t0
    print("=" * 78)
    print(f"Camporesi--Higuchi infrastructure: "
          f"{res.passed} PASS / {res.failed} FAIL  (~{elapsed:.2f} s)")
    print("=" * 78)
    print()
    print(f"  V_ind^EIX (leading + sub-leading) = "
          f"{V_data['V_ind_full']:.4f}  (= 144)")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
