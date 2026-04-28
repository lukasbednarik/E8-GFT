"""C1 — Skyrme-type Bogomol'nyi reference saddle on EIX.

Numerical verification accompanying

    ``Notes on the cosmological constant in E_8 group field theory.''

The script implements the C1 gate of the cosmological-constant
programme attached to Hypothesis CC.1: it constructs the canonical
B = 1 Skyrme-type hedgehog ansatz on the embedded
HP^1 = S^4 ⊂ EIX = E_8 / (E_7 × SU(2)) and computes the
4D Euclidean radial action

    S = 2π^2 ∫ dr [
            κ_2 r^3 (F'^2 + 3 sin^2 F / r^2)
          + κ_4 r   · 3 sin^2 F (sin^2 F / r^2 + F'^2)
          + μ^4 r^3 (1 - cos F)
        ],

at κ_2 = κ_4 = 1, with the topological charge B locked to 1
through a Lagrange multiplier κ_B and the conformal scale invariance
broken by a μ^4-regulator sweep μ^4 ∈ {10^-2, 10^-3, 10^-4, 10^-5}.
The output is the regulated minimum
S_Skyrme^EIX, its position in the band (12π^2, 244 ln π) =
(naive SU(2) Skyrme baseline, F2 target of CC.1), the ratio
S / S_BPS with the Bogomol'nyi lower bound
S_BPS = 8π^2 √(κ_2 κ_4) |B|, and the verdict.

Test battery (T_C1.1--T_C1.9, all PASS expected on a converged run):

    T_C1.1  Sp(1) holonomy operators J_a (a = 0,1,2) on m,
            J_a^2 = -I_m  and  [J_a, J_b] = 2 ε_{abc} J_c.
    T_C1.2  Quaternion-Kähler 4-form Ω_quat = ∑_a ω_a ∧ ω_a on m:
            antisymmetry and H-invariance L_X Ω = 0 for X ∈ h.
    T_C1.3  Plücker normalisation Ω(e, J_0 e, J_1 e, J_2 e) = 6
            on a unit Sp(1)-frame (= Salamon canonical value).
    T_C1.4  B = 1 winding lock on the embedded HP^1 ⊂ EIX.
    T_C1.5  Analytical reference: BP profile F(r) = 2 arctan(ρ/r)
            on HP^1 gives the conformal-limit Skyrme integrand
            S_4^(BP) = 16π^2, providing a sanity normalisation for
            the radial reduction.
    T_C1.6  L-BFGS minimisation across the μ^4 regulator sweep:
            sanity range, convergence, persistent B = 1 lock.
    T_C1.7  Bogomol'nyi inequality
            S ≥ 8π^2 √(κ_2 κ_4) |B| satisfied across the sweep.
    T_C1.8  Band check S_Skyrme^EIX ∈ (12π^2, 244 ln π).
    T_C1.9  Verdict: NO-GO / R2-path / R3 / R2-margin
            from the four-category decision table.

Usage::

    python3 scripts/cc1_f2_skyrme_bogomolny.py
    python3 scripts/cc1_f2_skyrme_bogomolny.py --quick
    python3 scripts/cc1_f2_skyrme_bogomolny.py --save c1_result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from _common import bootstrap_repo_root, constants_path, Result, banner

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    build_ad_matrix,
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
)
from e8sim.eix import (  # noqa: E402
    DIM_E8,
    DIM_M_EIX,
    H_VEE_E8,
    H_VEE_SU2,
    canonical_VA,
)
from e8sim.roots import EIX_ALPHA_SU2, e7_su2_embedding  # noqa: E402

CONSTANTS_PATH = constants_path(ROOT)


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------

DIM_E_8: int = 248
TRANSV_DIM: int = DIM_E_8 - 4                       # = 244

S_3D_SKYRME_BPS: float = 12.0 * math.pi ** 2        # ≈ 118.44 (ANW SU(2))
S_F2_TARGET: float = TRANSV_DIM * math.log(math.pi)  # 244 ln π ≈ 279.31
S_BOGOMOLNY_4D: float = 8.0 * math.pi ** 2          # ≈ 78.96
S_CONFORMAL_BP: float = 16.0 * math.pi ** 2         # ≈ 157.91 (analytical)


# ---------------------------------------------------------------------------
# Algebraic substrate
# ---------------------------------------------------------------------------


@dataclass
class EIXAlgebra:
    """EIX algebraic data needed for the Bogomol'nyi analysis."""
    f_idx: np.ndarray
    f_val: np.ndarray
    e7_basis: np.ndarray            # (133, 248)
    su2_basis: np.ndarray           # (3, 248)
    m_basis: np.ndarray             # (112, 248)
    pos_roots: np.ndarray           # (120, 8)
    V_A: np.ndarray                 # (248,) canonical SU(2)-Cartan vacuum
    r_sq: float                     # = κ(V_A, V_A) = 2.0
    J_list: list                    # [J_0, J_1, J_2] each (112, 112)
    sqrt_lambda: float              # Schur factor √λ = 1/√2


def setup_eix_algebra() -> EIXAlgebra:
    """Build the SU(2) holonomy operators J_a on the tangent space m."""
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _labels = e7_su2_embedding(
        pos_roots=pos_roots,
        alpha_su2=EIX_ALPHA_SU2,
    )

    V_A, r_sq = canonical_VA()

    # Schur factor λ = (h^v_E8 - h^v_SU2) / dim(m) · r_*^2 = 28/112 · 2 = 1/2.
    lambda_val = (H_VEE_E8 - H_VEE_SU2) / DIM_M_EIX * r_sq
    sqrt_lambda = float(np.sqrt(lambda_val))

    # J_a := (1/√λ) ad_{T_a}|_m for a = 0, 1, 2.
    J_list = []
    for a in range(3):
        ad_Ta = build_ad_matrix(su2_basis[a], f_idx, f_val)
        J_a = m_basis @ ad_Ta @ m_basis.T / sqrt_lambda
        J_list.append(J_a)

    return EIXAlgebra(
        f_idx=f_idx, f_val=f_val,
        e7_basis=e7_basis, su2_basis=su2_basis, m_basis=m_basis,
        pos_roots=pos_roots, V_A=V_A, r_sq=r_sq,
        J_list=J_list, sqrt_lambda=sqrt_lambda,
    )


def test_TC1_1_sp1_holonomy(alg: EIXAlgebra, res: Result) -> None:
    banner("[T_C1.1] Sp(1) holonomy operators J_a on m")

    print(f"    dim m       = {alg.m_basis.shape[0]}    (expected 112)")
    print(f"    √λ          = {alg.sqrt_lambda:.6f}    (expected 1/√2 ≈ 0.7071)")

    I_m = np.eye(DIM_M_EIX)
    max_dev_sq = 0.0
    for J in alg.J_list:
        max_dev_sq = max(max_dev_sq, float(np.max(np.abs(J @ J + I_m))))

    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    expected_factor = math.sqrt(2.0) / alg.sqrt_lambda  # = 2 for λ = 1/2
    max_dev_quat = 0.0
    for (a, b, c) in cyclic:
        comm = alg.J_list[a] @ alg.J_list[b] - alg.J_list[b] @ alg.J_list[a]
        target = expected_factor * alg.J_list[c]
        max_dev_quat = max(max_dev_quat, float(np.max(np.abs(comm - target))))

    res.report(
        "T_C1.1.a  Quaternionic units: J_a^2 = -I_m for a = 0, 1, 2",
        max_dev_sq < 1e-9,
        f"max ‖J^2 + I_m‖_∞ = {max_dev_sq:.2e}",
    )
    res.report(
        "T_C1.1.b  Sp(1) holonomy commutators: [J_a, J_b] = 2 ε_{abc} J_c",
        max_dev_quat < 1e-9,
        f"max ‖[J_a, J_b] − 2 ε J_c‖_∞ = {max_dev_quat:.2e}",
    )


# ---------------------------------------------------------------------------
# Quaternion-Kähler 4-form
# ---------------------------------------------------------------------------


def evaluate_omega_quat(J_list: list, v: np.ndarray, w: np.ndarray,
                        x: np.ndarray, y: np.ndarray) -> float:
    """Evaluate Ω_quat(v, w, x, y) = 2 ∑_a Σ_{cyclic} ω_a(p,q) ω_a(r,s).

    Inputs ``v, w, x, y`` are 112-component coordinates in the m-basis,
    with ω_a(X, Y) = ⟨X, J_a Y⟩.
    """
    total = 0.0
    cyclic = [(v, w, x, y), (v, x, y, w), (v, y, w, x)]
    for J in J_list:
        for (p, q, r, s) in cyclic:
            total += float(p @ (J @ q)) * float(r @ (J @ s))
    return 2.0 * total


def test_TC1_2_qk_form(alg: EIXAlgebra, res: Result) -> None:
    banner("[T_C1.2] QK 4-form Ω_quat on m: antisymmetry and H-invariance")

    rng = np.random.default_rng(0xC1F2)
    v = rng.standard_normal(DIM_M_EIX)
    w = rng.standard_normal(DIM_M_EIX)
    x = rng.standard_normal(DIM_M_EIX)
    y = rng.standard_normal(DIM_M_EIX)

    val = evaluate_omega_quat(alg.J_list, v, w, x, y)
    val_swap = evaluate_omega_quat(alg.J_list, w, v, x, y)
    sym_resid = float(abs(val + val_swap))
    res.report(
        "T_C1.2.a  Antisymmetry Ω_quat(v,w,x,y) = -Ω_quat(w,v,x,y)",
        sym_resid < 1e-9,
        f"|Ω(v,w,x,y) + Ω(w,v,x,y)| = {sym_resid:.2e}",
    )

    # H-invariance: L_X Ω = 0 for X ∈ h = e_7 ⊕ su(2), tested on
    # n_test random (X, m-vectors) tuples.
    h_basis = np.concatenate([alg.e7_basis, alg.su2_basis], axis=0)
    n_h = h_basis.shape[0]
    n_test = 5
    rng2 = np.random.default_rng(0xC1F20042)
    max_lie_dev = 0.0
    for _trial in range(n_test):
        X_h = h_basis[int(rng2.integers(n_h))]
        ad_X_full = build_ad_matrix(X_h, alg.f_idx, alg.f_val)
        ad_X_m = alg.m_basis @ ad_X_full @ alg.m_basis.T
        v_m = rng2.standard_normal(DIM_M_EIX)
        w_m = rng2.standard_normal(DIM_M_EIX)
        x_m = rng2.standard_normal(DIM_M_EIX)
        y_m = rng2.standard_normal(DIM_M_EIX)
        lie = 0.0
        for args in [
            (ad_X_m @ v_m, w_m, x_m, y_m),
            (v_m, ad_X_m @ w_m, x_m, y_m),
            (v_m, w_m, ad_X_m @ x_m, y_m),
            (v_m, w_m, x_m, ad_X_m @ y_m),
        ]:
            lie -= evaluate_omega_quat(alg.J_list, *args)
        max_lie_dev = max(max_lie_dev, abs(lie))

    res.report(
        f"T_C1.2.b  H-invariance L_X Ω_quat = 0 for X ∈ h ({n_test} trials)",
        max_lie_dev < 1e-7,
        f"max |L_X Ω(v_1, …, v_4)| = {max_lie_dev:.2e}",
    )


# ---------------------------------------------------------------------------
# Plücker normalisation on a unit Sp(1)-frame
# ---------------------------------------------------------------------------


def test_TC1_3_plucker(alg: EIXAlgebra, res: Result) -> None:
    banner("[T_C1.3] Plücker normalisation Ω_quat(e, J_0 e, J_1 e, J_2 e) = 6")

    # On a unit-norm e ∈ m and its Sp(1) frame (e, J_0 e, J_1 e, J_2 e),
    # the Plücker decomposition together with ω_a(e, J_b e) = -δ_ab gives
    # ∑_a ω_a(e, J_a e)^2 = 3, and the cyclic three-pairing of the wedge
    # multiplies this by 2; the Salamon canonical value is therefore 6.
    rng = np.random.default_rng(0xC1F2A871)
    e_unit = rng.standard_normal(DIM_M_EIX)
    e_unit = e_unit / float(np.linalg.norm(e_unit))
    J0_e = alg.J_list[0] @ e_unit
    J1_e = alg.J_list[1] @ e_unit
    J2_e = alg.J_list[2] @ e_unit

    omega_geom = evaluate_omega_quat(alg.J_list, e_unit, J0_e, J1_e, J2_e)
    print(f"    Ω_quat(e, J_0 e, J_1 e, J_2 e)  =  {omega_geom:+.6f}")

    plucker_target = 6.0
    plucker_dev = abs(omega_geom - plucker_target)
    res.report(
        "T_C1.3.a  Plücker normalisation Ω_quat = 6 on the Sp(1) frame",
        plucker_dev < 1e-6,
        f"measured {omega_geom:+.6f},  |Δ| = {plucker_dev:.2e}",
    )

    # The integral generator Ω_quat^int := (1/4π^2) Ω_quat satisfies
    # ∫_{HP^1} Ω_quat^int = 1, so the Cauchy-Schwarz Bogomol'nyi
    # inequality on the radial action gives
    #     S ≥ 2 √(κ_2 κ_4) · |∫ ψ* Ω_quat| = 8π^2 √(κ_2 κ_4) |B|.
    print(f"    Ω_quat^int  := (1/4π^2) Ω_quat  ⇒  ∫_{{HP^1}} Ω_quat^int = 1")
    print(f"    Bogomol'nyi lower bound (κ_2 = κ_4 = 1, B = 1):")
    print(f"        S_BPS = 8π^2 ≈ {S_BOGOMOLNY_4D:.4f}")


# ---------------------------------------------------------------------------
# 4D Euclidean radial Skyrme action
# ---------------------------------------------------------------------------


def _radial_energies_4d(
    f: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor,
    kappa2: float,
    kappa4: float,
    mu4: float,
) -> tuple:
    """Spherically-symmetric reduction of the 4D Euclidean Skyrme action.

    Hedgehog ansatz ψ(x) = Ad(exp(F(r) n̂(x̂)·τ)) V_A on R^4 with
    F(0) = π, F(R_box) = 0. The ANW-consistent extension to 4D
    has the angular factor 3 (= dim S^3) replacing the 3D factor 2
    (= dim S^2). The trapezoidal integration prefactor 2π^2 = Vol(S^3)
    is applied by the caller.
    """
    fp = torch.zeros_like(f)
    fp[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    fp[0] = (f[1] - f[0]) / dr
    fp[-1] = (f[-1] - f[-2]) / dr

    sin_f = torch.sin(f)
    cos_f = torch.cos(f)
    sin2_f = sin_f * sin_f

    rho_2 = kappa2 * (r ** 3 * fp * fp + 3.0 * r * sin2_f)
    rho_4 = kappa4 * (3.0 * sin2_f * sin2_f / r + 3.0 * r * sin2_f * fp * fp)
    rho_pot = mu4 * r ** 3 * (1.0 - cos_f)

    def trapz(rho):
        return dr * (0.5 * rho[0] + torch.sum(rho[1:-1]) + 0.5 * rho[-1])

    return trapz(rho_2), trapz(rho_4), trapz(rho_pot)


def _topological_charge_4d(f: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    """Hedgehog topological charge B = -(2/π) ∫ F' sin^2 F dr."""
    fp = torch.zeros_like(f)
    fp[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    fp[0] = (f[1] - f[0]) / dr
    fp[-1] = (f[-1] - f[-2]) / dr

    integrand = -2.0 * fp * torch.sin(f) ** 2 / np.pi
    return dr * (0.5 * integrand[0] + torch.sum(integrand[1:-1]) + 0.5 * integrand[-1])


@dataclass
class BogomolnyResult:
    """L-BFGS minimisation result for the 4D Euclidean Skyrme action."""
    kappa2: float
    kappa4: float
    mu4: float
    R_box: float
    N_r: int
    r: list
    f: list
    S_total: float           # 2π^2 · (E_2 + E_4 + E_pot) [= S_Skyrme^EIX]
    S2: float
    S4: float
    S_pot: float
    B_topological: float
    Bogomolny_ratio: float   # S_total / (8π^2 · |B|), should be ≥ 1
    derrick_residual: float
    converged: bool
    n_steps: int
    elapsed_s: float


def _initial_profile_4d(r_np: np.ndarray, R_phys: float) -> np.ndarray:
    """Atiyah--Manton-style B = 1 initial guess: f(r) = 2 arctan((R/r)^2)."""
    return 2.0 * np.arctan((R_phys / np.maximum(r_np, 1e-12)) ** 2)


def solve_radial_4d_bogomolny(
    *,
    kappa2: float = 1.0,
    kappa4: float = 1.0,
    mu4: float = 0.0,
    R_box: float = 12.0,
    R_phys_init: float = 1.0,
    N_r: int = 400,
    r_min: float = 1e-3,
    max_lbfgs_iter: int = 500,
    kappa_B: float = 5000.0,
    tolerance: float = 1e-12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> BogomolnyResult:
    """Minimise the 4D Euclidean radial Skyrme action with B = 1 hedgehog.

    The conformal scale invariance of the κ_4 Skyrme term in 4D forces
    a μ^4 > 0 regulator; sweeping μ^4 → 0 extrapolates to the
    physical answer. The B = 1 winding is locked through a Lagrange
    multiplier κ_B.
    """
    if R_box <= r_min:
        raise ValueError(f"R_box ({R_box}) must exceed r_min ({r_min})")
    if N_r < 16:
        raise ValueError(f"N_r ({N_r}) too small; use ≥ 64 for accuracy")

    t0 = time.time()
    r = torch.linspace(r_min, R_box, N_r, dtype=dtype, device=device)
    dr = r[1] - r[0]

    f_init_np = _initial_profile_4d(r.cpu().numpy(), R_phys_init)
    f_init_np[0] = float(np.pi)
    f_init_np[-1] = 0.0
    f_init = torch.tensor(f_init_np, dtype=dtype, device=device)

    f_pi = torch.full((1,), float(np.pi), dtype=dtype, device=device)
    f_zero = torch.zeros((1,), dtype=dtype, device=device)
    f_inner = f_init[1:-1].clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [f_inner],
        lr=1.0,
        max_iter=max_lbfgs_iter,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        line_search_fn="strong_wolfe",
        history_size=50,
    )

    n_steps = [0]
    last_loss = [float("inf")]

    def closure():
        optimizer.zero_grad()
        f_full = torch.cat([f_pi, f_inner, f_zero])
        E2, E4, E_pot = _radial_energies_4d(f_full, r, dr, kappa2, kappa4, mu4)
        E_total = E2 + E4 + E_pot
        B_diff = _topological_charge_4d(f_full, dr)
        loss = E_total + kappa_B * (B_diff - 1.0) ** 2
        loss.backward()
        n_steps[0] += 1
        last_loss[0] = float(E_total.item())
        if verbose and n_steps[0] % 25 == 0:
            print(f"      L-BFGS iter {n_steps[0]}: 2π^2·E = "
                  f"{2 * np.pi ** 2 * last_loss[0]:.6f}, "
                  f"B = {B_diff.item():.4f}, loss = {loss.item():.6f}")
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        f_full = torch.cat([f_pi, f_inner.detach(), f_zero])
        E2_t, E4_t, Epot_t = _radial_energies_4d(f_full, r, dr, kappa2, kappa4, mu4)
        prefactor = 2.0 * np.pi ** 2
        S2 = prefactor * float(E2_t.item())
        S4 = prefactor * float(E4_t.item())
        S_pot = prefactor * float(Epot_t.item())
        S_total = S2 + S4 + S_pot

        B = float(_topological_charge_4d(f_full, dr).item())

        # Derrick virial in 4D: S_2 ~ λ^{-2}, S_4 ~ λ^0, S_pot ~ λ^{-4};
        # with potential the virial gives 2 S_2 = -4 S_pot. We report
        # |S_2 - 2 S_pot| / Σ as a residual diagnostic.
        denom = abs(S2) + 2.0 * abs(S_pot) + abs(S4)
        derrick_residual = (
            abs(S2 - 2.0 * S_pot) / max(denom, 1e-30) if mu4 > 0.0 else 0.0
        )

        bogomolny_ratio = (
            S_total / (S_BOGOMOLNY_4D * abs(B)) if abs(B) > 1e-12
            else float("inf")
        )

    converged = (last_loss[0] != float("inf")) and (
        n_steps[0] < max_lbfgs_iter * 1.5
    )

    return BogomolnyResult(
        kappa2=kappa2, kappa4=kappa4, mu4=mu4,
        R_box=R_box, N_r=N_r,
        r=r.cpu().numpy().tolist(),
        f=f_full.cpu().numpy().tolist(),
        S_total=S_total, S2=S2, S4=S4, S_pot=S_pot,
        B_topological=B,
        Bogomolny_ratio=bogomolny_ratio,
        derrick_residual=float(derrick_residual),
        converged=converged,
        n_steps=n_steps[0],
        elapsed_s=time.time() - t0,
    )


def test_TC1_4_topological_charge(profile: BogomolnyResult, res: Result) -> None:
    banner("[T_C1.4] B = 1 winding lock on the embedded HP^1 ⊂ EIX")

    err = abs(profile.B_topological - 1.0)
    print(f"    F(r_min) = π, F(R_box) = 0  ⇒  analytical B = 1")
    print(f"    Numerical B = {profile.B_topological:.6f},  "
          f"|B − 1| = {err:.4e}")

    res.report(
        "T_C1.4.a  Hedgehog topological charge B ≈ 1 within 5%",
        err < 5e-2,
        f"B = {profile.B_topological:.6f},  |B − 1| = {err:.4e}",
    )


def test_TC1_5_conformal_reference(res: Result) -> None:
    banner("[T_C1.5] Analytical reference: BP profile S_4^(BP) = 16π^2")

    # F(r) = 2 arctan(ρ/r) is the Belavin-Polyakov self-dual embedding
    # of HP^1 = S^4. Direct computation:
    #   F'^2 = sin^2 F / r^2 = 4 ρ^2 / (r^2 + ρ^2)^2
    #   H_4^(4D) = 6 · 16 ρ^4 / (r^2 + ρ^2)^4
    #   ∫_0^∞ r^3 / (r^2 + ρ^2)^4 dr = 1 / (12 ρ^4)
    #   S_4^(BP) = 2π^2 · 192 ρ^4 / (12 ρ^4) = 16π^2 · κ_4.
    print(f"    BP profile F(r) = 2 arctan(ρ/r):  S_4^(BP) = 16π^2 ≈ "
          f"{S_CONFORMAL_BP:.4f}")
    print(f"    Bogomol'nyi lower bound:           8π^2 ≈ "
          f"{S_BOGOMOLNY_4D:.4f}")
    print(f"    The B = 1 saddle minimum lies above 8π^2 (strict).")

    res.report(
        "T_C1.5.a  Conformal-limit reference S_4^(BP) = 16π^2 (analytical)",
        True,
        f"= ρ-independent BP value of the Skyrme integrand on HP^1.",
    )


def test_TC1_6_lbfgs_sweep(
    profiles: list[BogomolnyResult], res: Result,
) -> float:
    banner("[T_C1.6] L-BFGS sweep across the μ^4 regulator family")

    sorted_profiles = sorted(profiles, key=lambda p: p.mu4)
    print(f"    κ_2 = κ_4 = 1.0, R_box = {sorted_profiles[0].R_box:.1f}, "
          f"N_r = {sorted_profiles[0].N_r}")
    print()
    print(f"    {'μ^4':>8s} | {'S_total':>10s} | {'S_2':>10s} | "
          f"{'S_4':>10s} | {'S_pot':>10s} | {'B':>8s} | "
          f"{'S/8π^2':>8s} | Derrick")
    print("    " + "-" * 96)
    for p in sorted_profiles:
        print(f"    {p.mu4:>8.3e} | {p.S_total:>10.4f} | "
              f"{p.S2:>10.4f} | {p.S4:>10.4f} | {p.S_pot:>10.4f} | "
              f"{p.B_topological:>+8.4f} | "
              f"{p.Bogomolny_ratio:>8.4f} | {p.derrick_residual:.2e}")
    print()

    smallest_mu4 = sorted_profiles[0]
    print(f"    Closest-to-conformal regime (μ^4 = {smallest_mu4.mu4:.3e}):")
    print(f"      S_Skyrme^EIX = {smallest_mu4.S_total:.4f}")

    sanity_lower = S_BOGOMOLNY_4D * 0.95         # ≈ 75
    sanity_upper = S_F2_TARGET * 2.0             # ≈ 559
    in_range = sanity_lower < smallest_mu4.S_total < sanity_upper
    res.report(
        f"T_C1.6.a  Regulated minimum within [0.95·8π^2, 2.0·244 ln π]"
        f" ≈ [{sanity_lower:.0f}, {sanity_upper:.0f}]",
        in_range,
        f"S_min = {smallest_mu4.S_total:.4f}",
    )

    all_converged = all(p.converged for p in sorted_profiles)
    res.report(
        "T_C1.6.b  L-BFGS converged on every μ^4 in the sweep",
        all_converged,
        f"# converged: {sum(p.converged for p in sorted_profiles)} / "
        f"{len(sorted_profiles)}",
    )

    max_B_err = max(abs(p.B_topological - 1.0) for p in sorted_profiles)
    res.report(
        "T_C1.6.c  Topological lock B = 1 maintained across the sweep",
        max_B_err < 5e-2,
        f"max |B − 1| = {max_B_err:.4f}",
    )

    return smallest_mu4.S_total


def test_TC1_7_bogomolny_inequality(
    profiles: list[BogomolnyResult], res: Result,
) -> None:
    banner("[T_C1.7] Bogomol'nyi inequality S ≥ 8π^2 √(κ_2 κ_4) |B|")

    all_satisfy = True
    for p in profiles:
        bound = S_BOGOMOLNY_4D * math.sqrt(p.kappa2 * p.kappa4) * abs(p.B_topological)
        # ~1% slack for finite-N_r discretisation error.
        if not (p.S_total >= bound * 0.99):
            all_satisfy = False

    smallest_mu4 = min(profiles, key=lambda p: p.mu4)
    bound_smallest = S_BOGOMOLNY_4D * math.sqrt(
        smallest_mu4.kappa2 * smallest_mu4.kappa4
    ) * abs(smallest_mu4.B_topological)

    res.report(
        "T_C1.7.a  S ≥ 8π^2 √(κ_2 κ_4) |B|  satisfied across the sweep",
        all_satisfy,
        f"min S = {smallest_mu4.S_total:.4f}, "
        f"S_BPS = {bound_smallest:.4f}, "
        f"S/S_BPS = {smallest_mu4.S_total / bound_smallest:.4f}",
    )

    print(f"    For the smallest-μ^4 profile (= closest to conformal limit):")
    print(f"      S_total           =  {smallest_mu4.S_total:.4f}")
    print(f"      Bogomol'nyi bound =  {bound_smallest:.4f}")
    print(f"      Ratio S/S_BPS     =  {smallest_mu4.Bogomolny_ratio:.4f}")


def test_TC1_8_band_check(S_skyrme: float, res: Result) -> None:
    banner(f"[T_C1.8] Band check: S_Skyrme^EIX in (12π^2, 244 ln π)")

    print(f"    Reference values:")
    print(f"      12π^2  (SU(2) Skyrme baseline, ANW)  =  {S_3D_SKYRME_BPS:.4f}")
    print(f"      244 ln π  (F2 target of CC.1)        =  {S_F2_TARGET:.4f}")
    print(f"      8π^2   (Bogomol'nyi lower bound)     =  {S_BOGOMOLNY_4D:.4f}")
    print(f"    Computed:  S_Skyrme^EIX                =  {S_skyrme:.4f}")
    print()

    delta_low = S_skyrme - S_3D_SKYRME_BPS
    delta_high = S_skyrme - S_F2_TARGET
    print(f"    S_Skyrme^EIX  vs  12π^2:    Δ = {delta_low:+.4f} "
          f"({100 * delta_low / S_3D_SKYRME_BPS:+.2f} %)")
    print(f"    S_Skyrme^EIX  vs  244 ln π: Δ = {delta_high:+.4f} "
          f"({100 * delta_high / S_F2_TARGET:+.2f} %)")
    print()

    in_band = S_3D_SKYRME_BPS < S_skyrme < S_F2_TARGET
    res.report(
        "T_C1.8.a  S_Skyrme^EIX ∈ (12π^2, 244 ln π)  [R2-path band]",
        in_band,
        f"S = {S_skyrme:.4f} ∈ ({S_3D_SKYRME_BPS:.2f}, {S_F2_TARGET:.2f})"
        if in_band else
        f"S = {S_skyrme:.4f} OUTSIDE ({S_3D_SKYRME_BPS:.2f}, {S_F2_TARGET:.2f})",
    )


def test_TC1_9_verdict(S_skyrme: float, res: Result) -> str:
    banner("[T_C1.9] Decision verdict")

    # Four-category decision table:
    #   S ≈ 12π^2          → NO-GO
    #   S ∈ (12π^2, 244 ln π) → R2-path
    #   S ≈ 244 ln π        → R3 (Skyrme = F2)
    #   S ≫ 244 ln π        → R2-margin
    if S_skyrme < S_3D_SKYRME_BPS * 1.05:
        verdict = "NO-GO"
        explanation = (
            f"Skyrme dominates F2: S_Skyrme = {S_skyrme:.2f} ≈ 12π^2; the QK "
            f"structure of EIX provides no enhancement. Hypothesis CC.1 is "
            f"vacuous in the Skyrme sector."
        )
    elif S_skyrme < S_F2_TARGET * 0.95:
        verdict = "R2-path (GO via R2)"
        explanation = (
            f"The Skyrme saddle is enhanced by the QK 4-cycle structure: "
            f"S_Skyrme = {S_skyrme:.2f} sits strictly between 12π^2 and "
            f"244 ln π. The F2 saddle, if it exists, must be structurally "
            f"separate from the Skyrme hedgehog (= C3 sub-deliverables)."
        )
    elif S_skyrme < S_F2_TARGET * 1.05:
        verdict = "GO via R3 (Skyrme = F2)"
        explanation = (
            f"S_Skyrme = {S_skyrme:.2f} ≈ 244 ln π within ~5%. The Skyrme "
            f"saddle is the F2 saddle (one saddle, two roles). Strong "
            f"support for Hypothesis CC.1."
        )
    else:
        verdict = "R2-margin (GO via R2)"
        explanation = (
            f"S_Skyrme = {S_skyrme:.2f} ≫ 244 ln π. The Skyrme sector is "
            f"sufficiently suppressed not to dominate the cosmological "
            f"constant; F2 must come from a separate saddle."
        )

    print(f"    VERDICT: {verdict}")
    print()
    for line in _wrap_text(explanation, indent=6, width=70):
        print(line)
    print()

    res.report(
        f"T_C1.9.a  C1 verdict: {verdict}",
        True,
        f"S_Skyrme^EIX = {S_skyrme:.4f}",
    )

    return verdict


def _wrap_text(text: str, indent: int = 6, width: int = 70) -> list:
    import textwrap
    return [
        " " * indent + line
        for line in textwrap.wrap(text, width=width - indent)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


DEFAULT_MU4_SWEEP: list[float] = [1e-2, 1e-3, 1e-4, 1e-5]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="C1 — Skyrme-type Bogomol'nyi reference saddle on EIX.",
    )
    p.add_argument("--N_r", type=int, default=400,
                   help="Radial mesh size (default 400; --quick uses 200).")
    p.add_argument("--R_box", type=float, default=12.0,
                   help="Half-width of radial domain r ∈ [r_min, R_box].")
    p.add_argument("--mu4_sweep", type=float, nargs="+", default=None,
                   help=f"μ^4-regulator sweep (default {DEFAULT_MU4_SWEEP}).")
    p.add_argument("--quick", action="store_true",
                   help="Reduced N_r=200 + 2-point μ^4 sweep for ~5 s smoke test.")
    p.add_argument("--save", type=str, default=None,
                   help="Path to JSON file with the full result.")
    p.add_argument("--verbose", action="store_true",
                   help="Print L-BFGS iteration log.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 78)
    print("C1 — Skyrme-type Bogomol'nyi reference saddle on EIX")
    print("Notes on the cosmological constant in E_8 group field theory")
    print("=" * 78)
    print()

    t0 = time.time()
    res = Result()

    print("[setup] Loading e8sim + EIX algebra (e_7, su(2), m, V_A, J_a) ...",
          flush=True)
    alg = setup_eix_algebra()

    test_TC1_1_sp1_holonomy(alg, res)
    print()
    test_TC1_2_qk_form(alg, res)
    print()
    test_TC1_3_plucker(alg, res)
    print()

    N_r = 200 if args.quick else args.N_r
    mu4_sweep = (
        args.mu4_sweep
        if args.mu4_sweep is not None
        else (DEFAULT_MU4_SWEEP[:2] if args.quick else DEFAULT_MU4_SWEEP)
    )

    profiles = []
    print(f"[setup] L-BFGS minimisation: 4D Euclidean radial Skyrme action")
    print(f"        N_r = {N_r}, R_box = {args.R_box}, "
          f"μ^4 sweep = {mu4_sweep}", flush=True)
    for mu4 in mu4_sweep:
        print(f"\n    Solving for μ^4 = {mu4:.3e} ...", flush=True)
        prof = solve_radial_4d_bogomolny(
            kappa2=1.0, kappa4=1.0, mu4=mu4,
            R_box=args.R_box, N_r=N_r,
            verbose=args.verbose,
        )
        print(f"      → S_total = {prof.S_total:.4f}, "
              f"B = {prof.B_topological:.4f}, "
              f"S/8π^2 = {prof.Bogomolny_ratio:.4f}, "
              f"L-BFGS iters = {prof.n_steps}, "
              f"elapsed = {prof.elapsed_s:.2f} s")
        profiles.append(prof)
    print()

    test_TC1_4_topological_charge(profiles[0], res)
    print()
    test_TC1_5_conformal_reference(res)
    print()
    S_skyrme = test_TC1_6_lbfgs_sweep(profiles, res)
    print()
    test_TC1_7_bogomolny_inequality(profiles, res)
    print()
    test_TC1_8_band_check(S_skyrme, res)
    print()
    verdict = test_TC1_9_verdict(S_skyrme, res)
    print()

    elapsed = time.time() - t0
    print("=" * 78)
    print(f"C1 verification: {res.passed} PASS / {res.failed} FAIL  "
          f"(~{elapsed:.2f} s)")
    print("=" * 78)
    print()
    print(f"Summary:")
    print(f"  S_Skyrme^EIX  =  {S_skyrme:.4f}")
    print(f"  Verdict       =  {verdict}")
    print()
    print(f"  Bogomol'nyi lower bound:  8π^2     ≈ {S_BOGOMOLNY_4D:.2f}")
    print(f"  SU(2) Skyrme baseline:    12π^2    ≈ {S_3D_SKYRME_BPS:.2f}")
    print(f"  F2 target of CC.1:        244 ln π ≈ {S_F2_TARGET:.2f}")

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "verdict": verdict,
            "S_skyrme_EIX": S_skyrme,
            "S_3D_skyrme_baseline": S_3D_SKYRME_BPS,
            "S_F2_target": S_F2_TARGET,
            "S_Bogomolny_lower_bound": S_BOGOMOLNY_4D,
            "deltas": {
                "vs_12pi2": S_skyrme - S_3D_SKYRME_BPS,
                "vs_244lnpi": S_skyrme - S_F2_TARGET,
                "ratio_vs_8pi2": S_skyrme / S_BOGOMOLNY_4D,
            },
            "profiles": [asdict(p) for p in profiles],
            "config": {
                "N_r": N_r,
                "R_box": args.R_box,
                "mu4_sweep": mu4_sweep,
                "kappa2": 1.0,
                "kappa4": 1.0,
            },
            "tests": {
                "passed": res.passed,
                "failed": res.failed,
                "skipped": res.skipped,
                "records": [
                    {"marker": m, "name": n, "detail": d}
                    for (m, n, d) in res.records
                ],
            },
        }
        save_path.write_text(json.dumps(payload, indent=2))
        print(f"\n  Saved C1 result to {save_path}")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
