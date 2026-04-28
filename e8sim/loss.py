"""Energy functional (Hamiltonian), topological charge (Hopf invariant) and
charge measurement.

Per Q2(b) of `docs/theory-wip-p6-do1.md` (4/2026): the EIX-native hedgehog
lives in the 3D su(2)_stab subspace, so the topological charge B is the
Hopf invariant of the projected unit-vector field ψ̂(x) ∈ S² (= the second
homotopy invariant of S² → S² composed with the Hopfova fibrace
SU(2) → S²).  `compute_topological_charge` evaluates
B = (1/(16π²)) ∫ A·F where F is the Berry curvature of ψ̂ and A is the
U(1) connection in Coulomb gauge ∂_i A_i = 0 (= ∇² A_i = (curl F)_i,
solved via FFT).  The 1/(16π²) prefactor calibrates the standard SU(2)
hedgehog with f(0) = π to baryon number B = 1; see
``compute_topological_charge_diff`` body for normalisation derivation
and `debug_plan/scripts/eo2_b_init_convergence.py` for the empirical
continuum-limit verification (B_init(N) → 1 as N → ∞).
"""

import torch
import numpy as np

from .algebra import commutator
from .fields import compute_derivatives, extract_su2_hopf


def compute_loss(psi, psi_vac, f_idx, f_val, dx, kappa1, kappa2, kappa_pot,
                 N, chunk_size, use_cp=False, mask=None):
    """Compute the Skyrme-like energy functional on the EIX-native ansatz.

    Returns ``(loss, H2, H4, H_pot, H_gauge)``.

    H_pot pulls ψ → ``psi_vac`` ( = V_A in the SU(2)-Cartan, per Q2(b)
    rezoluci) on the outer shell so the soliton sits inside a vacuum
    buffer.  Because ``psi_vac`` is a single 248-direction (= V_A), the
    formula ‖ψ − ψ_vac‖² already constrains all 248 components — so the
    legacy auxiliary H_gauge term (which only pinned the H_2..H_7 Cartan
    slots in the *old* (H_0, H_1) vacuum convention) is no longer needed.
    H_gauge is returned as a zero tensor for backwards-compatible
    tuple-arity (callers that destructure 5 values still work).

    H_4 = κ_2 · S_c (Skyrme term).  Per Tvrzení D-3
    (theory-wip-p5-do1.md) S_c is the unique dominant O(∂⁴) H-invariant
    on EIX modulo K4 (large-N_c).
    """
    dpsi_dx, dpsi_dy, dpsi_dz = compute_derivatives(psi, dx)

    vol = (dx**3) * mask if mask is not None else (dx**3)

    H2 = kappa1 * torch.sum((dpsi_dx**2 + dpsi_dy**2 + dpsi_dz**2) * vol)

    C_xy = commutator(dpsi_dx, dpsi_dy, f_idx, f_val, N, chunk_size, use_cp)
    H4 = kappa2 * torch.sum((C_xy**2) * vol)
    del C_xy

    C_yz = commutator(dpsi_dy, dpsi_dz, f_idx, f_val, N, chunk_size, use_cp)
    H4 = H4 + kappa2 * torch.sum((C_yz**2) * vol)
    del C_yz

    C_zx = commutator(dpsi_dz, dpsi_dx, f_idx, f_val, N, chunk_size, use_cp)
    H4 = H4 + kappa2 * torch.sum((C_zx**2) * vol)
    del C_zx

    r_sq = torch.linspace(-1, 1, N, device=psi.device) ** 2
    R2 = r_sq.view(N, 1, 1) + r_sq.view(1, N, 1) + r_sq.view(1, 1, N)

    # Potential only in the outer shell — interior (R² < 0.5) is free so that
    # the soliton core survives under H₂+H₄ Derrick balance.
    shell_mask = (R2 >= 0.5).float()
    boundary_penalty = shell_mask * (1.0 + 5.0 * (R2 > 0.8).float())

    H_pot = kappa_pot * torch.sum(((psi - psi_vac)**2) * boundary_penalty * vol)

    H_gauge = torch.zeros((), dtype=psi.dtype, device=psi.device)

    loss = H2 + H4 + H_pot
    return loss, H2, H4, H_pot, H_gauge


# ─── Hopf invariant (= topological B-charge per Q2(b)) ────────────────


def _berry_curvature(n_field, dx):
    """Compute Berry curvature components F^i = (1/2) ε^{ijk} F_{jk} where
    F_{jk} = ε_{abc} n̂_a (∂_j n̂_b)(∂_k n̂_c).  Returns (F_yz, F_zx, F_xy)
    = ("magnetic field" B^i in U(1)-bundle convention).
    """
    dn_dx, dn_dy, dn_dz = compute_derivatives(n_field, dx)

    def triple_dot(n, a, b):
        cross = torch.stack([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ], dim=0)
        return torch.sum(n * cross, dim=0)

    F_yz = triple_dot(n_field, dn_dy, dn_dz)
    F_zx = triple_dot(n_field, dn_dz, dn_dx)
    F_xy = triple_dot(n_field, dn_dx, dn_dy)

    return F_yz, F_zx, F_xy


def _solve_coulomb_A(F_yz, F_zx, F_xy, dx):
    """Solve ∇² A_i = -(curl B)_i via FFT (Coulomb gauge ∂_i A_i = 0).

    Inputs (F_yz, F_zx, F_xy) are the components of the dual 1-vector
    B^i = (1/2) ε^{ijk} F_{jk}.  Coulomb-gauge identity:

        ∂_j F_{ji} = ∇² A_i,
        F_{ji} = ε_{jik} B^k  ⇒  ∂_j F_{ji} = -ε_{ijk} ∂_j B^k = -(curl B)_i.

    Uses periodic FFT — a finite-volume approximation accurate when the
    soliton is well-localised inside the box (vacuum buffer ≥ R_phys).
    """
    N = F_yz.shape[-1]
    device = F_yz.device

    freqs = torch.fft.fftfreq(N, d=dx, device=device)
    k = 2.0 * np.pi * freqs
    kx = k.view(N, 1, 1)
    ky = k.view(1, N, 1)
    kz = k.view(1, 1, N)
    k2 = kx**2 + ky**2 + kz**2
    k2_safe = k2.clone()
    k2_safe[0, 0, 0] = 1.0

    Bx = F_yz.unsqueeze(0)
    By = F_zx.unsqueeze(0)
    Bz = F_xy.unsqueeze(0)

    dBx_dx, dBx_dy, dBx_dz = compute_derivatives(Bx, dx)
    dBy_dx, dBy_dy, dBy_dz = compute_derivatives(By, dx)
    dBz_dx, dBz_dy, dBz_dz = compute_derivatives(Bz, dx)

    curl_Bx = (dBz_dy[0] - dBy_dz[0])
    curl_By = (dBx_dz[0] - dBz_dx[0])
    curl_Bz = (dBy_dx[0] - dBx_dy[0])

    def fft_inverse_lap(rhs):
        # Solves ∇² A = rhs (real space) via -k² A_k = rhs_k.
        rhs_k = torch.fft.fftn(rhs.to(torch.complex64))
        A_k = -rhs_k / k2_safe
        A_k[0, 0, 0] = 0.0
        return torch.fft.ifftn(A_k).real.to(rhs.dtype)

    # ∇² A_i = -(curl B)_i
    Ax = fft_inverse_lap(-curl_Bx)
    Ay = fft_inverse_lap(-curl_By)
    Az = fft_inverse_lap(-curl_Bz)

    return Ax, Ay, Az


def compute_topological_charge(psi, psi_vac, su2_basis, dx):
    """Hopf invariant of the projected unit-vector field ψ̂(x) ∈ S².

    Per Q2(b) of `docs/theory-wip-p6-do1.md`: the EIX-native hedgehog
    lives in the 3D su(2)_stab subspace, so its topological invariant
    is B = H[ψ̂] = (1/(4π²)) ∫ d³x A^i F_i where F is the Berry
    curvature of the unit-vector field ψ̂(x) and A is the U(1) connection
    in Coulomb gauge.

    ``psi_vac`` is accepted (and ignored) only for backwards-compatible
    signature parity with the legacy 4D Skyrme charge implementation;
    Hopf invariant depends only on ψ and the choice of su(2)_stab basis.

    Returns a Python float (use the same function for the differentiable
    version inside the Phase-1 loss; PyTorch FFT propagates gradients
    natively).
    """
    return compute_topological_charge_diff(psi, psi_vac, su2_basis, dx).item()


def compute_topological_charge_diff(psi, psi_vac, su2_basis, dx):
    """Differentiable version of `compute_topological_charge`.

    Returns B as a scalar tensor (graph attached to ψ) so that the
    Phase-1 penalty κ_B·(B − 1)² can backpropagate through the Hopf
    invariant.  ``psi_vac`` is accepted for signature parity but is
    not used (Hopf invariant is defined purely from the ψ̂(x) field).
    """
    psi_su2 = extract_su2_hopf(psi, su2_basis)
    norm = torch.norm(psi_su2, dim=0, keepdim=True).clamp(min=1e-10)
    n = psi_su2 / norm

    F_yz, F_zx, F_xy = _berry_curvature(n, dx)
    Ax, Ay, Az = _solve_coulomb_A(F_yz, F_zx, F_xy, dx)

    integrand = Ax * F_yz + Ay * F_zx + Az * F_xy
    # Prefactor 1/(16π²) — calibrates to baryon number B = 1 for the
    # standard SU(2) hedgehog g = exp(if(r) n̂·σ) with f(0) = π,
    # f(∞) = 0 (= one full Skyrme winding deg(g) = 1).  Per Q2(b) §4
    # of `docs/theory-wip-p6-do1.md`, the Ad-image ψ̂ = g τ_3 g⁻¹/‖τ_3‖
    # carries Hopf charge H[ψ̂] = deg(g) = 1.
    #
    # In the `wolf_space_seed` Rodriguez SO(3) parametrization the
    # equivalent angle is c(r) = 2 f(r), so c(0) = 2π corresponds to
    # f(0) = π.  Empirical convergence test (eo2_b_init_convergence.py,
    # 4/2026) verifies B_init(N) → 1 in the continuum limit (N = 96
    # gives 0.93, N → ∞ extrapolation hits 1.0).
    #
    # Earlier prefactor 1/(8π²) was a Skyrme-winding (4D σ-target)
    # calibration carried over from the legacy hedgehog; on the
    # post-Q2(b) Hopf invariant (3D `su(2)_stab` target) it gave
    # B_init → 2 in the continuum (= "double-wound Hopfion"), which
    # destabilised Phase 1 via the κ_B (B − 1)² penalty.  The
    # 1/(16π²) factor restores topological consistency.
    H = (1.0 / (16.0 * np.pi**2)) * torch.sum(integrand) * dx**3
    return H


def measure_charges(psis, dxs, masks, psi_vac):
    """Integrate Cartan charges Q_k = ∫ (ψ_k − ψ_vac,k) over the box."""
    print("\n--- Kvantové měření uzlu ---")
    charges = []
    for k in range(8):
        Q_k = 0.0
        for level in range(len(psis)):
            volume_element = (dxs[level]**3) * masks[level]
            Q_k += torch.sum((psis[level][k] - psi_vac[k]) * volume_element).item()
        charges.append(Q_k)
        print(f"Q_{k} (Náboj H_{k}): {Q_k:.6e}")
    print("----------------------------\n")
    return charges
