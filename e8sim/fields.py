"""Field initialization (EIX-native hedgehog ansatz), derivatives, boundary
conditions, and SU(2)-projection field extraction.

Per Q2(b) of `docs/theory-wip-p6-do1.md` (4/2026): the hedgehog σ-direction
identifies with V_A in the SU(2)-Cartan, so the 4D Skyrme target
(σ, τ_1, τ_2, τ_3) structurally degenerates to 3D (= the F4' SU(2)
sub-algebra of the EIX stabiliser).  ψ(x) = Ad_{exp(f(r) n̂(x̂)^a τ_a)}(V_A)
lies pointwise inside this 3D `su(2)_stab` subspace; the topological charge
B is the Hopf invariant of the projected unit-vector field
ψ̂(x) ∈ S² ⊂ ℝ³ (= `extract_su2_hopf`), computed in `loss.compute_topological_charge`.
"""

import torch
import torch.nn.functional as F
import numpy as np


def wolf_space_seed(N, R_max, su2_basis, device='cuda', R_phys=3.0):
    """B = 1 hedgehog Skyrmion embedded in the EIX SU(2) factor (Q2(b) ansatz).

    Implements ψ(x) = Ad_{exp(f(r)·n̂(x̂)^a τ_a)}(V_A) in 248-space directly
    via Rodriguez rotation.  V_A ≡ ``su2_basis[0]`` is the SU(2)-Cartan
    direction of the F4' sub-algebra (= Sp(1) holonomy faktor of EIX, per
    Q1 4/2026), and τ_a ≡ ``su2_basis[a]`` for a = 0, 1, 2 are the three
    su(2)_stab generators with [τ_a, τ_b] = √2 ε_abc τ_c (E_8 Chevalley
    convention).

    Parameters
    ----------
    N : int
        Lattice side length.
    R_max : float
        Half-width of the simulation box (R_max ≥ R_phys).
    su2_basis : ndarray (3, 248)
        Output ``su2_basis`` of :func:`e8sim.roots.e7_su2_embedding`
        for the chosen α_su2 root.
    device : str
        Torch device.
    R_phys : float
        Physical radius of the hedgehog profile (f(0) = π, f(R_phys) = 0).

    Returns
    -------
    psi : (248, N, N, N) torch.float32
        EIX-native hedgehog seed; lies entirely in span(su2_basis[0..2]),
        pointwise unit Casimir κ(ψ, ψ) = 2 (= κ(V_A, V_A)).
    """
    s0 = torch.as_tensor(su2_basis[0], dtype=torch.float32, device=device)
    s1 = torch.as_tensor(su2_basis[1], dtype=torch.float32, device=device)
    s2 = torch.as_tensor(su2_basis[2], dtype=torch.float32, device=device)

    coords = torch.linspace(-R_max, R_max, N, device=device)
    Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
    r = torch.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

    # SO(3) rotation angle c(r) of the Ad-projected hedgehog ψ̂ = R(n̂, c)·V̂_A.
    # For SU(2) winding 1 (= unit baryon number B = 1; Manton-Sutcliffe
    # 2004 §9.10), the SO(3) angle covers one full 2π revolution as r
    # crosses the soliton core:
    #   c(0) = 2π,  c(R_phys) = 0.
    # Reason: standard Skyrme F(0) = π with U = exp(i F n̂·σ) ⇒
    # SO(3) rotation 2F(0) = 2π at the core (= identity in SO(3), but
    # SU(2) element −I = -1 in covering group, hence winding 1).
    c = 2.0 * np.pi * (torch.clamp(1.0 - r / R_phys, min=0.0) ** 2)
    return _hedgehog_psi_from_so3_angle(c, X, Y, Z, r, s0, s1, s2)


def _hedgehog_psi_from_so3_angle(c, X, Y, Z, r, s0, s1, s2):
    """Apply Rodriguez R(n̂(x̂), c(r)) · V_A to build the 248-dim hedgehog ψ.

    Helper shared by :func:`wolf_space_seed` (analytic quadratic c(r) ansatz)
    and :func:`wolf_space_seed_from_radial_profile` (1D-radial-Skyrmion
    c(r) = 2 f(r) ansatz).  ``s0, s1, s2`` are the three SU(2)-stab
    basis vectors in 248-space; ``c`` is the SO(3) rotation angle on
    the lattice (shape (N, N, N)); (X, Y, Z, r) are the coordinate grids
    used to build the radial unit vector n̂(x̂) ∈ S².

    Notes on the 2 × Skyrme convention: the standard SU(2) Skyrmion
    hedgehog is ``U(x) = exp(i f(r) n̂·σ)`` with f(0) = π, f(∞) = 0
    (Adkins-Nappi-Witten 1983 §5).  The Adjoint image
    ``Ad(U)·τ_3 = R(n̂, 2 f(r))·τ_3`` rotates the V_A direction by
    SO(3) angle ``c(r) = 2 f(r)`` — hence the factor 2 used in
    :func:`wolf_space_seed_from_radial_profile`.
    """
    cos_c = torch.cos(c)
    sin_c = torch.sin(c)
    one_m_cos = 1.0 - cos_c

    # Radial unit vector n̂(x̂) on S².
    theta = torch.acos(torch.clamp(Z / r, -1.0, 1.0))
    phi = torch.atan2(Y, X)
    n_x = torch.sin(theta) * torch.cos(phi)
    n_y = torch.sin(theta) * torch.sin(phi)
    n_z = torch.cos(theta)

    # Rodriguez R(n̂, c) · ẑ where ẑ ↔ s_0 (= V_A direction).
    # Mapping spatial (x, y, z) ↔ algebra (s_1, s_2, s_0):
    coef_s0 = cos_c + one_m_cos * n_z * n_z
    coef_s1 = sin_c * n_y + one_m_cos * n_z * n_x
    coef_s2 = -sin_c * n_x + one_m_cos * n_z * n_y

    psi = (
        s0.view(248, 1, 1, 1) * coef_s0.unsqueeze(0)
        + s1.view(248, 1, 1, 1) * coef_s1.unsqueeze(0)
        + s2.view(248, 1, 1, 1) * coef_s2.unsqueeze(0)
    )
    return psi


def wolf_space_seed_from_radial_profile(
    N,
    R_max,
    su2_basis,
    r_radial,
    f_radial,
    device='cuda',
):
    """B = 1 hedgehog Skyrmion built from a tabulated 1D radial profile.

    Drop-in replacement for :func:`wolf_space_seed` that consumes the
    analytic 1D Skyrmion profile ``f(r) ∈ [0, π]`` produced by
    ``debug_plan/scripts/e10_b1_skyrmion_profile.py`` (= L-BFGS minimum
    of the radial Skyrme energy on the EIX SU(2)-stab fibre, Q2(b)
    rezoluce).  Whereas :func:`wolf_space_seed` uses the *quadratic*
    ansatz ``c(r) = 2π·max(1 − r/R_phys, 0)²`` — a coarse first guess
    that is then relaxed by Phase-1 lattice optimisation in ``main.py``
    — this function uses the *true* SU(2) Skyrme minimum, so the
    resulting ψ is already at the action minimum and Phase-1 can be
    skipped (or run only as a polishing step).

    The 1D profile carries B = 1 *exactly* by construction (winding
    f(0) = π → f(∞) = 0; analytical identity, see
    `debug_plan/scripts/e10_b1_skyrmion_profile.py:_topological_charge_radial_diff`),
    versus B ≈ 0.52 obtained by Phase-1 lattice relaxation of the
    quadratic seed on N = 16 (lattice + FFT discretisation artefact;
    see `debug_plan/scripts/eo2_b_init_convergence.py` for the
    continuum-limit B(N) → 1 verification).  Embedding this analytic
    profile back into the 3D lattice still incurs a finite-N FFT
    discretisation error in the Hopf-invariant *measurement*, but the
    underlying ψ field is much closer to the true B = 1 minimum than
    the lattice-relaxed background.

    Parameters
    ----------
    N : int
        Lattice side length.
    R_max : float
        Half-width of the 3D simulation box.  The 1D profile must
        cover the box-corner distance ``R_max·√3 ≈ 1.73 R_max`` for
        the Dirichlet vacuum boundary to be respected without
        extrapolation.
    su2_basis : ndarray (3, 248)
        Same SU(2)-stab basis as :func:`wolf_space_seed`.
    r_radial : array-like (N_r,)
        Strictly increasing radial mesh (= ``ProfileResult.r`` from
        e10_b1).  Need not include r = 0; the value at r = 0 is
        obtained by linear extrapolation from r_radial[0] (or
        equivalently by the f(0) = π Dirichlet BC).
    f_radial : array-like (N_r,)
        Profile values f(r_i) ∈ [0, π] with f(r_radial[0]) ≈ π,
        f(r_radial[-1]) ≈ 0 (= ``ProfileResult.f`` from e10_b1).
    device : str
        Torch device.

    Returns
    -------
    psi : (248, N, N, N) torch.float32
        Same layout as :func:`wolf_space_seed`; pointwise unit Casimir.
    """
    s0 = torch.as_tensor(su2_basis[0], dtype=torch.float32, device=device)
    s1 = torch.as_tensor(su2_basis[1], dtype=torch.float32, device=device)
    s2 = torch.as_tensor(su2_basis[2], dtype=torch.float32, device=device)

    r_np = np.asarray(r_radial, dtype=np.float64)
    f_np = np.asarray(f_radial, dtype=np.float64)
    if r_np.shape != f_np.shape or r_np.ndim != 1 or r_np.size < 2:
        raise ValueError(
            f"r_radial / f_radial must be 1D arrays of equal length ≥ 2; "
            f"got shapes {r_np.shape} and {f_np.shape}"
        )
    if not np.all(np.diff(r_np) > 0):
        raise ValueError("r_radial must be strictly increasing")

    R_corner = float(np.sqrt(3.0) * R_max)
    if r_np[-1] < R_corner - 1e-6:
        raise ValueError(
            f"1D profile mesh ends at r={r_np[-1]:.4f}, but the 3D box "
            f"corner sits at R_max·√3 ≈ {R_corner:.4f}; rerun "
            f"e10_b1_skyrmion_profile with --R_box ≥ {R_corner:.4f} so "
            f"that interpolation does not extrapolate (= would inject "
            f"non-vacuum ψ at the Dirichlet boundary)"
        )

    coords = torch.linspace(-R_max, R_max, N, device=device)
    Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
    r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2 + 1e-8)

    # CPU NumPy interpolation → torch tensor on device.  ``np.interp``
    # uses the boundary value outside [r_radial[0], r_radial[-1]], so
    # f → π at r ≤ r_radial[0] and f → 0 at r ≥ r_radial[-1] are
    # honoured automatically (matches the e10_b1 Dirichlet BCs).
    r_flat = r.detach().cpu().numpy().reshape(-1)
    f_interp = np.interp(r_flat, r_np, f_np)
    f_3d = torch.as_tensor(
        f_interp.reshape(r.shape), dtype=torch.float32, device=device,
    )

    c = 2.0 * f_3d
    return _hedgehog_psi_from_so3_angle(c, X, Y, Z, r, s0, s1, s2)


def load_radial_profile(json_path, beta=0.0, beta_tol=1e-6):
    """Load (r, f) arrays from an e10_b1_skyrmion_profile.py JSON cache.

    Parameters
    ----------
    json_path : str | Path
        Path to a JSON file produced by
        ``debug_plan/scripts/e10_b1_skyrmion_profile.py --save <path>``.
        Contains a ``profiles`` list, one entry per β value scanned.
    beta : float
        Which β value to extract (default 0.0 = pure Skyrme, ANW
        reference).  Selected as the profile minimising
        ``|p['beta'] − beta|``; raises ValueError if the gap exceeds
        ``beta_tol``.
    beta_tol : float
        Tolerance for the β match.

    Returns
    -------
    r_radial : ndarray (N_r,)
    f_radial : ndarray (N_r,)
    metadata : dict
        Useful header fields (``M``, ``F_dimensionless``, ``derrick_residual``,
        ``B_topological``, ``R_box``, ``N_r``) for diagnostics + caching.
    """
    import json
    from pathlib import Path

    payload = json.loads(Path(json_path).read_text())
    profiles = payload.get('profiles')
    if not profiles:
        raise ValueError(
            f"JSON {json_path} does not contain a non-empty 'profiles' list"
        )
    best = min(profiles, key=lambda p: abs(float(p['beta']) - float(beta)))
    if abs(float(best['beta']) - float(beta)) > beta_tol:
        available = sorted(float(p['beta']) for p in profiles)
        raise ValueError(
            f"No profile within tol {beta_tol:.2e} of β = {beta}; "
            f"available β values: {available}"
        )
    r_radial = np.asarray(best['r'], dtype=np.float64)
    f_radial = np.asarray(best['f'], dtype=np.float64)
    metadata = {
        k: best.get(k)
        for k in (
            'beta', 'kappa2', 'kappa4', 'mu4', 'R_box', 'N_r',
            'M', 'F_dimensionless', 'derrick_residual',
            'B_topological', 'R_Skyrme', 'converged', 'n_steps',
        )
    }
    return r_radial, f_radial, metadata


def extract_su2_hopf(psi, su2_basis):
    """Project a 248-dim field ψ onto the 3D su(2)_stab subspace.

    Returns the 3-vector field ψ_{su2}(x) = ⟨su2_basis_a, ψ(x)⟩_E whose
    unit-normalised ψ̂(x) ∈ S² ⊂ ℝ³ carries the Hopf invariant (= topological
    B-charge per Q2(b)).

    Parameters
    ----------
    psi : (248, *spatial) torch.Tensor
    su2_basis : (3, 248) ndarray or torch.Tensor

    Returns
    -------
    psi_su2 : (3, *spatial) torch.Tensor
    """
    su2_t = torch.as_tensor(su2_basis, dtype=psi.dtype, device=psi.device)
    return torch.einsum('ij,j...->i...', su2_t, psi)


def compute_derivatives(psi, dx):
    """Central finite-difference spatial derivatives along x, y, z."""
    dpsi_dx = torch.zeros_like(psi)
    dpsi_dx[..., 1:-1] = (psi[..., 2:] - psi[..., :-2]) / (2 * dx)
    dpsi_dx[..., 0] = (psi[..., 1] - psi[..., 0]) / dx
    dpsi_dx[..., -1] = (psi[..., -1] - psi[..., -2]) / dx

    dpsi_dy = torch.zeros_like(psi)
    dpsi_dy[..., 1:-1, :] = (psi[..., 2:, :] - psi[..., :-2, :]) / (2 * dx)
    dpsi_dy[..., 0, :] = (psi[..., 1, :] - psi[..., 0, :]) / dx
    dpsi_dy[..., -1, :] = (psi[..., -1, :] - psi[..., -2, :]) / dx

    dpsi_dz = torch.zeros_like(psi)
    dpsi_dz[..., 1:-1, :, :] = (psi[..., 2:, :, :] - psi[..., :-2, :, :]) / (2 * dx)
    dpsi_dz[..., 0, :, :] = (psi[..., 1, :, :] - psi[..., 0, :, :]) / dx
    dpsi_dz[..., -1, :, :] = (psi[..., -1, :, :] - psi[..., -2, :, :]) / dx

    return dpsi_dx, dpsi_dy, dpsi_dz


def enforce_dirichlet(psi, psi_vac):
    """Pin the six boundary faces to the vacuum configuration."""
    with torch.no_grad():
        psi[:, 0, :, :] = psi_vac[:, 0, :, :]
        psi[:, -1, :, :] = psi_vac[:, -1, :, :]
        psi[:, :, 0, :] = psi_vac[:, :, 0, :]
        psi[:, :, -1, :] = psi_vac[:, :, -1, :]
        psi[:, :, :, 0] = psi_vac[:, :, :, 0]
        psi[:, :, :, -1] = psi_vac[:, :, :, -1]


def enforce_nested_boundaries(psis, psi_vac, R_maxs):
    """Enforce boundary conditions across nested refinement levels."""
    with torch.no_grad():
        enforce_dirichlet(psis[0], psi_vac)

        for level in range(1, len(psis)):
            scale = R_maxs[level] / R_maxs[level - 1]
            N = psis[level].shape[1]
            device = psis[level].device

            ticks = torch.linspace(-scale, scale, N, device=device)
            Z, Y, X = torch.meshgrid(ticks, ticks, ticks, indexing='ij')
            grid = torch.stack((X, Y, Z), dim=-1).unsqueeze(0)

            src = psis[level - 1].unsqueeze(0)
            interpolated = F.grid_sample(
                src, grid, mode='bilinear', padding_mode='border', align_corners=True,
            ).squeeze(0)

            psis[level][:, 0, :, :] = interpolated[:, 0, :, :]
            psis[level][:, -1, :, :] = interpolated[:, -1, :, :]
            psis[level][:, :, 0, :] = interpolated[:, :, 0, :]
            psis[level][:, :, -1, :] = interpolated[:, :, -1, :]
            psis[level][:, :, :, 0] = interpolated[:, :, :, 0]
            psis[level][:, :, :, -1] = interpolated[:, :, :, -1]
