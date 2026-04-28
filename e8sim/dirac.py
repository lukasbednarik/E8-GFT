"""Dirac scanner: phase-2 zero-mode search on a frozen E8 background.

The Hamiltonian couples an 8-component spinor (4 Dirac x 2 isospin) to a
frozen E8 hedgehog background via the chiral Yukawa interaction:

    H = -i (alpha_i (x) I_2) d_i
      + m_0 (beta (x) I_2) sigma(x)
      + i m_0 (beta gamma_5 (x) tau_a) phi_a(x)        a = 1, 2
      + r_w  (beta (x) I_2) nabla^2

The scalar field sigma(x) and pion field phi_a(x) are extracted from the
EIX-native hedgehog psi(x) (Q2(b) of `docs/theory-wip-p6-do1.md`) via
:func:`e8sim.fields.extract_su2_hopf`:

    psi_su2(x) = (su2_basis @ psi)[a, x]            a = 0, 1, 2
    sigma(x)   = psi_su2[0, x]                      (= projection onto V_A)
    phi_a(x)   = psi_su2[a, x]                      a = 1, 2

Only two Pauli isospin couplings (tau_1, tau_2) appear because per Q2(b)
the σ-direction identifies with τ_3 (= V_A in SU(2)-Cartan), so there is
no separate τ_3 pion field.  The standard 3-component Yukawa structure
collapses to a 2-component transverse coupling.
"""

import torch
import numpy as np

from .fields import extract_su2_hopf


def get_dirac_matrices(device):
    """4x4 Dirac matrices in the Dirac representation."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    I = torch.eye(2, dtype=torch.complex64, device=device)
    O = torch.zeros((2, 2), dtype=torch.complex64, device=device)

    alpha_x = torch.cat([torch.cat([O, sigma_x], dim=1),
                          torch.cat([sigma_x, O], dim=1)], dim=0)
    alpha_y = torch.cat([torch.cat([O, sigma_y], dim=1),
                          torch.cat([sigma_y, O], dim=1)], dim=0)
    alpha_z = torch.cat([torch.cat([O, sigma_z], dim=1),
                          torch.cat([sigma_z, O], dim=1)], dim=0)

    beta = torch.cat([torch.cat([I, O], dim=1),
                       torch.cat([O, -I], dim=1)], dim=0)
    gamma_5 = torch.cat([torch.cat([O, I], dim=1),
                           torch.cat([I, O], dim=1)], dim=0)

    return alpha_x, alpha_y, alpha_z, beta, gamma_5


def get_dirac_isospin_matrices(device):
    """8x8 Dirac-isospin matrices for the hedgehog Hamiltonian.

    The spinor lives in C^4_Dirac (x) C^2_isospin = C^8.  All matrices
    are built as Kronecker products of 4x4 Dirac and 2x2 isospin blocks.

    Returns a dict with keys:
        alpha_x, alpha_y, alpha_z  -- kinetic:  alpha_i (x) I_2
        beta                       -- mass/Wilson: beta (x) I_2
        gamma_5                    -- chirality: gamma_5 (x) I_2
        bg5_tau_x, bg5_tau_y, bg5_tau_z -- isospin coupling:
            i * (beta gamma_5) (x) tau_i   (Hermitian by construction)
    """
    alpha_x, alpha_y, alpha_z, beta4, gamma5_4 = get_dirac_matrices(device)
    I2 = torch.eye(2, dtype=torch.complex64, device=device)

    tau_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    tau_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    bg5 = beta4 @ gamma5_4

    return {
        'alpha_x': torch.kron(alpha_x, I2),
        'alpha_y': torch.kron(alpha_y, I2),
        'alpha_z': torch.kron(alpha_z, I2),
        'beta': torch.kron(beta4, I2),
        'gamma_5': torch.kron(gamma5_4, I2),
        'bg5_tau_x': 1j * torch.kron(bg5, tau_x),
        'bg5_tau_y': 1j * torch.kron(bg5, tau_y),
        'bg5_tau_z': 1j * torch.kron(bg5, tau_z),
    }


def fft_derivative(chi, dx):
    """Spectral (FFT) spatial derivatives of a spinor field.

    Works for any number of components (first dimension is batch).
    """
    N = chi.shape[1]
    freqs = torch.fft.fftfreq(N, d=dx, device=chi.device)

    kx = freqs.view(1, N, 1, 1)
    ky = freqs.view(1, 1, N, 1)
    kz = freqs.view(1, 1, 1, N)

    chi_f = torch.fft.fftn(chi, dim=(1, 2, 3))

    kx_ang = 2 * np.pi * kx
    ky_ang = 2 * np.pi * ky
    kz_ang = 2 * np.pi * kz

    dchi_dx = torch.fft.ifftn(1j * kx_ang * chi_f, dim=(1, 2, 3))
    dchi_dy = torch.fft.ifftn(1j * ky_ang * chi_f, dim=(1, 2, 3))
    dchi_dz = torch.fft.ifftn(1j * kz_ang * chi_f, dim=(1, 2, 3))

    return dchi_dx, dchi_dy, dchi_dz


def fft_laplacian(chi, dx):
    """Spectral Laplacian nabla^2 chi via FFT."""
    N = chi.shape[1]
    freqs = torch.fft.fftfreq(N, d=dx, device=chi.device)

    kx = freqs.view(1, N, 1, 1)
    ky = freqs.view(1, 1, N, 1)
    kz = freqs.view(1, 1, 1, N)

    chi_f = torch.fft.fftn(chi, dim=(1, 2, 3))
    k2 = (2 * np.pi * kx)**2 + (2 * np.pi * ky)**2 + (2 * np.pi * kz)**2

    return torch.fft.ifftn(-k2 * chi_f, dim=(1, 2, 3))


def apply_dirac_hamiltonian(chi, matrices, M_scalar, phi_vector,
                             dx, r_wilson=0.0, g_coupling=10.0):
    """Apply the 8-component Dirac-isospin Hamiltonian H to spinor chi.

    H chi = -i (alpha_i (x) I_2) d_i chi
          + (beta (x) I_2) M_scalar chi
          + g sum_a [i (beta gamma_5 (x) tau_a)] phi_a chi    a = 1, 2
          + r_w (beta (x) I_2) nabla^2 chi

    Args:
        chi: (8, N, N, N) complex spinor (4 Dirac × 2 isospin).
        matrices: dict from get_dirac_isospin_matrices (8x8).
        M_scalar: (N, N, N) real scalar mass field, or None.
        phi_vector: (phi_1, phi_2) pion field components (transverse to V_A;
            two components only because σ-direction identifies with τ_3 per
            Q2(b) of `docs/theory-wip-p6-do1.md`), or None.
        dx: lattice spacing.
        r_wilson: Wilson parameter (positive removes doublers).
        g_coupling: Yukawa coupling strength.
    """
    dchi_dx, dchi_dy, dchi_dz = fft_derivative(chi, dx)

    alpha_x = matrices['alpha_x']
    alpha_y = matrices['alpha_y']
    alpha_z = matrices['alpha_z']
    beta = matrices['beta']

    kinetic = (
        -1j * torch.einsum('ij,jklm->iklm', alpha_x, dchi_dx)
        + -1j * torch.einsum('ij,jklm->iklm', alpha_y, dchi_dy)
        + -1j * torch.einsum('ij,jklm->iklm', alpha_z, dchi_dz)
    )

    if r_wilson != 0.0:
        lap_chi = fft_laplacian(chi, dx)
        kinetic = kinetic + r_wilson * torch.einsum('ij,jklm->iklm', beta, lap_chi)

    mass_term = torch.zeros_like(chi)
    if M_scalar is not None:
        mass_term = mass_term + (
            torch.einsum('ij,jklm->iklm', beta, chi) * M_scalar.unsqueeze(0)
        )

    if phi_vector is not None:
        phi_1, phi_2 = phi_vector
        iso_coupling = (
            torch.einsum('ij,jklm->iklm', matrices['bg5_tau_x'], chi)
            * phi_1.unsqueeze(0)
            + torch.einsum('ij,jklm->iklm', matrices['bg5_tau_y'], chi)
            * phi_2.unsqueeze(0)
        )
        mass_term = mass_term + g_coupling * iso_coupling

    return kinetic + mass_term


def run_dirac_scanner(psi_frozen, su2_basis, dx, device, steps=500, lr=0.02,
                       init_mode='gaussian', sigma_init=None,
                       gamma_chir_init=0.0, gamma_chir_final=0.0):
    """Run the Dirac scanner to find zero modes on a frozen E8 background.

    Uses an 8-component spinor (4 Dirac x 2 isospin) with chiral Yukawa
    coupling to the hedgehog (σ, φ_1, φ_2) fields extracted from the
    EIX-native ψ(x) via :func:`extract_su2_hopf` (Q2(b) of
    `docs/theory-wip-p6-do1.md`).  Only τ_1, τ_2 Pauli isospin couplings
    are active; τ_3 coupling is absent because σ-direction identifies with
    τ_3 per Q2(b).

    Args:
        psi_frozen: (248, N, N, N) frozen Phase-1 background.
        su2_basis: (3, 248) ndarray for the chosen sector — used to project
            ψ onto the 3-dim su(2)_stab subspace.
        init_mode: 'random' (uniform noise) or 'gaussian' (Gaussian envelope
            centred at the defect, biases the optimizer toward the topological
            zero mode).
        sigma_init: width of the Gaussian envelope (defaults to box_radius/3).
        gamma_chir_init / gamma_chir_final: cosine-annealed chirality-projector
            penalty −γ_chir · ⟨χ|Γ₅|χ⟩².  The negative sign + square favours
            *any* definite chirality without preselecting the sign — the
            topological winding of the background is left to determine which
            sign is energetically preferred.  Anneal to 0 so the final state
            is still a true E² minimum.
    """
    print("\n--- Phase 2: Dirac Scanner (zero-mode search, EIX-native) ---")
    N = psi_frozen.shape[1]
    R_max = (N - 1) * dx / 2.0

    psi_su2 = extract_su2_hopf(psi_frozen, su2_basis)
    sigma_field = psi_su2[0]
    phi_vector = (psi_su2[1], psi_su2[2])

    m0 = 10.0
    M_scalar = m0 * sigma_field
    g_coupling = m0
    r_wilson = 0.5 * dx

    matrices = get_dirac_isospin_matrices(device)

    coords = torch.linspace(-R_max, R_max, N, device=device)
    Zg, Yg, Xg = torch.meshgrid(coords, coords, coords, indexing='ij')
    r2_field = (Xg**2 + Yg**2 + Zg**2)

    if init_mode == 'gaussian':
        if sigma_init is None:
            sigma_init = R_max / 3.0
        envelope = torch.exp(-r2_field / (2.0 * sigma_init**2)).to(torch.complex64)
        chi = torch.randn((8, N, N, N), dtype=torch.complex64, device=device)
        chi = chi * envelope.unsqueeze(0)
        print(f"  Gaussian init: sigma={sigma_init:.3f}")
    else:
        chi = torch.randn((8, N, N, N), dtype=torch.complex64, device=device)

    chi = chi / torch.sqrt(torch.sum(torch.abs(chi)**2) * (dx**3))
    chi.requires_grad_(True)

    if gamma_chir_init != 0.0:
        print(f"  Chirality-projector penalty: γ_chir = {gamma_chir_init:.2f} → "
              f"{gamma_chir_final:.2f} (cosine annealed)")

    optimizer = torch.optim.Adam([chi], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    gamma_5 = matrices['gamma_5']

    for step in range(steps):
        optimizer.zero_grad()

        chi_normed = chi / torch.sqrt(torch.sum(torch.abs(chi)**2) * (dx**3))
        H_chi = apply_dirac_hamiltonian(
            chi_normed, matrices, M_scalar, phi_vector, dx,
            r_wilson, g_coupling,
        )

        E2 = torch.sum(torch.abs(H_chi)**2) * (dx**3)

        loss = E2
        frac = step / max(steps - 1, 1)
        cos_frac = 0.5 * (1.0 + np.cos(np.pi * frac))

        if gamma_chir_init != 0.0:
            gam_c = gamma_chir_final + (gamma_chir_init - gamma_chir_final) * cos_frac
            chi_g5_chi = torch.einsum('iklm,ij,jklm->klm',
                                       chi_normed.conj(), gamma_5, chi_normed)
            C_loss = torch.sum(chi_g5_chi.real) * (dx**3)
            loss = loss - gam_c * C_loss * C_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                chi_normed = chi / torch.sqrt(torch.sum(torch.abs(chi)**2) * (dx**3))
                H_chi_eval = apply_dirac_hamiltonian(
                    chi_normed, matrices, M_scalar, phi_vector, dx,
                    r_wilson, g_coupling,
                )
                E2 = (torch.sum(torch.abs(H_chi_eval)**2) * (dx**3)).item()
                E = np.sqrt(E2)

                chi_gamma5_chi = torch.einsum('iklm,ij,jklm->klm',
                                               chi_normed.conj(), gamma_5,
                                               chi_normed)
                C = torch.sum(chi_gamma5_chi.real * (dx**3)).item()

                print(f"  Scanner step {step:>4}/{steps} | E^2 = {E2:.6e} "
                      f"| E = {E:.6e} | Chirality C = {C:.4f}")

    with torch.no_grad():
        chi_normed = chi / torch.sqrt(torch.sum(torch.abs(chi)**2) * (dx**3))
        H_chi_final = apply_dirac_hamiltonian(
            chi_normed, matrices, M_scalar, phi_vector, dx,
            r_wilson, g_coupling,
        )
        E2 = (torch.sum(torch.abs(H_chi_final)**2) * (dx**3)).item()
        E = np.sqrt(max(E2, 0))

        density = torch.sum(torch.abs(chi_normed)**2, dim=0)
        center_density = density[N // 2, N // 2, N // 2].item()
        max_density = torch.max(density).item()
        mean_density = torch.mean(density).item()

        chi_gamma5_chi = torch.einsum('iklm,ij,jklm->klm',
                                       chi_normed.conj(), gamma_5, chi_normed)
        C = torch.sum(chi_gamma5_chi.real * (dx**3)).item()

        com_x = (torch.sum(Xg * density) * (dx**3)).item()
        com_y = (torch.sum(Yg * density) * (dx**3)).item()
        com_z = (torch.sum(Zg * density) * (dx**3)).item()
        com_r = float(np.sqrt(com_x**2 + com_y**2 + com_z**2))

        # Spatial concentration: max/mean.  > 5 indicates a non-uniform mode.
        peakedness = max_density / max(mean_density, 1e-30)

        # Containment inside the box's physical radius (use R_max as proxy
        # for the soliton extent; the hedgehog has R_phys <= R_max).
        r2_grid = (Xg**2 + Yg**2 + Zg**2)
        containment_radius = R_max  # density should mostly live inside the box centre
        inside_mask = (r2_grid < containment_radius**2).float()
        frac_inside = (torch.sum(inside_mask * density) * (dx**3)).item()

        # Localization: mode is concentrated AND its centre-of-mass is
        # near the geometric centre of the box (where the defect lives).
        # This replaces the brittle single-point check `centre_density > 0.1*max`
        # which is sensitive to lattice discretization (Step 10C fix).
        is_concentrated = peakedness > 5.0
        is_centred = com_r < 0.5 * R_max
        is_localized = is_concentrated and is_centred

        print("\nScanner results:")
        print(f"  E^2 = {E2:.6e},  E = {E:.6e}")
        print(f"  Chirality C = {C:.4f}")
        print(f"  Max density:    {max_density:.6f}")
        print(f"  Mean density:   {mean_density:.6f}")
        print(f"  Centre density: {center_density:.6f}")
        print(f"  Peakedness (max/mean): {peakedness:.1f}  "
              f"(> 5 = concentrated)")
        print(f"  COM radius: {com_r:.3f}  (< {0.5*R_max:.2f} = centred)")
        print(f"  Fraction within |r|<{containment_radius:.1f}: {frac_inside:.3f}")
        if is_localized:
            print("  -> Fermion is LOCALIZED inside the defect.")
        else:
            why = []
            if not is_concentrated:
                why.append(f"not concentrated (peakedness {peakedness:.1f} ≤ 5)")
            if not is_centred:
                why.append(f"not centred (COM r={com_r:.2f} ≥ {0.5*R_max:.2f})")
            print(f"  -> Fermion NOT localized: {', '.join(why)}.")

    return {
        'E2': E2, 'E': E, 'chirality': C,
        'center_density': center_density,
        'max_density': max_density,
        'mean_density': mean_density,
        'peakedness': peakedness,
        'com_radius': com_r,
        'fraction_inside': frac_inside,
        'is_localized': is_localized,
    }


def _spinor_inner_product(phi, chi, dx):
    """Hermitian inner product ⟨phi|chi⟩ = Σ_a ∫ φ_a^* χ_a d³x.

    For two 8-component spinor fields φ, χ on an N³ lattice with spacing dx.
    Returns a complex scalar tensor (complex64).
    """
    return torch.sum(phi.conj() * chi) * (dx ** 3)


def _spinor_norm_sq(chi, dx):
    """Real-valued ⟨χ|χ⟩ = Σ_a ∫ |χ_a|² d³x for an 8-component spinor."""
    return (torch.sum(torch.abs(chi) ** 2) * (dx ** 3)).real


def _orthogonalize_in_place(chi, prev_modes, dx):
    """Project χ onto the orthogonal complement of {prev_modes}, in-place.

    Modified Gram-Schmidt:  χ ← χ − Σ_j ⟨χ_j | χ⟩ χ_j, then normalize.
    Operates on chi.data without an autograd graph (no .grad on result).

    Args:
        chi:        (8, N, N, N) complex64 leaf tensor (requires_grad=True OK).
        prev_modes: list of (8, N, N, N) complex64 tensors, each *normalized*.
        dx:         lattice spacing (float).
    """
    with torch.no_grad():
        chi_new = chi.data.clone()
        for chi_j in prev_modes:
            proj = _spinor_inner_product(chi_j, chi_new, dx)
            chi_new = chi_new - proj * chi_j
        norm_sq = _spinor_norm_sq(chi_new, dx)
        chi_new = chi_new / torch.sqrt(torch.clamp(norm_sq, min=1e-30))
        chi.data.copy_(chi_new)


def _initialize_spinor(N, dx, R_max, device, sigma_init=None,
                       seed=None, prev_modes=None):
    """Random Gaussian-enveloped 8-component spinor for Adam initialisation.

    For excited modes (prev_modes non-empty), the initialised spinor is
    immediately orthogonalised against prev_modes so the optimizer starts
    inside the orthogonal complement, not on the same eigenvector again.
    """
    if seed is not None:
        torch.manual_seed(seed)

    coords = torch.linspace(-R_max, R_max, N, device=device)
    Zg, Yg, Xg = torch.meshgrid(coords, coords, coords, indexing='ij')
    r2 = Xg ** 2 + Yg ** 2 + Zg ** 2

    if sigma_init is None:
        sigma_init = R_max / 3.0

    envelope = torch.exp(-r2 / (2.0 * sigma_init ** 2)).to(torch.complex64)
    chi = torch.randn((8, N, N, N), dtype=torch.complex64, device=device)
    chi = chi * envelope.unsqueeze(0)

    if prev_modes:
        with torch.no_grad():
            for chi_j in prev_modes:
                proj = _spinor_inner_product(chi_j, chi, dx)
                chi = chi - proj * chi_j

    norm_sq = _spinor_norm_sq(chi, dx)
    chi = chi / torch.sqrt(torch.clamp(norm_sq, min=1e-30))
    return chi


def _measure_mode(chi_normed, matrices, M_scalar, phi_vector,
                  dx, r_wilson, g_coupling, R_max):
    """Compute (E², spatial diagnostics, chirality) for a normalised spinor.

    Returns a dict with keys: E2, E, chirality, center_density,
    max_density, mean_density, peakedness, com_radius, fraction_inside,
    is_localized.  Used both inside multi-state scanner and for offline
    re-evaluation of stored spinor fields.
    """
    H_chi = apply_dirac_hamiltonian(
        chi_normed, matrices, M_scalar, phi_vector, dx,
        r_wilson, g_coupling,
    )
    E2 = (torch.sum(torch.abs(H_chi) ** 2) * (dx ** 3)).item()
    E = float(np.sqrt(max(E2, 0.0)))

    gamma_5 = matrices['gamma_5']
    chi_g5_chi = torch.einsum('iklm,ij,jklm->klm',
                               chi_normed.conj(), gamma_5, chi_normed)
    C = (torch.sum(chi_g5_chi.real) * (dx ** 3)).item()

    density = torch.sum(torch.abs(chi_normed) ** 2, dim=0)
    N = chi_normed.shape[1]
    center_density = density[N // 2, N // 2, N // 2].item()
    max_density = torch.max(density).item()
    mean_density = torch.mean(density).item()
    peakedness = max_density / max(mean_density, 1e-30)

    coords = torch.linspace(-R_max, R_max, N, device=chi_normed.device)
    Zg, Yg, Xg = torch.meshgrid(coords, coords, coords, indexing='ij')
    com_x = (torch.sum(Xg * density) * (dx ** 3)).item()
    com_y = (torch.sum(Yg * density) * (dx ** 3)).item()
    com_z = (torch.sum(Zg * density) * (dx ** 3)).item()
    com_r = float(np.sqrt(com_x ** 2 + com_y ** 2 + com_z ** 2))

    r2_grid = Xg ** 2 + Yg ** 2 + Zg ** 2
    inside_mask = (r2_grid < R_max ** 2).float()
    frac_inside = (torch.sum(inside_mask * density) * (dx ** 3)).item()

    is_localized = (peakedness > 5.0) and (com_r < 0.5 * R_max)

    return {
        'E2': E2, 'E': E, 'chirality': C,
        'center_density': center_density,
        'max_density': max_density,
        'mean_density': mean_density,
        'peakedness': peakedness,
        'com_radius': com_r,
        'fraction_inside': frac_inside,
        'is_localized': is_localized,
    }


def run_dirac_scanner_multistate(
    psi_frozen, su2_basis, dx, device, *,
    K_states=3,
    steps=1500, lr=0.02,
    init_mode='gaussian', sigma_init=None,
    gamma_chir_init=0.0, gamma_chir_final=0.0,
    deflation_penalty=0.0,
    log_every=100,
    return_spinors=False,
    base_seed=None,
):
    """Multi-state Dirac scanner: find first K excited bound modes.

    Implements §E.10 bod 2 task 1 from `docs/theory-wip-p6.md`:  power
    iteration with deflation = sequential Adam minimisation of E² with
    Modified Gram-Schmidt orthogonalisation against previously found
    modes.  This is the standard PyTorch-friendly alternative to scipy
    Lanczos when the Hamiltonian is only available as an operator
    (apply_dirac_hamiltonian) and not as a sparse matrix; for our 8 ×
    N³ DOF problem (16384 complex DOF on N=16) it converges to the K
    lowest E² modes in ~K × steps optimisation steps.

    The k-th eigenmode minimises  E²_k = ⟨χ_k | H² | χ_k⟩  subject to
    ⟨χ_k | χ_j⟩ = 0 for j < k.  We enforce orthogonality *after every
    Adam step* via Modified Gram-Schmidt + renormalisation; this keeps
    the optimisation problem differentiable while guaranteeing the
    orthogonal complement constraint at convergence.

    Args:
        psi_frozen:        (248, N, N, N) Phase-1 background.
        su2_basis:         (3, 248) ndarray, `su2_basis[0]` = V_A direction.
        dx:                lattice spacing.
        device:            torch device.
        K_states:          number of excited bound modes to extract (≥ 1).
        steps:             Adam steps per mode.
        lr:                Adam learning rate.
        init_mode:         'gaussian' (envelope) or 'random' (full noise).
        gamma_chir_init,
        gamma_chir_final:  cosine-annealed chirality penalty
                           −γ_chir⟨χ|Γ₅|χ⟩²; not used for the multi-state
                           scanner by default (γ = 0) because chirality
                           preselection conflicts with finding the lowest
                           excited states; left here for future API parity
                           with `run_dirac_scanner`.
        deflation_penalty: extra λ Σ_j |⟨χ_j|χ⟩|² penalty (default 0; the
                           Modified Gram-Schmidt projection already enforces
                           orthogonality without this, but a non-zero λ
                           may speed up Adam convergence by smoothing the
                           constraint surface).
        log_every:         print E² every N steps (0 = silent).
        return_spinors:    if True, include the converged χ_k tensors in
                           the result (large memory:  K × 8 × N³ complex64).
                           Use only when caller needs to compute overlap
                           matrices between modes.
        base_seed:         RNG seed for first mode (subsequent modes use
                           base_seed + k); reproducibility for tests.

    Returns:
        list of K dicts, one per mode, with keys:
            E2, E, chirality, center_density, max_density, mean_density,
            peakedness, com_radius, fraction_inside, is_localized
            (always present; same keys as `run_dirac_scanner`).
            And optionally:
            spinor:  (8, N, N, N) complex64 cpu tensor
                     (only if return_spinors=True).

    Notes:
        * The lowest mode (k=0) usually converges to ⟨E²⟩ ≈ 1 in lattice
          units (= near-zero JR mode in mass gap); excited modes climb
          slowly into the gap edge (~m_0² = 100).  Spectral ratios E²_k+1
          / E²_k = ~1.5–3 on N=16 lattice (artefact of finite N + Q3 γ
          sector dependency); continuum N → ∞ would lower the ratios.
        * Memory: K × 8 × N³ complex64 = K × 32 KiB for N=16, scales as
          N³.  For N=32 with K=4: 1 MiB per mode = trivial; OOM only
          becomes a concern at N ≥ 96.
        * Compute: ~K × steps Adam updates; on RTX 4060 N=16 K=3 takes
          ~3 min for steps=1500.  For N=32: factor ~8 in time.
    """
    print(f"\n--- Phase 2: Multi-state Dirac Scanner "
          f"(K={K_states} excited modes, EIX-native) ---")
    N = psi_frozen.shape[1]
    R_max = (N - 1) * dx / 2.0

    psi_su2 = extract_su2_hopf(psi_frozen, su2_basis)
    sigma_field = psi_su2[0]
    phi_vector = (psi_su2[1], psi_su2[2])

    m0 = 10.0
    M_scalar = m0 * sigma_field
    g_coupling = m0
    r_wilson = 0.5 * dx

    matrices = get_dirac_isospin_matrices(device)
    gamma_5 = matrices['gamma_5']

    if sigma_init is None:
        sigma_init = R_max / 3.0
    if base_seed is None:
        base_seed = 0

    found_modes = []
    found_results = []

    for k in range(K_states):
        seed_k = base_seed + 1000 * (k + 1)
        chi = _initialize_spinor(
            N, dx, R_max, device,
            sigma_init=sigma_init, seed=seed_k,
            prev_modes=found_modes,
        )
        chi.requires_grad_(True)

        optimizer = torch.optim.Adam([chi], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps,
        )

        if log_every > 0:
            print(f"\n  Mode {k} init: |χ|² = "
                  f"{_spinor_norm_sq(chi.detach(), dx).item():.4f},  "
                  f"deflated against {len(found_modes)} prev modes")

        for step in range(steps):
            optimizer.zero_grad()

            chi_normed = chi / torch.sqrt(
                torch.clamp(_spinor_norm_sq(chi, dx), min=1e-30)
            )
            H_chi = apply_dirac_hamiltonian(
                chi_normed, matrices, M_scalar, phi_vector, dx,
                r_wilson, g_coupling,
            )
            E2 = (torch.sum(torch.abs(H_chi) ** 2) * (dx ** 3)).real

            loss = E2

            frac = step / max(steps - 1, 1)
            cos_frac = 0.5 * (1.0 + np.cos(np.pi * frac))

            if gamma_chir_init != 0.0:
                gam_c = (gamma_chir_final
                         + (gamma_chir_init - gamma_chir_final) * cos_frac)
                chi_g5_chi = torch.einsum(
                    'iklm,ij,jklm->klm',
                    chi_normed.conj(), gamma_5, chi_normed,
                )
                C_loss = torch.sum(chi_g5_chi.real) * (dx ** 3)
                loss = loss - gam_c * C_loss * C_loss

            if deflation_penalty > 0.0 and found_modes:
                ortho_pen = chi.new_zeros(())
                for chi_j in found_modes:
                    proj = _spinor_inner_product(chi_j, chi_normed, dx)
                    ortho_pen = ortho_pen + (proj.conj() * proj).real
                loss = loss + deflation_penalty * ortho_pen

            loss.backward()
            optimizer.step()
            scheduler.step()

            if found_modes:
                _orthogonalize_in_place(chi, found_modes, dx)

            if log_every > 0 and (step % log_every == 0 or step == steps - 1):
                with torch.no_grad():
                    chi_eval = chi / torch.sqrt(
                        torch.clamp(_spinor_norm_sq(chi, dx), min=1e-30)
                    )
                    H_chi_eval = apply_dirac_hamiltonian(
                        chi_eval, matrices, M_scalar, phi_vector, dx,
                        r_wilson, g_coupling,
                    )
                    E2_v = (torch.sum(torch.abs(H_chi_eval) ** 2)
                            * (dx ** 3)).item()
                    chi_g5 = torch.einsum(
                        'iklm,ij,jklm->klm',
                        chi_eval.conj(), gamma_5, chi_eval,
                    )
                    C_v = (torch.sum(chi_g5.real) * (dx ** 3)).item()
                    print(f"    Mode {k}  step {step:>4}/{steps}  "
                          f"E² = {E2_v:.6e}  E = {np.sqrt(max(E2_v, 0)):.4e}  "
                          f"|C| = {abs(C_v):.4f}")

        with torch.no_grad():
            chi_final = chi / torch.sqrt(
                torch.clamp(_spinor_norm_sq(chi, dx), min=1e-30)
            )
            if found_modes:
                chi_tmp = chi_final.clone()
                for chi_j in found_modes:
                    proj = _spinor_inner_product(chi_j, chi_tmp, dx)
                    chi_tmp = chi_tmp - proj * chi_j
                norm_sq = _spinor_norm_sq(chi_tmp, dx)
                chi_final = chi_tmp / torch.sqrt(
                    torch.clamp(norm_sq, min=1e-30)
                )

            measurement = _measure_mode(
                chi_final, matrices, M_scalar, phi_vector,
                dx, r_wilson, g_coupling, R_max,
            )

            found_modes.append(chi_final.detach().clone())

            if return_spinors:
                measurement['spinor'] = chi_final.detach().cpu().clone()

            found_results.append(measurement)

            print(f"  Mode {k} converged:  E² = {measurement['E2']:.6e},  "
                  f"E = {measurement['E']:.4e},  C = {measurement['chirality']:.3f},  "
                  f"peak = {measurement['peakedness']:.1f},  "
                  f"COM = {measurement['com_radius']:.3f},  "
                  f"localized = {measurement['is_localized']}")

    sort_perm = sorted(range(len(found_results)),
                        key=lambda i: found_results[i]['E2'])
    sorted_results = [found_results[i] for i in sort_perm]
    needed_sort = sort_perm != list(range(len(found_results)))
    if needed_sort:
        print("\n  ── Post-hoc sort by E² (raw deflation order was non-monotonic) ──")
        for new_k, old_k in enumerate(sort_perm):
            print(f"    new k = {new_k}  ←  raw mode {old_k} "
                  f"(E² = {found_results[old_k]['E2']:.6e})")
    for k in range(len(sorted_results)):
        sorted_results[k]['raw_mode_index'] = sort_perm[k]
        sorted_results[k]['k_after_sort'] = k

    print("\n  ── Multi-state spectrum (this sector, sorted by E²) ──")
    for k, m in enumerate(sorted_results):
        print(f"    k = {k}:  E² = {m['E2']:.6e},  E = {m['E']:.4e},  "
              f"C = {m['chirality']:+.3f}")
    if len(sorted_results) >= 2:
        print("  ── Spectral ratios E²_{k+1}/E²_k ──")
        for k in range(1, len(sorted_results)):
            ratio = (sorted_results[k]['E2']
                     / max(sorted_results[k - 1]['E2'], 1e-30))
            print(f"    E²_{k} / E²_{k - 1} = {ratio:.4f}")

    return sorted_results


def overlap_matrix(spinors_a, spinors_b, dx):
    """Inter-sector spinor overlap matrix  M[i, j] = ⟨χ_i^(a) | χ_j^(b)⟩.

    Args:
        spinors_a:  list[(8, N, N, N)] from sector a.
        spinors_b:  list[(8, N, N, N)] from sector b.
        dx:         lattice spacing.

    Returns:
        complex tensor shape (len(a), len(b)).

    For physical PMNS interpretation we need spinors expressed in the same
    basis (same lattice grid + same Dirac/isospin frame).  Per the Q3(γ)
    resolution, sectors are Ad-conjugate gauge representatives — so the
    overlap M^a_b for a ≠ b carries the gauge-rotation matrix between
    triality reps; only the singular values (= overlap magnitudes) are
    gauge-invariant predictions for PMNS angles.
    """
    K_a = len(spinors_a)
    K_b = len(spinors_b)
    if K_a == 0 or K_b == 0:
        return torch.empty((K_a, K_b), dtype=torch.complex64)

    M = torch.zeros((K_a, K_b), dtype=torch.complex64,
                     device=spinors_a[0].device)
    for i in range(K_a):
        for j in range(K_b):
            M[i, j] = _spinor_inner_product(spinors_a[i], spinors_b[j], dx)
    return M
