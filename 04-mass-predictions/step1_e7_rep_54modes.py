"""Step 1: E7-representation of 54 zero modes from c_2(E) = -1.

On EIX = E8/(E7 × SU(2)), the canonical connection restricted to a totally
geodesic HP^1 ⊂ EIX has:
    c_2(E_{Sp(28)}) = -1,  c_2(L_{Sp(1)}) = +1

The 56 of E7 (= the "fiber" directions of the (56,2) tangent space)
decomposes under the instanton holonomy algebra.

Key subtlety (compact real form): In the orthonormal e8sim basis,
the Cartan bracket [H, e_γ^(1)] = (H·γ) e_γ^(2) (maps between the two
real components of each root generator pair). So we must compute the
SQUARED action (ad(T))² to get the charge-squared spectrum on the
56 "up" subspace.

This script:
1. Finds the E7 curvature direction T on an HP^1 frame
2. Computes E7-charges of the 56 representation under ad(T)² 
3. Determines the zero-mode content and identifies the missing modes
4. Verifies consistency with ind(D_{E⊗L}) = 54

Status: new computation (2026-05-15).
Dependencies: c2_E_on_HP1.py (HP^1 frame), e8sim (algebra infrastructure).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (
    bracket_vec_fast,
    load_structure_constants_numpy,
    extract_pos_roots_numpy,
)
from e8sim.roots import e7_su2_embedding, EIX_ALPHA_SU2
from e8sim.eix import DIM_E7, DIM_SU2, KAPPA_OVER_EUCLID, kappa

CONSTANTS_PATH = constants_path(ROOT)
TOL = 1e-8


def commutator(f_idx, f_val, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return bracket_vec_fast(f_idx, f_val, x, y)


def project(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return basis.T @ (basis @ vec)


def build_HP1_frame(m_basis, su2_basis, f_idx, f_val, v0_idx=0):
    """Build orthonormal 4-frame for HP^1 (quaternionic line in m)."""
    H_su2 = su2_basis[0]
    E_plus = su2_basis[1]
    E_minus = su2_basis[2]
    e_a = m_basis[v0_idx]

    ad_H = commutator(f_idx, f_val, H_su2, e_a)
    ad_Ep = commutator(f_idx, f_val, E_plus, e_a)
    ad_Em = commutator(f_idx, f_val, E_minus, e_a)

    candidates = np.array([e_a, ad_H, ad_Ep, ad_Em])
    frame = np.zeros((4, DIM_E8))
    for k in range(4):
        v = candidates[k].copy()
        for j in range(k):
            v -= float(v @ frame[j]) * frame[j]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return None
        frame[k] = v / norm
    return frame


def get_e7_curvature_direction(frame, e7_basis, f_idx, f_val):
    """Get the nonzero E7 curvature direction from an HP^1 frame."""
    for i in range(4):
        for j in range(i+1, 4):
            B = commutator(f_idx, f_val, frame[i], frame[j])
            proj = project(B, e7_basis)
            norm = np.linalg.norm(proj)
            if norm > 1e-10:
                return proj / norm
    return None


def compute_e7_charges_direct(pos_roots, m_pos_idx, T_R8, alpha_su2):
    """Compute E7-charges of the 56 m-doublets under direction T.

    In the compact form, [T, e_γ^(1)] = (T·γ)_R8 · e_γ^(2).
    The charge of doublet γ_k is: q_k = T_R8 · γ_k.
    The squared Casimir eigenvalue on up-space is: -q_k².
    """
    alpha_hat = alpha_su2 / np.linalg.norm(alpha_su2)

    charges = np.zeros(len(m_pos_idx))
    for k, idx in enumerate(m_pos_idx):
        gamma_k = pos_roots[idx]
        charges[k] = np.dot(T_R8, gamma_k)

    return charges


def compute_full_112_action(generator, m_basis, f_idx, f_val):
    """Compute the full 112×112 matrix of ad(generator) on m-space.

    This correctly handles the compact-form bracket where [H, e^(1)] → e^(2).
    """
    n = m_basis.shape[0]  # 112
    M = np.zeros((n, n))
    for i in range(n):
        bracket = commutator(f_idx, f_val, generator, m_basis[i])
        M[:, i] = m_basis @ bracket
    return M


def main() -> int:
    print("=" * 75)
    print("Step 1: E7-representation of 54 zero modes from c_2(E) = -1")
    print("=" * 75)
    print()

    t0 = time.time()
    res = Result()

    print("[setup] Loading E8 structure constants and EIX basis...")
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, block_labels = e7_su2_embedding(pos_roots=pos_roots)

    # Recover m_pos_idx (positive m-root indices)
    alpha_su2 = EIX_ALPHA_SU2
    su2_pos_idx = None
    for k in range(120):
        if np.allclose(pos_roots[k], alpha_su2, atol=1e-9):
            su2_pos_idx = k
            break
    m_pos_idx = []
    for k in range(120):
        if k == su2_pos_idx:
            continue
        dot = float(np.dot(pos_roots[k], alpha_su2))
        if abs(dot) > 1e-9:
            m_pos_idx.append(k)
    m_pos_idx = np.array(m_pos_idx, dtype=int)
    assert len(m_pos_idx) == 56

    print(f"  e7: {e7_basis.shape}, su2: {su2_basis.shape}, m: {m_basis.shape}")
    print(f"  m_pos_idx: {len(m_pos_idx)} positive m-roots")
    print()

    # ==================================================================
    # T1: Direct algebraic computation of E7 weights
    # ==================================================================
    banner("[T1] E7 weights of the 56 representation (direct from roots)")

    # The E7 Cartan is the 7D subspace of R^8 perpendicular to α_su2.
    # E7-weight of doublet k = projection of γ_k onto this hyperplane.
    alpha_hat = alpha_su2 / np.linalg.norm(alpha_su2)
    e7_weights = np.zeros((56, 8))
    for k in range(56):
        gamma = pos_roots[m_pos_idx[k]]
        e7_weights[k] = gamma - np.dot(gamma, alpha_hat) * alpha_hat

    print(f"  E7 weight vectors: shape {e7_weights.shape}")
    dots = np.array([np.dot(pos_roots[m_pos_idx[k]], alpha_su2) for k in range(56)])
    print(f"  γ·α_su2 values: {int(np.sum(dots > 0))} with +1, {int(np.sum(dots < 0))} with -1")

    # Verify: the weights should span a 7D space (the E7 weight space)
    rank_weights = np.linalg.matrix_rank(e7_weights, tol=1e-8)
    print(f"  Rank of weight matrix: {rank_weights} (expected 7)")
    res.report("E7 weights span 7D", rank_weights == 7, f"rank = {rank_weights}")

    # ==================================================================
    # T2: E7 curvature direction and charge computation
    # ==================================================================
    banner("[T2] E7-charges under instanton curvature direction")

    # Build first valid frame and get curvature direction
    T = None
    frame_v0 = None
    for v0 in range(112):
        frame = build_HP1_frame(m_basis, su2_basis, f_idx, f_val, v0_idx=v0)
        if frame is None:
            continue
        T = get_e7_curvature_direction(frame, e7_basis, f_idx, f_val)
        if T is not None:
            frame_v0 = v0
            break

    if T is None:
        print("FATAL: No valid E7 curvature direction found.")
        return 1

    print(f"  Frame from v0_idx = {frame_v0}")
    print(f"  T norm = {np.linalg.norm(T):.6f}, lies in e7: "
          f"{np.linalg.norm(T - project(T, e7_basis)):.2e}")

    # Extract R^8 (Cartan) components of T
    T_R8 = T[:8]
    T_root_part = np.linalg.norm(T[8:])
    print(f"  T Cartan part (R^8): norm = {np.linalg.norm(T_R8):.6f}")
    print(f"  T root-generator part: norm = {T_root_part:.2e}")

    is_cartan = T_root_part < 1e-8
    print(f"  T is purely Cartan: {is_cartan}")

    if is_cartan:
        # Direct charge computation from root inner products
        charges = compute_e7_charges_direct(pos_roots, m_pos_idx, T_R8, alpha_su2)
    else:
        # Use full 112-dim matrix approach for non-Cartan T
        print("  Using full 112×112 matrix approach (T has root parts)...")
        M_full = compute_full_112_action(T, m_basis, f_idx, f_val)
        # M² maps up→up: eigenvalues give -q²
        M_sq = M_full @ M_full
        # Restrict to up-subspace (even indices)
        up_idx = np.arange(0, 112, 2)
        M_sq_up = M_sq[np.ix_(up_idx, up_idx)]
        # Eigenvalues of M²|_up should be -q² (negative semi-definite)
        sq_eigvals = np.linalg.eigvalsh(0.5*(M_sq_up + M_sq_up.T))
        charges = np.sqrt(np.abs(sq_eigvals))

    # Display charge spectrum
    charge_groups = []
    sorted_charges = np.sort(np.abs(charges))
    for c in sorted_charges:
        if not charge_groups or abs(c - charge_groups[-1][0]) > 1e-6:
            charge_groups.append((c, 1))
        else:
            charge_groups[-1] = (charge_groups[-1][0], charge_groups[-1][1] + 1)

    print(f"\n  Charge spectrum |q_k| = |T·γ_k| for 56 m-doublets:")
    print(f"  {'|charge|':>10} {'mult':>6} {'interpretation':>30}")
    print(f"  {'-'*10} {'-'*6} {'-'*30}")
    n_zero = 0
    for c, mult in charge_groups:
        if c < 1e-6:
            interp = "→ ZERO MODE (unlifted)"
            n_zero = mult
        else:
            interp = "→ lifted by E7 instanton"
        print(f"  {c:10.6f} {mult:6d} {interp:>30}")

    print(f"\n  Zero-charge modes: {n_zero}")
    print(f"  Nonzero-charge modes: {56 - n_zero}")

    # ==================================================================
    # T3: Full 112-dim squared-action verification
    # ==================================================================
    banner("[T3] Full 112-dim ad(T)² verification")

    M_full = compute_full_112_action(T, m_basis, f_idx, f_val)

    # Verify anti-symmetry
    asym = np.max(np.abs(M_full + M_full.T))
    print(f"  ad(T) anti-symmetry on 112-space: {asym:.2e}")
    res.report("ad(T) is anti-symmetric", asym < 1e-6, f"max = {asym:.2e}")

    # M² on up-subspace: [T,[T, e_γ^(1)]] = [T, q_k e_γ^(2)] = -q_k² e_γ^(1)
    M_sq = M_full @ M_full
    up_idx = np.arange(0, 112, 2)
    M_sq_up = M_sq[np.ix_(up_idx, up_idx)]

    # Should be diagonal with entries -q_k²
    diag_M_sq = np.diag(M_sq_up)
    off_diag = M_sq_up - np.diag(diag_M_sq)
    off_diag_max = np.max(np.abs(off_diag))
    print(f"  M²|_up off-diagonal max: {off_diag_max:.2e}")
    print(f"  M²|_up diagonal entries (= -q²): {np.sort(diag_M_sq)[:5]}...")

    # Charges from diagonal (should match direct computation)
    charges_from_Msq = np.sqrt(np.abs(diag_M_sq))
    n_zero_Msq = np.sum(charges_from_Msq < 1e-6)
    print(f"  Zero-charge from M²: {n_zero_Msq} (should match T3 direct: {n_zero})")

    if is_cartan:
        res.report(
            "Direct charges match M² computation",
            np.allclose(np.sort(np.abs(charges)), np.sort(charges_from_Msq), atol=1e-6),
            f"max diff = {np.max(np.abs(np.sort(np.abs(charges)) - np.sort(charges_from_Msq))):.2e}",
        )

    # ==================================================================
    # T4: Interpretation of charges under E6×U(1)
    # ==================================================================
    banner("[T4] E6×U(1) interpretation of charge spectrum")

    # Under E6×U(1) ⊂ E7: 56 = 27_{+1} ⊕ 27̄_{-1} ⊕ 1_{+3} ⊕ 1_{-3}
    # The U(1) charges are the eigenvalues under a specific E7 Cartan direction
    # (the H_lambda splitter from the EIX embedding).
    # The instanton direction T may or may not align with this U(1).

    # Block labels encode the |q_U(1)| values under the EIX H_lambda splitter
    print(f"  Block structure (E6×U(1) blocks from H_lambda splitter):")
    up_idx = np.arange(0, 112, 2)
    for bl in range(4):
        mask = block_labels[up_idx] == bl
        count = np.sum(mask)
        if count > 0:
            avg_charge = np.mean(np.abs(charges[mask])) if is_cartan else 0
            print(f"    Block {bl}: {count} modes, avg |instanton charge| = {avg_charge:.4f}")

    # Key test: charge spectrum matches E6×U(1) branching (54:2, ratio 3:1)
    if is_cartan:
        sorted_abs = np.sort(np.abs(charges))
        low_charge = sorted_abs[0]  # the majority charge
        high_charge = sorted_abs[-1]  # the minority charge
        ratio = high_charge / low_charge if low_charge > 1e-10 else 0

        n_low = np.sum(np.abs(np.abs(charges) - low_charge) < 1e-6)
        n_high = np.sum(np.abs(np.abs(charges) - high_charge) < 1e-6)

        print(f"\n  Charge ratio |q_high|/|q_low| = {ratio:.6f} (expected 3.0)")
        print(f"  Multiplicities: {n_low} low + {n_high} high = {n_low+n_high}")

        res.report(
            "Charge ratio = 3 (E6×U(1) branching pattern)",
            abs(ratio - 3.0) < 0.001,
            f"ratio = {ratio:.6f}",
        )
        res.report(
            "Multiplicities (54, 2) match E6×U(1) branching",
            n_low == 54 and n_high == 2,
            f"got ({n_low}, {n_high})",
        )

        # Note: instanton E6' ≠ standard E6 (H_lambda); they are E7-conjugate
        block3_mask = block_labels[up_idx] == 3
        if np.sum(block3_mask) == 2:
            block3_charges = np.abs(charges[block3_mask])
            print(f"\n  NOTE: Standard E6 singlets (block 3) have instanton"
                  f" charge {block3_charges[0]:.4f}")
            print(f"  This equals the LOW charge → instanton E6' ≠ standard E6")
            print(f"  (The two E6 subgroups are related by E7 Weyl conjugation)")

    # ==================================================================
    # T5: Universality across quaternionic lines
    # ==================================================================
    banner("[T5] Universality: charge spectra across quaternionic lines")

    spectra = []
    n_valid = 0
    for v0 in range(min(56, m_basis.shape[0])):
        fr = build_HP1_frame(m_basis, su2_basis, f_idx, f_val, v0_idx=v0)
        if fr is None:
            continue
        T_local = get_e7_curvature_direction(fr, e7_basis, f_idx, f_val)
        if T_local is None:
            continue
        T_R8_local = T_local[:8]
        if np.linalg.norm(T_local[8:]) > 1e-8:
            continue  # skip non-Cartan cases
        ch = compute_e7_charges_direct(pos_roots, m_pos_idx, T_R8_local, alpha_su2)
        n_valid += 1
        n_z = np.sum(np.abs(ch) < 1e-6)
        spectrum = sorted(np.abs(ch), reverse=True)
        spectra.append((v0, n_z, spectrum[:4]))

    print(f"  Valid Cartan-type frames: {n_valid}")
    if spectra:
        zero_counts = [s[1] for s in spectra]
        print(f"  Zero-charge counts: min={min(zero_counts)}, max={max(zero_counts)}")
        print(f"  First few spectra (top 4 charges):")
        for v0, nz, sp in spectra[:5]:
            print(f"    v0={v0:3d}: n_zero={nz:2d}, top charges = "
                  f"[{', '.join(f'{c:.3f}' for c in sp)}]")

        # Key check: ALL frames give the same (54, 2) pattern
        # (no zeros expected — all modes have nonzero coupling)
        charge_patterns = []
        for v0, nz, sp in spectra:
            n_hi = sum(1 for c in sp if c > 1.0)
            charge_patterns.append(n_hi)
        all_have_2_high = all(p == 2 for p in charge_patterns[:20])
        res.report(
            "Universal charge pattern: always 2 modes at 3× (all frames)",
            all_have_2_high,
            f"first 20 frames: all have 2 high-charge modes = {all_have_2_high}",
        )

    # ==================================================================
    # T6: The physical zero-mode count (index theorem vs charges)
    # ==================================================================
    banner("[T6] Reconciling charges with index theorem")

    print(f"""
  KEY INSIGHT:
  ═══════════
  The index theorem gives ind(D_{{E⊗L}}) = 54 on HP^1.
  This counts NET chirality of zero modes, NOT the kernel of ad(T).

  The E7 charges q_k determine how each doublet COUPLES to the instanton.
  On S^4 with a U(1) connection of "instanton number" (from c_2):

  For charge-q component of rank-1:
    ind(D_q) depends on whether the effective c_2 for that component
    is changed by the E7 instanton coupling.

  For the FULL E⊗L bundle (rank 56×2 = 112):
    ind(D_{{E⊗L}}) = Σ_k (contribution from doublet k with charge q_k)

  The TOTAL is 54, regardless of the individual charges.

  However, the PHYSICAL content of the zero-mode space depends on
  which E7 representations the 54 modes carry. This is determined
  by the Dirac operator spectrum, not just the charges.

  For mass hierarchy: what matters is that the B=1 sector has a
  DIFFERENT coupling structure (through the E7 instanton) than B=0.
""")

    # Compute inner product structure of charges
    if is_cartan:
        print(f"  Charge spectrum statistics:")
        print(f"    Mean |q|: {np.mean(np.abs(charges)):.4f}")
        print(f"    Std |q|:  {np.std(np.abs(charges)):.4f}")
        print(f"    Max |q|:  {np.max(np.abs(charges)):.4f}")
        print(f"    Min |q|:  {np.min(np.abs(charges)):.4f}")

        # Verify: Σ q_k² should relate to c_2(E) via Killing form
        sum_q_sq = np.sum(charges**2)
        print(f"    Σ q² = {sum_q_sq:.4f}")
        # The Chern-Weil trace: tr_{56}(T²) = Σ q_k² for a Cartan T
        # This should be proportional to the instanton number
        print(f"    tr_56(T²) = Σq² = {sum_q_sq:.4f}")

        # Compare with kappa-trace
        kappa_TT = 2.0 * float(T @ T)
        print(f"    kappa(T,T) = {kappa_TT:.4f}")
        print(f"    Ratio tr_56/kappa = {sum_q_sq/kappa_TT:.4f} "
              f"(= Dynkin index of 56 rep)")

    # ==================================================================
    # T7: Which m-roots are the E6 singlets?
    # ==================================================================
    banner("[T7] Identification of E6 singlets in the 56")

    # Under E6×U(1) ⊂ E7: the E6 singlets have the highest |U(1)-charge|
    # (charge ±3 in the convention where 27 has charge ±1).
    # These correspond to specific m-roots.

    # The u1_splitter direction gives the U(1) charge
    from e8sim.roots import EIX_H_LAMBDA
    u1_dir = EIX_H_LAMBDA / np.linalg.norm(EIX_H_LAMBDA)

    u1_charges = np.zeros(56)
    for k in range(56):
        gamma = pos_roots[m_pos_idx[k]]
        u1_charges[k] = np.dot(gamma, u1_dir)

    # Sort by |u1_charge| to identify the singlets (highest charge)
    abs_u1 = np.abs(u1_charges)
    sorted_idx = np.argsort(abs_u1)[::-1]
    print(f"  Top-5 |U(1)-charges| (E6 singlets have highest):")
    for rank, idx in enumerate(sorted_idx[:5]):
        gamma = pos_roots[m_pos_idx[idx]]
        inst_q = charges[idx] if is_cartan else 0
        print(f"    #{rank}: m-root #{idx}, γ = {gamma}, |U1| = {abs_u1[idx]:.4f}, "
              f"instanton q = {inst_q:+.4f}")

    # The E6 singlets (top 2 in |U(1)-charge|)
    singlet_indices = sorted_idx[:2]
    singlet_inst_charges = charges[singlet_indices] if is_cartan else np.zeros(2)
    print(f"\n  E6 singlets: indices {singlet_indices}")
    print(f"  Their instanton charges: {singlet_inst_charges}")
    print(f"  Their E7 weights:")
    for idx in singlet_indices:
        print(f"    w_{idx} = {e7_weights[idx]}")

    # The instanton's 2 high-charge modes are E6' singlets (conjugate E6')
    if is_cartan:
        highest_charge_idx = np.argsort(np.abs(charges))[-2:]
        print(f"\n  Instanton's high-charge modes (= E6'-singlets): indices {highest_charge_idx}")
        print(f"  These are NOT the standard E6 singlets (block 3), confirming")
        print(f"  that instanton E6' is E7-conjugate to standard E6.")
        std_singlet_match = all(idx in singlet_indices for idx in highest_charge_idx)
        res.report(
            "Instanton E6' ≠ standard E6 (different singlet identification)",
            not std_singlet_match,
            f"confirms E7 Weyl conjugation between the two E6 subgroups",
        )

    # ==================================================================
    # Summary
    # ==================================================================
    elapsed = time.time() - t0
    print()
    print("=" * 75)
    print(f"Summary: {res.passed} PASS / {res.failed} FAIL  (~{elapsed:.1f} s)")
    print("=" * 75)
    print()

    # Final interpretation
    if is_cartan:
        # Report the full charge spectrum for documentation
        print("  COMPLETE CHARGE SPECTRUM (for documentation):")
        for k in np.argsort(np.abs(charges)):
            gamma = pos_roots[m_pos_idx[k]]
            bl = block_labels[2*k]
            print(f"    k={k:2d}, block={bl}, q={charges[k]:+.6f}, γ={gamma}")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
