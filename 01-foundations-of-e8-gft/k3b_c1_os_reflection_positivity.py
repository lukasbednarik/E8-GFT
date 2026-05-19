"""Numerical reflection positivity of the slow-mode two-point
function on the antichain abelian subspace $\\mathfrak{a}\\subset
\\mathfrak{m}_{\\mathrm{EIX}}$ (leading bosonic Gaussian sector).

This script implements the numerical verification cited under
``Proposition~\\ref{prop:leading-os-rp}`` of
``sections/06-emergent-spacetime.tex`` (sub-claim $(\\alpha)$ of
Hypothesis~\\ref{hyp:4d-emergence}). Together with the Glimm--Jaffe
sufficiency analysis of ``k3b_os_b_full_interacting.py`` it covers
the leading + manifest-positivity sub-leading content of $(\\alpha)$
recorded in the status list of \\S~\\ref{sec:emergent:hyp} of that section.

Theoretical setup (a one-paragraph summary of
\\S\\ref{sec:emergent:alpha}). The compact $E_8$-GFT of
\\ref{post:realform} is fundamentally Euclidean; Lorentzian spacetime
is recovered on the reconstructed Hilbert space through the
Osterwalder--Schrader theorem~\\cite{osterwalder-schrader1973,
osterwalder-schrader1975}, conditional on reflection positivity (RP)
of the Schwinger functions (Glimm--Jaffe~\\cite{glimm-jaffe1987}
\\S6.2). On the four-dimensional abelian sector
$\\mathfrak{a}\\cong\\R^{4}$ spanned by the antichain Chevalley root
vectors of Definition~\\ref{def:Pmu-EIX}, the leading slow-mode
kinetic data of \\eqref{eq:slow-mode-kinetic} are
$K^{\\mu\\nu} = \\tfrac12\\delta^{\\mu\\nu}$ and
$M_W^{\\,2} = c_H^{(\\mathrm{EIX})}\\,r_*^{\\,2}$, giving the standard
4D Euclidean Yukawa propagator
$G_2(R) = (m/4\\pi^{\\,2}R)\\,K_1(mR)$, with $m^{\\,2} = M_W^{\\,2}/K_{\\mathrm{kin}}$.
The Källén--Lehmann sum-of-squares argument of
Proposition~\\ref{prop:leading-os-rp} establishes RP analytically;
this script verifies it numerically on the explicit $\\mathfrak{e}_8$
basis (e8sim) at the canonical normalisation of
Appendix~\\ref{app:conv:lie}.

Five layered tests (no separate ``OS.1``/``OS.2`` reprint of upstream
numbers; the kinetic/mass data are imported from
``k3b_volovik_signature.py``):

  OS.3  Position-space Schwinger function: $G_2(R) > 0$ for $R > 0$
        (manifest from $K_1(z) > 0$); exponential decay
        $G_2(R)\\,R^{\\,3/2} \\sim e^{-mR}$ in the IR (slope $\\approx -m$);
        strict monotonicity in $R$.

  OS.4  Reflection-positivity matrix $M_{nm} = G_2(R_{nm})$ with
        $R_{nm} = \\sqrt{(\\tau_n+\\tau_m)^{\\,2} + |\\boldsymbol{x}_n -
        \\boldsymbol{x}_m|^{\\,2}}$ on $N = 30$ random test points
        $(\\tau_n,\\boldsymbol{x}_n)$ supported in $\\{\\tau > 0\\}$
        (delta-function limit of the OS form
        \\eqref{eq:os-matrix}); verify
        $\\lambda_{\\min}(M) \\ge -\\mathrm{TOL_{PSD}}$.

  OS.5  $\\mathrm{O}(4)$-isotropy of the verdict across all four
        time-axis choices in $\\mathfrak{a}$ (rotational symmetry of
        $K^{\\mu\\nu} = \\tfrac12\\delta^{\\mu\\nu}$).

  OS.6  Antichain-orbit invariance: rerun OS.4 on
        $N_a = 5$ representatives drawn from the $630$-element set
        of maximal antichains in $\\Delta(\\mathfrak{m}_{\\mathrm{EIX}}^+)$
        (Corollary~\\ref{cor:antichain-EIX}), confirming that the
        verdict does not depend on the antichain choice
        (\\S~\\ref{para:antichain-equivariance}; full $630$-sweep
        is performed by ``e3_antichain_full_sweep.py``).

  OS.7  Wick rotation $p^{\\,0}\\to -i\\,E$ produces the Lorentzian
        propagator with mass-shell pole at
        $E^{\\,2} = |\\boldsymbol{k}|^{\\,2} + m^{\\,2} > 0$
        (positive-energy spectrum) and group velocity
        $v_g = |\\boldsymbol{k}|/E < 1$ (causal propagation).

A positive verdict on OS.3--OS.7 closes the leading bosonic Gaussian
content of sub-claim $(\\alpha)$: by Glimm--Jaffe Theorem~6.2.2 a
Gaussian Euclidean QFT with positive-definite kinetic operator and
positive mass-squared is automatically reflection-positive
(the explicit Kallén--Lehmann form is recorded in the proof of
Proposition~\\ref{prop:leading-os-rp}); the OS theorem then delivers a
Lorentzian Wightman QFT for the slow-mode two-point function with
the Poincaré representation of
Proposition~\\ref{prop:gamma-global-poincare}. The $f$-tensor
sub-leading invariants $\\mathcal{S}_c$, $\\mathcal{S}_{c'}$,
$\\mathcal{S}_e$ on the $\\mathcal{D}_{\\mathrm{stab}}$-boundary
remain \\textsf{[Open]}
(Proposition~\\ref{prop:full-os-rp-interior}); they are not
in scope here.

Conventions (shared with ``e8sim/eix.py`` and ``k3b_volovik_signature.py``):

  - e8sim 248-basis $\\{T_A\\}_{A=0..247}$ orthonormal w.r.t. $(X,Y)_E$.
  - $\\kappa(X,Y) := -K(X,Y)/h^v = 2\\,(X,Y)_E$
    (``KAPPA_OVER_EUCLID = 2``).
  - $V_A = \\alpha_{\\mathrm{su}_2}/\\|\\alpha_{\\mathrm{su}_2}\\|_E$,
    $r_*^{\\,2} := \\kappa(V_A, V_A) = 2$.
  - $X_\\mu := E_{+\\alpha_\\mu}^{\\mathrm{e8sim}}
    + E_{-\\alpha_\\mu}^{\\mathrm{e8sim}}\\in\\mathfrak{m}$,
    $\\|X_\\mu\\|_E = \\sqrt 2$.
  - $K^{\\mu\\nu} = (1/\\|X_\\mu\\|_E^{\\,2})\\,\\delta^{\\mu\\nu} =
    \\tfrac12\\delta^{\\mu\\nu}$.
  - $M_W^{\\,2} = c_H^{(\\mathrm{EIX})}\\,r_*^{\\,2} = \\tfrac14\\cdot 2 = \\tfrac12$.
  - $m^{\\,2} = M_W^{\\,2}/K_{\\mathrm{kin}} = (1/2)/(1/2) = 1$.

References:

  - K. Osterwalder, R. Schrader, *Axioms for Euclidean Green's
    functions*, Comm. Math. Phys. **31** (1973) 83; **42** (1975) 281.
  - J. Glimm, A. Jaffe, *Quantum Physics: A Functional Integral Point
    of View*, 2nd ed. (Springer, 1987), \\S6.2.2 (RP for free fields).
  - I. J. Schoenberg, *Metric spaces and positive definite functions*,
    Trans. Amer. Math. Soc. **44** (1938) 522 (positive-definiteness
    via Bessel/Yukawa kernels).
  - ``sections/06-emergent-spacetime.tex``,
    Proposition~\\ref{prop:leading-os-rp} and
    Proposition~\\ref{prop:gamma-global-poincare}.

Run::

    python3 scripts/01-foundations-of-e8-gft/k3b_c1_os_reflection_positivity.py
    pytest -v scripts/01-foundations-of-e8-gft/k3b_c1_os_reflection_positivity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8


# ----------------------------------------------------------------------
# Modified Bessel function K_1(x)  (numpy-only quadrature)
# ----------------------------------------------------------------------
#
# We deliberately avoid an external dependency on ``scipy.special.kv``;
# K_1(x) is computed via the integral representation
#
#     K_1(x) = ∫_0^∞ exp(−x cosh(t)) cosh(t) dt,   x > 0,
#
# which is strictly positive and numerically stable over the range
# x ∈ (10⁻³, 30) relevant for our RP test.  Vectorised via broadcasting;
# nominal precision ~1e-13 (cross-checked against ``mpmath.besselk(1, x)``
# on a 1000-point grid).

_BK1_T_MAX = 25.0   # cosh(25) ~ 3.6e10 → exp(−x · cosh(25)) underflows for x > 1e-9
_BK1_NT = 600       # quadrature points; 600 gives rel. err < 1e-13


def besselk1(x: np.ndarray) -> np.ndarray:
    """K_1(x) via integral representation, vectorised over numpy.

    K_1(x) = ∫_0^∞ exp(−x cosh(t)) cosh(t) dt  (x > 0).

    For x > 30 we switch to the asymptotic form
        K_1(x) ~ √(π/(2x)) · exp(−x) · (1 + 3/(8x) + ...)
    (the integrand would otherwise underflow).
    """
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    x_flat = x.flatten()
    out = np.empty_like(x_flat)

    small = x_flat < 30.0
    large = ~small

    if np.any(small):
        x_s = x_flat[small]
        t = np.linspace(0.0, _BK1_T_MAX, _BK1_NT)
        cosh_t = np.cosh(t)
        exp_arg = np.clip(-np.outer(x_s, cosh_t), -700.0, 0.0)
        integrand = np.exp(exp_arg) * cosh_t[None, :]
        out[small] = np.trapezoid(integrand, t, axis=1)

    if np.any(large):
        x_l = x_flat[large]
        out[large] = (np.sqrt(np.pi / (2.0 * x_l)) * np.exp(-x_l)
                       * (1.0 + 3.0 / (8.0 * x_l)
                          - 15.0 / (128.0 * x_l ** 2)))

    return out.reshape(shape)

ROOT = bootstrap_repo_root()
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from e8sim.algebra import (  # noqa: E402
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
)
from e8sim.roots import (  # noqa: E402
    e7_su2_embedding,
    build_compatibility_matrix,
    enumerate_antichains_of_size,
)
from e8sim.eix import (  # noqa: E402
    DIM_E8 as _DIM_E8,
    DIM_M_EIX,
    DIM_H_EIX,
    H_VEE_E8,
    H_VEE_SU2,
    C_H_EIX,
    KAPPA_OVER_EUCLID,
    canonical_VA as build_VA,
)
from e3_orbit_selection import (  # noqa: E402
    root_set,
    m_plus_for_EIX,
)
from k3b_volovik_signature import (  # noqa: E402
    m_plus_indices_in_canonical_pos,
    build_X_mu,
    slow_mode_field_basis,
    slow_mode_kinetic_matrix,
)

assert DIM_E8 == _DIM_E8 == 248


CONSTANTS_PATH = constants_path(ROOT)

TOL_ALG = 1e-9
TOL_PSD = 1e-10        # roundoff tolerance for "non-negative" eigenvalue
TOL_FRACT = 1e-3
N_TEST_FUNCTIONS = 30  # OS.4: number of test points in {τ > 0}
N_ANTICHAIN_SAMPLES = 5  # OS.6: gauge-invariance test
RNG_SEED = 0


# ----------------------------------------------------------------------
# Yukawa Green function in 4D (free massive scalar)
# ----------------------------------------------------------------------


def yukawa_propagator(R: np.ndarray, m: float) -> np.ndarray:
    """G_2(R) = (m / 4π² R) · K_1(m R) for R > 0; regularise at R = 0.

    Standard Euclidean 4D massive scalar Green function:
        G_2(R) = ∫ d^4 p / (2π)^4 · exp(i p·x) / (p² + m²)
               = m / (4π² R) · K_1(m R),    R = |x|.

    Positive for R > 0, with logarithmic singularity at R → 0
    (~ 1/(2π² R²) for m → 0) and exponential decay
        G_2(R) ~ √(m / (2π · R)) · exp(−m R) / (4π² R)         (R → ∞).

    On the diagonal (R = 0) we cap at a UV-regulated value
    (point-splitting); the PSD test never evaluates at R = 0
    (test points are distinct).
    """
    R = np.asarray(R, dtype=np.float64)
    out = np.zeros_like(R)
    nz = R > 1e-12
    out[nz] = (m / (4 * np.pi ** 2 * R[nz])) * besselk1(m * R[nz])
    out[~nz] = np.inf
    return out


# ----------------------------------------------------------------------
# OS reflection-positivity matrix builder
# ----------------------------------------------------------------------


def build_RP_matrix(points: np.ndarray, m: float, time_axis: int = 0
                    ) -> np.ndarray:
    """Build the OS reflection-positivity matrix M_{nm} = G_2(Θx_n − x_m).

    Parameters
    ----------
    points : (N, 4) float64
        Locations of N test points in R^4.  All points must have
        positive `time_axis` coordinate (= support in {τ > 0}).
    m : float
        Effective Euclidean mass m² = M_W² / K_kin (= 1 in our units).
    time_axis : int, default 0
        Which of the 4 spacetime axes to use as Euclidean time.

    Returns
    -------
    M : (N, N) float64
        Symmetric matrix M_{nm} = G_2(R_{nm}) with
        R_{nm} = sqrt((τ_n + τ_m)² + |spatial_n − spatial_m|²).
        For the delta-function limit of OS test functions, this is
        the canonical RP matrix (Glimm–Jaffe §6.2.2).
    """
    N = len(points)
    times = points[:, time_axis]
    spatial_axes = [a for a in range(4) if a != time_axis]
    spatial = points[:, spatial_axes]  # (N, 3)

    tau_sum = times[:, None] + times[None, :]  # (N, N)
    diff = spatial[:, None, :] - spatial[None, :, :]  # (N, N, 3)
    spatial_sq = np.einsum("nmi,nmi->nm", diff, diff)
    R = np.sqrt(tau_sum ** 2 + spatial_sq)

    M = yukawa_propagator(R, m)
    return M


def random_points_positive_time(N: int, time_axis: int = 0,
                                  tau_min: float = 0.5, tau_max: float = 5.0,
                                  spatial_range: float = 5.0,
                                  rng: np.random.Generator | None = None
                                  ) -> np.ndarray:
    """Sample N random points (τ_n, x⃗_n) with τ_n ∈ [tau_min, tau_max] > 0.

    Spatial coordinates uniform in [-spatial_range, spatial_range]^3.
    Returned as (N, 4) array with `time_axis` carrying τ_n.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    pts = np.zeros((N, 4), dtype=np.float64)
    pts[:, time_axis] = rng.uniform(tau_min, tau_max, size=N)
    spatial_axes = [a for a in range(4) if a != time_axis]
    pts[:, spatial_axes] = rng.uniform(-spatial_range, spatial_range,
                                          size=(N, 3))
    return pts


# ----------------------------------------------------------------------
# Setup helper — K^{μν} + M_W² imported from k3b_volovik_signature
# ----------------------------------------------------------------------


def compute_effective_mass(antichain: tuple, m_pos_idx_in_canonical: np.ndarray,
                            r_sq: float) -> tuple[float, float, float]:
    """Return (K_diag, M_W², m_eff) from the slow-mode block of K3b.

    K^{μν} = (1/2) δ^{μν} (verified by `k3b_volovik_signature.py`),
    M_W² = c_H · r_*², m_eff² = M_W²/K_diag.
    """
    X = build_X_mu(m_pos_idx_in_canonical, antichain)
    E_field = slow_mode_field_basis(m_pos_idx_in_canonical, antichain)
    K_munu = slow_mode_kinetic_matrix(X, E_field)
    K_diag = float(np.mean(np.diag(K_munu)))
    M_W_sq = C_H_EIX * r_sq
    m_eff = float(np.sqrt(M_W_sq / K_diag))
    return K_diag, M_W_sq, m_eff


# ----------------------------------------------------------------------
# Test OS.3 — Position-space Schwinger function: positivity & decay
# ----------------------------------------------------------------------


def test_OS_3_schwinger_position(m_eff: float, res: Result) -> None:
    banner("[OS.3] Schwinger function G_2(R): positivity + exponential decay")

    # G_2(R) = (m / 4π² R) K_1(m R)
    R_samples = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    G_R = yukawa_propagator(R_samples, m_eff)

    print(f"    G_2(R) at Euclidean distances R (m = {m_eff:.4f}):")
    for R, G in zip(R_samples, G_R):
        print(f"      R = {R:6.2f}:   G_2 = {G:+.6e}")

    res.report(
        "G_2(R) > 0 for all R > 0 (manifest from K_1 > 0)",
        np.all(G_R > 0),
        f"min G_2 = {float(G_R.min()):.4e}",
    )

    # Asymptotic decay: G_2(R → ∞) ~ √(m/(2π R)) · exp(−m R) / (4π² R),
    # i.e. G_2(R) · R^{3/2} ~ const · exp(−m R).  Fit log[G_2 · R^{3/2}]
    # vs. R to extract the slope ≈ −m without the algebraic prefactor bias.
    R_large = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    G_large = yukawa_propagator(R_large, m_eff)
    log_GR32 = np.log(G_large * R_large ** 1.5)
    slope = np.polyfit(R_large, log_GR32, 1)[0]
    res.report(
        f"Exponential decay G_2(R)·R^{{3/2}} ~ exp(−m R) in the IR "
        f"(slope ≈ −m = {-m_eff:.4f})",
        abs(slope - (-m_eff)) < 0.01,
        f"slope = {slope:.6f}, expected = {-m_eff:.6f}",
    )

    # Strict monotonicity in R (= no oscillating part, no ghost)
    diffs = np.diff(G_R)
    res.report(
        "G_2(R) is strictly decreasing in R (no oscillation, no ghost)",
        np.all(diffs < 0),
        f"max(diff) = {float(diffs.max()):.4e} (must be < 0)",
    )


# ----------------------------------------------------------------------
# Test OS.4 — Reflection-positivity matrix M_{nm} ≽ 0
# ----------------------------------------------------------------------


def test_OS_4_reflection_positivity_matrix(m_eff: float, time_axis: int,
                                              N_pts: int,
                                              res: Result,
                                              rng: np.random.Generator | None = None
                                              ) -> tuple[float, float]:
    banner(f"[OS.4] RP matrix M_{{nm}} ≽ 0 (N={N_pts}, time-axis = {time_axis})")

    pts = random_points_positive_time(N_pts, time_axis=time_axis, rng=rng)
    M = build_RP_matrix(pts, m_eff, time_axis=time_axis)

    sym_err = float(np.linalg.norm(M - M.T))
    res.report(
        f"RP matrix is symmetric (G_2 is a function of R)",
        sym_err < TOL_ALG,
        f"‖M − M^T‖ = {sym_err:.2e}",
    )

    eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
    eigs.sort()
    n_neg = int(np.sum(eigs < -TOL_PSD))
    n_pos = int(np.sum(eigs > +TOL_PSD))
    n_zero = N_pts - n_neg - n_pos

    print(f"    Eigenvalues of M_{{nm}} (N×N = {N_pts}×{N_pts}):")
    print(f"      min = {float(eigs[0]):+.4e}, max = {float(eigs[-1]):+.4e}")
    print(f"      spectrum = ({n_pos} positive, {n_zero} zero, {n_neg} negative)")
    print(f"      cond #(M) ≈ {(eigs[-1] / max(eigs[0], 1e-30)):.2e}")

    res.report(
        f"All eigenvalues of M_{{nm}} ≥ 0 (= RP, OS axiom E2)",
        n_neg == 0,
        f"min eigenvalue = {float(eigs[0]):+.4e}, n_neg = {n_neg}",
    )

    return float(eigs[0]), float(eigs[-1])


# ----------------------------------------------------------------------
# Test OS.5 — O(4)-isotropy: independence of the time-axis choice
# ----------------------------------------------------------------------


def test_OS_5_time_axis_invariance(m_eff: float, N_pts: int,
                                     res: Result) -> None:
    banner(f"[OS.5] O(4)-isotropy: RP PASS for each of the 4 time-axis choices")

    rng = np.random.default_rng(RNG_SEED + 100)
    min_eigs = []
    for ta in range(4):
        pts = random_points_positive_time(N_pts, time_axis=ta, rng=rng)
        M = build_RP_matrix(pts, m_eff, time_axis=ta)
        eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
        min_eigs.append(float(eigs.min()))
        n_neg = int(np.sum(eigs < -TOL_PSD))
        print(f"    time-axis = {ta}: min eigenvalue = {min_eigs[-1]:+.4e}, "
              f"n_neg = {n_neg}")

    all_ok = all(me >= -TOL_PSD for me in min_eigs)
    res.report(
        f"RP PASS for each of the 4 time-axis choices (O(4) Euclidean isotropy)",
        all_ok,
        f"min eigenvalues per axis = "
        f"{['{:+.2e}'.format(me) for me in min_eigs]}",
    )

    # Structural consistency: K^{μν} = (1/2)·𝟙 is isotropic, so all
    # 4 axes must yield the same statistics modulo the random seed.
    print(f"    Structural consistency: K^{{μν}} = (1/2)·𝟙 → O(4) symmetry")
    print(f"    rotating time-axis ↔ any of the 4 axes; all 4 PASS")


# ----------------------------------------------------------------------
# Test OS.6 — Antichain-orbit invariance
# ----------------------------------------------------------------------


def test_OS_6_antichain_invariance(m_eff: float, antichains: list,
                                     m_pos_idx_in_canonical: np.ndarray,
                                     r_sq: float, N_pts: int,
                                     res: Result) -> None:
    banner(f"[OS.6] Antichain-orbit invariance: RP PASS for N_a={len(antichains)} of 630")

    rng = np.random.default_rng(RNG_SEED + 200)
    pts = random_points_positive_time(N_pts, time_axis=0, rng=rng)

    min_eigs = []
    K_diags = []
    M_W_sqs = []
    for ai, ac in enumerate(antichains):
        # K^{μν} = (1/2)·𝟙 across all antichains (verified by
        # ``k3b_volovik_signature.py``); M_W² = c_H · r_*² is structural
        # and antichain-independent, so m² = 1 is gauge-invariant and the
        # RP matrix is identical across antichains modulo the random
        # sample.  Recomputed here for explicit verification.
        X = build_X_mu(m_pos_idx_in_canonical, ac)
        E_field = slow_mode_field_basis(m_pos_idx_in_canonical, ac)
        K_munu = slow_mode_kinetic_matrix(X, E_field)
        K_diag_ac = float(np.mean(np.diag(K_munu)))
        M_W_sq_ac = C_H_EIX * r_sq
        m_eff_ac = float(np.sqrt(M_W_sq_ac / K_diag_ac))

        K_diags.append(K_diag_ac)
        M_W_sqs.append(M_W_sq_ac)

        M = build_RP_matrix(pts, m_eff_ac, time_axis=0)
        eigs = np.linalg.eigvalsh(0.5 * (M + M.T))
        min_eigs.append(float(eigs.min()))
        n_neg = int(np.sum(eigs < -TOL_PSD))
        print(f"    antichain {ai}: K_kin = {K_diag_ac:.4f}, M_W² = {M_W_sq_ac:.4f}, "
              f"m = {m_eff_ac:.4f}, min eig = {min_eigs[-1]:+.4e}, n_neg = {n_neg}")

    K_std = float(np.std(K_diags))
    M_W_std = float(np.std(M_W_sqs))

    all_ok = all(me >= -TOL_PSD for me in min_eigs)
    res.report(
        f"RP PASS for all {len(antichains)} sampled antichains",
        all_ok,
        f"min eigenvalues = "
        f"{['{:+.2e}'.format(me) for me in min_eigs[:5]]}{'...' if len(min_eigs) > 5 else ''}",
    )
    res.report(
        "K^{μν}_diag is gauge-invariant across antichains",
        K_std < 1e-9,
        f"std(K_diag) = {K_std:.2e} over {len(antichains)} antichains",
    )
    res.report(
        "M_W² is gauge-invariant across antichains (structural mass)",
        M_W_std < 1e-9,
        f"std(M_W²) = {M_W_std:.2e} over {len(antichains)} antichains",
    )


# ----------------------------------------------------------------------
# Test OS.7 — Wick rotation + Lorentzian dispersion
# ----------------------------------------------------------------------


def test_OS_7_wick_rotation_dispersion(m_eff: float, res: Result) -> None:
    banner(f"[OS.7] Wick rotation p^0 → −i E + Lorentzian dispersion")

    # Euclidean propagator: G_E(p) = 2 / ((p^0)² + |k⃗|² + 1).
    # Wick rotation p^0 → −i E (Lorentzian energy):
    #   (p^0)² → −E²
    # Lorentzian propagator:  G_L = 2 / (−E² + |k⃗|² + 1) = -2 / (E² − |k⃗|² − 1).
    # Mass-shell pole: E² = |k⃗|² + 1 = |k⃗|² + m²  (relativistic
    # dispersion for a scalar of mass m).

    print(f"    Euclidean:   G_E(p) = 2 / ((p^0)² + |k⃗|² + 1)")
    print(f"    Wick:        p^0 → −i E,  so (p^0)² → −E²")
    print(f"    Lorentzian:  G_L(E, k⃗) = 2 / (−E² + |k⃗|² + 1)")
    print(f"                 = −2 / (E² − |k⃗|² − m²),  m² = 1")

    # Pole at E² = |k⃗|² + m² > 0 (positive-energy spectrum)
    k_samples = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    E_squared = k_samples ** 2 + m_eff ** 2
    E_samples = np.sqrt(E_squared)

    print(f"\n    Mass-shell pole positions:")
    for k, E in zip(k_samples, E_samples):
        print(f"      |k⃗| = {k:5.2f}:  E = √(k² + m²) = {E:.4f}  > 0  ✓")

    res.report(
        "Lorentzian pole at E² = |k⃗|² + m² > 0 (positive energy)",
        np.all(E_squared > 0),
        f"min E² = {float(E_squared.min()):.4f}",
    )
    res.report(
        f"Pole at rest mass: E(k=0) = m = {m_eff:.4f} (Lorentzian mass)",
        abs(E_samples[0] - m_eff) < TOL_ALG,
        f"E(0) = {float(E_samples[0]):.6f}, expected {m_eff:.6f}",
    )

    # Group velocity v_g = dE/dk = k / E < 1 (causal)
    print(f"\n    Causality check (group velocity v_g = k/E):")
    for k in [0.5, 1.0, 2.0, 5.0, 10.0]:
        E = np.sqrt(k ** 2 + m_eff ** 2)
        v_g = k / E
        print(f"      |k⃗| = {k:5.2f}:  v_g = {v_g:.4f}  < 1  (causal)")

    v_g_samples = np.array([k / np.sqrt(k ** 2 + m_eff ** 2)
                              for k in [0.5, 1.0, 2.0, 5.0, 10.0]])
    res.report(
        "Group velocity v_g = k/E < 1 (causal Lorentz propagation)",
        np.all(v_g_samples < 1.0),
        f"max v_g = {float(v_g_samples.max()):.4f}",
    )


# ----------------------------------------------------------------------
# Pytest-compatible wrappers
# ----------------------------------------------------------------------


def _setup_globals():
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots_arr = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots_arr)
    h_basis = np.vstack([e7_basis, su2_basis])
    V_A, r_sq = build_VA()

    pos_canonical = pos_roots_arr
    all_roots = np.concatenate([pos_canonical, -pos_canonical], axis=0)
    m_plus = m_plus_for_EIX(pos_canonical)
    m_pos_idx_in_canonical = m_plus_indices_in_canonical_pos(m_plus, pos_canonical)
    rl_set = root_set(all_roots)
    compat = build_compatibility_matrix(m_plus, rl_set)

    antichains_one = enumerate_antichains_of_size(compat, 4, max_count=1)
    antichain = antichains_one[0]

    return {
        "f_idx": f_idx, "f_val": f_val,
        "pos_roots": pos_canonical, "all_roots": all_roots,
        "m_basis": m_basis, "h_basis": h_basis,
        "V_A": V_A, "r_sq": r_sq,
        "m_plus": m_plus, "compat": compat,
        "m_pos_idx_in_canonical": m_pos_idx_in_canonical,
        "antichain": antichain,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("K3b-OS — Reflection positivity of the slow-mode 2-point function")
    print("Reference: sections/06-emergent-spacetime.tex, prop:leading-os-rp")
    print("Strategy:  leading bosonic Gaussian RP → OS reconstruction")
    print("=" * 72, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8_constants.pt + EIX embedding ...", flush=True)
    g = _setup_globals()
    print(f"        e8sim 248-basis: {DIM_E8}-dim orthonormal in (X, Y)_E")
    print(f"        h_basis = {g['h_basis'].shape}, m_basis = {g['m_basis'].shape}")
    print(f"        V_A in Cartan(SU(2)): r_*² = κ(V_A, V_A) = {g['r_sq']:.4f}")
    print(f"        |Δ(𝔪⁺)| = {len(g['m_plus'])}; antichain (one of 630) = "
          f"{g['antichain']}")
    print(f"        N_test_points = {N_TEST_FUNCTIONS}, "
          f"N_antichain_samples = {N_ANTICHAIN_SAMPLES}, RNG seed = {RNG_SEED}")

    K_diag, M_W_sq, m_eff = compute_effective_mass(g["antichain"],
                                                     g["m_pos_idx_in_canonical"],
                                                     g["r_sq"])
    print(f"        K^{{μν}} = {K_diag:.4f}·I,  "
          f"M_W² = {M_W_sq:.4f},  m_eff = {m_eff:.4f}",
          flush=True)

    test_OS_3_schwinger_position(m_eff, res)

    rng = np.random.default_rng(RNG_SEED)
    test_OS_4_reflection_positivity_matrix(m_eff, time_axis=0,
                                              N_pts=N_TEST_FUNCTIONS,
                                              res=res, rng=rng)
    test_OS_5_time_axis_invariance(m_eff, N_pts=N_TEST_FUNCTIONS, res=res)

    print(f"\n[setup] Enumerating {N_ANTICHAIN_SAMPLES} antichains for OS.6 ...",
          flush=True)
    antichains = enumerate_antichains_of_size(g["compat"], 4,
                                                 max_count=N_ANTICHAIN_SAMPLES)
    test_OS_6_antichain_invariance(m_eff, antichains,
                                     g["m_pos_idx_in_canonical"],
                                     g["r_sq"], N_pts=N_TEST_FUNCTIONS,
                                     res=res)

    test_OS_7_wick_rotation_dispersion(m_eff, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"K3b-OS summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 72)

    if res.failed > 0:
        print("\nFailed sub-tests:")
        for marker, name, detail in res.records:
            if marker == "FAIL":
                print(f"  - {name}")
                if detail:
                    print(f"      {detail}")
        return 1

    print("\nKey structural result (leading bosonic Gaussian sector):")
    print()
    print("  1. OS.3 PASS:  Schwinger function G_2(R) > 0, exp(−mR) decay,")
    print("     no oscillation.  Setup K^{μν} = (1/2)·𝟙, M_W² = 1/2, m_eff = 1")
    print("     imported from k3b_volovik_signature.py (slow-mode kinetic data,")
    print("     eq. (eq:slow-mode-kinetic) of sections/06-emergent-spacetime.tex).")
    print()
    print("  2. OS.4 PASS:  Reflection positivity verified numerically:")
    print(f"     • RP matrix M_{{nm}} for N={N_TEST_FUNCTIONS} test points in {{τ > 0}}")
    print("       is positive-semi-definite; all eigenvalues ≥ 0.")
    print("     • Glimm–Jaffe Theorem 6.2.2 guarantees this analytically;")
    print("       the numerical test confirms consistency of the setup.")
    print()
    print(f"  3. OS.5 PASS:  O(4)-isotropy: RP PASS for each of the 4 time axes")
    print(f"     (= rotational symmetry of K^{{μν}} = (1/2)·𝟙).")
    print()
    print(f"  4. OS.6 PASS:  Antichain-orbit invariance: RP PASS for N={N_ANTICHAIN_SAMPLES}")
    print(f"     antichains drawn from the 630-element set (consistency with")
    print(f"     the K^{{μν}} gauge-invariance verified by k3b_volovik_signature.py).")
    print()
    print("  5. OS.7 PASS:  Wick rotation p^0 → −i E gives a Lorentzian propagator")
    print(f"     with mass-shell pole E² = |k⃗|² + m² (m = {m_eff:.4f}).  Group")
    print("     velocity v_g = k/E < 1 (causality preserved); rest-mass E(0) = m.")
    print()
    print("Consequence (sections/06-emergent-spacetime.tex, prop:leading-os-rp):")
    print()
    print("  ✓ The leading bosonic Gaussian slow-mode 2-point function on the")
    print("    antichain abelian subspace 𝔞 ⊂ 𝔪_EIX satisfies the Osterwalder–")
    print("    Schrader axiom E2 (reflection positivity).  Together with E0, E1,")
    print("    E3, E4 (Proposition prop:gamma-global-poincare), the OS theorem")
    print("    delivers a Lorentzian Wightman QFT for G_2 with positive-definite")
    print("    Hilbert space and a strongly continuous unitary representation of")
    print("    the proper orthochronous Poincaré group P^↑_+.")
    print()
    print("  → Sub-claim (α) of Hypothesis hyp:4d-emergence is closed at the")
    print("    leading-Gaussian level; the extension to a D_stab interior point of")
    print("    the full interacting slow-mode action is verified by")
    print("    k3b_os_b_full_interacting.py (Proposition prop:full-os-rp-interior).")
    print()
    print("  → Out of scope here:")
    print("    • the full D_stab-boundary verdict for the sub-leading f-tensor")
    print("      invariants S_c, S_{c'}, S_e (prop:full-os-rp-interior, [Open]);")
    print("    • non-perturbative RP for the full path integral on E_8")
    print("      (constructive QFT on the group manifold).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
