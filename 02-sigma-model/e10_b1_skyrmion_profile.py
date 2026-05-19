"""Numerical verification of the Derrick virial identity on a B = 1
radial Skyrme-type profile, and of the operative sign
$\\lambda_{\\mathrm{sk}} > 0$.

Reference: companion paper *"Sigma model on the exceptional Wolf
space EIX = E_8/(E_7 x SU(2)): Wang--Ziller uniqueness, single
Skyrme term, and the Derrick sign $\\lambda_{\\rm sk} > 0$"*
(Proposition: Derrick virial identity on EIX; Corollary: operative
sign $\\lambda_{\\rm sk} > 0$).

What this script does
---------------------

We solve the radial Skyrme-type problem in the $B = 1$ topological
sector by minimising the static energy

    E[f] = 4 pi int_0^infty dr
              [ kappa_2 * ( r^2 (f')^2 + 2 sin^2 f )
              + lambda_sk * sin^2 f * ( 2 (f')^2 + sin^2 f / r^2 )
              + mu4 * r^2 * ( 1 - cos f ) ]

with Dirichlet boundary conditions $f(0) = \\pi$, $f(R) = 0$ on a
uniform 1D radial grid, using PyTorch L-BFGS in float64. The
minimiser is then used to verify two distinct claims of the paper:

(i)   the Derrick virial identity $E_4 = E_2 + 3 E_{\\rm pot}$ at the
      stationary point (with and without the potential); and

(ii)  the operative sign constraint $\\lambda_{\\rm sk} > 0$: with
      $\\lambda_{\\rm sk} > 0$ the minimiser is non-trivial and
      satisfies the virial identity, while with
      $\\lambda_{\\rm sk} < 0$ the static energy is unbounded below
      under L-BFGS and no stationary point compatible with the
      virial identity is found.

The radial 1D reduction is the standard SU(2) hedgehog ansatz; we use
it here only as a concrete non-trivial sector in which to test the
two claims numerically. No claim about $E_8/H$-specific dynamics is
made on the basis of this script.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from _common import bootstrap_repo_root, Result, banner

bootstrap_repo_root()


# ----------------------------------------------------------------------
# Radial energy functional
# ----------------------------------------------------------------------


@dataclass
class ProfileResult:
    """Output of the radial L-BFGS minimisation."""

    kappa2: float
    lambda_sk: float
    mu4: float
    R_box: float
    N_r: int
    f: np.ndarray
    M: float
    E2: float
    E4: float
    E_pot: float
    derrick_residual: float
    converged: bool
    n_steps: int
    elapsed_s: float


def _trapz(rho: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    return dr * (0.5 * rho[0] + torch.sum(rho[1:-1]) + 0.5 * rho[-1])


def _radial_derivative(f: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    fp = torch.zeros_like(f)
    fp[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    fp[0] = (f[1] - f[0]) / dr
    fp[-1] = (f[-1] - f[-2]) / dr
    return fp


def _radial_energies(
    f: torch.Tensor,
    r: torch.Tensor,
    dr: torch.Tensor,
    kappa2: float,
    lambda_sk: float,
    mu4: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trapezoidal integrals of the radial densities $E_2$, $E_4$, $E_{\\rm pot}$.

    The overall factor $4\\pi$ from the angular integration is applied
    by the caller; the densities here are the radial integrands only.
    Boundary derivatives use one-sided finite differences; interior
    derivatives are central.
    """
    fp = _radial_derivative(f, dr)

    sin_f = torch.sin(f)
    cos_f = torch.cos(f)
    sin2_f = sin_f * sin_f

    rho_2 = kappa2 * (r * r * fp * fp + 2.0 * sin2_f)
    rho_4 = lambda_sk * sin2_f * (2.0 * fp * fp + sin2_f / (r * r))
    rho_pot = mu4 * r * r * (1.0 - cos_f)

    return _trapz(rho_2, dr), _trapz(rho_4, dr), _trapz(rho_pot, dr)


def _winding_charge(f: torch.Tensor, dr: torch.Tensor) -> torch.Tensor:
    """Hedgehog winding charge $B = -(1/\\pi) [f - \\sin f \\cos f]_0^R$.

    For Dirichlet data $f(0) = \\pi$, $f(R) = 0$ the analytic
    answer is $B = 1$; we compute it numerically from the radial
    density $-(2/\\pi) f' \\sin^2 f$ to detect lattice-level
    unwinding during L-BFGS.
    """
    fp = _radial_derivative(f, dr)
    integrand = -2.0 * fp * torch.sin(f) ** 2 / np.pi
    return _trapz(integrand, dr)


def _initial_profile(r_np: np.ndarray, R_phys: float) -> np.ndarray:
    """Atiyah--Manton-style initial guess $f(r) = 2\\arctan((R/r)^2)$.

    The ansatz has the correct asymptotics for the $B = 1$ sector
    ($f(0) = \\pi$, $f \\to 0$ at infinity with $1/r^2$ tail) and gives
    a fast, robust start for L-BFGS.
    """
    return 2.0 * np.arctan((R_phys / np.maximum(r_np, 1e-12)) ** 2)


def solve_radial_skyrmion(
    *,
    kappa2: float = 1.0,
    lambda_sk: float = 1.0,
    mu4: float = 0.0,
    R_box: float = 12.0,
    R_phys_init: float = 1.0,
    N_r: int = 400,
    r_min: float = 1e-3,
    kappa_B: float = 1000.0,
    max_lbfgs_iter: int = 2000,
    tolerance: float = 1e-12,
    dtype: torch.dtype = torch.float64,
) -> ProfileResult:
    """Minimise the 1D radial static energy with Dirichlet $B = 1$ data.

    The interior values $f[1:-1]$ are optimised by PyTorch L-BFGS in
    float64; the boundary values $f[0] = \\pi$ and $f[-1] = 0$ are
    fixed by construction. A topological penalty
    $\\kappa_B (B - 1)^2$ is added to suppress lattice-level
    unwinding; it vanishes identically at $B = 1$ and therefore does
    not bias the variational stationarity condition tested below.
    The full result (including separate $E_2, E_4, E_{\\rm pot}$
    contributions and the Derrick residual
    $|E_4 - E_2 - 3 E_{\\rm pot}| / \\Sigma$) is returned to the
    caller.
    """
    if R_box <= r_min:
        raise ValueError(f"R_box ({R_box}) must exceed r_min ({r_min})")
    if N_r < 64:
        raise ValueError(f"N_r ({N_r}) too small; use >= 64 for accuracy")

    t0 = time.time()
    device = "cpu"

    r = torch.linspace(r_min, R_box, N_r, dtype=dtype, device=device)
    dr = r[1] - r[0]

    f_init_np = _initial_profile(r.cpu().numpy(), R_phys_init)
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
    last_E = [float("inf")]
    diverged = [False]

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        f_full = torch.cat([f_pi, f_inner, f_zero])
        E2, E4, E_pot = _radial_energies(
            f_full, r, dr, kappa2, lambda_sk, mu4
        )
        E_total = E2 + E4 + E_pot
        B = _winding_charge(f_full, dr)
        loss = E_total + kappa_B * (B - 1.0) ** 2
        loss.backward()
        n_steps[0] += 1
        last_E[0] = float(E_total.item())
        if not np.isfinite(last_E[0]) or last_E[0] < -1e6:
            diverged[0] = True
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        diverged[0] = True

    with torch.no_grad():
        f_full = torch.cat([f_pi, f_inner.detach(), f_zero])
        E2_t, E4_t, Epot_t = _radial_energies(
            f_full, r, dr, kappa2, lambda_sk, mu4
        )
        E2 = 4.0 * np.pi * float(E2_t.item())
        E4 = 4.0 * np.pi * float(E4_t.item())
        E_pot = 4.0 * np.pi * float(Epot_t.item())
        M = E2 + E4 + E_pot

        denom = abs(E4) + abs(E2) + 3.0 * abs(E_pot)
        derrick = abs(E4 - E2 - 3.0 * E_pot) / max(denom, 1e-30)

    converged = (
        not diverged[0]
        and np.isfinite(last_E[0])
        and n_steps[0] < int(max_lbfgs_iter * 1.5)
    )

    return ProfileResult(
        kappa2=kappa2,
        lambda_sk=lambda_sk,
        mu4=mu4,
        R_box=R_box,
        N_r=N_r,
        f=f_full.cpu().numpy(),
        M=M,
        E2=E2,
        E4=E4,
        E_pot=E_pot,
        derrick_residual=float(derrick),
        converged=converged,
        n_steps=n_steps[0],
        elapsed_s=time.time() - t0,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_derrick_virial(prof: ProfileResult, res: Result, label: str) -> None:
    """Verify the Derrick virial identity $E_4 = E_2 + 3 E_{\\rm pot}$.

    The identity is the variational stationarity condition under the
    one-parameter rescaling $f \\mapsto f \\circ (\\lambda \\cdot)$ in
    the radial sector. We require the relative residual
    $|E_4 - E_2 - 3 E_{\\rm pot}| / (|E_4| + |E_2| + 3|E_{\\rm pot}|)$
    to be below $5 \\times 10^{-3}$ at the L-BFGS stationary point.
    """
    res.report(
        f"Derrick virial identity ({label}): "
        f"|E_4 - E_2 - 3 E_pot| / Sigma < 5e-3",
        prof.derrick_residual < 5e-3,
        f"residual = {prof.derrick_residual:.3e}; "
        f"E_2 = {prof.E2:.4f}, E_4 = {prof.E4:.4f}, "
        f"3 E_pot = {3.0 * prof.E_pot:.4f}",
    )


def test_nontrivial(prof: ProfileResult, res: Result, label: str) -> None:
    """Verify that the minimiser is non-trivial: $E_2 > 0$ at the stationary point.

    Triviality (constant configuration $f \\equiv 0$ or
    $f \\equiv \\pi$) would give $E_2 = 0$. The Dirichlet boundary
    data $f(0) = \\pi$, $f(R) = 0$ together with $E_2 > 0$ certify
    that the minimiser is a genuine non-trivial static configuration
    in the $B = 1$ sector.
    """
    res.report(
        f"Non-trivial static configuration ({label}): E_2 > 0",
        prof.E2 > 1e-3,
        f"E_2 = {prof.E2:.4f}",
    )


def test_negative_lambda_collapse(
    prof: ProfileResult, res: Result, label: str
) -> None:
    """Verify the corollary's collapse claim in the $\\lambda_{\\rm sk} < 0$ regime.

    With $\\lambda_{\\rm sk} < 0$ the quartic energy $E_4$ is
    non-positive on every configuration, while the virial identity
    requires $E_4 = E_2 + 3 E_{\\rm pot} > 0$ on a non-trivial
    static configuration. The two are incompatible: no static
    finite-energy minimiser exists. We verify the operative
    consequence on the radial profile: the L-BFGS minimisation is
    either non-convergent (energy unbounded below) or settles on a
    configuration that violates the virial identity at order one.
    """
    failed_to_converge = not prof.converged
    virial_violated = prof.derrick_residual > 0.1
    energy_negative = prof.M < 0.0
    collapses = failed_to_converge or virial_violated or energy_negative

    res.report(
        f"lambda_sk < 0 regime ({label}): "
        f"no stationary point compatible with the virial identity",
        collapses,
        f"converged = {prof.converged}, M = {prof.M:.4e}, "
        f"derrick residual = {prof.derrick_residual:.3e}",
    )


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------


def print_summary_row(label: str, prof: ProfileResult) -> None:
    print(
        f"    {label:<28s}  "
        f"M = {prof.M:>+10.4f}   "
        f"E_2 = {prof.E2:>+9.4f}   "
        f"E_4 = {prof.E4:>+9.4f}   "
        f"3 E_pot = {3.0 * prof.E_pot:>+9.4f}   "
        f"derrick = {prof.derrick_residual:>9.2e}   "
        f"iters = {prof.n_steps:>4d}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Numerical verification of the Derrick virial identity "
            "and of the operative sign lambda_sk > 0 on a B = 1 "
            "radial Skyrme-type profile."
        ),
    )
    p.add_argument("--N_r", type=int, default=600)
    p.add_argument("--R_box", type=float, default=12.0)
    p.add_argument("--R_phys_init", type=float, default=1.0)
    p.add_argument("--r_min", type=float, default=1e-3)
    p.add_argument("--kappa2", type=float, default=1.0)
    p.add_argument("--lambda_sk", type=float, default=1.0)
    p.add_argument("--mu4", type=float, default=0.0)
    p.add_argument("--kappa_B", type=float, default=1000.0)
    p.add_argument("--max_iter", type=int, default=4000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print(
        "Numerical verification: Derrick virial identity + "
        "lambda_sk > 0 corollary"
    )
    print(
        "Sector: B = 1 radial Skyrme-type profile, "
        "Dirichlet BCs f(0) = pi, f(R) = 0"
    )
    print("=" * 72, flush=True)

    print("\n  Configuration:")
    print(f"    kappa_2     = {args.kappa2}")
    print(f"    lambda_sk   = {args.lambda_sk}")
    print(f"    mu4         = {args.mu4}")
    print(f"    kappa_B     = {args.kappa_B}  (topological penalty)")
    print(f"    N_r         = {args.N_r}")
    print(f"    R_box       = {args.R_box}")
    print(f"    R_phys_init = {args.R_phys_init}")
    print(f"    dtype       = float64")
    print()

    res = Result()
    t_global = time.time()

    # ----------------------------------------------------------------
    # Run 1: lambda_sk > 0, mu4 = 0  (pure Skyrme, virial E_4 = E_2)
    # ----------------------------------------------------------------
    banner("[run 1] lambda_sk > 0, mu4 = 0")
    prof_pure = solve_radial_skyrmion(
        kappa2=args.kappa2,
        lambda_sk=abs(args.lambda_sk),
        mu4=0.0,
        R_box=args.R_box,
        R_phys_init=args.R_phys_init,
        N_r=args.N_r,
        r_min=args.r_min,
        kappa_B=args.kappa_B,
        max_lbfgs_iter=args.max_iter,
    )
    print_summary_row("lambda_sk > 0, mu4 = 0", prof_pure)
    print()
    test_derrick_virial(prof_pure, res, "lambda_sk > 0, mu4 = 0")
    test_nontrivial(prof_pure, res, "lambda_sk > 0, mu4 = 0")

    # ----------------------------------------------------------------
    # Run 2: lambda_sk > 0, mu4 > 0  (with potential, virial E_4 = E_2 + 3 E_pot)
    # ----------------------------------------------------------------
    banner("[run 2] lambda_sk > 0, mu4 > 0")
    prof_pot = solve_radial_skyrmion(
        kappa2=args.kappa2,
        lambda_sk=abs(args.lambda_sk),
        mu4=max(args.mu4, 0.1),
        R_box=args.R_box,
        R_phys_init=args.R_phys_init,
        N_r=args.N_r,
        r_min=args.r_min,
        kappa_B=args.kappa_B,
        max_lbfgs_iter=args.max_iter,
    )
    print_summary_row("lambda_sk > 0, mu4 > 0", prof_pot)
    print()
    test_derrick_virial(prof_pot, res, "lambda_sk > 0, mu4 > 0")
    test_nontrivial(prof_pot, res, "lambda_sk > 0, mu4 > 0")

    # ----------------------------------------------------------------
    # Run 3: lambda_sk < 0  (collapse: no stationary point compatible
    # with the virial identity)
    # ----------------------------------------------------------------
    banner("[run 3] lambda_sk < 0  (collapse regime)")
    prof_neg = solve_radial_skyrmion(
        kappa2=args.kappa2,
        lambda_sk=-abs(args.lambda_sk),
        mu4=0.0,
        R_box=args.R_box,
        R_phys_init=args.R_phys_init,
        N_r=args.N_r,
        r_min=args.r_min,
        kappa_B=args.kappa_B,
        max_lbfgs_iter=args.max_iter,
    )
    print_summary_row("lambda_sk < 0, mu4 = 0", prof_neg)
    print()
    test_negative_lambda_collapse(prof_neg, res, "lambda_sk < 0")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_global
    print()
    print("=" * 72)
    print(
        f"Summary: {res.passed} PASS / {res.failed} FAIL"
        + (f" / {res.skipped} SKIP" if res.skipped else "")
        + f"   ({elapsed:.1f}s total)"
    )
    print("=" * 72)

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
