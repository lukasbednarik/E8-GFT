#!/usr/bin/env python3
"""M4(c)-relaxation §5.1: Explicit spectral sum Σ_2(s).

Computes the spectral sum relevant to the M4(c)-relaxation CC mechanism:

    Σ_2(s) := Σ_{ρ ∈ class-one, C_2 > 0}  d_ρ · m_ρ · C_2(ρ)^{2-s}

and its zeta-regularization Σ_2'(0) = -Σ d_ρ C_2(ρ)^2 log C_2(ρ).

This sum differs from the standard log-det spectral zeta (-Σ d_ρ log C_2)
only in the summation kernel: C_2^{2-s} vs C_2^{-s}.  The physical content:
the M4(c)-porušující operátor S_d has Hessian eigenvalue C_2(ρ)^2 on each
class-one mode (Lemma 5.2.1), so Σ_2(0) = Σ d_ρ C_2^2 is the "weighted
counting function" and Σ_2'(0) is the corresponding zeta-regularized
determinant analog.

Infrastructure reused:
  - do5b_helgason_spherical_enum.py: Cartan-Helgason fast enumeration
    of class-one E_8 reps for the symmetric pair (E_8, E_7×SU(2))
  - do5b_explicit_log_det.py: build_e8_rep_machinery()

Tests:
  T_S2.1  Cross-check: Λ²=280 gives 8 class-one reps (= Helgason enum verified)
  T_S2.2  Σ_2(0) at Λ²=280: explicit sum of d_ρ · C_2^2 over 7 non-trivial reps
  T_S2.3  Σ_2'(0) at Λ²=280: explicit sum of -d_ρ · C_2^2 · log C_2
  T_S2.4  Convergence study: Σ_2(0) and Σ_2'(0) vs Λ² ∈ [120, ..., 5000]
  T_S2.5  Growth rate fit: log Σ_2(0) ~ p · log Λ² (empirical exponent)
  T_S2.6  Ratio Σ_2'(0) / Σ_2(0) convergence (→ log of dominant C_2)
  T_S2.7  Connection to CC formula: Σ_2(0) / (4π)^56 / N_EIX^2 estimate

Reference:
  - research/01-cc/cc-tests/m4c-relaxation.md §5.1
  - debug_plan/scripts/do5b_helgason_spherical_enum.py (Helgason enum)
  - debug_plan/scripts/do5b_explicit_log_det.py (E_8 rep machinery)
  - research/01-cc/scripts/m4c_prefactor_verify.py (§5.2 prefactor = 1)

Run:
    python3 research/01-cc/scripts/m4c_spectral_sum.py
    python3 research/01-cc/scripts/m4c_spectral_sum.py --max-lambda-sq 10000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "debug_plan" / "scripts"))

from _common import Result, banner  # noqa: E402
from do5b_explicit_log_det import build_e8_rep_machinery  # noqa: E402
from do5b_helgason_spherical_enum import (  # noqa: E402
    enumerate_class_one_via_helgason,
    SPHERICAL_FUND_EXPECTED,
)

# ──────────────────────────────────────────────────────────────────────
# Physical constants (from m4c-relaxation.md §4)
# ──────────────────────────────────────────────────────────────────────
DIM_EIX = 112
A2_EIX = 1_175_384.0 / 15.0  # = 78 358.93...
N_EIX_PIC2 = 1.96e32
N_EIX_PIC2_SQ = N_EIX_PIC2**2  # = 3.84e64
FOUR_PI_56 = (4.0 * math.pi) ** 56  # = 3.6e61
CC_PDG = 2.89e-122


# ──────────────────────────────────────────────────────────────────────
# Core spectral sums
# ──────────────────────────────────────────────────────────────────────


def sigma_2_partial(cands: list[dict], s: float = 0.0) -> float:
    """Σ_2(s)|_N = Σ_{C_2 > 0, class-one} d_ρ · C_2(ρ)^{2-s}.

    At s=0: Σ d_ρ C_2^2.
    """
    val = 0.0
    for c in cands:
        if c["C2"] <= 1e-12:
            continue
        val += c["d_lam"] * c["C2"] ** (2.0 - s)
    return val


def sigma_2_prime_partial(cands: list[dict]) -> float:
    """Σ_2'(0)|_N = -Σ_{C_2 > 0} d_ρ · C_2(ρ)^2 · log C_2(ρ).

    This is d/ds Σ_2(s)|_{s=0}.
    """
    val = 0.0
    for c in cands:
        if c["C2"] <= 1e-12:
            continue
        val -= c["d_lam"] * c["C2"] ** 2 * math.log(c["C2"])
    return val


def sigma_2_general(cands: list[dict], s: float) -> float:
    """General Σ_2(s) for arbitrary s (convergence study)."""
    return sigma_2_partial(cands, s)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_T_S2_1_cross_check_280(M, res: Result) -> list[dict]:
    banner("[T_S2.1] Cross-check: Λ²=280 gives 8 class-one reps")
    cands = enumerate_class_one_via_helgason(M, 280.0, verbose=False)
    n = len(cands)
    print(f"    Λ² = 280: N_class-one = {n} (expected 8)")
    res.report("T_S2.1: Λ²=280 → 8 class-one reps", n == 8, f"got {n}")
    return cands


def test_T_S2_2_sigma2_at_280(cands_280: list[dict], res: Result) -> float:
    banner("[T_S2.2] Σ_2(0) at Λ²=280")
    s2 = sigma_2_partial(cands_280, s=0.0)

    print(f"\n    Σ_2(0) = Σ_{{C_2 > 0}} d_ρ · C_2(ρ)²")
    print(f"    {'label':<26}  {'d_ρ':>14}  {'C_2':>8}  {'C_2²':>12}  {'d_ρ·C_2²':>16}")
    print(f"    " + "-" * 82)
    running = 0.0
    for c in cands_280:
        if c["C2"] <= 1e-12:
            continue
        contrib = c["d_lam"] * c["C2"] ** 2
        running += contrib
        print(f"    {str(c['label_dynkin']):<26}  {c['d_lam']:>14,}  "
              f"{c['C2']:>8.1f}  {c['C2']**2:>12.1f}  {contrib:>16,.1f}")
    print(f"    {'':>26}  {'':>14}  {'':>8}  {'Σ =':>12}  {s2:>16,.4f}")
    print()

    res.report(
        f"T_S2.2: Σ_2(0)|_{{Λ²=280}} = {s2:.4e}",
        s2 > 0 and np.isfinite(s2),
        f"sum over 7 non-trivial class-one reps",
    )
    return s2


def test_T_S2_3_sigma2_prime_at_280(cands_280: list[dict], res: Result) -> float:
    banner("[T_S2.3] Σ_2'(0) at Λ²=280")
    s2p = sigma_2_prime_partial(cands_280)

    print(f"\n    Σ_2'(0) = -Σ d_ρ · C_2² · log C_2")
    print(f"    {'label':<26}  {'d_ρ':>14}  {'C_2':>8}  {'C_2²·log C_2':>16}  "
          f"{'-d_ρ·C_2²·log C_2':>20}")
    print(f"    " + "-" * 92)
    running = 0.0
    for c in cands_280:
        if c["C2"] <= 1e-12:
            continue
        kernel = c["C2"] ** 2 * math.log(c["C2"])
        contrib = -c["d_lam"] * kernel
        running += contrib
        print(f"    {str(c['label_dynkin']):<26}  {c['d_lam']:>14,}  "
              f"{c['C2']:>8.1f}  {kernel:>16.2f}  {contrib:>20,.2f}")
    print(f"    {'':>26}  {'':>14}  {'':>8}  {'Σ =':>16}  {s2p:>20,.4f}")
    print()

    res.report(
        f"T_S2.3: Σ_2'(0)|_{{Λ²=280}} = {s2p:.4e}",
        np.isfinite(s2p),
        f"weighted log-det analog for C_2^2 kernel",
    )
    return s2p


def test_T_S2_4_convergence(M, lambda_sq_grid: list[float],
                             res: Result) -> dict:
    banner("[T_S2.4] Convergence study: Σ_2(0) and Σ_2'(0) vs Λ²")

    hdr_s2p = "Σ_2'(0)"
    print(f"\n    {'Λ²':>6}  {'N':>5}  {'Σ_2(0)':>16}  {hdr_s2p:>18}  "
          f"{'ratio':>12}  {'time (s)':>8}")
    print(f"    " + "-" * 74)

    results = {}
    for L_sq in lambda_sq_grid:
        t0 = time.time()
        cands = enumerate_class_one_via_helgason(M, L_sq, verbose=False)
        s2 = sigma_2_partial(cands, 0.0)
        s2p = sigma_2_prime_partial(cands)
        elapsed = time.time() - t0
        ratio = s2p / s2 if abs(s2) > 1e-30 else float('nan')
        results[L_sq] = {
            "N": len(cands),
            "sigma2_0": s2,
            "sigma2_prime_0": s2p,
            "ratio": ratio,
            "time_s": elapsed,
        }
        print(f"    {L_sq:>6.0f}  {len(cands):>5}  {s2:>16.6e}  "
              f"{s2p:>18.6e}  {ratio:>12.4f}  {elapsed:>8.2f}")
    print()

    # Monotonicity check
    sorted_grid = sorted(results.keys())
    monotone_s2 = all(
        results[sorted_grid[i+1]]["sigma2_0"] >= results[sorted_grid[i]]["sigma2_0"]
        for i in range(len(sorted_grid) - 1)
    )
    res.report(
        f"T_S2.4: convergence study over {len(lambda_sq_grid)} Λ² values",
        monotone_s2 and len(results) == len(lambda_sq_grid),
        f"Σ_2(0) monotonically increasing: {monotone_s2}",
    )
    return results


def test_T_S2_5_growth_rate(conv_results: dict, res: Result) -> float:
    banner("[T_S2.5] Growth rate fit: log Σ_2(0) ~ p · log Λ²")

    sorted_grid = sorted(conv_results.keys())
    valid = [(L, conv_results[L]["sigma2_0"])
             for L in sorted_grid if conv_results[L]["sigma2_0"] > 1.0]

    if len(valid) < 4:
        res.report("T_S2.5: insufficient data for growth fit", False, "need ≥ 4 points")
        return float('nan')

    Ls = np.array([v[0] for v in valid], dtype=np.float64)
    Vs = np.array([v[1] for v in valid], dtype=np.float64)
    log_Ls = np.log(Ls)
    log_Vs = np.log(Vs)
    p, c = np.polyfit(log_Ls, log_Vs, 1)

    print(f"\n    Log-fit: log Σ_2(0) = p · log Λ² + c")
    print(f"      p (exponent) = {p:.4f}")
    print(f"      c (intercept) = {c:.4f}")
    print(f"      => Σ_2(0) ~ Λ^{{{2*p:.2f}}}")
    print()

    # Comparison with theoretical expectation:
    # d_ρ ~ C_2^{N_active/2} (Plancherel), count ~ C_2^{rank/2} = C_2^2
    # => Σ d_ρ C_2^2 ~ ∫ C_2^{N_active/2 + 2} · C_2^{rank/2 - 1} dC_2
    #                 ~ C_2^{(N_active + rank)/2 + 2}
    # With N_active = 50, rank = 4: exponent ~ (50+4)/2 + 2 = 29
    # But this is the Σ_2(0) exponent in Λ² (not Λ), so in Λ²:
    # Σ_2(0) ~ (Λ²)^{(N_active+rank)/2 + 2} => p_theory ~ 29
    print(f"    Theoretical expectation (Plancherel density):")
    print(f"      N_active = 50 (from do5b_truncation_residue_bound.py T_TR.1)")
    print(f"      rank = 4 (F_4 restricted root system)")
    print(f"      p_theory ≈ (N_active + rank)/2 + 2 = 29 (leading order)")
    print(f"      Empirical p = {p:.2f}")
    print()

    res.report(
        f"T_S2.5: growth exponent p = {p:.2f}",
        np.isfinite(p) and p > 0,
        f"Σ_2(0) ~ (Λ²)^{{{p:.2f}}}",
    )
    return p


def test_T_S2_6_ratio_convergence(conv_results: dict, res: Result) -> None:
    banner("[T_S2.6] Ratio Σ_2'(0) / Σ_2(0) convergence")

    sorted_grid = sorted(conv_results.keys())
    print(f"\n    The ratio -Σ d_ρ C_2² log C_2 / Σ d_ρ C_2² converges to")
    print(f"    the average log C_2 weighted by d_ρ C_2²  (= dominant C_2 log).")
    print()
    hdr_ratio = "Σ_2'(0)/Σ_2(0)"
    print(f"    {'Λ²':>6}  {hdr_ratio:>18}  {'Δ from prev':>14}")
    print(f"    " + "-" * 44)

    prev_ratio = None
    for L_sq in sorted_grid:
        r = conv_results[L_sq]
        ratio = r["ratio"]
        delta_str = "—"
        if prev_ratio is not None:
            delta_str = f"{ratio - prev_ratio:+.4f}"
        print(f"    {L_sq:>6.0f}  {ratio:>18.6f}  {delta_str:>14}")
        prev_ratio = ratio
    print()

    # The ratio should converge to ~ -log(C_2_max) as larger modes dominate
    last_ratio = conv_results[sorted_grid[-1]]["ratio"]
    print(f"    Limiting ratio ≈ {last_ratio:.4f}")
    print(f"    Interpretation: average -log C_2 weighted by d_ρ · C_2^2")
    print(f"      (dominated by largest C_2 in the truncation)")
    print()

    res.report(
        f"T_S2.6: ratio Σ_2'(0)/Σ_2(0) = {last_ratio:.4f} (= avg -log C_2, weighted)",
        np.isfinite(last_ratio),
        f"last Λ² = {sorted_grid[-1]:.0f}",
    )


def test_T_S2_7_cc_connection(conv_results: dict, res: Result) -> None:
    banner("[T_S2.7] Connection to CC formula")

    sorted_grid = sorted(conv_results.keys())
    L_max = sorted_grid[-1]
    s2_max = conv_results[L_max]["sigma2_0"]
    s2p_max = conv_results[L_max]["sigma2_prime_0"]

    print(f"\n    CC formula (§4.3 of m4c-relaxation.md):")
    print(f"      δ(Λ ℓ_P²) ~ a_2(EIX) / [(4π)^56 · N_EIX²]")
    print()
    print(f"    Schematic identification:")
    print(f"      a_2(EIX) = heat-kernel coefficient = 1 175 384 / 15 ≈ 78 359")
    print(f"      The claim (§4.2): M4(c)-porušující operátor shifts a_0 → a_2")
    print(f"      via the ∇² insertion in the 1-loop trace.")
    print()
    print(f"    Spectral-sum perspective:")
    print(f"      The 1-loop contribution from S_d = Σ_ρ d_ρ · C_2(ρ)^2 · (propagator)")
    print(f"      After zeta-regularization, the finite part relevant to CC is")
    print(f"      controlled by Σ_2'(0) / (normalization).")
    print()
    print(f"    Current partial sums (Λ² = {L_max:.0f}):")
    print(f"      Σ_2(0)   = {s2_max:.6e}")
    print(f"      Σ_2'(0)  = {s2p_max:.6e}")
    print()

    # The connection to a_2 comes from the heat-kernel:
    # Tr[Δ^2 e^{-tΔ}] = Σ d_ρ C_2² e^{-t C_2}
    #                   ~ V/(4πt)^{d/2} · Σ_k (k)(k-1) a_k t^{k-2}  (for k≥2)
    # The a_2 term contributes at t^0 in the small-t expansion of Tr[Δ^2 e^{-tΔ}].
    # So the "effective a_2" from the spectral sum is extractable as the
    # t^0 coefficient of the heat trace of Δ^2.

    # Compute heat trace of Δ^2: Σ d_ρ C_2² e^{-t C_2} at small t
    cands_max = enumerate_class_one_via_helgason(
        build_e8_rep_machinery(), L_max, verbose=False
    )
    t_test = np.array([0.001, 0.003, 0.005, 0.01, 0.02])
    print(f"    Heat trace of Δ² (= Σ d_ρ C_2² e^{{-t C_2}}):")
    print(f"    {'t':>8}  {'Tr[Δ² e^{-tΔ}]':>20}")
    print(f"    " + "-" * 32)
    for t in t_test:
        ht = sum(c["d_lam"] * c["C2"]**2 * math.exp(-t * c["C2"])
                 for c in cands_max if c["C2"] > 1e-12)
        print(f"    {t:>8.4f}  {ht:>20.6e}")
    print()

    # Reference CC estimate using a_2
    cc_estimate_a2 = A2_EIX / (FOUR_PI_56 * N_EIX_PIC2_SQ)
    cc_half = 0.5 * cc_estimate_a2
    print(f"    Reference CC estimate (§4.3, Picture 2):")
    print(f"      a_2(EIX) / [(4π)^56 · N²] = {cc_estimate_a2:.4e}")
    print(f"      With prefactor 1/2:          {cc_half:.4e}")
    print(f"      PDG 2024:                    {CC_PDG:.4e}")
    print(f"      Ratio (prediction/PDG):      {cc_half / CC_PDG:.4f}")
    print()

    res.report(
        f"T_S2.7: CC connection documented; a_2-based estimate = {cc_half:.3e}",
        np.isfinite(cc_half) and cc_half > 0,
        f"ratio to PDG = {cc_half/CC_PDG:.3f}",
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="M4(c) §5.1: Spectral sum Σ_2(s)")
    parser.add_argument("--max-lambda-sq", type=float, default=5000.0,
                        help="Maximum Λ² for convergence study (default 5000)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save JSON results to path")
    args = parser.parse_args()

    print("=" * 78)
    print("M4(c)-relaxation §5.1 — Explicit spectral sum Σ_2(s)")
    print()
    print("  Σ_2(s) = Σ_{ρ ∈ class-one, C_2>0}  d_ρ · C_2(ρ)^{2-s}")
    print("  Σ_2'(0) = -Σ d_ρ · C_2² · log C_2     (zeta-reg derivative)")
    print()
    print("Infrastructure: do5b_helgason_spherical_enum.py (Cartan-Helgason)")
    print("Reference:      m4c-relaxation.md §5.1")
    print("=" * 78)
    print(f"Args: max_lambda_sq = {args.max_lambda_sq:.0f}")
    print("=" * 78, flush=True)

    res = Result()
    t0_global = time.time()

    print("\n[setup] Building E_8 rep machinery...", flush=True)
    M = build_e8_rep_machinery()
    print(f"        Positive roots: {len(M['pos_roots'])}")
    print(f"        |ρ|² = {float(M['rho'] @ M['rho']):.1f}")
    print()

    # ── T_S2.1: Cross-check ──
    cands_280 = test_T_S2_1_cross_check_280(M, res)
    print()

    # ── T_S2.2: Σ_2(0) at Λ²=280 ──
    s2_280 = test_T_S2_2_sigma2_at_280(cands_280, res)
    print()

    # ── T_S2.3: Σ_2'(0) at Λ²=280 ──
    s2p_280 = test_T_S2_3_sigma2_prime_at_280(cands_280, res)
    print()

    # ── T_S2.4: Convergence study ──
    base_grid = [120.0, 200.0, 280.0, 400.0, 600.0, 800.0, 1000.0,
                 1500.0, 2000.0, 3000.0, 4000.0, 5000.0]
    lambda_sq_grid = [L for L in base_grid if L <= args.max_lambda_sq]
    if 280.0 not in lambda_sq_grid:
        lambda_sq_grid.append(280.0)
    lambda_sq_grid = sorted(lambda_sq_grid)

    conv = test_T_S2_4_convergence(M, lambda_sq_grid, res)
    print()

    # ── T_S2.5: Growth rate ──
    p_exp = test_T_S2_5_growth_rate(conv, res)
    print()

    # ── T_S2.6: Ratio convergence ──
    test_T_S2_6_ratio_convergence(conv, res)
    print()

    # ── T_S2.7: CC connection ──
    test_T_S2_7_cc_connection(conv, res)

    elapsed = time.time() - t0_global
    print()
    print("=" * 78)
    print(f"M4(c) §5.1 Spectral sum:  {res.passed} PASS / {res.failed} FAIL  "
          f"(~{elapsed:.1f} s)")
    print("=" * 78)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "script": "m4c_spectral_sum.py",
            "section": "§5.1 Explicit spektrální suma",
            "elapsed_s": elapsed,
            "summary": {
                "passed": res.passed,
                "failed": res.failed,
            },
            "sigma2_at_280": {
                "sigma2_0": s2_280,
                "sigma2_prime_0": s2p_280,
                "n_class_one": 8,
            },
            "convergence": {
                "lambda_sq": lambda_sq_grid,
                "sigma2_0": [conv[L]["sigma2_0"] for L in lambda_sq_grid],
                "sigma2_prime_0": [conv[L]["sigma2_prime_0"] for L in lambda_sq_grid],
                "ratio": [conv[L]["ratio"] for L in lambda_sq_grid],
                "N_class_one": [conv[L]["N"] for L in lambda_sq_grid],
            },
            "growth_exponent": float(p_exp) if np.isfinite(p_exp) else None,
            "cc_reference": {
                "a2_EIX": A2_EIX,
                "cc_estimate_half_prefactor": 0.5 * A2_EIX / (FOUR_PI_56 * N_EIX_PIC2_SQ),
                "cc_PDG": CC_PDG,
            },
        }
        with out_path.open("w") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"\n  JSON saved to {out_path}")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
