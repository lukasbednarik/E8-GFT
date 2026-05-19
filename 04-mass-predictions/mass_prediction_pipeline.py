"""Step B+C+D: Fermion mass prediction pipeline.

Full mass formula:

  y_f = β₀ + α₅C₅ + α₁q + γ₁CΛ + θ₀s + θ₁C₅s + θ₂C₅²s + δ_GJ

where δ_GJ is the Georgi-Jarlskog correction for down-type fermions:
  quarks:   +J(gen)·ln(3)/2
  leptons:  -J(gen)·ln(3)/2
  up-type:  0
with J = {+1, -1, 0} for gen = {1, 2, 3}.

The GJ factor 3 = N_c is NOT a free parameter; it is the number of
colours determined by the SU(3) subgroup of SU(5) ⊂ SO(10) ⊂ E₆ ⊂ E₇.

Inputs (6):     m_t, m_u, m_c, m_b, m_d, m_s  (PDG → GUT RGE)
Ab initio (2):  γ₁ = 0.670,  GJ factor = N_c = 3
Predictions:    e, μ, τ  (3 genuine, ZERO free parameters in lepton sector)

CLI:
  python3 research/03-a-fermions/debug_plan/scripts/mass_prediction_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import Result, banner  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

Y_TREE = 0.194
V_H = 246.0  # GeV

# PDG 2024 pole masses [GeV]
PDG = {
    'u':  0.00216, 'd':  0.00467, 'e':  0.000511,
    'c':  1.27,    's':  0.0934,  'mu': 0.10566,
    't':  173.2,   'b':  4.18,    'tau': 1.777,
}

# GUT-scale running masses (SM 2-loop RGE from PDG → M_GUT ≈ 2e16 GeV)
# Ratios are more robust than absolute values.
# Using standard RGE running factors:
#   m_q(M_GUT) / m_q(m_Z) ≈ 0.4 (quarks, QCD running)
#   m_l(M_GUT) / m_l(m_Z) ≈ 1.0 (leptons, no QCD)
# Ref: Xing-Zhang-Zhou 2008 for RGE tables.
RGE_QUARK = 0.40    # average QCD running factor M_GUT/m_Z
RGE_LEPTON = 1.02   # negligible QED running

GUT_MASSES = {
    'u':  PDG['u']   * RGE_QUARK,
    'd':  PDG['d']   * RGE_QUARK,
    'e':  PDG['e']   * RGE_LEPTON,
    'c':  PDG['c']   * RGE_QUARK,
    's':  PDG['s']   * RGE_QUARK,
    'mu': PDG['mu']  * RGE_LEPTON,
    't':  PDG['t']   * RGE_QUARK,
    'b':  PDG['b']   * RGE_QUARK,
    'tau': PDG['tau'] * RGE_LEPTON,
}

# Profile constants from Krok A (variational solver + ab initio)
C_H = 202.4       # Derrick-invariant energy (solver, ≈ lit. 200.5)
GAMMA1 = 0.670    # ab initio from q^2 I_2(56) C_sin2G / Dq

# E7 instanton charge structure (from Step 1)
DQ_E7 = 2.0 / np.sqrt(6)   # 0.8165

# Generation quantum numbers
FERMIONS = {
    # name: (generation, type, C5, q_E7, Q_X)
    'u':   (1, 'up',   +1, +1/np.sqrt(6), -1.0),
    'd':   (1, 'down', +1, +1/np.sqrt(6), +3.0),
    'e':   (1, 'down', +1, +1/np.sqrt(6), +3.0),
    'c':   (2, 'up',    0, -1/np.sqrt(6), -1.0),
    's':   (2, 'down',  0, -1/np.sqrt(6), +3.0),
    'mu':  (2, 'down',  0, -1/np.sqrt(6), +3.0),
    't':   (3, 'up',   -1, -1/np.sqrt(6), -1.0),
    'b':   (3, 'down', -1, -1/np.sqrt(6), +3.0),
    'tau': (3, 'down', -1, -1/np.sqrt(6), +3.0),
}


# ═══════════════════════════════════════════════════════════════════
#  Krok B: Fix sigma-model parameters from (m_t, m_u)
# ═══════════════════════════════════════════════════════════════════

def fix_parameters():
    """Fix (β₀, α₅, α₁, θ₀, θ₁, θ₂) from 6 quark masses.

    System (6 quarks → 6 params, γ₁ ab initio, GJ from N_c=3):

      y_u = β₀ + α₅ + α₁/√6 - θ₀ - θ₁ - θ₂
      y_c = β₀       - α₁/√6 - θ₀
      y_t = β₀ - α₅ - α₁/√6 + γ₁CΛ - θ₀ + θ₁ - θ₂
      y_d = β₀ + α₅ + α₁/√6 + θ₀ + θ₁ + θ₂
      y_s = β₀       - α₁/√6 + θ₀
      y_b = β₀ - α₅ - α₁/√6 + γ₁CΛ + θ₀ - θ₁ + θ₂
    """
    CL = 0.548

    y = {k: np.log(GUT_MASSES[k]) for k in ('t', 'u', 'c', 'b', 'd', 's')}

    # (s)-(c):  y_s - y_c = 2θ₀
    theta0 = (y['s'] - y['c']) / 2.0

    # (t)-(b):  y_t - y_b = -2θ₀ + 2θ₁ - 2θ₂   ... (A)
    # (d)-(u):  y_d - y_u = +2θ₀ + 2θ₁ + 2θ₂   ... (B)
    A = y['t'] - y['b']       # = -2θ₀ + 2θ₁ - 2θ₂
    B = y['d'] - y['u']       # = +2θ₀ + 2θ₁ + 2θ₂

    theta1 = (A + B) / 4.0    # (A+B)/4 = θ₁
    theta2 = (B - A) / 4.0 - theta0   # (B-A)/4 = θ₀ + θ₂ → θ₂

    # (t)-(c):  y_t - y_c = -α₅ + γ₁CΛ + θ₁ - θ₂
    alpha5 = -(y['t'] - y['c']) + GAMMA1 * CL + theta1 - theta2

    # (u)-(c):  y_u - y_c = α₅ + 2α₁/√6 - θ₁ - θ₂
    alpha1 = (y['u'] - y['c'] - alpha5 + theta1 + theta2) * np.sqrt(6) / 2.0

    # β₀ from (c):  y_c = β₀ - α₁/√6 - θ₀
    beta0 = y['c'] + alpha1 / np.sqrt(6) + theta0

    return {
        'alpha5': alpha5, 'alpha1': alpha1,
        'theta0': theta0, 'theta1': theta1, 'theta2': theta2,
        'beta0': beta0, 'gamma1': GAMMA1,
        'C_Lambda_B1': CL, 'm_0': np.exp(beta0),
    }


# ═══════════════════════════════════════════════════════════════════
#  Krok C: Predict all fermion masses
# ═══════════════════════════════════════════════════════════════════

N_C = 3  # number of colours — determines GJ Clebsch factor

GJ_FACTOR = {1: +1, 2: -1, 3: 0}  # Georgi-Jarlskog quantum number J(gen)


def predict_masses(params):
    """Compute GUT-scale masses for all 9 fermions.

    Quarks: reproduced exactly from 6 inputs.
    Leptons: predicted from GJ correction (N_c=3, zero free params).
    """
    a5 = params['alpha5']
    a1 = params['alpha1']
    g1 = params['gamma1']
    t0 = params['theta0']
    t1 = params['theta1']
    t2 = params['theta2']
    b0 = params['beta0']
    CL = params['C_Lambda_B1']

    predictions = {}
    for name, (gen, ftype, C5, q_E7, Q_X) in FERMIONS.items():
        s = -1.0 if ftype == 'up' else +1.0
        is_lepton = name in ('e', 'mu', 'tau')

        y = b0 + a5 * C5 + a1 * q_E7
        y += t0 * s + t1 * C5 * s + t2 * C5 ** 2 * s

        if gen == 3:
            y += g1 * CL

        # Georgi-Jarlskog: modifies LEPTON Yukawa relative to quark
        if ftype == 'down' and is_lepton:
            J = GJ_FACTOR[gen]
            y -= J * np.log(N_C)

        predictions[name] = {
            'gen': gen, 'type': ftype,
            'm_pred_GUT': np.exp(y),
            'm_exp_GUT': GUT_MASSES[name],
            'C5': C5, 'q_E7': q_E7, 'Q_X': Q_X,
        }

    return predictions


def run_to_pole(predictions):
    """Convert GUT-scale predictions to pole masses (inverse RGE)."""
    for name, p in predictions.items():
        is_quark = name not in ('e', 'mu', 'tau')
        rge = RGE_QUARK if is_quark else RGE_LEPTON
        p['m_pred_pole'] = p['m_pred_GUT'] / rge
        p['m_exp_pole'] = PDG[name]


# ═══════════════════════════════════════════════════════════════════
#  Krok D: Comparison with PDG
# ═══════════════════════════════════════════════════════════════════

def compare_pdg(predictions, params):
    """Compute chi^2 and summary statistics."""
    chi2_log = 0.0
    n_pred = 0
    correct_ordering = True

    for name, p in predictions.items():
        ratio = p['m_pred_GUT'] / max(p['m_exp_GUT'], 1e-30)
        p['ratio'] = ratio
        p['log_ratio'] = np.log10(ratio)

        log_err = (np.log(p['m_pred_GUT']) - np.log(p['m_exp_GUT'])) ** 2
        # Use sigma = 0.5 in log-mass (generous uncertainty from RGE + model)
        p['chi2_contrib'] = log_err / 0.5 ** 2
        chi2_log += p['chi2_contrib']
        n_pred += 1

    # Check generation ordering
    for ftype_set in [('u', 'c', 't'), ('d', 's', 'b'), ('e', 'mu', 'tau')]:
        masses = [predictions[f]['m_pred_GUT'] for f in ftype_set]
        if not (masses[0] < masses[1] < masses[2]):
            correct_ordering = False

    return {
        'chi2_total': chi2_log,
        'chi2_per_dof': chi2_log / max(n_pred - 6, 1),
        'n_predictions': n_pred,
        'n_inputs': 6,
        'correct_ordering': correct_ordering,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    banner("Mass Prediction Pipeline — Kroky B+C+D")
    res = Result()

    # ── Krok B: Fix parameters ────────────────────────────────────
    banner("Krok B: Fix sigma-model parameters from (m_t, m_u)")
    params = fix_parameters()

    print(f"  Inputs (6 quark masses → 6 params):")
    for name in ['t', 'b', 'c', 's', 'u', 'd']:
        print(f"    m_{name}(GUT) = {GUT_MASSES[name]:.4e} GeV  [INPUT]")
    print(f"  Predictions (3 lepton masses, GJ from N_c={N_C}):")
    for name in ['tau', 'mu', 'e']:
        print(f"    m_{name}(GUT) = {GUT_MASSES[name]:.4e} GeV  [PREDICT]")

    print(f"\n  Derived coefficients:")
    print(f"    β₀     = {params['beta0']:+.4f}  "
          f"(m₀ = {params['m_0']:.4f} GeV)")
    print(f"    α₅     = {params['alpha5']:+.4f}  (topological sector)")
    print(f"    α₁     = {params['alpha1']:+.4f}  (E₇ instanton charge)")
    print(f"    γ₁     = {params['gamma1']:+.4f}  (ab initio, NOT fitted)")
    print(f"    θ₀     = {params['theta0']:+.4f}  (universal up/down)")
    print(f"    θ₁     = {params['theta1']:+.4f}  (linear cross-term C₅·s)")
    print(f"    θ₂     = {params['theta2']:+.4f}  (quadratic cross-term C₅²·s)")

    t = params
    print(f"\n  Effective θ_eff = θ₀ + θ₁C₅ + θ₂C₅²:")
    for gen, C5 in [(1, +1), (2, 0), (3, -1)]:
        te = t['theta0'] + t['theta1'] * C5 + t['theta2'] * C5 ** 2
        pairs = {1: ('d', 'u'), 2: ('s', 'c'), 3: ('b', 't')}
        d, u = pairs[gen]
        print(f"    Gen {gen} (C₅={C5:+d}): θ_eff={te:+.4f}  "
              f"→ m_{d}/m_{u} = {np.exp(2*te):.4f} "
              f"(exp: {GUT_MASSES[d]/GUT_MASSES[u]:.4f})")

    print(f"\n  GJ correction (down-type only, N_c={N_C}):")
    for gen in [1, 2, 3]:
        J = GJ_FACTOR[gen]
        print(f"    Gen {gen}: J={J:+d} → quark/lepton = {N_C**J:.1f}×")

    res.report("α₅ is negative (3rd gen heavier)",
               params['alpha5'] < 0,
               f"α₅ = {params['alpha5']:+.4f}")
    res.report("α₁ is finite and O(1)",
               0.01 < abs(params['alpha1']) < 100,
               f"α₁ = {params['alpha1']:+.4f}")

    # ── Krok C: Predict masses ────────────────────────────────────
    banner("Krok C: Predict 12 fermion masses")
    predictions = predict_masses(params)
    run_to_pole(predictions)

    # ── Krok D: PDG comparison ────────────────────────────────────
    banner("Krok D: PDG comparison")
    stats = compare_pdg(predictions, params)

    print(f"\n  GUT-scale comparison:")
    print(f"  {'Fermion':<6s} {'Gen':>3s} {'Type':<5s} "
          f"{'m_pred(GUT)':>12s} {'m_exp(GUT)':>12s} "
          f"{'ratio':>8s} {'log₁₀(r)':>9s}")
    print("  " + "-" * 68)

    order = ['u', 'd', 'e', 'c', 's', 'mu', 't', 'b', 'tau']
    for name in order:
        p = predictions[name]
        print(f"  {name:<6s} {p['gen']:>3d} {p['type']:<5s} "
              f"{p['m_pred_GUT']:12.4e} {p['m_exp_GUT']:12.4e} "
              f"{p['ratio']:8.2f} {p['log_ratio']:+9.3f}")

    print(f"\n  Pole-mass comparison:")
    print(f"  {'Fermion':<6s} {'m_pred':>10s} {'m_PDG':>10s} "
          f"{'ratio':>8s} {'Status':>8s}")
    print("  " + "-" * 50)
    for name in order:
        p = predictions[name]
        ratio = p['m_pred_pole'] / max(p['m_exp_pole'], 1e-30)
        within_factor = 'OK' if 0.1 < ratio < 10 else 'MISS'
        unit = 'MeV' if p['m_pred_pole'] < 1 else 'GeV'
        m_p = p['m_pred_pole'] * (1000 if unit == 'MeV' else 1)
        m_e = p['m_exp_pole'] * (1000 if unit == 'MeV' else 1)
        print(f"  {name:<6s} {m_p:10.3f} {m_e:10.3f} "
              f"{ratio:8.2f}× {within_factor:>8s}  {unit}")

    print(f"\n  χ² statistics (log-mass, σ=0.5):")
    print(f"    χ²_total     = {stats['chi2_total']:.2f}")
    print(f"    χ²/DOF       = {stats['chi2_per_dof']:.2f}  "
          f"(DOF = {stats['n_predictions']} - {stats['n_inputs']})")
    print(f"    Correct ordering: {stats['correct_ordering']}")

    res.report("Correct generational ordering (m_1 < m_2 < m_3)",
               stats['correct_ordering'],
               "u<c<t, d<s<b, e<mu<tau")

    n_within_10 = sum(1 for p in predictions.values()
                      if 0.1 < p['ratio'] < 10)
    res.report(f"At least 8/9 masses within factor 10 of PDG",
               n_within_10 >= 8,
               f"{n_within_10}/9 within factor 10")

    n_within_3 = sum(1 for p in predictions.values()
                     if 1/3 < p['ratio'] < 3)
    res.report(f"At least 4/9 masses within factor 3 of PDG",
               n_within_3 >= 4,
               f"{n_within_3}/9 within factor 3")

    # ── Summary ───────────────────────────────────────────────────
    banner("Pipeline summary")

    print(f"\n  INPUT:  6 quark masses (PDG pole → GUT RGE)")
    print(f"  AB INITIO: γ₁ = {GAMMA1}, GJ factor = N_c = {N_C}")
    print(f"  FIXED:  β₀={params['beta0']:+.3f}, α₅={params['alpha5']:+.3f}, "
          f"α₁={params['alpha1']:+.3f}")
    print(f"          θ₀={params['theta0']:+.3f}, θ₁={params['theta1']:+.3f}, "
          f"θ₂={params['theta2']:+.3f}")
    print(f"  PRED:   {stats['n_predictions'] - stats['n_inputs']} genuine predictions "
          f"(3 leptons, ZERO free params in lepton sector)")
    print(f"  RESULT: χ²/DOF = {stats['chi2_per_dof']:.2f}, "
          f"ordering {'✓' if stats['correct_ordering'] else '✗'}")
    print(f"          {n_within_3}/9 within ×3, {n_within_10}/9 within ×10")
    print()

    print(f"  {res.summary('Mass pipeline B+C+D')}\n")

    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
