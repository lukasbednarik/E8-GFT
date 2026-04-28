"""PDG 2024 / CODATA 2022 / NuFit 5.2 phenomenological constants.

Unified frozen dataclass so that every script in ``debug_plan/scripts/``
references one source of truth for particle masses, mixing angles, and
cosmological scales.

Sources:
    * PDG 2024 — https://pdg.lbl.gov/
    * CODATA 2022
    * NuFit 5.2 (normal ordering, without SK atmospheric)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PDG2024:
    """Central values + uncertainties of the observables used in E9 tests."""

    # Lepton masses (MeV/c^2)
    m_e:        float = 0.51099895069
    sig_m_e:    float = 1.6e-10
    m_mu:       float = 105.6583755
    sig_m_mu:   float = 2.3e-6
    m_tau:      float = 1776.93        # PDG 2024 mean
    sig_m_tau:  float = 0.09

    # Quark masses (PDG 2024).  Convention:
    #   • Light quarks (u, d, s) — MS-bar at μ = 2 GeV (MeV/c^2);
    #   • Heavy quarks c, b — MS-bar at μ = m_q (MeV/c^2);
    #   • Top quark — direct (pole-mass) measurement (MeV/c^2).
    # Asymmetric PDG errors are symmetrised by taking the larger half.
    m_u_MSbar2GeV:    float = 2.16
    sig_m_u:          float = 0.49
    m_d_MSbar2GeV:    float = 4.67
    sig_m_d:          float = 0.48
    m_s_MSbar2GeV:    float = 93.4
    sig_m_s:          float = 8.6
    m_c_MSbar:        float = 1.27e3       # = 1.27 GeV
    sig_m_c:          float = 0.02e3
    m_b_MSbar:        float = 4.18e3       # = 4.18 GeV
    sig_m_b:          float = 0.03e3
    m_t_pole:         float = 172.69e3     # = 172.69 GeV (PDG 2024 direct)
    sig_m_t:          float = 0.30e3

    # Nucleon masses (MeV/c^2)
    m_p:        float = 938.27208816
    sig_m_p:    float = 2.9e-6
    m_n:        float = 939.5654205
    sig_m_n:    float = 5.4e-7

    # Weak mixing angle (sin^2 theta_W, MS-bar at M_Z)
    sin2_theta_W:     float = 0.23122
    sig_sin2_theta_W: float = 4.0e-5

    # PMNS angles in degrees (NuFit 5.2, normal ordering, no SK atm)
    theta12_PMNS_deg: float = 33.45
    sig_theta12:      float = 0.75
    theta23_PMNS_deg: float = 49.0
    sig_theta23:      float = 1.0
    theta13_PMNS_deg: float = 8.62
    sig_theta13:      float = 0.12
    delta_CP_PMNS_deg: float = 230.0
    sig_delta_CP:     float = 30.0

    # Unified couplings at M_Z (PDG 2024, MS-bar)
    M_Z_GeV:           float = 91.1876
    alpha_em_inv_MZ:   float = 127.95
    alpha_s_MZ:        float = 0.1180

    # Planck scale (CODATA 2022)
    M_P_GeV:           float = 1.220890e19
    M_P_reduced_GeV:   float = 2.435323e18

    # Higgs VEV (= electroweak scale; PDG 2024)
    v_EW_GeV:          float = 246.220       # = (sqrt(2) G_F)^{-1/2}

    # Cosmological constant (order of magnitude, PDG 2024)
    Lambda_lP2_obs:    float = 2.9e-122

    # Bagdonaite limit for proton-electron mass ratio drift
    mu_dot_over_mu_limit: float = 2.0e-16  # /yr (Bagdonaite 2013 NH3)


PDG = PDG2024()
