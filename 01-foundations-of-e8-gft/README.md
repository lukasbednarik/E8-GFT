# Verification scripts — Foundations of an $E_8$ group field theory

Numerical verification scripts accompanying the paper
*Foundations of an $E_8$ group field theory: action uniqueness,
vacuum selection, and a four-dimensional algebraic substrate*.

## Usage

Each script is a standalone Python file importing the
[`e8sim`](../e8sim/) library from the sibling directory. Run
directly from this folder:

```bash
cd scripts/01-foundations-of-e8-gft
python3 e3_orbit_selection.py
```

Shared infrastructure (`bootstrap_repo_root`, `Result`, banner
printer) lives in [`_common.py`](_common.py).

## Script ↔ claim map

Each script verifies one or more claims in `sections/*.tex`.
Scripts marked `[Proven-num]` are the primary proof of the
corresponding claim; *additional* scripts provide independent
numerical checks beyond the analytical proof.

### Section 3 — Action uniqueness (Appendix A2)

| Script | Claim | Type |
|---|---|---|
| [`e0_algebra_base.py`](e0_algebra_base.py) | Lemma 3.1, 3.2 (struct. constants, Killing, $h^\vee = 30$, Casimir-degree spectrum) | additional |
| [`e1_action_form.py`](e1_action_form.py) | Theorem 3.1 (Hilbert series $\Sym^n\,\mathrm{adj}\,\mathfrak{e}_8$, dim of $(n,k)$-sectors) | additional |
| [`e1_verify_hypotheses.py`](e1_verify_hypotheses.py) | Theorem 3.1 (numerical certificates: antisym $f$, Jacobi, Killing, identically-zero sectors, $\mathcal{S}_a/\mathcal{S}_b/\mathcal{S}_c$) | additional |
| [`e1_open_points.py`](e1_open_points.py) | Appendix A2 (rank tests `test_4_2_classification`, `test_4_4_completeness`: $\dim^{(4,2)} = 3$, $\dim^{(4,4)} = 5$) | primary `[Proven-num]` |
| [`e1_o5_plethysm.py`](e1_o5_plethysm.py) | Appendix A2 (analytic plethysm upper bound $\dim^{(4,4)} \le 5$ via $f \times f$) | additional |

### Section 4 — BEC condensate

| Script | Claim | Type |
|---|---|---|
| [`e2_bec_phase.py`](e2_bec_phase.py) | Theorem 4.x (BEC vacuum existence, $r_*^2 = -c_2/(2c_4)$, $V_{\mathrm{eff}} = -c_2^2/(4c_4)$), Lemma (Goldstone count), Proposition (radial Higgs mass) | additional |

### Section 6 — Emergent spacetime (Wolf space EIX)

#### §6.4 — Algebraic four-dimensional sector

| Script | Claim | Type |
|---|---|---|
| [`e3_orbit_selection.py`](e3_orbit_selection.py) | Lemma (rank EIX $= 4$), Corollary (number of maximal antichains $= 630$) | primary `[Proven-num]` |

#### §6.5–6.7 — Sub-claims $(\alpha)$, $(\gamma)$, $(\delta)$ of the 4D hypothesis

| Script | Claim | Type |
|---|---|---|
| [`k3b_volovik_signature.py`](k3b_volovik_signature.py) | Slow-mode $K^{\mu\nu} = (1/2)\delta^{\mu\nu}$, signature $(+,+,+,+)$ on $\mathfrak{a}$ | primary `[Proven-num]` |
| [`k3b_c1_os_reflection_positivity.py`](k3b_c1_os_reflection_positivity.py) | Proposition (leading-Gaussian OS reflection positivity) | primary `[Proven-num]` |
| [`k3b_os_b_full_interacting.py`](k3b_os_b_full_interacting.py) | Proposition (full interacting OS RP at $\mathcal{D}_{\mathrm{stab}}$ interior point) | primary `[Proven-num, $\mathcal{D}_{\mathrm{stab}}$ interior]` |
| [`e3_antichain_full_sweep.py`](e3_antichain_full_sweep.py) | Sweep over all 630 maximal antichains $\Delta(\mathfrak{m}_{\mathrm{EIX}}^+)$ for $(\alpha)$, $(\gamma)$ | primary `[Proven-num]` |
| [`r4_eix_v_ind_camporesi_higuchi.py`](r4_eix_v_ind_camporesi_higuchi.py) | Proposition (Camporesi–Higuchi: $\mathcal{V}_{\mathrm{ind}}^{(\mathrm{EIX})} = 432/3 = 144$, leading + BV-BRST sub-leading) | primary `[Proven-num, leading + sub-leading]` |
| [`do5b_eix_log_determinant.py`](do5b_eix_log_determinant.py) | BV-BRST sub-leading input $c_H''^{(\mathrm{QK})} = -2\pi^2$ via Sp(1)-ghost factor | primary `[Proven-mat, BV-BRST]` |

#### §6.8 — Topological consistency

| Script | Claim | Type |
|---|---|---|
| [`e5_topology.py`](e5_topology.py) | Theorem (Lower homotopy of EIX): $\pi_0 = \pi_1 = \pi_2(\mathrm{EIX}) = 0$ | primary `[Proven-num]` |

## Shared infrastructure

| File | Description |
|---|---|
| [`_common.py`](_common.py) | Bootstraps repo path (exposes `e8sim` on `sys.path`), `Result` PASS/FAIL/SKIP reporter, banner printer. |
