"""E3 -- Maximal antichains in Delta(m^+) for the EIX orbit.

Numerical verification of the rank/antichain content of
``sections/06-emergent-spacetime.tex``:

  - Corollary~``cor:antichain-EIX``  max antichain in Delta(m_EIX^+) = 4
  - paragraph after Cor.            #(maximal antichains in EIX) = 630

By Suter's rank-antichain identity the maximal-antichain cardinality
in Delta(m^+) equals the rank of the corresponding symmetric space
G/K. We compute it directly on the explicit 240-root realisation of
E_8 (e8sim canonical basis) for the EIX orbit:

  T3.1   max antichain in Delta(m_EIX^+) = 4
  T3.4   count of size-4 antichains in EIX = 630

Reference: Suter 2004, ``Maximal abelian subalgebras of root
systems'', Adv.~Math. 190, 175.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, Result, banner

ROOT = bootstrap_repo_root()

from e8sim import generate_roots, EIX_ALPHA_SU2  # noqa: E402
from e8sim.roots import (  # noqa: E402
    build_compatibility_matrix,
    max_antichain_size,
    count_antichains_of_size,
)


# ----------------------------------------------------------------------
# Root system helpers
# ----------------------------------------------------------------------


def positive_roots() -> np.ndarray:
    """Return the 120 positive E_8 roots in R^8 (e8sim canonical ordering)."""
    all_roots = generate_roots()  # (240, 8)
    pos = []
    for r in all_roots:
        nz = np.nonzero(np.abs(r) > 0.01)[0]
        if len(nz) > 0 and r[nz[0]] > 0:
            pos.append(r)
    return np.array(pos)


def root_set(all_roots: np.ndarray) -> set[tuple]:
    """Hashable set of all roots for membership query (4-decimal rounding)."""
    return {tuple(np.round(r, 4)) for r in all_roots}


# ----------------------------------------------------------------------
# Orbit-specific m^+ extraction
# ----------------------------------------------------------------------


def m_plus_for_EIX(pos_roots: np.ndarray) -> np.ndarray:
    """Positive roots in m^+ for EIX = E_8 / (E_7 x SU(2)).

    h = e_7 + su(2) consists of roots either equal to +/- alpha_SU(2)
    or orthogonal to alpha_SU(2) (the E_7 part). The complement
    m_EIX^+ has 56 positive roots, equal to the dimension of the 56
    factor of the isotropy representation (56, 2) (cf.
    eq.~``isotropy-EIX'' in section 6).
    """
    alpha_su2 = EIX_ALPHA_SU2
    m_plus = []
    for r in pos_roots:
        if np.allclose(r, alpha_su2, atol=1e-6):
            continue
        if np.allclose(r, -alpha_su2, atol=1e-6):
            continue
        if abs(float(np.dot(r, alpha_su2))) < 1e-6:
            continue
        m_plus.append(r)
    return np.array(m_plus)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_T3_1_EIX(pos_roots: np.ndarray, all_roots: np.ndarray,
                   res: Result) -> None:
    banner("[T3.1] Max antichain in Delta(m_EIX^+) = 4 (rank of EIX)")

    m_plus = m_plus_for_EIX(pos_roots)
    res.report(
        "|Delta(m_EIX^+)| = 56 (= 56 positive roots in (56, 2))",
        len(m_plus) == 56,
        f"got {len(m_plus)}",
    )

    root_lookup = root_set(all_roots)
    print(f"    Building compatibility matrix ({len(m_plus)} x {len(m_plus)})...",
          flush=True)
    compat = build_compatibility_matrix(m_plus, root_lookup)

    print("    Searching max antichain...", flush=True)
    t0 = time.time()
    max_size = max_antichain_size(compat)
    dt = time.time() - t0
    res.report(
        "max antichain = 4 (Lemma rank-EIX, Corollary antichain-EIX)",
        max_size == 4,
        f"measured {max_size} (in {dt:.1f}s)",
    )


def test_T3_4_EIX_count630(pos_roots: np.ndarray, all_roots: np.ndarray,
                            res: Result) -> None:
    banner("[T3.4] Number of size-4 antichains in Delta(m_EIX^+) = 630")

    m_plus = m_plus_for_EIX(pos_roots)
    root_lookup = root_set(all_roots)
    compat = build_compatibility_matrix(m_plus, root_lookup)

    print("    Counting all antichains of size 4...", flush=True)
    t0 = time.time()
    count = count_antichains_of_size(compat, 4)
    dt = time.time() - t0

    # The 630-element set is the support of the e3_antichain_full_sweep.py
    # sweep verifying the per-antichain constancy of the (alpha)/(gamma)/
    # (epsilon) verdicts (paragraph after Corollary~``cor:antichain-EIX''
    # and the equivariance paragraph ``para:antichain-equivariance'' in section 6).
    res.report(
        "number of size-4 antichains in Delta(m_EIX^+) = 630",
        count == 630,
        f"measured {count} (in {dt:.1f}s)",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("E3 -- Maximal antichains in Delta(m^+) for the EIX orbit")
    print("Reference: sections/06-emergent-spacetime.tex,")
    print("           Corollary antichain-EIX (cardinality 4 and count 630)")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Generating 240 E_8 roots and extracting positive ones...",
          flush=True)
    all_roots = generate_roots()
    pos_roots = positive_roots()
    print(f"        |Phi| = {len(all_roots)}, |Phi^+| = {len(pos_roots)}",
          flush=True)

    test_T3_1_EIX(pos_roots, all_roots, res)
    test_T3_4_EIX_count630(pos_roots, all_roots, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"E3 summary: {res.passed} PASS / {res.failed} FAIL ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed == 0:
        print("\n[PASS] E3 -- antichain content of EIX verified numerically.")
        return 0
    else:
        print("\n[FAIL] E3 failed -- inspect the per-test diagnostics above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
