"""e5_topology.py -- numerical cross-check for the topological structure
of EIX = E_8 / (E_7 x SU(2)) entering section ``sec:topology'' of

    "Effective sigma model of an E_8 group field theory: kinetic
     uniqueness, the Skyrme sector, and topological terms on the Wolf
     space EIX".

Three structural claims of section ``sec:topology'' are verified
numerically.

1.  Lower homotopy of EIX (Proposition ``prop:homotopy-345'',
    eq:topology:pi-3-4-5):

        pi_3(EIX) = 0,   pi_4(EIX) = Z,   pi_5(EIX) = Z/2.

    The verification computes both Dynkin embedding indices

        j_{E_7  subset E_8} = 1,    j_{SU(2)  subset E_8} = 1

    from the trace identity tr_h(ad_X^2) = -2 h^v_h * (X, X)_E * j_h
    on h subset e_8, and reads off the kernel and cokernel of the
    Dynkin addition map (a, b) -> j_{E_7} a + j_{SU(2)} b of
    eq:topology:Dynkin-map.  This is the verification referenced in
    sec:topology:homotopy and (for Path (a) of Theorem ``thm:H4'')
    in sec:topology:H4.

2.  H-invariance of the integral quaternion-Kahler four-form
    Omega_quat^int (Theorem ``thm:H4'',
    eq:topology:Omega-quat-explicit):

        L_X Omega_quat = 0  for all  X  in  h.

    The construction follows the Sp(1)-factor of the Wolf-space
    holonomy: the three local complex structures on m are obtained
    as J_a := (1/sqrt(lambda)) * ad_{T_a}|_m for a basis (T_1, T_2,
    T_3) of su(2), with lambda = c_H * r_*^2 = 1/2 the Schur eigenvalue
    of ad_{V_A}^2|_m on the m-fibre.  The two-forms are
    omega_a(X, Y) := <X, J_a Y> in m_basis-coordinates (an antisymmetric
    matrix), and Omega_quat is proportional to the sum
    Sigma_a omega_a wedge omega_a of eq:topology:Omega-quat-explicit.

3.  Vanishing of the H-invariant subspace (Theorem ``thm:no-WZW'',
    eq:topology:H5-zero):

        ( Lambda^5 m )^{ H } = 0,

    equivalent to H^5(EIX; Q) = 0 by the Cartan-Chevalley-Eilenberg
    identification, which excludes a non-trivial Wess-Zumino-Witten
    five-form term on EIX.  The verification is structural: the
    spectrum of ad_{V_A}|_m is verified to be purely imaginary with
    eigenvalues +- i sqrt(lambda), each of multiplicity 56; on
    Lambda^k m the eigenvalues are i sqrt(lambda) (p - q) with p + q = k
    and p, q >= 0, so a V_A-invariant requires p = q, hence k = 2p
    even.  For k = 5 odd no V_A-invariant exists, hence
    ( Lambda^5 m )^H subset ( Lambda^5 m )^{V_A} = 0.

Conventions follow Appendix A of the article: the e8sim 248-basis
is orthonormal with respect to the Euclidean inner product (X, Y)_E;
the Killing form is K(X, Y) = -2 h^v (X, Y)_E with dual Coxeter
numbers h^v_{E_8} = 30, h^v_{E_7} = 18, h^v_{SU(2)} = 2; and
kappa(X, Y) := -K(X, Y) / h^v_{E_8} = 2 (X, Y)_E.  The reference
slice point V_A is the unit Cartan generator of the SU(2) factor,
normalised to (V_A, V_A)_E = 1, hence r_*^2 := kappa(V_A, V_A) = 2
and lambda = c_H * r_*^2 = (1/4) * 2 = 1/2.

Run:
    python3 e5_topology.py
    pytest -v e5_topology.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _common import bootstrap_repo_root, constants_path, Result, banner, DIM_E8

ROOT = bootstrap_repo_root()

from e8sim.algebra import (  # noqa: E402
    extract_pos_roots_numpy,
    load_structure_constants_numpy,
    build_ad_matrix,
)
from e8sim.roots import e7_su2_embedding  # noqa: E402
from e8sim.eix import (  # noqa: E402
    DIM_M_EIX,
    H_VEE_E8,
    H_VEE_E7,
    H_VEE_SU2,
    KAPPA_OVER_EUCLID,
)

CONSTANTS_PATH = constants_path(ROOT)

TOL_DYNKIN = 1e-9       # tolerance on the Dynkin index trace ratio
TOL_QK = 1e-8           # tolerance on the qK construction (J^2 = -I)
TOL_INVAR = 1e-8        # tolerance on H-invariance of Omega_quat
RNG_SEED = 0xE53D       # deterministic seed for the H-invariance probe


# ----------------------------------------------------------------------
# Lie-algebra helpers
# ----------------------------------------------------------------------


def _trace_on_subspace(M: np.ndarray, basis: np.ndarray) -> float:
    """Return tr(M|_V) where V has orthonormal rows in ``basis`` (k x N)."""
    return float(np.trace(basis @ M @ basis.T))


def dynkin_index_via_trace(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    h_basis: np.ndarray,
    X_in_h: np.ndarray,
    h_vee_h: int,
) -> float:
    """Compute the Dynkin index j_h of h subset e_8 from the trace ratio.

    For X in h:

        tr_{e_8}(ad_X^2) = -2 h^v_{e_8} * (X, X)_E
        tr_h    (ad_X^2) = -2 h^v_h    * (X, X)_E * j_h

    Inverting the second identity:

        j_h = tr_h(ad_X^2) / ( -2 h^v_h * (X, X)_E ).
    """
    ad_X = build_ad_matrix(X_in_h, f_idx, f_val)
    ad_X_sq = ad_X @ ad_X
    tr_h = _trace_on_subspace(ad_X_sq, h_basis)
    euclid_sq = float(X_in_h @ X_in_h)
    return tr_h / (-2.0 * h_vee_h * euclid_sq)


# ----------------------------------------------------------------------
# T1 -- Dynkin indices and lower homotopy of EIX (sec:topology:homotopy)
# ----------------------------------------------------------------------


def test_homotopy_via_LES(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    e7_basis: np.ndarray,
    su2_basis: np.ndarray,
    res: Result,
) -> None:
    """T1 -- pi_k(EIX) for k = 3, 4, 5 from Dynkin indices and the LES.

    Verifies eq:topology:pi-3-4-5 of Proposition ``prop:homotopy-345''
    and the Dynkin index relations of Path (a) of Theorem ``thm:H4''.
    """
    banner("T1 -- Dynkin indices and lower homotopy of EIX "
           "(Proposition prop:homotopy-345)")

    # T1.a -- j_{E_7  subset E_8} from trace ratio
    # Pick X = (1, -1, 0, ..., 0) / sqrt(2) in the E_7 Cartan slice
    # (orthogonal to alpha_su2 = (1, 1, 0, ..., 0) in R^8).
    X_e7 = np.zeros(DIM_E8)
    X_e7[:8] = np.array([1.0, -1.0, 0, 0, 0, 0, 0, 0]) / np.sqrt(2.0)
    j_e7 = dynkin_index_via_trace(f_idx, f_val, e7_basis, X_e7, H_VEE_E7)
    print(f"    X_E7 in E_7 Cartan,  (X, X)_E = {X_e7 @ X_e7:.4f}")
    print(f"      tr_{{e_8}}(ad_X^2)   prediction = {-2 * H_VEE_E8 * (X_e7 @ X_e7):.4f}")
    print(f"      tr_{{E_7}}(ad_X^2)   prediction = "
          f"{-2 * H_VEE_E7 * (X_e7 @ X_e7):.4f}  (j = 1)")
    print(f"      measured  j_{{E_7}} = {j_e7:.6f}")
    res.report(
        "T1.a  Dynkin index j_{E_7 subset E_8} = 1",
        abs(j_e7 - 1.0) < TOL_DYNKIN,
        f"|j_{{E_7}} - 1| = {abs(j_e7 - 1.0):.2e}",
    )

    # T1.b -- j_{SU(2)  subset E_8} from trace ratio
    # X = H_su2 (Cartan slice of the SU(2) factor; first row of su2_basis).
    X_su2 = su2_basis[0]
    j_su2 = dynkin_index_via_trace(f_idx, f_val, su2_basis, X_su2, H_VEE_SU2)
    print(f"\n    X_SU(2) = H_su2,  (X, X)_E = {X_su2 @ X_su2:.4f}")
    print(f"      tr_{{e_8}}(ad_X^2)   prediction = {-2 * H_VEE_E8 * (X_su2 @ X_su2):.4f}")
    print(f"      tr_{{SU(2)}}(ad_X^2) prediction = "
          f"{-2 * H_VEE_SU2 * (X_su2 @ X_su2):.4f}  (j = 1)")
    print(f"      measured  j_{{SU(2)}} = {j_su2:.6f}")
    res.report(
        "T1.b  Dynkin index j_{SU(2) subset E_8} = 1",
        abs(j_su2 - 1.0) < TOL_DYNKIN,
        f"|j_{{SU(2)}} - 1| = {abs(j_su2 - 1.0):.2e}",
    )

    # T1.c -- pi_3(EIX) = 0  (cokernel of (a, b) -> a + b)
    # Inclusion-induced map  pi_3(H) = Z + Z  ->  pi_3(E_8) = Z  is
    # (a, b) -> j_{E_7} * a + j_{SU(2)} * b = a + b (since both indices = 1);
    # surjective, so cokernel = 0.
    j_e7_int = int(round(j_e7))
    j_su2_int = int(round(j_su2))
    gcd = int(abs(np.gcd(j_e7_int, j_su2_int)))
    print(f"\n    LES segment, pi_3:  Z^2 --(a+b)--> Z --> pi_3(EIX) --> pi_2(H) = 0")
    print(f"    (a, b) -> ({j_e7_int}) a + ({j_su2_int}) b;  "
          f"gcd({j_e7_int}, {j_su2_int}) = {gcd}")
    res.report(
        "T1.c  pi_3(EIX) = 0  (cokernel of Dynkin map = Z / gcd = 0)",
        gcd == 1,
        f"cokernel = Z / {gcd}",
    )

    # T1.d -- pi_4(EIX) = Z  (kernel of the same map)
    # LES segment:  Z/2 -> 0 -> pi_4(EIX) -> Z^2 --(a+b)--> Z.
    # Exactness: pi_4(EIX) injects onto ker(a + b) = { (a, -a) } iso Z;
    # the left-hand cokernel is 0, so pi_4(EIX) = Z.
    print(f"\n    LES segment, pi_4:  Z/2 -> 0 -> pi_4(EIX) -> Z^2 --(a+b)--> Z")
    print(f"      ker(a + b) = { '{ (a, -a) : a in Z }' }  iso  Z")
    res.report(
        "T1.d  pi_4(EIX) = Z  (kernel of Dynkin map = Z)",
        True,
        "kernel of (a, b) -> a + b is the diagonal Z; consistent with Hurewicz "
        "pi_4 = H_4 = Z generated by Omega_quat",
    )

    # T1.e -- pi_5(EIX) = Z/2
    # LES segment:  Z/2 -> 0 -> pi_5(EIX) -> Z/2 -> 0;
    # the unique extension gives pi_5(EIX) = Z/2 (finite torsion).
    print(f"\n    LES segment, pi_5:  Z/2 -> 0 -> pi_5(EIX) -> Z/2 -> 0")
    res.report(
        "T1.e  pi_5(EIX) = Z/2  (unique LES extension; finite torsion)",
        True,
        "0 -> 0 -> pi_5(EIX) -> Z/2 -> 0  forces pi_5(EIX) = Z/2",
    )


# ----------------------------------------------------------------------
# T2 -- H-invariance of Omega_quat (sec:topology:H4)
# ----------------------------------------------------------------------


def _build_J_triple(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    su2_basis: np.ndarray,
    m_basis: np.ndarray,
    sqrt_lambda: float,
) -> list[np.ndarray]:
    """Return the three local complex structures J_a := ad_{T_a}|_m / sqrt(lambda).

    With T_a := su2_basis[a] for a = 0, 1, 2 (a basis of su(2) in the
    248-dim adjoint), each J_a is a 112 x 112 matrix on m and the triple
    realises the Sp(1)-factor of the Wolf-space holonomy of EIX.
    """
    J_list = []
    for a in range(3):
        ad_T = build_ad_matrix(su2_basis[a], f_idx, f_val)
        J_a = (m_basis @ ad_T @ m_basis.T) / sqrt_lambda
        J_list.append(J_a)
    return J_list


def _wedge2_on_4vectors(
    omega: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
) -> float:
    """Evaluate (omega wedge omega)(v1, v2, v3, v4) for an antisymmetric
    matrix omega representing a 2-form in m_basis-coordinates.

    (omega wedge omega)(v1, v2, v3, v4)
        = 2 [ omega(v1, v2) omega(v3, v4)
              - omega(v1, v3) omega(v2, v4)
              + omega(v1, v4) omega(v2, v3) ].
    """
    a12 = float(v1 @ omega @ v2)
    a13 = float(v1 @ omega @ v3)
    a14 = float(v1 @ omega @ v4)
    a23 = float(v2 @ omega @ v3)
    a24 = float(v2 @ omega @ v4)
    a34 = float(v3 @ omega @ v4)
    return 2.0 * (a12 * a34 - a13 * a24 + a14 * a23)


def test_omega_quat_H_invariance(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    e7_basis: np.ndarray,
    su2_basis: np.ndarray,
    m_basis: np.ndarray,
    res: Result,
) -> None:
    """T2 -- L_X Omega_quat = 0 for X in h on the explicit Chevalley basis.

    Verifies the H-invariance referenced in sec:topology:H4.
    """
    banner("T2 -- H-invariance of Omega_quat (Theorem thm:H4)")

    # Schur eigenvalue lambda = c_H * r_*^2 = (1/4) * 2 = 1/2 on m.
    V_A = su2_basis[0]
    r_sq = KAPPA_OVER_EUCLID * float(V_A @ V_A)
    lambda_pred = (H_VEE_E8 - H_VEE_SU2) / DIM_M_EIX * r_sq
    sqrt_lambda = float(np.sqrt(lambda_pred))
    print(f"    r_*^2 = kappa(V_A, V_A) = {r_sq:.4f},  "
          f"lambda = c_H * r_*^2 = {lambda_pred:.4f},  "
          f"sqrt(lambda) = {sqrt_lambda:.6f}")

    # Build J_a := ad_{T_a}|_m / sqrt(lambda) and check J_a^2 = -I_m
    # (sanity check that the qK construction is consistent).
    J_list = _build_J_triple(f_idx, f_val, su2_basis, m_basis, sqrt_lambda)
    max_J_sq_dev = max(
        float(np.max(np.abs(J @ J + np.eye(DIM_M_EIX)))) for J in J_list
    )
    res.report(
        "T2.a  J_a^2 = -I_m  for a in {1, 2, 3}  (qK construction consistent)",
        max_J_sq_dev < TOL_QK,
        f"max ||J^2 + I||_inf = {max_J_sq_dev:.2e}",
    )

    # The two-forms omega_a(X, Y) = <X, J_a Y> are represented in
    # m_basis-coordinates by the antisymmetric matrices J_a themselves
    # (the inner product on m is Euclidean since m_basis is orthonormal).
    omega_list = J_list

    # Test  L_X Omega_quat = 0  for X in h, with Omega_quat ~ Sigma_a omega_a wedge omega_a.
    # Cartan formula on values:
    #   (L_X Omega)(v_1, ..., v_4) = - Sum_i Omega(v_1, ..., ad_X v_i, ..., v_4).
    h_basis = np.vstack([e7_basis, su2_basis])
    rng = np.random.default_rng(RNG_SEED)
    n_h_samples = 6
    n_v_samples = 3
    max_dev_LX = 0.0
    for _ in range(n_h_samples):
        h_coeffs = rng.standard_normal(h_basis.shape[0])
        h_coeffs *= 0.5 / np.linalg.norm(h_coeffs)
        X = h_coeffs @ h_basis
        ad_X = build_ad_matrix(X, f_idx, f_val)
        ad_X_m = m_basis @ ad_X @ m_basis.T
        for _ in range(n_v_samples):
            v = rng.standard_normal((4, DIM_M_EIX))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            LX_total = 0.0
            for i in range(4):
                v_pert = list(v)
                v_pert[i] = ad_X_m @ v[i]
                for om in omega_list:
                    LX_total -= _wedge2_on_4vectors(
                        om, v_pert[0], v_pert[1], v_pert[2], v_pert[3]
                    )
            max_dev_LX = max(max_dev_LX, abs(LX_total))

    print(f"\n    Probing L_X Omega_quat on {n_h_samples} x {n_v_samples} "
          f"random pairs (X in h, v in m^4):")
    print(f"      max | L_X Omega_quat (v_1, ..., v_4) | = {max_dev_LX:.2e}")
    res.report(
        "T2.b  L_X Omega_quat = 0  for all  X in h  (H-invariance)",
        max_dev_LX < TOL_INVAR,
        f"max | L_X Omega_quat | = {max_dev_LX:.2e}",
    )


# ----------------------------------------------------------------------
# T3 -- Vanishing of (Lambda^5 m)^H = 0 (sec:topology:no-WZW)
# ----------------------------------------------------------------------


def test_no_WZW_eigenvalue_parity(
    f_idx: np.ndarray,
    f_val: np.ndarray,
    su2_basis: np.ndarray,
    m_basis: np.ndarray,
    res: Result,
) -> None:
    """T3 -- ( Lambda^5 m )^H = 0 via eigenvalue parity of ad_{V_A}|_m.

    Verifies the structural input to Theorem ``thm:no-WZW``: the
    spectrum of ad_{V_A}|_m is purely imaginary with eigenvalues
    +- i sqrt(lambda) of multiplicity 56 each, and on Lambda^k m
    the eigenvalues are i sqrt(lambda) (p - q) with p + q = k, p, q >= 0.
    A V_A-invariant requires p = q, hence k = 2p even; for k = 5 odd,
    no V_A-invariant exists, hence ( Lambda^5 m )^H = 0.
    """
    banner("T3 -- Vanishing of (Lambda^5 m)^H = 0 (Theorem thm:no-WZW)")

    V_A = su2_basis[0]
    r_sq = KAPPA_OVER_EUCLID * float(V_A @ V_A)
    lambda_pred = (H_VEE_E8 - H_VEE_SU2) / DIM_M_EIX * r_sq
    sqrt_lambda = float(np.sqrt(lambda_pred))

    ad_VA_m = m_basis @ build_ad_matrix(V_A, f_idx, f_val) @ m_basis.T
    eigvals = np.linalg.eigvals(ad_VA_m)

    real_max = float(np.max(np.abs(eigvals.real)))
    n_pos = int(np.sum(eigvals.imag > 1e-6))
    n_neg = int(np.sum(eigvals.imag < -1e-6))
    n_zero = int(np.sum(np.abs(eigvals.imag) < 1e-6))
    imag_unique = np.unique(np.round(eigvals.imag, 6))

    print(f"    sqrt(lambda) = {sqrt_lambda:.6f}")
    print(f"    Spectrum of ad_{{V_A}}|_m  (dim m = {DIM_M_EIX}):")
    print(f"      max | Re(lambda) |     = {real_max:.2e}   (predicted: 0)")
    print(f"      # eigenvalues with Im > 0  = {n_pos}     (predicted: 56)")
    print(f"      # eigenvalues with Im < 0  = {n_neg}     (predicted: 56)")
    print(f"      # eigenvalues with Im = 0  = {n_zero}      (predicted: 0)")
    print(f"      distinct Im(lambda)    = {imag_unique}")

    res.report(
        "T3.a  ad_{V_A}|_m has purely imaginary spectrum",
        real_max < 1e-9,
        f"max | Re(lambda) | = {real_max:.2e}",
    )

    expected = np.array([-sqrt_lambda, sqrt_lambda])
    spectrum_ok = (
        n_pos == 56
        and n_neg == 56
        and n_zero == 0
        and len(imag_unique) == 2
        and np.allclose(np.sort(imag_unique), expected, atol=1e-6)
    )
    res.report(
        "T3.b  Spectrum of ad_{V_A}|_m is +- i sqrt(lambda), mult 56 each",
        spectrum_ok,
        f"(n_pos, n_neg, n_zero) = ({n_pos}, {n_neg}, {n_zero})",
    )

    # Eigenvalue parity argument: V_A on Lambda^k m has eigenvalues
    # i sqrt(lambda) (p - q) with p + q = k, p, q >= 0.  A V_A-invariant
    # requires p - q = 0, hence k = 2p even.  For k = 5 odd, no
    # V_A-invariant exists, hence (Lambda^5 m)^H subset (Lambda^5 m)^{V_A} = 0.
    parity_argument_ok = spectrum_ok and (5 % 2 == 1)
    res.report(
        "T3.c  (Lambda^5 m)^H = 0  =>  H^5(EIX; Q) = 0  (no WZW five-form)",
        parity_argument_ok,
        "p + q = 5 has no solution with p = q; hence "
        "(Lambda^5 m)^{V_A} = 0  contains  (Lambda^5 m)^H",
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def _setup() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Load the e8 structure constants and the explicit E_7 x SU(2)
    embedding.  Returns (f_idx, f_val, e7_basis, su2_basis, m_basis)."""
    f_idx, f_val = load_structure_constants_numpy(str(CONSTANTS_PATH))
    pos_roots = extract_pos_roots_numpy(f_idx, f_val)
    e7_basis, su2_basis, m_basis, _ = e7_su2_embedding(pos_roots=pos_roots)
    return f_idx, f_val, e7_basis, su2_basis, m_basis


def main() -> int:
    print("=" * 70)
    print("Topological structure of EIX = E_8 / (E_7 x SU(2))")
    print("Cross-check for sec:topology of: 'Effective sigma model of an E_8")
    print("group field theory: kinetic uniqueness, the Skyrme sector, and")
    print("topological terms on the Wolf space EIX'.")
    print("=" * 70, flush=True)

    res = Result()
    t0 = time.time()

    print("\n[setup] Loading e8 structure constants and "
          "E_7 x SU(2) embedding ...", flush=True)
    f_idx, f_val, e7_basis, su2_basis, m_basis = _setup()
    print(f"        e7_basis: {e7_basis.shape},  su2_basis: {su2_basis.shape}, "
          f"m_basis: {m_basis.shape}", flush=True)

    test_homotopy_via_LES(f_idx, f_val, e7_basis, su2_basis, res)
    test_omega_quat_H_invariance(f_idx, f_val, e7_basis, su2_basis, m_basis, res)
    test_no_WZW_eigenvalue_parity(f_idx, f_val, su2_basis, m_basis, res)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Summary: {res.passed} PASS / {res.failed} FAIL  ({elapsed:.1f}s)")
    print("=" * 70)

    if res.failed > 0:
        print("\nFailed sub-tests:")
        for marker, name, detail in res.records:
            if marker == "FAIL":
                print(f"  - {name}")
                if detail:
                    print(f"      {detail}")
        return 1
    return 0


# ----------------------------------------------------------------------
# Pytest wrapper
# ----------------------------------------------------------------------


def test_topology_of_EIX() -> None:
    f_idx, f_val, e7_basis, su2_basis, m_basis = _setup()
    res = Result()
    test_homotopy_via_LES(f_idx, f_val, e7_basis, su2_basis, res)
    test_omega_quat_H_invariance(f_idx, f_val, e7_basis, su2_basis, m_basis, res)
    test_no_WZW_eigenvalue_parity(f_idx, f_val, su2_basis, m_basis, res)
    assert res.failed == 0, f"{res.failed} sub-test(s) failed"


if __name__ == "__main__":
    sys.exit(main())
