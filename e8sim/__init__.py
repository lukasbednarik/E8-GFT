"""E8 Skyrmion simulation package."""

from .roots import (
    TRIALITY_SECTOR_MAP,
    E8_SIMPLE_ROOTS,
    E8_CARTAN_MATRIX_KNOWN,
    EIX_ALPHA_SU2,
    EIX_H_LAMBDA,
    generate_roots,
    cartan_matrix,
    dynkin_coords,
    positive_root_indices,
    root_lookup,
    classify_root_triality,
    extract_root_vectors,
    find_triality_roots,
    e7_su2_embedding,
)
from .algebra import (
    load_structure_constants,
    load_structure_constants_numpy,
    commutator,
    quadratic_casimir,
    bracket_vec_fast,
    extract_pos_roots_numpy,
    build_dense_f,
    build_ad_matrix,
    killing_form,
    adjoint_exp,
)
from .fields import (
    wolf_space_seed,
    wolf_space_seed_from_radial_profile,
    load_radial_profile,
    extract_su2_hopf,
    compute_derivatives,
    enforce_dirichlet,
    enforce_nested_boundaries,
)
from .loss import (
    compute_loss,
    compute_topological_charge,
    compute_topological_charge_diff,
    measure_charges,
)
from .dirac import run_dirac_scanner, run_dirac_scanner_multistate, overlap_matrix
from .roots import (
    is_strongly_orthogonal,
    build_compatibility_matrix,
    max_antichain_size,
    count_antichains_of_size,
    enumerate_antichains_of_size,
)
