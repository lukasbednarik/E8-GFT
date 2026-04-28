"""E8 Lie algebra operations: commutators, structure constant loading,
Casimir invariants, and dense bracket helpers for testing/analysis."""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def quadratic_casimir(psi):
    """Pointwise quadratic Casimir C₂(x) = Σ_i ψ_i(x)² of a 248D field.

    For a field on the adjoint orbit, C₂ is preserved by the Lie update
    exp(ad_X)ψ.  Returns a scalar (spatial) tensor.
    """
    return torch.sum(psi ** 2, dim=0)


def load_structure_constants(path="e8_constants.pt", device="cpu"):
    """Load pre-computed structure constants from sparse tensor file.

    Returns (f_idx, f_val) where f_idx is (3, nnz) indices and f_val is (nnz,) values.
    """
    e8 = torch.load(path, map_location=device, weights_only=True).coalesce()
    return e8.indices(), e8.values()


def commutator_block(A, B, i_b, j_b, k_b, val_b, N, chunk_size):
    C = torch.zeros_like(A)
    val_b = val_b.to(A.dtype)
    num_nz = val_b.shape[0]
    for start in range(0, num_nz, chunk_size):
        end = min(start + chunk_size, num_nz)
        i_c = i_b[start:end]
        j_c = j_b[start:end]
        k_c = k_b[start:end]
        val_c = val_b[start:end].view(-1, 1, 1, 1)

        prod = A[i_c] * B[j_c] * val_c
        idx_expand = k_c.view(-1, 1, 1, 1).expand(-1, N, N, N)
        C.scatter_add_(0, idx_expand, prod)
    return C


def commutator(A, B, f_idx, f_val, N, chunk_size=1024, use_cp=False):
    """Compute [A, B]^k = sum f_{ij}^k A^i B^j using sparse structure constants."""
    i = f_idx[0]
    j = f_idx[1]
    k = f_idx[2]

    C = torch.zeros_like(A)
    num_nz = f_val.shape[0]

    block_size = chunk_size * 8 if use_cp else num_nz

    for start in range(0, num_nz, block_size):
        end = min(start + block_size, num_nz)
        i_b = i[start:end]
        j_b = j[start:end]
        k_b = k[start:end]
        val_b = f_val[start:end]

        if use_cp and A.requires_grad:
            C_block = checkpoint(
                commutator_block, A, B, i_b, j_b, k_b, val_b, N, chunk_size,
                use_reentrant=False,
            )
        else:
            C_block = commutator_block(A, B, i_b, j_b, k_b, val_b, N, chunk_size)

        C = C + C_block

    return C


# ─── Dense bracket helpers (for testing / analysis) ──────────────


def load_structure_constants_numpy(path="e8_constants.pt"):
    """Load structure constants as numpy arrays.

    Returns (f_idx, f_val) where f_idx is (3, nnz) int and f_val is (nnz,) float.
    """
    e8 = torch.load(path, map_location="cpu", weights_only=True).coalesce()
    return e8.indices().numpy(), e8.values().numpy()


def bracket_vec_fast(f_idx_np, f_val_np, a, b):
    """Vectorised [a, b] for 248-component numpy vectors."""
    result = np.zeros(248)
    np.add.at(result, f_idx_np[2], f_val_np * a[f_idx_np[0]] * b[f_idx_np[1]])
    return result


def extract_pos_roots_numpy(f_idx, f_val):
    """Recover 120 positive root vectors from [W_k, U_a] = α_a[k] V_a.

    Works with numpy index/value arrays.
    """
    pos_roots = np.zeros((120, 8))
    n_entries = len(f_val) if hasattr(f_val, '__len__') else f_val.shape[0]
    for n in range(n_entries):
        i = int(f_idx[0, n])
        j = int(f_idx[1, n])
        k = int(f_idx[2, n])
        if i < 8 and 8 <= j < 128 and k == 128 + (j - 8):
            pos_roots[j - 8, i] = float(f_val[n])
    return pos_roots


# ─── General Lie-algebra helpers (dense, for analysis scripts) ─────


_DIM = 248


def build_dense_f(f_idx, f_val):
    """Scatter sparse structure constants into a dense (248, 248, 248) array.

    Accepts numpy arrays (from :func:`load_structure_constants_numpy`).
    """
    F = np.zeros((_DIM, _DIM, _DIM), dtype=np.float64)
    F[f_idx[0], f_idx[1], f_idx[2]] = f_val
    return F


def build_ad_matrix(X, f_idx, f_val):
    """Build the 248x248 adjoint-action matrix (ad_X)^B_C = X^A f_{AC}^B.

    Works with sparse numpy index/value arrays from
    :func:`load_structure_constants_numpy`.
    """
    M = np.zeros((_DIM, _DIM), dtype=np.float64)
    np.add.at(M, (f_idx[2], f_idx[1]), X[f_idx[0]] * f_val)
    return M


def killing_form(F):
    """Killing tensor K_{AB} = sum_{C,D} f_{ACD} f_{BCD} from dense F.

    For compact E_8 with orthonormal Chevalley basis this equals
    2 h^v delta_{AB} with h^v = 30.
    """
    return np.einsum("ACD,BCD->AB", F, F, optimize=True)


def adjoint_exp(X, f_idx, f_val):
    """Compute exp(ad_X) as a (248, 248) numpy matrix via torch matrix_exp."""
    ad = build_ad_matrix(X, f_idx, f_val)
    return torch.linalg.matrix_exp(torch.from_numpy(ad)).numpy()
