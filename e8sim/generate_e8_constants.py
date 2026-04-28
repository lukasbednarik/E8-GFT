import torch
import numpy as np

def generate_e8_constants_chevalley():
    # 1. Generate roots
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    r = np.zeros(8)
                    r[i] = s1
                    r[j] = s2
                    roots.append(r)
    for i in range(256):
        signs = np.array([1 if (i & (1 << k)) else -1 for k in range(8)])
        if np.prod(signs) == 1:
            roots.append(signs * 0.5)
            
    roots = np.array(roots) # 240, 8
    
    simple_roots = np.array([
        [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]
    ])
    
    C = np.zeros((8, 8), dtype=int)
    for i in range(8):
        for j in range(8):
            C[i, j] = int(np.round(np.dot(simple_roots[i], simple_roots[j])))
            
    M = np.zeros((8, 8), dtype=int)
    for i in range(8):
        M[i, i] = 1
        for j in range(i+1, 8):
            M[i, j] = C[i, j]
            
    inv_simple = np.linalg.inv(simple_roots.T)
    c_coords = np.array([np.round(inv_simple @ r).astype(int) for r in roots])
    
    pos_indices = []
    for i, c in enumerate(c_coords):
        idx = np.nonzero(c)[0][0]
        if c[idx] > 0:
            pos_indices.append(i)
            
    pos_roots = roots[pos_indices]
    pos_c = c_coords[pos_indices]
    
    # map root tuple to index in pos_roots
    root_to_pos_idx = {}
    for i, r in enumerate(pos_roots):
        root_to_pos_idx[tuple(np.round(r, 5))] = i
        
    def eps(c1, c2):
        power = int(np.round(c1.T @ M @ c2))
        return 1 if power % 2 == 0 else -1
        
    def N(r1, r2):
        """Returns the structure constant N(r1, r2) for roots r1, r2.
        Assumes r1+r2 is a root."""
        # 1. Determine if r1 is pos or neg
        sgn1, idx1 = (1, root_to_pos_idx[tuple(np.round(r1, 5))]) if tuple(np.round(r1, 5)) in root_to_pos_idx else (-1, root_to_pos_idx[tuple(np.round(-r1, 5))])
        sgn2, idx2 = (1, root_to_pos_idx[tuple(np.round(r2, 5))]) if tuple(np.round(r2, 5)) in root_to_pos_idx else (-1, root_to_pos_idx[tuple(np.round(-r2, 5))])
        
        if sgn1 == 1 and sgn2 == 1:
            return eps(pos_c[idx1], pos_c[idx2])
        elif sgn1 == -1 and sgn2 == -1:
            return -eps(pos_c[idx1], pos_c[idx2])
        elif sgn1 == 1 and sgn2 == -1: # r1 pos, r2 neg. Let alpha=r1, beta=-r2
            alpha = r1
            beta = -r2
            gamma = alpha - beta # which is r1 + r2
            sgn3, idx3 = (1, root_to_pos_idx[tuple(np.round(gamma, 5))]) if tuple(np.round(gamma, 5)) in root_to_pos_idx else (-1, root_to_pos_idx[tuple(np.round(-gamma, 5))])
            if sgn3 == 1:
                # γ>0: α = β+γ, Jacobi ⇒ N(α,-β) = -N(β,γ)
                return -eps(pos_c[idx2], pos_c[idx3])
            else:
                # γ<0: δ=-γ>0, β = α+δ, Jacobi ⇒ N(α,-β) = -N(α,δ)
                return -eps(pos_c[idx1], pos_c[idx3])
        else: # sgn1 == -1, sgn2 == 1
            return -N(r2, r1)

    f_ijk = []
    def add_f(i, j, k, val):
        if abs(val) > 1e-5:
            f_ijk.append((i, j, k, val))
            f_ijk.append((j, i, k, -val))

    # Generators:
    # 0..7: W_k = i H_k
    # 8..127: U_a = 1/sqrt(2) (E_a - E_-a)
    # 128..247: V_a = i/sqrt(2) (E_a + E_-a)
    
    # 1. [W_k, U_a] = alpha[k] V_a
    #    [W_k, V_a] = -alpha[k] U_a
    for k in range(8):
        for a in range(120):
            alpha = pos_roots[a]
            U_a = 8 + a
            V_a = 128 + a
            add_f(k, U_a, V_a, alpha[k])
            add_f(k, V_a, U_a, -alpha[k])
            
            # [U_a, V_a] = \sum alpha[k] W_k
            add_f(U_a, V_a, k, alpha[k])
            
    # 3. [U_a, U_b], [V_a, V_b], [U_a, V_b]
    for a in range(120):
        for b in range(a+1, 120):
            r_a = pos_roots[a]
            r_b = pos_roots[b]
            U_a, V_a = 8 + a, 128 + a
            U_b, V_b = 8 + b, 128 + b
            
            # Roots a+b and a-b
            for s_b, r_b_eff in [(1, r_b), (-1, -r_b)]:
                r_c = r_a + r_b_eff
                
                # Check if r_c is a root
                c_key = tuple(np.round(r_c, 5))
                if c_key in root_to_pos_idx or tuple(np.round(-r_c, 5)) in root_to_pos_idx:
                    # Yes, it is a root.
                    n_val = N(r_a, r_b_eff)
                    
                    # Is r_c positive or negative?
                    if c_key in root_to_pos_idx:
                        sgn_c = 1
                        idx_c = root_to_pos_idx[c_key]
                    else:
                        sgn_c = -1
                        idx_c = root_to_pos_idx[tuple(np.round(-r_c, 5))]
                        
                    U_c = 8 + idx_c
                    V_c = 128 + idx_c
                    
                    val = n_val / np.sqrt(2)
                    
                    # Commutators
                    if s_b == 1: # a+b is root
                        if sgn_c == 1:
                            add_f(U_a, U_b, U_c, val)
                            add_f(V_a, V_b, U_c, -val)
                            add_f(U_a, V_b, V_c, val)
                            add_f(V_a, U_b, V_c, val)
                        else:
                            add_f(U_a, U_b, U_c, -val)
                            add_f(V_a, V_b, U_c, val)
                            add_f(U_a, V_b, V_c, val)
                            add_f(V_a, U_b, V_c, val)
                    else: # a-b is root
                        if sgn_c == 1:
                            add_f(U_a, U_b, U_c, -val)
                            add_f(V_a, V_b, U_c, -val)
                            add_f(U_a, V_b, V_c, val)
                            add_f(V_a, U_b, V_c, -val)
                        else:
                            add_f(U_a, U_b, U_c, val)
                            add_f(V_a, V_b, U_c, val)
                            add_f(U_a, V_b, V_c, val)
                            add_f(V_a, U_b, V_c, -val)

    # Save to sparse tensor
    print(f"Generated {len(f_ijk)} structure constants.")
    indices = torch.tensor([[i, j, k] for i, j, k, v in f_ijk], dtype=torch.long).T
    values = torch.tensor([v for i, j, k, v in f_ijk], dtype=torch.float64)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(248, 248, 248))
    torch.save(sparse_tensor, "e8_constants.pt")
    print("Saved to e8_constants.pt")

generate_e8_constants_chevalley()
