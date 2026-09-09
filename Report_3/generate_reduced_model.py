# -*- coding: utf-8 -*-
"""
Generate Reduced System for Exercise 11
Based on Exercise 8 Logic with a Single User-Defined Step
"""

import numpy as np
import scipy as sp

# =====================================================
# USER SETTINGS
# =====================================================

# >>>>> SELECT YOUR STEP SIZE HERE <<<<<
# Choose the step size you validated in Exercise 8 (e.g., 4, 6, 8)
# 1 = Full model, 10 = Very coarse
USER_STEP = 4 

# Define which parts of the structure to KEEP as Dynamic
# (Based on Ex 8/11 instructions: Eliminate Monopile & Drivetrain)
# Adjust node ranges if your specific model definition differs
tower_start = 46  # First node above seabed/monopile
tower_end = 65    
hub_start = 70
hub_end = 72

# =====================================================
# NODE GENERATION (Exact logic from Ex 8)
# =====================================================

def part_nodes(start, end, step):
    """Generates nodes ensuring start and end are always included."""
    nodes = list(np.arange(start, end + 1, step))
    if nodes[-1] != end:
        nodes.append(end)
    if nodes[0] != start:
        nodes.insert(0, start)
    return np.unique(np.array(nodes))

# Generate Dynamic Node Lists
support_nodes = part_nodes(tower_start, tower_end, USER_STEP)
hub_nodes = part_nodes(hub_start, hub_end, USER_STEP)
blade1_nodes = part_nodes(73, 141, USER_STEP)
blade2_nodes = part_nodes(142, 210, USER_STEP)
blade3_nodes = part_nodes(211, 279, USER_STEP)

# Combine all dynamic nodes
dynamic_nodes = np.unique(np.concatenate([
    support_nodes,
    hub_nodes,
    blade1_nodes,
    blade2_nodes,
    blade3_nodes
]))

print(f"--- Reduction Settings ---")
print(f"User Step Size: {USER_STEP}")
print(f"Dynamic Nodes Count: {len(dynamic_nodes)}")
print(f"Total Dynamic DOFs: {len(dynamic_nodes) * 6}")

# =====================================================
# LOAD FULL MODEL
# =====================================================

data = np.load('owt_XKM.npz')
X = data['X']
K_full = data['K']
M_full = data['M']
print("Loaded owt_XKM.npz successfully.")


n_total_nodes = 279
n_total_dofs = 6 * n_total_nodes

# =====================================================
# SYSTEM REDUCTION (Ex 8 Logic)
# =====================================================

# 1. Create Index Sets
block = np.arange(1, 7)
Id_matrix = (dynamic_nodes[:, None] - 1) * 6 + block
Id = Id_matrix.ravel()  # Dynamic DOFs (1-based)

All_DOFs = np.arange(1, n_total_dofs + 1)
Is = np.setdiff1d(All_DOFs, Id) # Static DOFs (1-based)

# Convert to 0-based for Python
id_idx = Id - 1
is_idx = Is - 1

# 2. Partition Matrices
Kdd = K_full[np.ix_(id_idx, id_idx)]
Kds = K_full[np.ix_(id_idx, is_idx)]
Ksd = K_full[np.ix_(is_idx, id_idx)]
Kss = K_full[np.ix_(is_idx, is_idx)]

Mdd = M_full[np.ix_(id_idx, id_idx)]
Mds = M_full[np.ix_(id_idx, is_idx)]
Msd = M_full[np.ix_(is_idx, id_idx)]
Mss = M_full[np.ix_(is_idx, is_idx)]

# 3. Calculate Transformation Matrix S
# S maps Dynamic Displacements to Static Displacements: u_s = S * u_d
S = -np.linalg.solve(Kss, Ksd)

# 4. Calculate Reduced Matrices
# Using efficient matrix multiplication
K_temp = Ksd + Kss @ S
K_red = Kdd + S.T @ K_temp

M_temp = Msd + Mss @ S
M_red = Mdd + S.T @ M_temp

print(f"Reduced K shape: {K_red.shape}")
print(f"Reduced M shape: {M_red.shape}")

# =====================================================
# MODAL ANALYSIS (To get Omega)
# =====================================================
n_modes_needed = 20 # Ensure we get at least mode 13
lam, U_red = sp.linalg.eigh(K_red, M_red, subset_by_index=[0, n_modes_needed-1])

omega = np.sqrt(np.real(lam))
freq = omega / (2 * np.pi)

# Sort
iw = np.argsort(omega)
omega = omega[iw]
freq = freq[iw]
U_red = U_red[:, iw]

print(f"Mode 2 Freq: {freq[1]:.4f} Hz")
print(f"Mode 13 Freq: {freq[12]:.4f} Hz")

# =====================================================
# SAVE TO FILE
# =====================================================
# Note: Renamed 'is' to 'is_vec' to avoid Python keyword error
np.savez(
    'vibprop_sysred.npz',
    K=K_red,
    M=M_red,
    S=S,
    omega=omega,
    U=U_red,
    id=Id,
    is_vec=Is,      # Renamed to avoid syntax error
    step_size=USER_STEP
)

print("\nSuccess! Saved 'vibprop_sysred.npz'.")
print("You can now run your Exercise 11 time integration script.")