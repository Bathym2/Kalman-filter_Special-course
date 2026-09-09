# =====================================================
# IMPORTS
# =====================================================

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.close('all')

# =====================================================
# USER SETTINGS
# =====================================================

modes_to_analyze = [2, 13]
step = 3

# =====================================================
# LOAD DATA
# =====================================================

npzfile = np.load('owt_XKM.npz')
X = npzfile['X']
K_full = npzfile['K']
M_full = npzfile['M']

# =====================================================
# NODE SET GENERATOR
# =====================================================

def part_nodes(start, end, step):
    nodes = list(np.arange(start, end+1, step))
    if nodes[-1] != end:
        nodes.append(end)
    if nodes[0] != start:
        nodes.insert(0, start)
    return np.array(nodes)

# Support = nodes 1–65 (as header states)
support = part_nodes(1, 45, 1)

drivetrain = part_nodes(66, 69, step)
hub = part_nodes(70, 72, step)
blade1 = part_nodes(73, 141, step)
blade2 = part_nodes(142, 210, step)
blade3 = part_nodes(211, 279, step)

nodes = np.unique(np.concatenate((
    support,
    drivetrain,
    hub,
    blade1,
    blade2,
    blade3
)))

# =====================================================
# STATIC CONDENSATION
# =====================================================

block = np.arange(1, 7)
Id_matrix = (nodes[:, None] - 1) * 6 + block
Id = Id_matrix.ravel()

All_DOFs = np.arange(1, 6*279 + 1)
Is = np.setdiff1d(All_DOFs, Id)

id_idx = Id - 1
is_idx = Is - 1

# Partition matrices
Mdd = M_full[np.ix_(id_idx, id_idx)]
Mds = M_full[np.ix_(id_idx, is_idx)]
Msd = M_full[np.ix_(is_idx, id_idx)]
Mss = M_full[np.ix_(is_idx, is_idx)]

Kdd = K_full[np.ix_(id_idx, id_idx)]
Kds = K_full[np.ix_(id_idx, is_idx)]
Ksd = K_full[np.ix_(is_idx, id_idx)]
Kss = K_full[np.ix_(is_idx, is_idx)]

# Static condensation
S = -np.linalg.solve(Kss, Ksd)

K_red = Kdd + S.T @ Ksd + Kds @ S + S.T @ Kss @ S
M_red = Mdd + S.T @ Msd + Mds @ S + S.T @ Mss @ S

# =====================================================
# MODAL ANALYSIS
# =====================================================

lam, U = sp.linalg.eigh(
    K_red,
    M_red,
    subset_by_index=[0, max(modes_to_analyze)-1]
)

omega = np.sqrt(np.real(lam))
iw = np.argsort(omega)

omega = omega[iw]
freq = omega / (2*np.pi)
U = U[:, iw]

# Normalize modes to max value = 1
U = U / U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])]

# =====================================================
# OUTPUT INFO
# =====================================================

print(f"--- Reduction Settings ---")
print(f"Step Size: {step}")
print(f"Dynamic Nodes Count: {len(nodes)}")
print(f"Total Dynamic DOFs: {len(nodes)*6}")
print(f"Mode 2 Frequency: {freq[1]:.4f} Hz")
print(f"Mode 13 Frequency: {freq[12]:.4f} Hz")

# =====================================================
# SAVE
# =====================================================

np.savez(
    'ReducedOutput.npz',
    X=X,
    K=K_red,
    M=M_red,
    S=S,
    omega=omega,
    U=U,
    id_vec=Id,
    is_vec=Is,
    step_size=step
)

print("\nSuccess! Saved 'ReducedOutput.npz'.")