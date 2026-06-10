# @Luka

import numpy as np

# --- Team sizes (paper Section 4) ---
K1 = 2          # agents in Team 1
K2 = 4          # agents in Team 2
K  = K1 + K2

# --- Strategy dimensions (paper Section 4) ---
M1 = 5          # dimension of x (Team 1 strategy vector)
M2 = 10         # dimension of y (Team 2 strategy vector)
M  = M1 + M2

# --- Step sizes (paper Figure 1, setting 1; always MU2 = MU1/2) ---
MU1 = 0.001
MU2 = 0.0005

# --- Simulation ---
NUM_ITERS = 200_000
RANDOM_SEED = 42

# --- Topologies (used by network.py and topology_test.py) ---
TOPOLOGIES = ["fully_connected", "ring", "single_edge"]

# --- Step-size sweep for Figure 3 (all four paper settings; MU2 = MU1/2 throughout) ---
MU_SWEEP = [
    (0.001,  0.0005),
    (0.0008, 0.0004),
    (0.0006, 0.0003),
    (0.0004, 0.0002),
]

# --- Noise (sigma^2; uniform across agents) ---
NOISE_VAR = 0.1

# --- Agent weights (uniform; p_k^(t) = 1/K_t for all k in team t) ---
# Used by @Marija for Assumption 2 computation and Nash equilibrium derivation.
# Not used directly in the ATC loop.
P1 = np.ones(K1) / K1   # [0.5, 0.5]
P2 = np.ones(K2) / K2   # [0.25, 0.25, 0.25, 0.25]

# --- Game parameters (@Marija: replace Z_STAR and R_U with analytically derived values) ---
# Defaults use z*=0 and identity covariance.  With R_U=I, delta_eff=0 so Assumption 2
# holds for any positive step sizes — useful for initial testing before real values arrive.
Z_STAR = np.zeros(M)                          # Nash equilibrium col{x*, y*}, shape (M,)
R_U    = [np.eye(M) for _ in range(K)]        # per-agent regression covariance, each (M, M)
