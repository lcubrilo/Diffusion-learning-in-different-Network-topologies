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
# Regular/single-run figures use the paper's full horizon (2×10⁵). The Monte-Carlo
# moment panels (Figure 1, mc mode) run MC_SEEDS × 4 μ-settings, so 200k there is far
# too slow; they keep the shorter 41k horizon. This preserves the existing 41k MC
# figures while letting the regular figures move to 200k.
NUM_ITERS    = 200_000   # regular / single-run: Fig 1 (regular), Fig 2, Fig 3
MC_NUM_ITERS = 41_000    # Monte-Carlo moment panels only: Fig 1 (mc)
RANDOM_SEED = 42

# Monte-Carlo realisations for the error-moment panels in Figure 1.
# The paper plots E[‖z̃‖⁴] and E[‖z̃‖] (averaged over many runs). On a SINGLE run the
# two dB curves are locked at exactly 4× (fourth_dB = 4·first_dB); only averaging the
# two powers separately over independent seeds breaks that lock and lets the steady-state
# levels separate the way the paper shows. More seeds → smoother curves (and slower).
MC_SEEDS = 20

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
P1 = np.ones(K1) / K1   # [0.5, 0.5]
P2 = np.ones(K2) / K2   # [0.25, 0.25, 0.25, 0.25]

# --- R_U: per-agent regression covariance R_{k,u} in R^{M x M} ---
# The Hessian of E[Q_k] equals R_{k,u}, so its block structure drives everything:
#   R_xx  = R_U[:M1, :M1]   (5x5)  → ν_eff^(1)  (paper eq 9b) — Team 1 curvature
#   R_yy  = R_U[M1:, M1:]   (10x10)→ ν_eff^(2)  (paper eq 9c) — Team 2 curvature
#   R_xy  = R_U[:M1, M1:]   (5x10) → δ_eff^(21) (paper eq 9a) — cross-team coupling
#   R_yx  = R_U[M1:, :M1]   (10x5) → δ_eff^(12) (paper eq 9a) — cross-team coupling
#
# With R_U = I: R_xy = 0 → δ_eff = 0 → teams decouple → topology has no effect (Fig 2 flat).
# COUPLING = 0.5 adds off-diagonal blocks. PD: Schur complement min eigenvalue = 0.75 > 0.
COUPLING = 0.5

def _make_R() -> np.ndarray:
    R = np.eye(M)
    R[:M1, M1:] = COUPLING * np.eye(M1, M2)
    R[M1:, :M1] = COUPLING * np.eye(M2, M1)
    return R

R_U = [_make_R() for _ in range(K)]

# --- Per-agent target vectors z_k* (paper eq 26: d(k,i) = u_{k,i}^T z_k* + v(k,i)) ---
# Each agent has its OWN true solution. With a single shared z*, teams trivially agree
# → no real competition → Nash equilibrium is degenerate (always equals that shared z*).
# A separate RNG (seed + 999) keeps ZK_STAR independent of the simulation RNG.
_rng_zk = np.random.default_rng(RANDOM_SEED + 999)
ZK_STAR = [_rng_zk.standard_normal(M) * 0.5 for _ in range(K)]   # list of K (M,) vectors

TEAM = [1] * K1 + [2] * K2           # team assignment: TEAM[k] ∈ {1, 2}
P    = np.concatenate([P1, P2])       # combined weight vector, shape (K,)


def _compute_nash(R_list, zk_star_list, team_list, p_list, M1, M2):
    """Solve for Nash equilibrium z* = col{x*, y*}.

    Nash condition — zero aggregate gradient for each team:
      Team 1: sum_{k in T1} p_k [R_xx_k (x* - x_k*) + R_xy_k (y* - y_k*)] = 0
      Team 2: sum_{k in T2} p_k [R_yx_k (x* - x_k*) + R_yy_k (y* - y_k*)] = 0

    Rearranges to a linear system  A_mat @ z* = b_vec, solved exactly.
    """
    Axx = np.zeros((M1, M1)); Axy = np.zeros((M1, M2)); bx = np.zeros(M1)
    Ayx = np.zeros((M2, M1)); Ayy = np.zeros((M2, M2)); by = np.zeros(M2)

    for k, (R, zk, t, pk) in enumerate(zip(R_list, zk_star_list, team_list, p_list)):
        Rxx = R[:M1, :M1];  Rxy = R[:M1, M1:]
        Ryx = R[M1:, :M1];  Ryy = R[M1:, M1:]
        xk  = zk[:M1];      yk  = zk[M1:]
        if t == 1:
            Axx += pk * Rxx;  Axy += pk * Rxy
            bx  += pk * (Rxx @ xk + Rxy @ yk)
        else:
            Ayx += pk * Ryx;  Ayy += pk * Ryy
            by  += pk * (Ryx @ xk + Ryy @ yk)

    A_mat = np.block([[Axx, Axy], [Ayx, Ayy]])
    b_vec = np.concatenate([bx, by])
    return np.linalg.solve(A_mat, b_vec)


# Nash equilibrium z* = col{x*, y*}: the point the CD algorithm converges to.
# Used in metrics.py for MSD computation (we measure ‖estimate − Z_STAR‖).
Z_STAR = _compute_nash(R_U, ZK_STAR, TEAM, P, M1, M2)
