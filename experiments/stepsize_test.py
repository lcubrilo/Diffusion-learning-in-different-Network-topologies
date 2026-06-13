# @Luka
#
# Figure 3: steady-state MSD floor vs step size μ₁. Fixed topology (fully_connected).
# Theory (Theorem 1) predicts floor = O(μ_max) — expect slope ≈ 1 on log x-axis.
#
# Run: python -m experiments.stepsize_test  (from project root)

import os, sys
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

from config import MU_SWEEP, K1, K2, Z_STAR, ZK_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from metrics import compute_msd, to_db
from game import QuadraticGame

import numpy as np


def stepsize_test():
    A1   = build_within_team_matrix("fully_connected", K1)
    A2   = build_within_team_matrix("fully_connected", K2)
    mats = build_cross_team_matrices("fully_connected")
    sanity_check({**mats, "A1": A1, "A2": A2})

    floors = {}
    for idx, (mu1, mu2) in enumerate(MU_SWEEP, 1):
        game    = QuadraticGame(ZK_STAR, R_U)
        history = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=mu1, mu2=mu2,
                         label=f"μ₁={mu1} ({idx}/{len(MU_SWEEP)})")
        msd_db  = to_db(compute_msd(history, Z_STAR))
        tail    = msd_db[int(0.9 * len(msd_db)):]   # last 10% ≈ steady state
        floors[mu1] = float(np.mean(tail))

    mu_vals   = list(floors.keys())
    floor_vals = list(floors.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(mu_vals, floor_vals, "o-")
    ax.set_xlabel("μ₁ (log scale)")
    ax.set_ylabel("Steady-state MSD floor (dB)")
    ax.set_title("Figure 3: MSD Floor vs Step Size")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    plt.savefig(os.path.join(FIGS, "figure3.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    stepsize_test()
