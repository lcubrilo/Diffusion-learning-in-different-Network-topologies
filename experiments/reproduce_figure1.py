# @Luka
#
# Figure 1: reproduce paper MSD plot for all four step-size settings.
# Uses fully-connected topology and paper parameters (Section 4).
# Visual match across four curves validates the implementation is correct.
#
# Run: python -m experiments.reproduce_figure1  (from project root)

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")

from config import K1, K2, MU_SWEEP, Z_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from metrics import compute_msd, to_db, plot_msd
from game import QuadraticGame


def reproduce_figure1():
    A1   = build_within_team_matrix("fully_connected", K1)
    A2   = build_within_team_matrix("fully_connected", K2)
    mats = build_cross_team_matrices("fully_connected")
    sanity_check({**mats, "A1": A1, "A2": A2})

    curves = {}
    for mu1, mu2 in MU_SWEEP:
        game    = QuadraticGame(Z_STAR, R_U)
        history = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=mu1, mu2=mu2)
        label   = f"μ₁={mu1}, μ₂={mu2}"
        curves[label] = to_db(compute_msd(history, Z_STAR))

    plot_msd(curves,
             title="Figure 1: MSD vs Iterations (paper reproduction)",
             save_path=os.path.join(FIGS, "figure1.png"))


if __name__ == "__main__":
    reproduce_figure1()
