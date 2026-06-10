# @Luka
#
# Figure 2: MSD convergence for fully_connected vs ring vs single_edge.
# Fixed step sizes (MU1, MU2 from config). One curve per topology.
#
# Run: python -m experiments.topology_test  (from project root)

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")

from config import TOPOLOGIES, MU1, MU2, K1, K2, Z_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from metrics import compute_msd, to_db, plot_msd
from game import QuadraticGame


def topology_test():
    results = {}

    for topology in TOPOLOGIES:
        A1   = build_within_team_matrix(topology, K1)
        A2   = build_within_team_matrix(topology, K2)
        mats = build_cross_team_matrices(topology)
        sanity_check({**mats, "A1": A1, "A2": A2})

        game    = QuadraticGame(Z_STAR, R_U)
        history = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=MU1, mu2=MU2)
        results[topology] = to_db(compute_msd(history, Z_STAR))

    plot_msd(results,
             title=f"Figure 2: MSD vs Topology (μ₁={MU1}, μ₂={MU2})",
             save_path=os.path.join(FIGS, "figure2.png"))


if __name__ == "__main__":
    topology_test()
