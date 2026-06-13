# @Luka
#
# Figure 2: MSD convergence for fully_connected vs ring vs single_edge.
# Fixed step sizes (MU1, MU2 from config). One curve per topology.
#
# Run: python -m experiments.topology_test  (from project root)

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")
DATA = os.path.join(ROOT, "results", "data")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from config import TOPOLOGIES, MU1, MU2, K1, K2, Z_STAR, ZK_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from metrics import compute_msd, to_db
from game import QuadraticGame


def topology_test():
    results = {}

    for topology in TOPOLOGIES:
        A1   = build_within_team_matrix(topology, K1)
        A2   = build_within_team_matrix(topology, K2)
        mats = build_cross_team_matrices(topology)
        sanity_check({**mats, "A1": A1, "A2": A2})

        # Print matrix norms so we can confirm topologies actually differ
        print(f"\n[{topology}]")
        print(f"  A1 =\n{A1.round(4)}")
        print(f"  A2 =\n{A2.round(4)}")
        print(f"  A21 =\n{mats['A21'].round(4)}")

        game    = QuadraticGame(ZK_STAR, R_U)
        history = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=MU1, mu2=MU2,
                         label=topology)
        msd_db = to_db(compute_msd(history, Z_STAR))
        results[topology] = msd_db

        # Save raw data for external inspection
        os.makedirs(DATA, exist_ok=True)
        np.save(os.path.join(DATA, f"topology_{topology}_msd.npy"), msd_db)
        tail = msd_db[int(0.8 * len(msd_db)):]
        print(f"  steady-state floor: mean={tail.mean():.2f} dB  std={tail.std():.2f} dB")

    # --- Overlay plot (Figure 2a) ---
    os.makedirs(FIGS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fully_connected": "tab:blue", "ring": "tab:orange", "single_edge": "tab:green"}
    for topology, msd_db in results.items():
        ax.plot(msd_db, label=topology, color=colors[topology])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSD (dB)")
    ax.set_title(f"Figure 2: MSD vs Topology (μ₁={MU1}, μ₂={MU2})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "figure2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Separate subplot per topology (Figure 2b) for debugging ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(f"Figure 2 (separate): MSD per Topology (μ₁={MU1}, μ₂={MU2})", fontweight="bold")
    for ax, (topology, msd_db) in zip(axes, results.items()):
        floor = msd_db[int(0.8 * len(msd_db)):].mean()
        ax.plot(msd_db, color=colors[topology], linewidth=0.8)
        ax.axhline(floor, color="black", linestyle="--", linewidth=1,
                   label=f"floor ≈ {floor:.1f} dB")
        ax.set_title(topology)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSD (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "figure2_separate.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure2.png and figure2_separate.png")
    print(f"Raw MSD arrays saved to {DATA}/topology_*.npy")
    print("To inspect: np.load('results/data/topology_fully_connected_msd.npy')")


if __name__ == "__main__":
    topology_test()
