# @Luka
#
# Figure 1: 6-panel reproduction of paper Figure 1.
#   Top row:    4th-order error moment | 1st-order error moment  (all 4 μ settings overlaid)
#   Bottom 4:   Per-team + per-agent MSD, one subplot per μ setting
#
# Run: python -m experiments.reproduce_figure1  (from project root)

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from config import K1, K2, MU_SWEEP, Z_STAR, ZK_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from game import QuadraticGame
from metrics import (compute_msd_team1, compute_msd_team2, compute_msd_agent,
                     compute_fourth_order_moment, compute_first_order_moment, to_db)


def reproduce_figure1():
    A1   = build_within_team_matrix("fully_connected", K1)
    A2   = build_within_team_matrix("fully_connected", K2)
    mats = build_cross_team_matrices("fully_connected")
    sanity_check({**mats, "A1": A1, "A2": A2})

    results = []
    for idx, (mu1, mu2) in enumerate(MU_SWEEP, 1):
        label = f"μ₁={mu1}, μ₂={mu2}"
        game  = QuadraticGame(ZK_STAR, R_U)
        hist  = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=mu1, mu2=mu2,
                       label=f"run {idx}/{len(MU_SWEEP)}")

        msd_a1t1 = to_db(compute_msd_agent(hist, Z_STAR, team=1, agent=0))
        msd_a1t2 = to_db(compute_msd_agent(hist, Z_STAR, team=2, agent=0))
        err4 = to_db(compute_fourth_order_moment(hist, Z_STAR))   # ‖z̃‖⁴, paper eq 21b
        err1 = to_db(compute_first_order_moment(hist, Z_STAR))   # ‖E[z̃]‖ via running mean, paper eq 21c

        # Empirical floor: mean of last 20% as proxy for theoretical performance level.
        # Paper plots team MSD as a flat horizontal line (Theorem 2 prediction); we
        # approximate that with the simulated steady-state average.
        tail = slice(int(0.8 * len(msd_a1t1)), None)
        floor_t1 = float(msd_a1t1[tail].mean())
        floor_t2 = float(msd_a1t2[tail].mean())
        print(f"  {label}  →  floor(team1) ≈ {floor_t1:.2f} dB  |  floor(team2) ≈ {floor_t2:.2f} dB")

        results.append(dict(label=label, mu1=mu1, mu2=mu2,
                            floor_t1=floor_t1, floor_t2=floor_t2,
                            msd_a1t1=msd_a1t1, msd_a1t2=msd_a1t2,
                            err1=err1, err4=err4))

    # --- Build 6-panel figure matching paper layout ---
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("Figure 1: Simulation Results", fontsize=13, fontweight="bold")

    # Top row: error moments (all 4 settings overlaid)
    ax_e4 = fig.add_subplot(3, 2, 1)
    ax_e1 = fig.add_subplot(3, 2, 2)
    for r in results:
        ax_e4.plot(r["err4"], label=r["label"])
        ax_e1.plot(r["err1"], label=r["label"])
    for ax, title in [(ax_e4, "Fourth-order error moment"), (ax_e1, "First-order error moment")]:
        ax.set_title(title)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error (dB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Bottom 4: per-team floor (horizontal) + per-agent MSD (falling), one subplot per setting.
    # Flat lines match paper Fig 1 layout: team MSD shown as the theoretical performance level
    # (approximated here as the empirical steady-state floor from the last 20% of iterations).
    for i, r in enumerate(results):
        ax = fig.add_subplot(3, 2, 3 + i)
        ax.axhline(r["floor_t1"], color="black", linewidth=1.2,
                   label=f"MSD(team1), μ₁={r['mu1']}")
        ax.axhline(r["floor_t2"], color="blue",  linewidth=1.2,
                   label=f"MSD(team2), μ₂={r['mu2']}")
        ax.plot(r["msd_a1t1"], color="magenta", linewidth=0.8,
                label=f"MSD(agent1,team1), μ₁={r['mu1']}")
        ax.plot(r["msd_a1t2"], color="red",     linewidth=0.8,
                label=f"MSD(agent1,team2), μ₂={r['mu2']}")
        ax.set_title(f"MSD (setting {i + 1})")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("MSD (dB)")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    save_path = os.path.join(FIGS, "figure1.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    reproduce_figure1()
