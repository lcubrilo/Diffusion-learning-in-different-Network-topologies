# THROWAWAY diagnostic — not part of the project. Safe to delete.
# Short run (few iters) for all 3 topologies with IDENTICAL obs + init, to see the
# early-iteration consensus separation in MSD (dB AND linear) + within-team disagreement.

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from config import TOPOLOGIES, MU1, MU2, K1, K2, Z_STAR, ZK_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices
from algorithm import run_cd
from metrics import compute_msd, compute_within_team_disagreement, to_db
from game import QuadraticGame

N_ITERS = 40        # total run length
ZOOM    = 15        # iterations to display
colors  = {"fully_connected": "tab:blue", "ring": "tab:orange", "single_edge": "tab:green"}

msd_lin, msd_db, disagree = {}, {}, {}
for topo in TOPOLOGIES:
    A1   = build_within_team_matrix(topo, K1)
    A2   = build_within_team_matrix(topo, K2)
    mats = build_cross_team_matrices(topo)
    game = QuadraticGame(ZK_STAR, R_U)          # fresh game, same seed -> identical obs
    hist = run_cd(game, {**mats, "A1": A1, "A2": A2}, mu1=MU1, mu2=MU2,
                  num_iters=N_ITERS, use_cache=False, label=topo)
    m = compute_msd(hist, Z_STAR)               # linear
    msd_lin[topo]  = m
    msd_db[topo]   = to_db(m)
    disagree[topo] = compute_within_team_disagreement(hist)

# --- numeric table, first 8 iters ---
print("\niter |        MSD linear (fully / ring / single)        |   disagreement (fully / ring / single)")
for i in range(8):
    ml = [msd_lin[t][i] for t in TOPOLOGIES]
    dd = [disagree[t][i] for t in TOPOLOGIES]
    print(f"{i:>4} | {ml[0]:8.4f} {ml[1]:8.4f} {ml[2]:8.4f}  | "
          f"{dd[0]:8.4f} {dd[1]:8.4f} {dd[2]:8.4f}")

# --- plot ---
fig, (ax_db, ax_lin, ax_dis) = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"THROWAWAY: first {ZOOM} iterations, identical obs+init (μ₁={MU1}, μ₂={MU2})",
             fontweight="bold")

for t in TOPOLOGIES:
    ax_db.plot(range(ZOOM),  msd_db[t][:ZOOM],  "o-", color=colors[t], label=t)
    ax_lin.plot(range(ZOOM), msd_lin[t][:ZOOM], "o-", color=colors[t], label=t)
    ax_dis.plot(range(ZOOM), disagree[t][:ZOOM], "o-", color=colors[t], label=t)

ax_db.set_title("MSD-to-Nash (dB)");      ax_db.set_ylabel("MSD (dB)")
ax_lin.set_title("MSD-to-Nash (linear)"); ax_lin.set_ylabel(r"mean $\|est-z^*\|^2$ (linear)")
ax_dis.set_title("Within-team disagreement (linear)"); ax_dis.set_ylabel(r"mean $\|x_k-\bar x\|^2$ (linear)")
for ax in (ax_db, ax_lin, ax_dis):
    ax.set_xlabel("Iteration"); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "early_iters_throwaway.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
