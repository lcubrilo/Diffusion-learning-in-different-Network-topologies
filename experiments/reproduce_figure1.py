# @Luka
#
# Figure 1: 6-panel reproduction of paper Figure 1.
#   Top row:    4th-order moment E[‖z̃‖⁴] | 1st-order moment ‖E[z̃]‖ (paper eq 21b/21c),
#               Monte-Carlo averaged over MC_SEEDS runs (all 4 μ settings overlaid).
#               MC averaging is REQUIRED: on a single run the two dB curves are locked at
#               exactly 4×. The first-order panel averages the error VECTOR (so noise
#               cancels → bias), the fourth-order averages the SCALAR ‖z̃‖⁴; the two
#               operators differ, which is what reproduces the paper's separated levels.
#   Bottom 4:   Per-team floor + per-agent MSD (single noisy run), one subplot per μ.
#   Also saves a simple all-agent MSD-vs-iterations overlay (one curve per μ).
#
# Two modes (separate output files, never overwrite each other):
#   regular (mc=False) → single noisy run; moments locked at 4×; fast.
#                        → figure1.png,    figure1_overlay.png
#   mc      (mc=True)  → Monte-Carlo averaged moments (E[‖z̃‖⁴], ‖E[z̃]‖); slow.
#                        → figure1_mc.png, figure1_overlay_mc.png
#
# Run: python -m experiments.reproduce_figure1 [regular|mc|both]   (default: regular)

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from config import K1, K2, MU_SWEEP, MC_SEEDS, MC_NUM_ITERS, Z_STAR, ZK_STAR, R_U
from network import build_within_team_matrix, build_cross_team_matrices, sanity_check
from algorithm import run_cd
from game import QuadraticGame
from metrics import (compute_msd, compute_msd_team1, compute_msd_team2, compute_msd_agent,
                     compute_fourth_order_moment, compute_first_order_moment,
                     compute_error_vector, to_db)
# NOTE: compute_msd_team1/2 average over ALL agents in a team (the paper's
# "team MSD" / theoretical performance level); compute_msd_agent is a single agent.


def _mc_error_moments(matrices, mu1, mu2, n_seeds, tag):
    """Monte-Carlo estimate of the paper's two error-moment panels (eq 21b / 21c):

        fourth-order  E[‖z̃_i‖⁴]   — average the SCALAR ‖z̃‖⁴ over seeds.
        first-order   ‖E[z̃_i]‖    — average the error VECTOR over seeds, then norm.

    The two use different operators (scalar-power vs vector mean), which is exactly
    what frees them from the deterministic 4× dB lock a single run imposes. Crucially
    the first-order panel averages the vector: zero-mean gradient noise cancels and
    only the bias survives, so it lands far below the 4×-prediction (matching the
    paper). Averaging the NORM instead (E[‖z̃‖]) barely breaks the lock — wrong quantity.

    Returns (err4_db, err1_db, hist_seed0); seed-0 history feeds the noisy MSD panels.
    """
    e4_sum = zvec_sum = None
    hist0 = None
    for s in range(n_seeds):
        game = QuadraticGame(ZK_STAR, R_U, rng=np.random.default_rng(s))
        hist = run_cd(game, matrices, mu1=mu1, mu2=mu2, num_iters=MC_NUM_ITERS,
                      seed=s, use_cache=False,
                      label=f"{tag} seed {s + 1}/{n_seeds}")
        e4_sum = e4_sum + compute_fourth_order_moment(hist, Z_STAR) if e4_sum is not None \
            else compute_fourth_order_moment(hist, Z_STAR)
        ze = compute_error_vector(hist, Z_STAR)                       # (T, D)
        zvec_sum = ze if zvec_sum is None else zvec_sum + ze
        if s == 0:
            hist0 = hist
    err4 = to_db(e4_sum / n_seeds)                                    # E[‖z̃‖⁴]
    err1 = to_db(np.linalg.norm(zvec_sum / n_seeds, axis=1))         # ‖E[z̃]‖ (bias)
    return err4, err1, hist0


def reproduce_figure1(mc=True):
    """Build Figure 1 (6-panel) + the simple MSD overlay.

    mc=True  → error-moment panels are Monte-Carlo averaged over MC_SEEDS runs
               (paper-faithful: E[‖z̃‖⁴] and ‖E[z̃]‖; the 4× dB lock is broken).
               Saved as figure1_mc.png / figure1_overlay_mc.png. Slow (~minutes).
    mc=False → error-moment panels come from ONE noisy run (raw ‖z̃‖⁴ and ‖z̃‖, whose
               dB curves are locked at exactly 4×). Saved as figure1.png /
               figure1_overlay.png. Fast (single cached run).

    The MSD panels and the overlay are a single noisy realisation in BOTH modes.
    """
    suffix = "_mc" if mc else ""
    A1   = build_within_team_matrix("fully_connected", K1)
    A2   = build_within_team_matrix("fully_connected", K2)
    mats = build_cross_team_matrices("fully_connected")
    sanity_check({**mats, "A1": A1, "A2": A2})

    if mc:
        e4_title = f"Fourth-order error moment  E[‖z̃‖⁴]  ({MC_SEEDS}-run MC avg)"
        e1_title = f"First-order error moment  ‖E[z̃]‖  ({MC_SEEDS}-run MC avg)"
    else:
        e4_title = "Fourth-order error moment  ‖z̃‖⁴  (single run, 4×-locked)"
        e1_title = "First-order error moment  ‖z̃‖  (single run, 4×-locked)"

    matrices = {**mats, "A1": A1, "A2": A2}
    results = []
    for idx, (mu1, mu2) in enumerate(MU_SWEEP, 1):
        label = f"μ₁={mu1}, μ₂={mu2}"

        if mc:
            # Paper-faithful moments: average ‖z̃‖⁴ and the z̃ vector over MC_SEEDS runs.
            err4, err1, hist = _mc_error_moments(matrices, mu1, mu2, MC_SEEDS,
                                                 tag=f"set {idx}/{len(MU_SWEEP)}")
        else:
            # Single cached run: raw instantaneous moments (dB curves locked at 4×).
            game = QuadraticGame(ZK_STAR, R_U)
            hist = run_cd(game, matrices, mu1=mu1, mu2=mu2,
                          label=f"run {idx}/{len(MU_SWEEP)}")
            err4 = to_db(compute_fourth_order_moment(hist, Z_STAR))
            err1 = to_db(compute_first_order_moment(hist, Z_STAR))

        # MSD panels + overlay: single realisation, kept noisy on purpose.
        msd_team1 = to_db(compute_msd_team1(hist, Z_STAR))        # team-average MSD (all T1 agents)
        msd_team2 = to_db(compute_msd_team2(hist, Z_STAR))        # team-average MSD (all T2 agents)
        msd_a1t1 = to_db(compute_msd_agent(hist, Z_STAR, team=1, agent=0))
        msd_a1t2 = to_db(compute_msd_agent(hist, Z_STAR, team=2, agent=0))
        msd_all = to_db(compute_msd(hist, Z_STAR))                # all-agent MSD (for overlay)

        # Empirical floor: mean of last 20% as proxy for theoretical performance level.
        # Paper plots team MSD as a flat horizontal line (Theorem 2 prediction); we
        # approximate that with the simulated steady-state average of the TEAM MSD
        # (averaged over all agents in the team — not a single agent).
        tail = slice(int(0.8 * len(msd_team1)), None)
        floor_t1 = float(msd_team1[tail].mean())
        floor_t2 = float(msd_team2[tail].mean())
        print(f"  {label}  →  floor(team1) ≈ {floor_t1:.2f} dB  |  floor(team2) ≈ {floor_t2:.2f} dB")

        results.append(dict(label=label, mu1=mu1, mu2=mu2,
                            floor_t1=floor_t1, floor_t2=floor_t2,
                            msd_a1t1=msd_a1t1, msd_a1t2=msd_a1t2,
                            err1=err1, err4=err4, msd_all=msd_all))

    # --- Build 6-panel figure matching paper layout ---
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Figure 1: Simulation Results ({'MC moments' if mc else 'single-run moments'})",
                 fontsize=13, fontweight="bold")

    # Top row: error moments (all 4 settings overlaid)
    ax_e4 = fig.add_subplot(3, 2, 1)
    ax_e1 = fig.add_subplot(3, 2, 2)
    for r in results:
        ax_e4.plot(r["err4"], label=r["label"])
        ax_e1.plot(r["err1"], label=r["label"])
    for ax, title in [(ax_e4, e4_title), (ax_e1, e1_title)]:
        ax.set_title(title)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error (dB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Bottom 4: per-team floor (horizontal) + per-agent MSD (falling), one subplot per setting.
    # Flat lines match paper Fig 1 layout: the TEAM MSD (averaged over all agents in the team)
    # shown as the theoretical performance level, approximated by its steady-state floor
    # (mean of last 20% of iterations). The falling curves are individual agents.
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
    save_path = os.path.join(FIGS, f"figure1{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")

    # --- Simple single-panel overlay: all-agent MSD vs iterations, one curve per μ ---
    fig2, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(r["msd_all"], label=r["label"], linewidth=0.8)
    ax.set_title("Figure 1: MSD vs Iterations (paper reproduction)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSD (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(FIGS, f"figure1_overlay{suffix}.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {overlay_path}")


def run_figure1(mode="regular"):
    """Dispatch by mode: 'regular' (single-run, fast), 'mc' (Monte-Carlo, slow), 'both'."""
    if mode in ("regular", "single", "both"):
        reproduce_figure1(mc=False)
    if mode in ("mc", "both"):
        reproduce_figure1(mc=True)


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "regular"
    if mode not in ("regular", "single", "mc", "both"):
        print(f"Unknown mode '{mode}'.  Usage: python -m experiments.reproduce_figure1 "
              f"[regular|mc|both]"); sys.exit(1)
    print(f"Figure 1 mode = {mode}   "
          f"(regular → figure1.png, fast;  mc → figure1_mc.png, averages {MC_SEEDS} "
          f"runs/setting ≈ minutes)")
    run_figure1(mode)
