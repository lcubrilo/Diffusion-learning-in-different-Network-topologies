# @Luka
#
# Visualize within-team and cross-team network graphs for all three topologies.
# Run: python -m experiments.visualize_topology  (from project root)

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from config import K1, K2, TOPOLOGIES
from network import build_within_team_matrix, build_cross_team_matrices


def _positions(K, team, x_offset):
    """Arrange K nodes vertically, centered, offset horizontally by team."""
    return {i: (x_offset, (K - 1) / 2 - i) for i in range(K)}


def plot_topology(topology: str, ax):
    A1  = build_within_team_matrix(topology, K1)
    A2  = build_within_team_matrix(topology, K2)
    mats = build_cross_team_matrices(topology)
    A21 = mats["A21"]   # (K2 x K1): Team1 agent k → observed by Team2 agent ℓ
    A12 = mats["A12"]   # (K1 x K2): Team2 agent k → observed by Team1 agent ℓ

    # Build a single directed graph over K1+K2 nodes
    # Team 1: nodes 0..K1-1  (left, blue)
    # Team 2: nodes K1..K-1  (right, red)
    G = nx.DiGraph()
    G.add_nodes_from(range(K1 + K2))

    # Within-team edges (A1): a1[ℓ,k] > 0 means k sends to ℓ
    for k in range(K1):
        for ell in range(K1):
            if k != ell and A1[ell, k] > 1e-9:
                G.add_edge(k, ell, kind="within")

    # Within-team edges (A2)
    for k in range(K2):
        for ell in range(K2):
            if k != ell and A2[ell, k] > 1e-9:
                G.add_edge(K1 + k, K1 + ell, kind="within")

    # Cross-team edges: A21[ℓ, k] > 0 means Team1-k is observed by Team2-ℓ
    for k in range(K1):
        for ell in range(K2):
            if A21[ell, k] > 1e-9:
                G.add_edge(k, K1 + ell, kind="cross")

    # Cross-team edges: A12[ℓ, k] > 0 means Team2-k is observed by Team1-ℓ
    for k in range(K2):
        for ell in range(K1):
            if A12[ell, k] > 1e-9:
                G.add_edge(K1 + k, ell, kind="cross")

    pos = {**_positions(K1, 1, 0.0), **{K1 + i: v for i, v in _positions(K2, 2, 2.0).items()}}
    colors = ["#4C72B0"] * K1 + ["#DD8452"] * K2
    labels = {i: f"T1-{i}" for i in range(K1)}
    labels.update({K1 + i: f"T2-{i}" for i in range(K2)})

    within_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "within"]
    cross_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "cross"]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=600)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=within_edges, ax=ax,
                           edge_color="#555555", arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.15", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                           edge_color="#2ca02c", arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1", width=2.0, style="dashed")

    ax.set_title(topology.replace("_", " ").title(), fontsize=11)
    ax.axis("off")


def visualize_topology():
    fig, axes = plt.subplots(1, len(TOPOLOGIES), figsize=(5 * len(TOPOLOGIES), 5))
    fig.suptitle("Network Topologies", fontsize=13, fontweight="bold")

    for ax, topology in zip(axes, TOPOLOGIES):
        plot_topology(topology, ax)

    legend = [
        mpatches.Patch(color="#4C72B0", label=f"Team 1 ({K1} agents)"),
        mpatches.Patch(color="#DD8452", label=f"Team 2 ({K2} agents)"),
        mpatches.Patch(color="#555555", label="Within-team link"),
        mpatches.Patch(color="#2ca02c", label="Cross-team link"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(FIGS, exist_ok=True)
    save_path = os.path.join(FIGS, "topologies.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    visualize_topology()
