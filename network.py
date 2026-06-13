# @Luka

import numpy as np
import networkx as nx
from config import K1, K2


def build_within_team_matrix(topology: str, K: int) -> np.ndarray:
    """Left-stochastic primitive combination matrix for a team of K agents.

    Topology governs the within-team graph:
      fully_connected — complete graph (every agent hears every teammate)
      ring / single_edge — cycle graph (each agent hears two ring-neighbours)

    Self-loops are added before normalisation so the resulting matrix is
    primitive for any K >= 1.
    """
    if topology == "fully_connected":
        G = nx.complete_graph(K)
    else:
        # ring and single_edge both use a cycle within each team;
        # only the cross-team matrices differ between those two cases.
        G = nx.cycle_graph(K)

    for node in G.nodes():
        G.add_edge(node, node)

    W = nx.to_numpy_array(G)   # symmetric adjacency
    return _normalize_left_stochastic(W)


def build_cross_team_matrices(topology: str) -> dict:
    """Return dict with keys A21, A12, A11, A22.

    Block structure (Assumption 1 of the paper):
        A21_blk = [A21; A11]  shape (K, K1) — left-stochastic
        A12_blk = [A12; A22]  shape (K, K2) — left-stochastic

    A21 (K2 x K1), A12 (K1 x K2): direct cross-team observation weights.
        At least one entry must be positive (Assumption 1, semi-weakly positive).
    A11 (K1 x K1), A22 (K2 x K2): within-team belief-propagation weights.
        Must be primitive.

    Cross-team weight fraction alpha = 0.5:
        each column of [A21;A11] gives alpha total to cross-team observations
        and (1-alpha) total to within-team belief propagation.
    """
    alpha = 0.5

    if topology == "fully_connected":
        # Every Team-1 agent observes every Team-2 agent equally.
        A21 = np.full((K2, K1), alpha / K2)
        A11 = np.full((K1, K1), (1 - alpha) / K1)
        # Every Team-2 agent observes every Team-1 agent equally.
        A12 = np.full((K1, K2), alpha / K1)
        A22 = np.full((K2, K2), (1 - alpha) / K2)

    elif topology == "ring":
        # Agent k in Team-1 directly observes Team-2 agent (k % K2).
        # The remaining (1-alpha) weight is spread uniformly among Team-1
        # agents for belief propagation.
        A21 = np.zeros((K2, K1))
        for k in range(K1):
            A21[k % K2, k] = alpha
        A11 = np.full((K1, K1), (1 - alpha) / K1)

        # Agent k in Team-2 directly observes Team-1 agent (k % K1).
        A12 = np.zeros((K1, K2))
        for k in range(K2):
            A12[k % K1, k] = alpha
        A22 = np.full((K2, K2), (1 - alpha) / K2)

    elif topology == "single_edge":
        # Only agent-0 from Team-1 observes agent-0 from Team-2 (and vice versa).
        # This is the minimum connectivity case the paper proves still converges.
        A21 = np.zeros((K2, K1))
        A21[0, 0] = alpha                        # one strictly positive entry

        # Column k=0 of A11 gets the remaining (1-alpha); other columns get 1.
        A11 = np.full((K1, K1), (1 - alpha) / K1)
        for k in range(1, K1):
            A11[:, k] = 1.0 / K1                 # full weight to belief propagation

        A12 = np.zeros((K1, K2))
        A12[0, 0] = alpha

        A22 = np.full((K2, K2), (1 - alpha) / K2)
        for k in range(1, K2):
            A22[:, k] = 1.0 / K2

    else:
        raise ValueError(f"Unknown topology '{topology}'. "
                         f"Choose from: fully_connected, ring, single_edge.")

    return {"A21": A21, "A11": A11, "A12": A12, "A22": A22}


def _normalize_left_stochastic(W: np.ndarray) -> np.ndarray:
    """Column-normalise W so each column sums to 1 (left-stochastic)."""
    col_sums = W.sum(axis=0, keepdims=True)
    return W / col_sums


def sanity_check(matrices: dict):
    """Assert all Assumption 1 conditions. Raises AssertionError on violation.

    Expected keys: A1, A2, A21, A12, A11, A22
    """
    A1  = matrices["A1"]
    A2  = matrices["A2"]
    A21 = matrices["A21"]
    A12 = matrices["A12"]
    A11 = matrices["A11"]
    A22 = matrices["A22"]

    # A1, A2 left-stochastic
    assert np.allclose(A1.sum(axis=0), 1, atol=1e-9), "A1 is not left-stochastic"
    assert np.allclose(A2.sum(axis=0), 1, atol=1e-9), "A2 is not left-stochastic"

    # Block matrices left-stochastic
    blk_21 = np.vstack([A21, A11])
    blk_12 = np.vstack([A12, A22])
    assert np.allclose(blk_21.sum(axis=0), 1, atol=1e-9), "[A21;A11] is not left-stochastic"
    assert np.allclose(blk_12.sum(axis=0), 1, atol=1e-9), "[A12;A22] is not left-stochastic"

    # Semi-weakly positive: at least one strictly positive entry
    assert (A21 > 0).any(), "A21 has no positive entry (violates Assumption 1 semi-weakly positive)"
    assert (A12 > 0).any(), "A12 has no positive entry (violates Assumption 1 semi-weakly positive)"

    # Primitivity via strong connectivity of the directed graph
    for mat, name in [(A1, "A1"), (A2, "A2"), (A11, "A11"), (A22, "A22")]:
        G = nx.from_numpy_array(mat, create_using=nx.DiGraph)
        assert nx.is_strongly_connected(G), \
            f"{name} is not primitive (directed graph is not strongly connected)"


if __name__ == "__main__":
    from config import K1, K2, TOPOLOGIES
    for topology in TOPOLOGIES:
        A1   = build_within_team_matrix(topology, K1)
        A2   = build_within_team_matrix(topology, K2)
        mats = build_cross_team_matrices(topology)
        sanity_check({**mats, "A1": A1, "A2": A2})
        print(f"{topology}: OK")
    print("All matrices passed Assumption 1 checks.")
