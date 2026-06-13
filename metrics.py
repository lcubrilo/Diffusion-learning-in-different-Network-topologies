# @Luka

import os
import numpy as np
import matplotlib.pyplot as plt
from config import K, K1, K2, M1


# ---------------------------------------------------------------------------
# Core error metrics — all return shape (T,), linear scale
# ---------------------------------------------------------------------------

def compute_msd_team1(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Average MSD across all Team-1 agents: mean_k ‖x_{k,i} − x*‖²."""
    x_star = z_star[:M1]
    err = history["x"] - x_star          # (T, K1, M1)
    return (err ** 2).sum(axis=2).mean(axis=1)   # (T,)


def compute_msd_team2(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Average MSD across all Team-2 agents: mean_k ‖y_{k,i} − y*‖²."""
    y_star = z_star[M1:]
    err = history["y"] - y_star          # (T, K2, M2)
    return (err ** 2).sum(axis=2).mean(axis=1)   # (T,)


def compute_msd_agent(history: dict, z_star: np.ndarray, team: int, agent: int) -> np.ndarray:
    """MSD of a single agent.

    team=1, agent=k  →  ‖x_{k,i} − x*‖²
    team=2, agent=k  →  ‖y_{k,i} − y*‖²
    """
    if team == 1:
        x_star = z_star[:M1]
        err = history["x"][:, agent, :] - x_star   # (T, M1)
    else:
        y_star = z_star[M1:]
        err = history["y"][:, agent, :] - y_star   # (T, M2)
    return (err ** 2).sum(axis=1)                   # (T,)


def compute_msd(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Average MSD across ALL agents (Team 1 + Team 2)."""
    t1 = compute_msd_team1(history, z_star) * K1
    t2 = compute_msd_team2(history, z_star) * K2
    return (t1 + t2) / K


def compute_fourth_order_moment(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Proxy for E[‖z̃_i‖⁴] from a single run — paper Theorem 1 eq 21b.

    Returns ‖z̃_i‖⁴ as a time series (shape (T,)).
    In dB: 10*log10(‖z̃_i‖⁴) = 40*log10(‖z̃_i‖).
    """
    x_star = z_star[:M1]
    y_star = z_star[M1:]
    x_err = history["x"] - x_star    # (T, K1, M1)
    y_err = history["y"] - y_star    # (T, K2, M2)
    sq = (x_err ** 2).sum(axis=(1, 2)) + (y_err ** 2).sum(axis=(1, 2))   # (T,)
    return sq ** 2                    # ‖z̃‖⁴


def compute_first_order_moment(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Proxy for ‖E[z̃_i]‖ from a single run — paper Theorem 1 eq 21c.

    E[z̃_i] is estimated via cumulative running mean of the error vectors
    (ergodic estimator).  The running mean cancels zero-mean noise, so this
    converges to the bias and looks visually smoother than the 4th-order curve.

    Returns ‖running_mean(z̃_i)‖ as a time series (shape (T,)).
    """
    x_star = z_star[:M1]
    y_star = z_star[M1:]

    # Flatten all agent errors into one long vector per timestep: (T, K1*M1 + K2*M2)
    x_err = (history["x"] - x_star).reshape(len(history["x"]), -1)   # (T, K1*M1)
    y_err = (history["y"] - y_star).reshape(len(history["y"]), -1)   # (T, K2*M2)
    z_err = np.concatenate([x_err, y_err], axis=1)                   # (T, K1*M1+K2*M2)

    # Cumulative mean: mean_i = (1/(i+1)) * sum_{t=0}^{i} z_err[t]
    running_mean = np.cumsum(z_err, axis=0) / np.arange(1, len(z_err) + 1)[:, None]

    return np.linalg.norm(running_mean, axis=1)   # ‖E[z̃_i]‖ proxy, shape (T,)


def compute_error_moment(history: dict, z_star: np.ndarray, order: int) -> np.ndarray:
    """Legacy wrapper kept for compatibility. Use compute_fourth_order_moment /
    compute_first_order_moment directly where possible."""
    if order == 4:
        return compute_fourth_order_moment(history, z_star)
    if order == 1:
        return compute_first_order_moment(history, z_star)
    x_star = z_star[:M1]
    y_star = z_star[M1:]
    x_err = history["x"] - x_star
    y_err = history["y"] - y_star
    sq = (x_err ** 2).sum(axis=(1, 2)) + (y_err ** 2).sum(axis=(1, 2))
    return np.sqrt(np.maximum(sq, 1e-300)) ** order


# ---------------------------------------------------------------------------
# dB conversion
# ---------------------------------------------------------------------------

def to_db(arr: np.ndarray) -> np.ndarray:
    """10 · log10(arr), clipped to avoid log(0)."""
    return 10.0 * np.log10(np.maximum(arr, 1e-30))


# ---------------------------------------------------------------------------
# Generic single-axis plot helper (used by topology_test, stepsize_test)
# ---------------------------------------------------------------------------

def plot_msd(curves: dict, title: str, save_path: str = None):
    """Plot one MSD-in-dB curve per entry in curves dict."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, msd_db in curves.items():
        ax.plot(msd_db, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSD (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
