# @Luka

import os
import numpy as np
import matplotlib.pyplot as plt
from config import K, M1


def compute_msd(history: dict, z_star: np.ndarray) -> np.ndarray:
    """Mean-square deviation over all agents at each iteration, linear scale.

    history keys: 'x' (T, K1, M1), 'y' (T, K2, M2).
    z_star shape: (M,) = (M1+M2,).
    Returns array of shape (T,).
    """
    x_star = z_star[:M1]   # (M1,)
    y_star = z_star[M1:]   # (M2,)
    T = history["x"].shape[0]
    msd = np.zeros(T)
    for i in range(T):
        err = (np.sum((history["x"][i] - x_star) ** 2)
             + np.sum((history["y"][i] - y_star) ** 2))
        msd[i] = err / K   # mean over all K agents
    return msd


def to_db(msd: np.ndarray) -> np.ndarray:
    """Convert linear-scale MSD to decibels: 10 * log10(msd)."""
    return 10.0 * np.log10(np.maximum(msd, 1e-30))


def plot_msd(curves: dict, title: str, save_path: str = None):
    """Plot one MSD-in-dB curve per entry in curves dict.

    curves: {label: msd_db_array}.
    Saves to save_path if provided, otherwise calls plt.show().
    """
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
