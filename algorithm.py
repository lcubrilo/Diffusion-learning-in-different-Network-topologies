# @Milica
#
# CD algorithm: within-team ATC diffusion + cross-team inference, iterated.

import os
import hashlib
import numpy as np

from config import K1, K2, M1, M2, MU1, MU2, NUM_ITERS, RANDOM_SEED
from gradients import stochastic_grad_x, stochastic_grad_y

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "data")


def _obs_cache_path(game, num_iters: int) -> str:
    """Cache key includes a hash of ZK_STAR + R_U so it auto-invalidates on parameter changes."""
    zk_bytes = np.concatenate([z.flatten() for z in game.zk_star]).tobytes()
    r_bytes  = np.concatenate([R.flatten() for R in game.R_u]).tobytes()
    content_hash = hashlib.md5(zk_bytes + r_bytes).hexdigest()[:10]
    return os.path.join(_DATA_DIR, f"obs_seed{RANDOM_SEED}_iters{num_iters}_{content_hash}.npz")


def load_or_generate_observations(game, num_iters: int) -> tuple:
    """Return (u_all, d_all) from cache file if it exists, else generate and save.

    u_all: (K, num_iters, M)  — regression vectors, same for all algorithm runs
    d_all: (K, num_iters)     — scalar observations

    Cache is keyed on RANDOM_SEED, num_iters, and a hash of ZK_STAR + R_U.
    Changing any game parameter automatically creates a fresh cache file.
    """
    cache_path = _obs_cache_path(game, num_iters)
    if os.path.exists(cache_path):
        print(f"  Loading observations from cache: {os.path.basename(cache_path)}")
        data = np.load(cache_path)
        return data["u_all"], data["d_all"]

    print(f"  Generating {num_iters:,} observations for {K1 + K2} agents "
          f"(seed={RANDOM_SEED}) ...")
    u_all, d_all = game.generate_all_observations(num_iters)
    os.makedirs(_DATA_DIR, exist_ok=True)
    np.savez(cache_path, u_all=u_all, d_all=d_all)
    print(f"  Saved → {cache_path}")
    return u_all, d_all


def run_cd(game, matrices: dict, mu1=MU1, mu2=MU2, num_iters=NUM_ITERS, label="") -> dict:
    """Run CD algorithm using pre-generated (cached) observations.

    Returns history dict with keys:
        x: shape (num_iters, K1, M1)
        y: shape (num_iters, K2, M2)
    """
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Starting {num_iters:,} iterations (μ₁={mu1}, μ₂={mu2})...")

    u_all, d_all = load_or_generate_observations(game, num_iters)

    rng = np.random.default_rng(RANDOM_SEED)
    x       = rng.normal(size=(K1, M1))
    y       = rng.normal(size=(K2, M2))
    y_prime = rng.normal(size=(K1, M2))
    x_prime = rng.normal(size=(K2, M1))

    hist_x = np.zeros((num_iters, K1, M1))
    hist_y = np.zeros((num_iters, K2, M2))

    report_every = num_iters // 10
    for i in range(num_iters):
        if i > 0 and i % report_every == 0:
            print(f"{prefix}  {100 * i // num_iters}% done ({i:,}/{num_iters:,})")

        x_prev, y_prev = x, y

        x, y = _within_team_diffusion(
            x=x_prev, y_prime=y_prime,
            y=y_prev, x_prime=x_prime,
            u_all=u_all, d_all=d_all, step=i,
            matrices=matrices, mu1=mu1, mu2=mu2,
        )
        x_prime, y_prime = _cross_team_inference(
            x=x_prev, y=y_prev,
            x_prime=x_prime, y_prime=y_prime,
            matrices=matrices,
        )

        hist_x[i] = x
        hist_y[i] = y

    print(f"{prefix}Done.")
    return {"x": hist_x, "y": hist_y}


def _within_team_diffusion(x, y_prime, y, x_prime, u_all, d_all, step, matrices, mu1, mu2):
    """ATC step: adapt by stochastic gradient, then combine inside each team."""
    A1 = matrices["A1"]
    A2 = matrices["A2"]

    psi_x = np.zeros_like(x)
    for ell in range(K1):
        u, d = u_all[ell, step], d_all[ell, step]
        psi_x[ell] = x[ell] - mu1 * stochastic_grad_x(x[ell], y_prime[ell], u, d)

    psi_y = np.zeros_like(y)
    for ell in range(K2):
        u, d = u_all[K1 + ell, step], d_all[K1 + ell, step]
        psi_y[ell] = y[ell] - mu2 * stochastic_grad_y(y[ell], x_prime[ell], u, d)

    # A is left-stochastic: x_new[k] = sum_ell A[ell,k]*psi[ell] = (A.T @ psi)[k]
    return A1.T @ psi_x, A2.T @ psi_y


def _cross_team_inference(x, y, x_prime, y_prime, matrices):
    """Inference step: agents update estimates of the opponent's strategy."""
    A21 = matrices["A21"]   # (K2, K1)
    A11 = matrices["A11"]   # (K1, K1)
    A12 = matrices["A12"]   # (K1, K2)
    A22 = matrices["A22"]   # (K2, K2)

    # y_prime: (K1, M2).  A21.T:(K1,K2) @ y:(K2,M2) = (K1,M2) ✓
    y_prime_new = A21.T @ y + A11.T @ y_prime
    # x_prime: (K2, M1).  A12.T:(K2,K1) @ x:(K1,M1) = (K2,M1) ✓
    x_prime_new = A12.T @ x + A22.T @ x_prime

    return x_prime_new, y_prime_new
