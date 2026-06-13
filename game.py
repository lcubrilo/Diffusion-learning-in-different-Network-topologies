# @Milica
#
# Quadratic game:
# Q_k(z) = 0.5 * (u_{k,i}^T z - d(k,i))^2
# d(k,i) = u_{k,i}^T z_k* + noise   (paper eq 26: per-agent z_k*, NOT a shared z*)

import numpy as np
from typing import List

from config import M, NOISE_VAR, RANDOM_SEED


class QuadraticGame:
    def __init__(self, zk_star: List[np.ndarray], R_u: List[np.ndarray], rng=None):
        """
        Args:
            zk_star: per-agent target vectors z_k*, list of K arrays each shape (M,).
                     Paper eq 26 uses a distinct z_k* per agent — this creates genuine
                     competition since teams implicitly pull toward different targets.
            R_u:     list of K covariance matrices, each shape (M, M)
        """
        self.zk_star = [np.asarray(z, dtype=float) for z in zk_star]
        self.R_u     = R_u
        self.rng     = np.random.default_rng(RANDOM_SEED) if rng is None else rng

        K = len(zk_star)
        assert len(R_u) == K
        for z, R in zip(self.zk_star, self.R_u):
            assert z.shape == (M,), f"z_k* must be ({M},), got {z.shape}"
            assert R.shape == (M, M)

    def generate_observation(self, k: int) -> tuple:
        """Return one streaming observation (u_{k,i}, d(k,i)) for agent k."""
        u = self.rng.multivariate_normal(np.zeros(M), self.R_u[k])
        d = u @ self.zk_star[k] + self.rng.normal(0.0, np.sqrt(NOISE_VAR))
        return u, d

    def generate_all_observations(self, num_iters: int) -> tuple:
        """Pre-generate all observations for all agents across all iterations.

        Returns:
            u_all: shape (K, num_iters, M)  — regression vectors
            d_all: shape (K, num_iters)     — scalar observations
        """
        K = len(self.R_u)
        u_all = np.zeros((K, num_iters, M))
        d_all = np.zeros((K, num_iters))
        for k in range(K):
            u_all[k] = self.rng.multivariate_normal(np.zeros(M), self.R_u[k], size=num_iters)
            d_all[k] = u_all[k] @ self.zk_star[k] + self.rng.normal(0.0, np.sqrt(NOISE_VAR), size=num_iters)
        return u_all, d_all

    def cost(self, z: np.ndarray, u: np.ndarray, d: float) -> float:
        """Evaluate Q_k(z)."""
        err = u @ z - d
        return 0.5 * err**2
