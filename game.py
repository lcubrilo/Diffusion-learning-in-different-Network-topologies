# @Milica
#
# Quadratic game:
# Q_k(z) = 0.5 * (u_{k,i}^T z - d(k,i))^2
# d(k,i) = u_{k,i}^T z_star + noise

import numpy as np
from typing import List

from config import M, NOISE_VAR, RANDOM_SEED


class QuadraticGame:
    def __init__(self, z_star: np.ndarray, R_u: List[np.ndarray], rng=None):
        """
        Args:
            z_star: Nash equilibrium col{x*, y*}, shape (M,)
            R_u: list of K covariance matrices, each shape (M, M)
        """
        self.z_star = np.asarray(z_star)
        self.R_u = R_u
        self.rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng

        assert self.z_star.shape == (M,)
        for R in self.R_u:
            assert R.shape == (M, M)

    def generate_observation(self, k: int) -> tuple:
        """Return one streaming observation (u_{k,i}, d(k,i)) for agent k."""
        R = self.R_u[k]
        u = self.rng.multivariate_normal(np.zeros(M), R)

        noise = self.rng.normal(0.0, np.sqrt(NOISE_VAR))
        d = u @ self.z_star + noise

        return u, d

    def cost(self, z: np.ndarray, u: np.ndarray, d: float) -> float:
        """Evaluate Q_k(z)."""
        err = u @ z - d
        return 0.5 * err**2
