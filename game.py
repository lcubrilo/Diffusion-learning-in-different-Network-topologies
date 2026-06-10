# @Milica  (z* provided by @Marija)
#
# Quadratic game: Q_k(z) = 0.5 * (u_{k,i}^T z - d(k,i))^2
# where d(k,i) = u_{k,i}^T z* + noise
#
# TODO: implement __init__    — store z_star, R_u, init rng
# TODO: implement generate_observation(k) — sample u ~ N(0, R_u[k]), compute d
# TODO: implement cost(z, u, d)           — evaluate Q_k
# TODO: confirm R_u matrices with @Marija (must match paper's numerical setup)

import numpy as np
from typing import List
from config import NOISE_VAR, RANDOM_SEED


class QuadraticGame:

    def __init__(self, z_star: np.ndarray, R_u: List[np.ndarray], rng=None):
        """
        Args:
            z_star: Nash equilibrium col{x*, y*}, shape (M,)  — from @Marija
            R_u:    list of K covariance matrices, each shape (M, M)
        """
        pass

    def generate_observation(self, k: int) -> tuple:
        """Return (u_ki, d_ki) for agent k."""
        pass

    def cost(self, z: np.ndarray, u: np.ndarray, d: float) -> float:
        """0.5 * (u^T z - d)^2"""
        pass
