# @Milica
#
# Stochastic gradients for quadratic loss:
# Q_k(z) = 0.5 * (u^T z - d)^2

import numpy as np

from config import M1


def stochastic_grad_x(x: np.ndarray, y_prime: np.ndarray, u: np.ndarray, d: float) -> np.ndarray:
    """Gradient with respect to x for Team 1."""
    z = np.concatenate([x, y_prime])
    err = u @ z - d
    return err * u[:M1]


def stochastic_grad_y(y: np.ndarray, x_prime: np.ndarray, u: np.ndarray, d: float) -> np.ndarray:
    """Gradient with respect to y for Team 2."""
    z = np.concatenate([x_prime, y])
    err = u @ z - d
    return err * u[M1:]
