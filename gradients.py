# @Milica
#
# Stochastic gradients for the quadratic loss Q_k(z) = 0.5 * (u^T z - d)^2:
#   grad_x = (u^T z - d) * u[:M1]
#   grad_y = (u^T z - d) * u[M1:]
#
# TODO: implement stochastic_grad_x(x, y_prime, u, d) -> np.ndarray shape (M1,)
# TODO: implement stochastic_grad_y(y, x_prime, u, d) -> np.ndarray shape (M2,)
# TODO: verify formula against paper's within-team diffusion equations

import numpy as np
from config import M1


def stochastic_grad_x(x: np.ndarray, y_prime: np.ndarray, u: np.ndarray, d: float) -> np.ndarray:
    pass


def stochastic_grad_y(y: np.ndarray, x_prime: np.ndarray, u: np.ndarray, d: float) -> np.ndarray:
    pass
