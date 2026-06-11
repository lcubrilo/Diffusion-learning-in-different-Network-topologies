# @Milica
#
# CD algorithm: within-team ATC diffusion + cross-team inference, iterated.

import numpy as np

from config import K1, K2, M1, M2, MU1, MU2, NUM_ITERS, RANDOM_SEED
from gradients import stochastic_grad_x, stochastic_grad_y


def run_cd(game, matrices: dict, mu1=MU1, mu2=MU2, num_iters=NUM_ITERS) -> dict:
    """
    Run CD algorithm.

    Returns history dict with keys:
        x: shape (num_iters, K1, M1)
        y: shape (num_iters, K2, M2)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    x = rng.normal(size=(K1, M1))
    y = rng.normal(size=(K2, M2))

    y_prime = rng.normal(size=(K1, M2))   # Team 1 estimates Team 2 strategy
    x_prime = rng.normal(size=(K2, M1))   # Team 2 estimates Team 1 strategy

    hist_x = np.zeros((num_iters, K1, M1))
    hist_y = np.zeros((num_iters, K2, M2))

    for i in range(num_iters):
        x, y = _within_team_diffusion(
            x=x,
            y_prime=y_prime,
            y=y,
            x_prime=x_prime,
            game=game,
            matrices=matrices,
            mu1=mu1,
            mu2=mu2,
        )

        x_prime, y_prime = _cross_team_inference(
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            matrices=matrices,
        )

        hist_x[i] = x
        hist_y[i] = y

    return {"x": hist_x, "y": hist_y}


def _within_team_diffusion(x, y_prime, y, x_prime, game, matrices, mu1, mu2):
    """ATC step: adapt by stochastic gradient, then combine inside each team."""
    A1 = matrices["A1"]
    A2 = matrices["A2"]

    psi_x = np.zeros_like(x)
    psi_y = np.zeros_like(y)

    for ell in range(K1):
        u, d = game.generate_observation(ell)
        grad = stochastic_grad_x(x[ell], y_prime[ell], u, d)
        psi_x[ell] = x[ell] - mu1 * grad

    x_new = np.zeros_like(x)
    for k in range(K1):
        for ell in range(K1):
            x_new[k] += A1[ell, k] * psi_x[ell]

    for ell in range(K2):
        u, d = game.generate_observation(K1 + ell)
        grad = stochastic_grad_y(y[ell], x_prime[ell], u, d)
        psi_y[ell] = y[ell] - mu2 * grad

    y_new = np.zeros_like(y)
    for k in range(K2):
        for ell in range(K2):
            y_new[k] += A2[ell, k] * psi_y[ell]

    return x_new, y_new


def _cross_team_inference(x, y, x_prime, y_prime, matrices):
    """Inference step: agents update estimates of the opponent's strategy."""
    A21 = matrices["A21"]
    A11 = matrices["A11"]
    A12 = matrices["A12"]
    A22 = matrices["A22"]

    y_prime_new = np.zeros_like(y_prime)
    for k in range(K1):
        for ell in range(K2):
            y_prime_new[k] += A21[ell, k] * y[ell]
        for ell in range(K1):
            y_prime_new[k] += A11[ell, k] * y_prime[ell]

    x_prime_new = np.zeros_like(x_prime)
    for k in range(K2):
        for ell in range(K1):
            x_prime_new[k] += A12[ell, k] * x[ell]
        for ell in range(K2):
            x_prime_new[k] += A22[ell, k] * x_prime[ell]

    return x_prime_new, y_prime_new
