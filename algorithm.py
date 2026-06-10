# @Milica
#
# CD algorithm: within-team ATC diffusion + cross-team inference, iterated.
#
# TODO: implement run_cd(game, matrices, mu1, mu2, num_iters) -> history dict
# TODO: implement _within_team_diffusion — paper eq. (1)-(2)
# TODO: implement _cross_team_inference  — paper eq. (3)-(4)
# TODO: decide initialization (zeros vs random) for x, y, x_prime, y_prime
# TODO: inference uses x_{i-1}, y_{i-1} — confirm no look-ahead on updated values

import numpy as np
from config import K1, K2, M1, M2, MU1, MU2, NUM_ITERS
from gradients import stochastic_grad_x, stochastic_grad_y


def run_cd(game, matrices: dict, mu1=MU1, mu2=MU2, num_iters=NUM_ITERS) -> dict:
    """Run CD algorithm. Returns history dict with keys 'x' (T,K1,M1) and 'y' (T,K2,M2)."""
    # TODO: initialize agent states
    # TODO: allocate history arrays
    # TODO: loop — call _within_team_diffusion then _cross_team_inference each step
    pass


def _within_team_diffusion(x, y_prime, y, x_prime, game, matrices, mu1, mu2):
    """ATC step — returns x_new (K1,M1), y_new (K2,M2)."""
    # TODO: adapt: gradient step per agent
    # TODO: combine: matrix multiply by A1 / A2
    pass


def _cross_team_inference(x, y, x_prime, y_prime, matrices):
    """Inference step — returns y_prime_new (K1,M2), x_prime_new (K2,M1)."""
    # TODO: y_prime_new = A21 @ y + A11 @ y_prime
    # TODO: x_prime_new = A12 @ x + A22 @ x_prime
    pass
