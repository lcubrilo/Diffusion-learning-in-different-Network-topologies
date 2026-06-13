# Team Reference Notes

Notes for Milica (algorithm.py, gradients.py, game.py) and Marija (theory).
These are implementation sketches and cross-checks — not authoritative, but meant to
help avoid integration bugs and to compare against what you actually deliver.

> **STATUS: code delivered (final).** These were pre-implementation sketches; a few no
> longer match the shipped code. Two important deviations:
> 1. **Per-agent targets.** The game uses a distinct target z_k* per agent (paper eq 26),
>    so `QuadraticGame.__init__(self, zk_star: list, R_u: list, rng=None)` takes a LIST of
>    K target vectors — not a single `z_star`. Gradients/algorithm/dimension notes below
>    are still accurate.
> 2. **config.py is filled in.** `ZK_STAR`, `R_U` (COUPLING=0.5) and the Nash `Z_STAR`
>    (via `_compute_nash`) are already computed in code — the "For Marija / replace these
>    placeholder lines" section below is obsolete.

---

## For Milica

### Function signatures Luka's code depends on

```python
# game.py  (DELIVERED signature — per-agent targets)
class QuadraticGame:
    def __init__(self, zk_star: list, R_u: list, rng=None): ...   # zk_star = list of K (M,) vectors
    def generate_observation(self, k: int) -> tuple:  # returns (u, d)
    def generate_all_observations(self, num_iters) -> tuple:      # (u_all, d_all), used by run_cd
    def cost(self, z: np.ndarray, u: np.ndarray, d: float) -> float: ...

# gradients.py
def stochastic_grad_x(x, y_prime, u, d) -> np.ndarray  # shape (M1,)
def stochastic_grad_y(y, x_prime, u, d) -> np.ndarray  # shape (M2,)

# algorithm.py
def run_cd(game, matrices: dict, mu1=MU1, mu2=MU2, num_iters=NUM_ITERS) -> dict
    # returns {'x': (T, K1, M1), 'y': (T, K2, M2)}
```

The history dict **only** needs `'x'` and `'y'` — Luka's `compute_msd` reads those two keys.
Beliefs `x'` and `y'` don't need to be stored in history.

### game.py sketch

```python
def __init__(self, z_star, R_u, rng=None):
    self.z_star   = z_star                          # shape (M,)
    self.R_u      = R_u                             # list of K matrices, each (M, M)
    self.rng      = rng or np.random.default_rng(RANDOM_SEED)
    self._noise_std = np.sqrt(NOISE_VAR)

def generate_observation(self, k):
    u = self.rng.multivariate_normal(np.zeros(len(self.z_star)), self.R_u[k])
    v = self.rng.normal(0, self._noise_std)
    d = u @ self.z_star + v
    return u, d     # u: (M,), d: scalar

def cost(self, z, u, d):
    return 0.5 * (u @ z - d) ** 2
```

### gradients.py sketch

For Q_k(z) = ½(u^T z − d)² with z = col{x, y_prime} (Team 1) or col{x_prime, y} (Team 2):

```python
def stochastic_grad_x(x, y_prime, u, d):
    z        = np.concatenate([x, y_prime])
    residual = u @ z - d
    return residual * u[:M1]    # shape (M1,)

def stochastic_grad_y(y, x_prime, u, d):
    z        = np.concatenate([x_prime, y])
    residual = u @ z - d
    return residual * u[M1:]    # shape (M2,)
```

### algorithm.py sketch

**Critical rule:** both ATC and inference steps must read **only i−1 values**.
Save copies at the top of each iteration *before* any update.

```python
def run_cd(game, matrices, mu1=MU1, mu2=MU2, num_iters=NUM_ITERS):
    A1, A2   = matrices['A1'], matrices['A2']
    A21, A11 = matrices['A21'], matrices['A11']
    A12, A22 = matrices['A12'], matrices['A22']

    x       = np.zeros((K1, M1))   # Team 1 strategy estimates
    y       = np.zeros((K2, M2))   # Team 2 strategy estimates
    x_prime = np.zeros((K2, M1))   # Team 2's belief about x*
    y_prime = np.zeros((K1, M2))   # Team 1's belief about y*

    hist_x = np.zeros((num_iters, K1, M1))
    hist_y = np.zeros((num_iters, K2, M2))

    for i in range(num_iters):
        x_old, y_old   = x.copy(), y.copy()
        xp_old, yp_old = x_prime.copy(), y_prime.copy()

        x, y             = _within_team_diffusion(x_old, yp_old, y_old, xp_old,
                                                   game, matrices, mu1, mu2)
        y_prime, x_prime = _cross_team_inference(x_old, y_old, xp_old, yp_old, matrices)

        hist_x[i], hist_y[i] = x, y

    return {'x': hist_x, 'y': hist_y}
```

**Matrix multiply convention — important.**
Combination matrices are left-stochastic: A[ℓ, k] = weight agent k gives agent ℓ.
To combine a matrix of row-vectors (rows = agents), use `A.T @ matrix`:

```python
def _within_team_diffusion(x, y_prime, y, x_prime, game, matrices, mu1, mu2):
    A1, A2 = matrices['A1'], matrices['A2']

    adapted_x = np.zeros((K1, M1))
    for k in range(K1):
        u, d = game.generate_observation(k)           # global index k
        adapted_x[k] = x[k] - mu1 * stochastic_grad_x(x[k], y_prime[k], u, d)

    adapted_y = np.zeros((K2, M2))
    for k in range(K2):
        u, d = game.generate_observation(K1 + k)      # global index K1+k
        adapted_y[k] = y[k] - mu2 * stochastic_grad_y(y[k], x_prime[k], u, d)

    return A1.T @ adapted_x, A2.T @ adapted_y


def _cross_team_inference(x, y, x_prime, y_prime, matrices):
    A21, A11 = matrices['A21'], matrices['A11']
    A12, A22 = matrices['A12'], matrices['A22']

    # A21: (K2×K1) → A21.T: (K1×K2) → A21.T @ y (K2×M2) = (K1×M2) ✓
    y_prime_new = matrices['A21'].T @ y + matrices['A11'].T @ y_prime

    # A12: (K1×K2) → A12.T: (K2×K1) → A12.T @ x (K1×M1) = (K2×M1) ✓
    x_prime_new = matrices['A12'].T @ x + matrices['A22'].T @ x_prime

    return y_prime_new, x_prime_new
```

### Dimension quick-reference

| Variable | Shape | Notes |
|---|---|---|
| x | (K1, M1) = (2, 5) | Team 1 estimates |
| y | (K2, M2) = (4, 10) | Team 2 estimates |
| y_prime | (K1, M2) = (2, 10) | Team 1's belief about y* |
| x_prime | (K2, M1) = (4, 5) | Team 2's belief about x* |
| u | (M,) = (15,) | Regression vector for one agent |
| d | scalar | Observation for one agent |
| A1 | (K1, K1) = (2, 2) | |
| A2 | (K2, K2) = (4, 4) | |
| A21 | (K2, K1) = (4, 2) | Upper block of A21_blk |
| A11 | (K1, K1) = (2, 2) | Lower block of A21_blk |
| A12 | (K1, K2) = (2, 4) | Upper block of A12_blk |
| A22 | (K2, K2) = (4, 4) | Lower block of A12_blk |

---

## For Marija

### Where to put your computed values  — ⚠️ OBSOLETE (already done in config.py)

This is now implemented. `config.py` contains:
- `ZK_STAR` — list of K per-agent targets z_k* (`standard_normal(M)*0.5`).
- `R_U` — list of K matrices: identity + `COUPLING=0.5` off-diagonal blocks (same per agent).
- `Z_STAR` — Nash z* = `_compute_nash(R_U, ZK_STAR, TEAM, P, M1, M2)` (solved exactly).

`Z_STAR` shape `(M,)`=`(15,)`=col{x*,y*}; `R_U` = list of K=6 (15,15) matrices.
Agent indices: 0..K1-1 = Team 1, K1..K-1 = Team 2. (Kept below: the Assumption-2 check still
applies and confirms the four MU_SWEEP pairs are valid for the shipped R_U.)

### Assumption 2 quick check (for the quadratic game)

With the quadratic loss Q_k(z) = ½‖u^T_{k,i} z − d(k,i)‖², the Hessians are:

- ∇²_x J^(1)_k = R_{k,u}[:M1, :M1]   (top-left 5×5 block of R_{k,u})
- ∇²_y J^(2)_k = R_{k,u}[M1:, M1:]   (bottom-right 10×10 block of R_{k,u})
- ∇²_{xy} J^(t)_k = R_{k,u}[:M1, M1:]  (off-diagonal 5×10 block)

The effective parameters (with uniform p_k = 1/K_t) become:

- ν^(1)_eff = λ_min( (1/K1) Σ_{k∈Team1} R_{k,u}[:M1,:M1] )
- ν^(2)_eff = λ_min( (1/K2) Σ_{k∈Team2} R_{k,u}[M1:,M1:] )
- δ^(21)_eff = ‖ (1/K1) Σ_{k∈Team1} R_{k,u}[:M1,M1:] ‖
- δ^(12)_eff = ‖ (1/K2) Σ_{k∈Team2} R_{k,u}[M1:,:M1] ‖

Assumption 2 (Eq. 7 of paper): 4(μ₂/μ₁) ν^(1)_eff ν^(2)_eff − (δ^(21)_eff + (μ₂/μ₁) δ^(12)_eff)² ≥ ε

With the identity default R_{k,u} = I₁₅: δ_eff = 0, so any positive step sizes satisfy it.
Your real R_{k,u} will have off-diagonal blocks → need to verify the inequality holds for
the four (μ₁, μ₂) pairs in MU_SWEEP.

### Open question from Google Docs

"Can A^(1) and A^(11) be set to the same matrix in practice?"
Mathematically they are independent. Setting them equal (A1 = A11) is a valid
simplification — Luka's network.py does *not* enforce equality but also does not
require them to differ. For the theory, they should probably differ (A^(1) moves actual
iterates; A^(11) propagates inferred beliefs), but it won't break anything to make them equal.
