# Project Context for Claude

## What this project is

Team implementation of the **Competing Diffusion (CD) algorithm** for the DOA course at PMF Novi Sad (Master AI, Prof. Dušan Jakovetić). Goal: implement the algorithm from the base paper, vary step size μ and network topology, observe MSD convergence behavior.

## Key resources

| Resource | Location |
|---|---|
| **Base paper** | `base paper/STABILITY AND PERFORMANCE ANALYSIS...pdf` |
| **GitHub repo** | https://github.com/lcubrilo/Diffusion-learning-in-different-Network-topologies |
| **Internal planning doc** | https://docs.google.com/document/d/18KfD_D7FXlJx_ppzYVmYB4swAn4_z3pDby0cpKHUroI/edit |
| **Paper draft (Overleaf)** | https://www.overleaf.com/project/6a1d6e5031222265a9f8d091 |

## Deliverables

**Report deadline:** June 14, 2026 (email to Prof. Jakovetić or TA Davor Kumozec)

Three target figures:
1. **Figure 1** — Reproduce paper's MSD plots (validates correctness)
2. **Figure 2** — MSD convergence for varying topology (fully connected vs ring vs single cross-team edge)
3. **Figure 3** — MSD floor as a function of μ for fixed topology

## File structure

```
main.py                  — runs all experiments end-to-end
config.py                — all hyperparameters
network.py               — combination matrix construction (NetworkX)
game.py                  — quadratic game: loss functions, Nash equilibrium z*
gradients.py             — stochastic gradient computation
algorithm.py             — CD algorithm main loop
metrics.py               — MSD calculation and dB conversion

experiments/
  reproduce_figure1.py   — Figure 1: reproduce paper MSD plots
  topology_test.py       — Figure 2: MSD vs topology
  stepsize_test.py       — Figure 3: MSD floor vs μ

results/
  figures/               — saved plot outputs (.png/.pdf)
  data/                  — cached simulation arrays (.npy)

base paper/              — source PDF (read-only)
```

## Team roles

**Milica — Algorithm**
Owns: `algorithm.py`, `gradients.py`, `game.py`
Implements ATC within-team diffusion + cross-team inference. Generates synthetic observations (u_{k,i}, d(k,i)) using z* from Marija.

**Luka — Network & Experiments**
Owns: `network.py`, `config.py`, `metrics.py`, `experiments/`, `main.py`, all plots.
Builds combination matrices. Runs topology and step-size experiments. Produces all figures.

**Marija — Theory**
Owns: Nash equilibrium z*, step-size validity bounds, stability analysis section of paper.
Analytically computes z* and valid μ range from Assumption 2. Hands z* to Milica and Luka.

## Algorithm (CD = Competing Diffusion)

Two teams (size K₁, K₂). Each agent sees only local observations and limited signals from the other team.

**Step 1 — Within-team diffusion (ATC):**
```
x_{k,i} = Σ_{ℓ∈N¹} a¹_{ℓk} [x_{ℓ,i-1} − μ¹ · ∇̂_x J¹_ℓ(x_{ℓ,i-1}, y'_{ℓ,i-1})]
y_{k,i} = Σ_{ℓ∈N²} a²_{ℓk} [y_{ℓ,i-1} − μ² · ∇̂_y J²_ℓ(x'_{ℓ,i-1}, y_{ℓ,i-1})]
```

**Step 2 — Cross-team inference:**
```
y'_{k,i} = Σ_{ℓ∈N²} a²¹_{ℓk} y_{ℓ,i-1}  +  Σ_{ℓ∈N¹} a¹¹_{ℓk} y'_{ℓ,i-1}
x'_{k,i} = Σ_{ℓ∈N¹} a¹²_{ℓk} x_{ℓ,i-1}  +  Σ_{ℓ∈N²} a²²_{ℓk} x'_{ℓ,i-1}
```

## Key variables

| Symbol | Meaning |
|---|---|
| K₁, K₂ | Agent counts per team |
| M₁, M₂ | Strategy vector dimensions |
| z* = col{x*, y*} | Nash equilibrium — ground truth the algorithm converges to |
| μ¹, μ² | Step sizes (must satisfy Assumption 2) |
| A¹, A² | Within-team combination matrices (left-stochastic, primitive) |
| A²¹, A¹² | Cross-team direct observation sub-matrices (at least one strictly positive entry) |
| A¹¹, A²² | Within-team belief-propagation sub-matrices (primitive) |
| x_{k,i}, y_{k,i} | Agent k's estimate of its own team's strategy at iteration i |
| y'_{k,i}, x'_{k,i} | Agent k's inferred belief about the opponent's strategy |
| u_{k,i} ∈ ℝ^M | Regression vector observed at agent k, drawn from N(0, R_{k,u}) |
| d(k,i) | Scalar observation = u_{k,i}^T z* + v(k,i) |
| MSD | 10·log10(mean over agents of ‖estimate − z*‖²), plotted in dB |

**Quadratic loss:** Q_k(z, ξ) = ½ ‖u_{k,i}^T z − d(k,i)‖²

## Combination matrix requirements (Assumption 1)

The cross-team matrices are defined as block matrices:
- **A²¹_blk** = [A²¹; A¹¹] ∈ ℝ^(K×K₁) — used by team-1 agents to infer team-2's strategy
- **A¹²_blk** = [A¹²; A²²] ∈ ℝ^(K×K₂) — used by team-2 agents to infer team-1's strategy

Requirements:
- **A¹, A²**: left-stochastic + primitive
- **A²¹, A¹²** (upper blocks): at least one strictly positive entry
- **A¹¹, A²²** (lower blocks): primitive
- **A²¹_blk, A¹²_blk** (full block matrices): left-stochastic

## Simulation parameters (paper Section 4)

- K₁=2, K₂=4, M₁=5, M₂=10
- Iterations: 2×10⁵
- Game: quadratic
- Step sizes: μ₁=0.001, μ₂=0.0005 (always μ₂=μ₁/2); see MU_SWEEP in config.py for all four paper settings
- Topologies: fully connected, ring, single cross-team edge
- Z_STAR and R_U: placeholder zeros/identity in config.py — @Marija to replace with analytically computed values
- p_k weights: uniform (1/K_t per agent); only used by Marija for Assumption 2, not in ATC loop code
