# Diffusion Learning in Different Network Topologies

Stability and performance analysis of the Competing Diffusion (CD) algorithm on two-network problems, for varying step sizes and graph topologies. Team project for *Distributed Optimization and Applications* (Master AI, PMF Novi Sad).

## Module map

```
config.py       — all hyperparameters (K1, K2, M, mu, topologies, sweep values)
game.py         — quadratic loss definition, stochastic observation generator
gradients.py    — stochastic grad_x and grad_y for the quadratic loss
algorithm.py    — CD main loop: ATC within-team diffusion + cross-team inference
network.py      — combination matrix construction (A1, A2, A21, A12, A11, A22)
metrics.py      — MSD, error moments (E[‖z̃‖⁴], ‖E[z̃]‖), within-team disagreement, dB conversion

experiments/
  reproduce_figure1.py  — reproduce paper Fig 1 (validates implementation); regular & MC modes
  topology_test.py      — Fig 2: MSD vs topology (+ within-team disagreement panel)
  stepsize_test.py      — Fig 3: MSD floor vs step size mu
  visualize_topology.py — network graphs of the three topologies (topologies.png)

results/
  figures/      — saved plots (.png)
  data/         — cached observations (.npz) + raw topology MSD arrays (.npy); gitignored

main.py         — runs all three experiments end-to-end
```

## Dependency graph

![Dependency graph - https://app.eraser.io/workspace/LKQwy4Etd7aqs9V6M7Ju?origin=share](proj_struc.png)

## Team

| Role | Owner | Files |
|---|---|---|
| Algorithm | Milica | `algorithm.py`, `gradients.py`, `game.py` |
| Network & Experiments | Luka | `network.py`, `config.py`, `metrics.py`, `experiments/`, `main.py` |
| Theory & Nash equilibrium | Marija | provides `z*`, valid `μ` range, `R_u` — handed to Milica and Luka |

## Simulation parameters (paper Section 4)

| Parameter | Value | Source |
|---|---|---|
| K₁ | 2 | paper |
| K₂ | 4 | paper |
| M₁ | 5 | paper |
| M₂ | 10 | paper |
| μ₁ | 0.001 (default) | paper Figure 1 |
| μ₂ | 0.0005 (= μ₁/2) | paper Figure 1 |
| ZK_STAR | per-agent target z_k* = `standard_normal(15)*0.5` | paper eq 26 (per-agent targets) |
| R_U | identity + `COUPLING=0.5` off-diagonal blocks, same for all agents | makes teams couple |
| Z_STAR | computed Nash z* = `config._compute_nash(...)` from the {z_k*} | derived, not a placeholder |
| NOISE_VAR | 0.1 (σ²) | our choice |
| NUM_ITERS | 41,000 | our choice |
| MC_SEEDS | 20 (Monte-Carlo runs for Figure 1 moment panels) | our choice |

## Running

```bash
python main.py                            # all figures; Figure 1 = single-run (fast)
python main.py mc                         # Figure 1 = Monte-Carlo moments (slow, ~minutes)
python main.py both                       # Figure 1 = both versions

python -m experiments.reproduce_figure1 [regular|mc|both]   # default: regular
python -m experiments.topology_test
python -m experiments.stepsize_test
```

All scripts must be run from the project root directory.

### Figure 1 output files (regular vs MC never overwrite each other)

| mode | 6-panel | MSD overlay |
|---|---|---|
| `regular` (single run, fast; moment curves locked at 4×) | `figure1.png` | `figure1_overlay.png` |
| `mc` (Monte-Carlo `MC_SEEDS` runs; E[‖z̃‖⁴] & ‖E[z̃]‖ decoupled) | `figure1_mc.png` | `figure1_overlay_mc.png` |
