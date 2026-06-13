# Project Context for Claude

## What this project is

Team implementation of the **Competing Diffusion (CD) algorithm** for the DOA course at PMF Novi Sad (Master AI, Prof. Dušan Jakovetić). Goal: implement the algorithm from the base paper, vary step size μ and network topology, observe MSD convergence behavior.

**Team:** Marija Kapriš · Luka Čubrilo · Milica Cvetić

## Key resources

| Resource | Location |
|---|---|
| **Base paper** | `base paper/original paper.pdf` (Shashkov, Pavan, Cai, Sayed — IEEE MLSP 2025) |
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
config.py                — all hyperparameters and sim settings
network.py               — combination matrix construction (NetworkX)
game.py                  — quadratic game: loss functions, data generation
gradients.py             — stochastic gradient computation
algorithm.py             — CD algorithm main loop
metrics.py               — MSD calculation and dB conversion

experiments/
  reproduce_figure1.py   — Figure 1: reproduce paper MSD plots
  topology_test.py       — Figure 2: MSD vs topology
  stepsize_test.py       — Figure 3: MSD floor vs μ

base paper/              — source PDF (read-only)
```

**Status: FULLY IMPLEMENTED.** All modules and the three experiments are complete and
run end-to-end via `main.py`. Core algorithm verified against the paper (ATC + cross-team
inference match eqs 3a/3b/4a/4b; per-agent z_k* matches eq 26; Nash solver verified;
Assumption 1 matrices pass `sanity_check`; Assumption 2 satisfied). All figures generated.

### Current implementation specifics (not in the original plan)

- **Per-agent targets.** `config.ZK_STAR` is a list of K per-agent target vectors z_k*
  (paper eq 26: `d(k,i)=u_{k,i}ᵀ z_k* + v`), drawn `standard_normal(M)*0.5`. The Nash
  point `Z_STAR` is computed from these by `config._compute_nash` (NOT a hand-supplied z*).
- **R_U / coupling.** `R_U` is identical for every agent: identity with `COUPLING=0.5`
  off-diagonal blocks (so teams actually couple; with R=I topology would have no effect).
  `NOISE_VAR=0.1`.
- **Sim length.** `NUM_ITERS = 41_000` (was originally planned at 2×10⁵). `RANDOM_SEED=42`.
- **Figure 1 has two modes** (`reproduce_figure1(mc=...)`, dispatch via `run_figure1(mode)`
  or `python main.py [regular|mc|both]`):
  - `regular` → single noisy run; moment dB-curves locked at 4×; fast → `figure1.png`,
    `figure1_overlay.png`.
  - `mc` → Monte-Carlo over `MC_SEEDS=20` runs; 4th-order = E[‖z̃‖⁴], **first-order =
    ‖E[z̃]‖** (paper eq 21c, vector-averaged bias — NOT E[‖z̃‖]); breaks the 4× lock →
    `figure1_mc.png`, `figure1_overlay_mc.png`.
- **Figure 2 finding:** MSD-to-Nash is **topology-invariant** here (gradient timescale ≫
  graph-mixing timescale; all graphs connected). The figure shows this (overlay + transient)
  plus a within-team-disagreement panel where topology IS visible (fully_connected reaches
  consensus in 1 step; ring/single_edge over ~5 steps).
- **`run_cd(..., seed=, use_cache=)`** added for the MC realisations (per-seed fresh obs/init).
- Known small caveats: Fig-1 moment *absolute* levels differ from the paper (NOISE_VAR scale);
  visualize_topology draws cross-team arrows reversed (cosmetic).

## Team roles

**Milica — Algorithm**
Owns: `algorithm.py`, `gradients.py`, `game.py`
Implements ATC within-team diffusion + cross-team inference. Generates synthetic observations (u_{k,i}, d(k,i)) using z* from Marija.

**Luka — Network & Experiments**
Owns: `network.py`, `config.py`, `metrics.py`, `experiments/`, all plots.
Builds combination matrices. Runs topology and step-size experiments. Produces all figures.

**Marija — Theory**
Owns: Nash equilibrium z*, step-size validity bounds, stability analysis section of paper.
Analytically computes z* and valid μ range from Assumption 2. Hands z* to Milica and Luka.

## Algorithm: exact update equations

Two teams (size K₁, K₂). Each agent sees only local observations and limited signals from the other team. At each iteration i, two steps run for all agents in parallel:

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

**Stochastic gradient for quadratic loss:**
```
Q_k(z, ξ) = ½ ‖u_{k,i}^T z − d(k,i)‖²
∇_x Q_k^(1) = u_{k,i}[0:M₁] · (u_{k,i}^T z − d(k,i))
∇_y Q_k^(2) = u_{k,i}[M₁:]  · (u_{k,i}^T z − d(k,i))
```

**Data generation per agent per step:**
- `u_{k,i} ~ N(0, R_{k,u})` where R_{k,u} is M×M (in code: same for all agents, COUPLING=0.5)
- `d(k,i) = u_{k,i}^T z_k* + v(k,i)`,  `v(k,i) ~ N(0, σ²_{k,v})` — **per-agent** target z_k*
  (paper eq 26), not the shared Nash z*. The Nash z* is then derived from the {z_k*}.

## Key variables

| Symbol | Meaning |
|---|---|
| K₁, K₂ | Agent counts per team |
| M₁, M₂ | Strategy vector dimensions |
| z* = col{x*, y*} | Nash equilibrium — ground truth; computed in code by `config._compute_nash` from the {z_k*} |
| μ¹, μ² | Step sizes (must satisfy Assumption 2); μ_max = max(μ¹, μ²) |
| A¹, A² | Within-team combination matrices for strategy iterates (left-stochastic, primitive) |
| A²¹, A¹² | Cross-team direct observation sub-matrices (≥1 strictly positive entry) |
| A¹¹, A²² | Within-team belief-propagation sub-matrices (primitive; independent of A¹, A²) |
| x_{k,i}, y_{k,i} | Agent k's estimate of its own team's strategy at iteration i |
| y'_{k,i}, x'_{k,i} | Agent k's inferred belief about the opponent's strategy |
| u_{k,i} ∈ ℝ^M | Regression vector drawn from N(0, R_{k,u}) each step |
| d(k,i) | Scalar observation = u_{k,i}^T z* + noise |
| z_k* | Per-agent target vector (paper eq 26); `config.ZK_STAR`. Distinct from Nash z*. |
| MSD | 10·log10(mean ‖estimate − z*‖²) per agent, plotted in dB over NUM_ITERS (=41,000) iterations |

## Combination matrix requirements (Assumption 1)

Block structure of cross-team matrices:
- **A²¹_blk** = [A²¹; A¹¹] ∈ ℝ^{K×K₁} — used by Team 1 agents to infer Team 2's strategy
- **A¹²_blk** = [A¹²; A²²] ∈ ℝ^{K×K₂} — used by Team 2 agents to infer Team 1's strategy

Requirements:
- A¹, A²: left-stochastic + primitive
- A²¹, A¹² (upper blocks): ≥1 strictly positive entry
- A¹¹, A²² (lower blocks): primitive
- A²¹_blk, A¹²_blk (full block matrices): left-stochastic overall

In practice: uniform weights based on graph topology (each neighbor gets equal weight, column sums normalized to 1).

## Assumption 2: step-size validity condition (Marija's job)

Step sizes are valid if:
```
4·(μ²/μ¹)·ν_eff¹·ν_eff²  −  (δ_eff²¹ + (μ²/μ¹)·δ_eff¹²)²  ≥ ε
```
- ν_eff^(t) = inf of weighted sum of own-Hessian norms — effective strong convexity per team
- δ_eff^(t't) = sup of weighted sum of cross-Hessian norms — cross-team coupling strength

Intuition: stabilizing curvature must dominate destabilizing competition coupling.

## Theorems (summary)

**Theorem 1 (Stability):** Under Assumptions 1–4, for sufficiently small μ_max = max(μ¹, μ²):
- `limsup E‖z̃_i‖²  = O(μ_max)` — 2nd moment bounded
- `limsup E‖z̃_i‖⁴  = O(μ_max²)` — 4th moment bounded
- `limsup ‖z̃_i‖    = O(μ_max)` — almost sure

**Theorem 2 (MSD):** Under Assumptions 1–5, steady-state MSD per agent scales as O(μ_max). Convergence rate α = 1 − Ω(μ_max). Closed-form expression via Lyapunov equation — **not implemented** (too complex, simulated curves only).

## What to skip

**Theoretical MSD prediction via Lyapunov** (computing X to overlay Tr(J_k X) on plots) — too complex for project scope. Focus on simulated MSD curves only.

## Paper status (Overleaf)

Sections with real content written:
- Introduction (4 references, motivating examples)
- Problem Formulation (objective functions, ATC update equations, cross-team inference equations)
- Theoretical Analysis: Assumptions 1–5 fully written; Theorems 1–2 properly stated with O(μ_max) bounds; full error dynamics (Θ, A, M, H, B, S, X defined)

Stub / placeholder sections:
- Implementation (generic text, no real parameters)
- Simulation Results (placeholder1/2/3.png — to be replaced with actual figures)
- Extended Experiments, Discussion, Conclusion (generic filler)

Open question in paper (red text): whether to use uniform combination weights or vary them across topology experiments. Decision is Luka's.
