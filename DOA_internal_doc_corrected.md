# DOA Projekat interni dokument

**Links:**
- Git Repo
- This document
- Overleaf (LaTeX of the paper)

## Project idea

Roughly speaking: let's implement the CD algorithm from the given paper, vary mu and network topology, and see how it behaves depending on those changes.

Final goal/deliverable? Unclear, one option is aiming for these three plots:
- **Figure 1:** reproduce the paper's MSD plots as validation that implementation is correct
- **Figure 2:** MSD convergence for varying topology sparsity — fully connected vs ring vs single cross-team edge. *(Finding from the runs: the MSD-to-Nash floor is essentially topology-invariant in this regime — the gradient timescale 1/μ greatly exceeds the graph-mixing timescale, and every connected graph reaches the same steady state. The figure presents this invariance, plus a within-team-disagreement panel where topology IS visible: fully connected reaches consensus in 1 step, ring/single-edge over ~5 steps. This is consistent with the paper's claim that a single cross-team edge already converges.)*
- **Figure 3:** MSD floor as a function of μ for fixed topology

## Zadaci po osobi:

Ovo je okviran predlog za početak, i menjaće se kroz vreme

### Role 1: The Algorithm - Milica
Objective: Build the core logic of the Competing Diffusion (CD) algorithm and the simulated "World."
- Implement Algorithm 1: Code the iterative update loops found in the source material. This includes:
  - Within-Team Diffusion: Implementing the "Adapt-then-Combine" (ATC) step where agents take a gradient step and average with teammates.
  - Cross-Team Inference: Coding the step where agents estimate the opponent's move using limited cross-team observations.
- Implement stochastic gradient formula for the quadratic loss, based on the formula documented by Marija.
- Data Generation: Create a "streaming data" class to simulate the Quadratic Game. You will generate the observations (u_{k,i} and d(k,i)) for each agent at every time step using **that agent's own ground-truth vector z_k\*** (one per agent — paper eq. 26), parameterized by the mathematician. **Important:** data is generated from the per-agent targets z_k\*, NOT from the shared Nash equilibrium z\*. (With a single shared z\* the teams would trivially agree and there would be no real competition.) The Nash z\* is derived from the set of {z_k\*} and is used only to measure error.
- Modular Integration: Ensure the algorithm is modular so it can easily accept different graph structures and step-size parameters for testing.
- Paper writing: explaining the implementation section, and any shared parts (eg intro)

### Role 2: The Network & Performance - Luka
Objective: Handle the infrastructure (graph theory) and visualize how well the system learns.
- Network Topology (Graph Design): Use NetworkX to build the connection matrices (A). You must ensure they meet the paper's requirements: they must be left-stochastic (columns sum to 1) and primitive (the network is connected).
- Independent Experimentation: Run two separate sets of simulations to "stress test" the algorithm's performance:
  - Topology Test: Compare how the algorithm converges when teams are fully connected vs. when they are connected by only a single link. Could be multiple plots for a handful of topologies, or many topologies, in which case topology goes into x axis (actually metric). Which metric? Testing which one has biggest correlation?
  - Learning Rate Test: Compare the steady-state "error floor" across different step-sizes (μ). The paper's Fig. 1 displays MSD at four step-size settings (its four MSD panels); **Figure 3 consolidates those floors into a single floor-vs-μ curve — that consolidation is our own plot**, motivated by the O(μ_max) steady-state bound (Theorem 1/2).
- Call the MSD prediction function to overlay theoretical vs simulated curves on plots. Probably too complex to implement, so skipped.
- Visualization: Calculate the Mean-Square Deviation (MSD) — the average squared distance from the truth — and plot it in dB (decibels) over the simulation horizon (NUM_ITERS = 41,000 iterations in our runs; the paper uses 2×10⁵).
- Paper writing: explaining the graph topology section and the plots/visualization and any shared parts (eg intro).
- Available as much as needed for the presentation/paper defense.

### Role 3: The Theory & Strategy - Marija
Objective: Define the "rules" of the game and translate the paper's stability requirements into code parameters.
- Problem Formulation: Define the specific Quadratic Game parameters (like covariance matrices R_{k,u} and noise levels σ_{k,v}) **and the per-agent target vectors z_k\***.
- Establish the "Ground Truth": Define the per-agent targets z_k\*, then compute the Nash Equilibrium (z\*) from the set of {z_k\*} (an exact linear solve). **Milica needs the per-agent z_k\*** to generate each agent's data; **Luka needs the Nash z\*** to measure the error (MSD). Note the dependency: the z_k\* are the inputs, z\* is derived from them.
- Document the stochastic gradient formula so Milica can implement it
- Parameter Tuning: Determine the valid range for the step-sizes (μ). You will use the paper's Assumption 2 to ensure the chosen values are "sufficiently small" to guarantee that the system remains stable and doesn't "blow up".
- Theoretical MSD Prediction Function: using Lyapunov equation. Probably too complex, skipped.
- Analysis: Lead the report-writing by explaining why the algorithm is stable in your tests. You will link the observed results (e.g., why a smaller μ results in a lower error floor) back to the paper's first- and fourth-order stability claims.
- Looking into literature: finding if any relevant similar papers exist to be listed in bibliography, maybe specific examples/applications, or descriptions how CD algorithm responds to mu/topology changes, or on diffusion learning for context, or on network games. Not a full literature review.

---

## Glossary/rečnik ključnih varijabli/konstanti

Pošto ima puno grčkih slova, latiničnih slova, podvučeno, nadvučeno, subskript, superscript - na jednom mestu ćemo ih sve definisati prostim rečima, pa pojasniti teorijski i provući kroz ilustrativan primer prodavnica da bismo lakše intuitivno razumeli šta se dešava.

### 1. Problem Given
Everything here is fixed before the algorithm starts. You define these once when setting up the simulation.

- **K₁** — How many agents are in Team 1. FreshMart/QuickShop: K₁ = 2 (two FreshMart store locations).
- **K₂** — How many agents are in Team 2. FreshMart/QuickShop: K₂ = 4 (four QuickShop store locations).
- **K** — Total agents across both teams. K = K₁ + K₂. FreshMart/QuickShop: K = 6.
- **M₁** — How many numbers Team 1 is optimizing — the length of their strategy vector. FreshMart/QuickShop: M₁ = 2 (FreshMart sets one price for bread, one for milk).
- **M₂** — Same as M₁ but for Team 2. FreshMart/QuickShop: M₂ = 3 (QuickShop sets prices for bread, milk, cheese).
- **M** — Total strategy dimension across both teams. M = M₁ + M₂. FreshMart/QuickShop: M = 5.
- **N^(1)** — The set of labels identifying Team 1's agents. N^(1) = {1, …, K₁}. FreshMart/QuickShop: N^(1) = {1, 2}.
- **N^(2)** — The set of labels identifying Team 2's agents. N^(2) = {K₁+1, …, K}. FreshMart/QuickShop: N^(2) = {3, 4, 5, 6}.
- **N** — All agents together. N = N^(1) ∪ N^(2). FreshMart/QuickShop: N = {1, 2, 3, 4, 5, 6}.

- **p_k^(t)** — How much agent k's loss counts toward its team's total objective. A positive scalar set by design. All weights within a team sum to 1. In the simplest case every agent is weighted equally. FreshMart/QuickShop: p₁^(1) = p₂^(1) = 0.5 — both FreshMart stores contribute equally to FreshMart's total loss. **Note: the p_k weights appear in the team objective J^(t) and in Marija's Assumption-2 computation. They do NOT appear in the ATC update loop — each agent simply steps along its own local gradient ∇Q_k with no explicit weighting. The weighting is implicit: with the chosen (uniform, doubly-stochastic) combination matrices, the dynamics converge to the p_k-weighted Nash equilibrium.**

- **Q_k^(t)(x, y, ξ)** — The loss agent k computes from a single observed data sample ξ. One noisy estimate of how bad the current strategy is. For the quadratic game this is (1/2)‖u_{k,i}^T z − d(k,i)‖². FreshMart/QuickShop: How poorly FreshMart store 1's current pricing explains today's customer demand — computed from a single day's observation.
- **J_k^(t)(x, y)** — The true expected loss of agent k — the average of Q_k^(t) over all possible data samples. This is what the agent would minimize if it had infinite data. In practice it can't be computed directly, only approximated sample by sample. FreshMart/QuickShop: FreshMart store 1's average pricing error across all possible market days, given both chains' current prices.
- **J^(1)(x, y)** — Team 1's total objective. Weighted sum of all Team 1 agents' expected losses: J^(1) = Σ p_k^(1) J_k^(1). This is what Team 1 collectively minimizes by choosing x. FreshMart/QuickShop: FreshMart's overall pricing performance — a weighted average across both store locations.
- **J^(2)(x, y)** — Same as J^(1) but for Team 2, minimized over y. FreshMart/QuickShop: QuickShop's overall pricing performance across all four stores.

- **z_k\*** — Agent k's OWN ground-truth target vector, ∈ R^M. The "true" parameters behind agent k's local observations: d(k,i) = u_{k,i}^T z_k\* + v(k,i) (paper eq. 26). Each agent has its own z_k\*, and they generally differ across agents — that difference is what creates genuine competition (with one shared target the teams would trivially agree). Fixed before training; defined by Marija. FreshMart/QuickShop: each store's local market responds a bit differently, so each store k carries its own private "true" demand/price model z_k\* ∈ R^5.
- **x\*** — The price vector Team 1 would settle on if both teams had fully adapted to each other. A fixed point: Team 1 can't do better by changing x\* if Team 2 is already at y\*. Unknown at runtime — finding it is the whole point of running the algorithm. FreshMart/QuickShop: e.g. x\* = [2.50, 1.80] — equilibrium prices for bread and milk at FreshMart.
- **y\*** — Same as x\* but for Team 2. FreshMart/QuickShop: e.g. y\* = [2.30, 1.90, 4.50] — equilibrium prices for bread, milk, cheese at QuickShop.
- **z\*** — x\* and y\* stacked into one vector. The full Nash equilibrium. z\* = col{x\*, y\*} ∈ R^M. The ground truth the whole algorithm converges toward. **It is DERIVED from the per-agent {z_k\*} by solving the Nash condition (an exact linear solve) — it is not handed in directly and is not used to generate data.** Luka uses z\* to measure MSD (how far each estimate is from z\*); the algorithm never sees it directly, only chases it. FreshMart/QuickShop: z\* = [2.50, 1.80, 2.30, 1.90, 4.50] — the chain-wide equilibrium prices that emerge from all the stores' competing local models.

#### Combination Matrices
Six matrices total, all fixed before training starts. They encode who communicates with whom and how much they trust each other. Understanding the difference between them is one of the trickier parts of this paper.

Organized in two groups:
- **Group 1 — Within-team diffusion (2 matrices):** Used in the within-team diffusion step. Govern how each team shares its actual strategy iterates internally.
- **Group 2 — Cross-team inference (4 matrices):** Used in the cross-team inference step. Govern how each team builds and spreads its beliefs about the opponent's strategy. These four come as two larger block matrices in the paper:

```
A^(21)_blk = [ A^(21) ]    total shape: K×K₁   (here 6×2)
             [ A^(11) ]

A^(12)_blk = [ A^(12) ]    total shape: K×K₂   (here 6×4)
             [ A^(22) ]
```

The _blk suffix is notation specific to this paper meaning "this is the full block matrix, which splits into two sub-matrices stacked vertically." The upper block is the direct cross-team observation; the lower block is the within-team propagation of inferred beliefs.

- **A^(1)** ∈ R^{K₁×K₁} — Within-Team-1 trust matrix for actual strategy iterates. Entry (ℓ,k) is the weight agent k places on agent ℓ's current strategy estimate when combining. Columns sum to 1. Must be primitive — all agents can eventually influence all others. FreshMart/QuickShop: 2×2 matrix. e.g. A^(1) = [[0.6, 0.4], [0.4, 0.6]] — each FreshMart store weights itself 60% and its teammate 40%.
- **A^(2)** ∈ R^{K₂×K₂} — Same as A^(1) but for Team 2's actual strategy iterates. FreshMart/QuickShop: 4×4 matrix governing how QuickShop stores share their strategy estimates.
- **A^(21)** ∈ R^{K₂×K₁} — How much each Team 1 agent weights direct observations of Team 2 agents' actual iterates when forming its belief about y\*. At least one entry must be nonzero — this is the minimum connectivity condition between the two teams that the paper's convergence result relies on. FreshMart/QuickShop: 4×2 matrix. Entry (ℓ,k): how much FreshMart store k trusts its direct observation of QuickShop store ℓ's current prices.
- **A^(11)** ∈ R^{K₁×K₁} — How Team 1 agents share their inferred beliefs about y\* among themselves. Despite the name this is not the same as A^(1). A^(1) moves actual strategy iterates x_{k,i}; A^(11) moves inferred beliefs y'_{k,i}. The two matrices are structurally independent and can differ. FreshMart/QuickShop: 2×2 matrix. How much FreshMart stores share and blend their guesses about QuickShop's equilibrium prices with each other.
- **A^(12)** ∈ R^{K₁×K₂} — Same as A^(21) but in the other direction: how much Team 2 agents weight direct observations of Team 1's actual iterates when forming beliefs about x\*. FreshMart/QuickShop: 2×4 matrix. How much each QuickShop store trusts its direct observation of FreshMart's current prices.
- **A^(22)** ∈ R^{K₂×K₂} — Same as A^(11) but for Team 2: how QuickShop stores share their inferred beliefs about x\* among themselves. FreshMart/QuickShop: 4×4 matrix.

**✓ Resolved (was an open question): A^(1) and A^(11) both operate within Team 1 but move different things — actual iterates vs inferred beliefs. They CAN be set to the same matrix in practice without breaking correctness — both only need to be left-stochastic and primitive. The paper keeps them independent because they conceptually govern different quantities, but there is no theoretical requirement that they differ.**

### 2. Hyperparameters
Chosen before training. Not updated by the algorithm. The only real hyperparameters here are the step sizes.

- **μ^(1)** — Step size for Team 1. Controls how big a gradient step each Team 1 agent takes each iteration. Must be small enough to satisfy Assumption 2 (the convergence condition). Constant throughout training. FreshMart/QuickShop: e.g. μ^(1) = 0.001.
- **μ^(2)** — Step size for Team 2. Can differ from μ^(1) because the two loss landscapes may have different curvature. (The paper's own four experiments happen to use μ^(2) = μ^(1)/2, but Assumption 2 only constrains the ratio, so other valid pairs exist.) FreshMart/QuickShop: e.g. μ^(2) = 0.0008.
- **μ_max** — The larger of the two step sizes: max{μ^(1), μ^(2)}. Not an independent choice — derived from the two above. Directly controls the steady-state MSD floor: smaller μ_max means agents converge closer to z\* but adapt more slowly. Appears in every convergence bound in the paper. FreshMart/QuickShop: μ_max = 0.001.

### 3. Stochastic Observations
What each agent receives from the environment each iteration. The algorithm reads these but has no control over them. In the simulation these are generated synthetically.

- **u_{k,i}** ∈ R^{1×M} — The demand sensitivity vector observed at agent k on iteration i. Each of its M entries says: if this product's price changes by $1, how many customers does store k gain or lose? Sampled fresh every iteration from N(0, R_{k,u}). FreshMart/QuickShop: For a FreshMart store, u_{k,i} has 5 entries — sensitivities to FreshMart bread, FreshMart milk, QuickShop bread, QuickShop milk, QuickShop cheese. e.g. u_{k,i} = [-0.8, +0.2, +0.5, -0.1, +0.3]: raising FreshMart bread price by $1 loses 0.8 customers; QuickShop raising their bread price by $1 sends 0.5 customers to FreshMart.
- **d(k,i)** — The actual demand change observed at store k on day i. A single scalar. Generated as d(k,i) = u_{k,i}^T z_k\* + v(k,i) — agent k's OWN true model (per-agent z_k\*, not the shared Nash z\*) plus noise. This is the learning signal the algorithm uses each iteration. FreshMart/QuickShop: e.g. 3.40 — store k gained a net 3.4 customers today relative to baseline.
- **v(k,i)** — Observation noise on demand. A small random scalar, independent across stores and days. Drawn from N(0, σ²_{k,v}). Represents unmodeled factors — weather, a local event, random footfall. FreshMart/QuickShop: e.g. 0.05 — a tiny random disturbance on today's demand reading.
- **ξ_{k,i}** — Shorthand for the full random draw at agent k on iteration i: the pair (u_{k,i}, v(k,i)) together. Used in the loss function definition to make clear that all randomness at this agent on this day comes from one draw. FreshMart/QuickShop: Everything random that FreshMart store 1 experiences on day i — the sensitivity vector it observes and the noise on its demand reading.

### 4. Algorithm Iterates
The internal state the algorithm maintains and updates every iteration. This is what actually runs during training.

- **x_{k,i}** ∈ R^{M₁} — Agent k's current guess of x\* at iteration i. Not the store's actual prices — its current best estimate of what Team 1's optimal price vector should be. All K₁ Team 1 agents are chasing the same x\*; they start with different random guesses and align through diffusion over time. Updated by the within-team diffusion step (eq. 3a). FreshMart/QuickShop: FreshMart store 1's guess at iteration i. e.g. [1.20, 3.10] early in training (random, wrong), gradually converging toward x\* = [2.50, 1.80].
- **y_{k,i}** ∈ R^{M₂} — Agent k's current guess of y\* at iteration i. For k ∈ N^(2): same role as x_{k,i} but within Team 2. Updated by eq. 3b. FreshMart/QuickShop: QuickShop store 2's current best guess at QuickShop's equilibrium prices for bread, milk, and cheese.
- **y'_{k,i}** ∈ R^{M₂} — Agent k ∈ N^(1)'s inferred belief about y\*. Team 1 can't see Team 2's internal strategy discussions directly. It builds this belief by combining two sources: (a) direct observations of neighboring Team 2 agents' actual iterates y_{ℓ,i}, weighted by A^(21), and (b) its own teammates' previous inferred beliefs y'_{ℓ,i-1}, weighted by A^(11). Updated by the cross-team inference step (eq. 4a). This is what makes competition possible without full information. FreshMart/QuickShop: FreshMart store 1's running belief about what QuickShop's equilibrium prices are. Starts wrong, gradually tracks y\* as signals leak through the network.
- **x'_{k,i}** ∈ R^{M₁} — Agent k ∈ N^(2)'s inferred belief about x\*. Symmetric to y'_{k,i} — built by Team 2 agents observing Team 1. Updated by eq. 4b. FreshMart/QuickShop: QuickShop store 3's running belief about FreshMart's equilibrium prices.
- **z_i** — All iterates across all agents stacked into one vector. z_i = col{ {x_{k,i}}, {x'_{k,i}}, {y_{k,i}}, {y'_{k,i}} }. Not computed explicitly by any single agent — a theoretical convenience for writing the recursion compactly. Useful when thinking about the system as a whole. FreshMart/QuickShop: All 6 agents' current guesses stacked — a vector of total length M·K = 30.

### 5. Analysis Machinery
These variables exist to prove Theorems 1 and 2. Milica and Luka do not need these to implement or run the algorithm. Marija needs X and B to compute the theoretical MSD predictor. Everyone will encounter these symbols when reading Section 3 of the paper.

- **s_{k,i}^(t)** — The gradient noise at agent k on iteration i. The difference between the stochastic gradient computed from one sample and the true expected gradient. Represents the error from learning off one day instead of averaging over all days. Has zero mean — it doesn't bias the algorithm, it just adds variance around the true direction. FreshMart/QuickShop: The error FreshMart store 1 makes by estimating the gradient from today's single demand observation rather than the true expectation.
- **F_{i-1}** — The complete history of all iterates across both teams up to iteration i-1. Used to state conditional expectations precisely in the noise assumptions. Not computed anywhere — purely a mathematical bookkeeping device for the proofs.
- **z̃_i** — How far all agents collectively are from Nash equilibrium at iteration i. The difference between the stacked target and the current z_i. The theorems bound the moments of its norm — proving this stays small is what convergence means formally.
- **B_{i-1}, B** — The matrix driving the error recursion forward each step. B_{i-1} is the iteration-varying version; B is its deterministic limit evaluated at Nash equilibrium. Whether the algorithm converges reduces to whether B is stable — specifically whether its spectral radius is less than 1.
- **Θ** — A large block matrix encoding the full network topology — both within-team and cross-team connections for both teams — in one object. Appears in the definition of B = Θ^T − A^T M H.
- **H_i, H** — Integrated Hessian matrices capturing the local curvature of the loss landscape near z\*. Needed to linearize the error recursion around Nash equilibrium. H is their limit evaluated exactly at z\*. Required to build B and therefore to compute the theoretical MSD predictor.
- **b** — Bias vector: stacked gradients of each agent's local loss evaluated at z\*. Nonzero because individual agents' gradients don't vanish at Nash equilibrium — only the weighted team sum does.
- **X** — Steady-state error covariance matrix. Defined as X = (Σ_{n=0}^∞ B^{nT} B^n) · A^T M S M A. Summarizes the accumulated effect of gradient noise on long-run deviation from z\*. The key quantity in Theorem 2 — MSD_k = Tr(J_k^(t) X). Computed by solving a discrete Lyapunov equation using scipy.
- **S, S^(t)** — Steady-state gradient noise covariance. Stacks the per-agent noise covariances evaluated at Nash equilibrium. Captures how noisy the gradient estimates are once the system has converged. Input to the formula for X.
- **J_k^(1), J_k^(2)** — Selector matrices for agent k. Binary diagonal matrices that extract agent k's block from the full stacked error vector. Used to go from the global MSD bound (covering all agents) to the per-agent MSD bound in Theorem 2. Note: these are calligraphic J in the paper — different from the loss functions J^(1), J^(2).
- **MSD_k** — Mean square deviation of agent k at steady state. How close agent k's estimate gets to Nash equilibrium in the long run, on average. The main performance metric. Theorem 2 gives MSD_k = Tr(J_k^(t) X). Plotted in dB over iterations in all the paper's figures. FreshMart/QuickShop: How far FreshMart store 1's converged price estimate is from x\* on average, in ($/product)².

### 6. Regularity Constants
Properties of the problem itself — not set by anyone, not updated by the algorithm. You verify these hold for your specific game parameters. If they do, and step sizes are small enough, the theory guarantees convergence. Presented as a table since entries are short.

| Symbol | What it bounds | Plain meaning |
|---|---|---|
| δ_k^(1), δ_k^(2) | Lipschitz constant of agent k's gradient | How fast the gradient can change as strategy changes. Also bounds the Hessian magnitude. |
| κ_d | Hessian Lipschitz constant | How fast the second derivative changes near z\*. Controls higher-order error terms in the Taylor expansion. |
| ν_eff^(1), ν_eff^(2) | Strong monotonicity constant per team | Lower bound on each team's loss curvature in its own strategy direction. Guarantees a unique Nash equilibrium exists. |
| δ_eff^(21), δ_eff^(12) | Cross-gradient coupling strength | How strongly Team 1's loss reacts to Team 2's strategy and vice versa. If too large relative to ν_eff, no step size can guarantee convergence. |
| β̄_k^(t), σ̄_k^(t) | Gradient noise magnitude bounds | β̄ controls how noise grows with iterate magnitude; σ̄ controls the irreducible noise floor. Both appear in the 4th moment bound on gradient noise. |
| γ ∈ (0,4], L | Noise covariance regularity near z\* | How smoothly noise statistics vary as you move away from Nash equilibrium. Needed for the MSD analysis to be tight. |
| ε | Smallness threshold | Appears in "sufficiently small" conditions throughout the assumptions. Makes inequalities strict. Not something you compute. |
