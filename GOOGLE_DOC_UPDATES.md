# Pending Google Doc Updates

Changes that should be made to the internal planning Google Doc when someone has time.
The doc URL: https://docs.google.com/document/d/18KfD_D7FXlJx_ppzYVmYB4swAn4_z3pDby0cpKHUroI/edit

---

## 1. Glossary: K₂ example is wrong

**Current:** FreshMart/QuickShop example uses K₂ = 3.
**Fix:** Change to K₂ = 4 to match the paper (Section 4: "K₁=2, K₂=4").

---

## 2. Glossary: step-size examples are approximate

**Current:** μ⁽¹⁾ = 0.001 and μ⁽²⁾ = 0.0008 (inconsistent with the paper's 2:1 ratio).
**Fix:** Use the paper's exact four settings:
- (μ₁, μ₂) = (0.001, 0.0005)
- (μ₁, μ₂) = (0.0008, 0.0004)
- (μ₁, μ₂) = (0.0006, 0.0003)
- (μ₁, μ₂) = (0.0004, 0.0002)
The paper always uses μ₂ = μ₁ / 2.

---

## 3. Luka's role description: Figure 3 misattributed to paper

**Current:** "Learning Rate Test: Compare the 'error floor' when using different step-sizes (μ),
*as shown in Fig. 1 of the paper*."
**Fix:** Remove "as shown in Fig. 1 of the paper". Figure 3 is **original work** — the paper's
Figure 1 shows MSD convergence curves over iterations (not a floor-vs-μ plot). The floor-vs-μ
plot is the team's own contribution, motivated by Theorem 1's O(μ_max) bound.

---

## 4. Add note about p_k weights in the ATC loop

**Where:** near the p_k^(t) glossary entry.
**Add:** "Note: p_k^(t) weights appear in the team objective J^(t) and in Marija's Assumption 2
computation. They do **not** appear in the ATC update loop — each agent computes its gradient
from its own local loss Q_k without any explicit weighting. The weights are implicit in the fact
that the algorithm converges to the Nash equilibrium of the weighted game."

---

## 5. Clarify A^(1) vs A^(11) open question

**Where:** the ❓ open question block.
**Add resolution:** "For implementation purposes, A^(1) and A^(11) can be set to the same
matrix (A^(1) = A^(11)) without breaking correctness. The paper treats them as independent
because they conceptually govern different quantities — actual iterates vs inferred beliefs —
but there is no theoretical requirement that they differ."
