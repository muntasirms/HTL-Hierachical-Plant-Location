# Assumptions Register

> **Status legend:**
> - 🔴 **UNVERIFIED** — carried over from original script, needs domain-expert review
> - 🟡 **PLAUSIBLE** — reasonable default but should be calibrated to specific deployment
> - 🟢 **VERIFIED** — confirmed against cited source

---

## A1 — Transport Cost Factor

| Field | Value |
|---|---|
| **Parameter** | `economics.transport_cost_per_unit_km` |
| **Value** | 138.8 $/MMGal/km |
| **Status** | 🔴 UNVERIFIED |
| **Derivation** | `0.059 $/m³/mi ÷ 1.61 km/mi ÷ 264 gal/m³ × 1,000,000 gal/MMGal` |
| **Source** | Table 4 of [Seider et al. (2015)](https://doi.org/10.1016/j.jclepro.2015.01.018) — cited in original script |
| **Notes** | Includes a dewatering rate adjustment. Assumes truck transport. Pipeline or rail transport would have very different cost structures. Does not account for return-trip (empty haul) costs. |
| **Action needed** | Verify the $/m³/mi figure against the cited paper. Confirm dewatering rate. Consider whether the cost should vary by region or distance tier. |

---

## A2 — Capital Cost Power Law

| Field | Value |
|---|---|
| **Parameter** | `economics.capital_cost_coef` = **−0.68**, `economics.capital_cost_exponent` = **2.0** |
| **Status** | 🔴 UNVERIFIED |
| **Formula** | `C_capital = coef × Σⱼ(loadⱼ ^ exponent)` |
| **Notes** | The **negative coefficient** with a **quadratic exponent** means capital cost is negative and grows more negative with plant size. This effectively acts as an *incentive* for larger plants. In typical engineering economics, capital cost is positive and scales sub-linearly (exponent < 1) to represent economies of scale. The current formulation may be intentionally modelling a net capital benefit (e.g., amortised cost recovery) or may be a placeholder. |
| **Action needed** | Clarify whether the negative coefficient is intentional. If capital cost should be positive, the sign needs to flip and the exponent should likely be < 1 (e.g., 0.6–0.8 for chemical process plants). |

---

## A3 — Revenue Coefficient

| Field | Value |
|---|---|
| **Parameter** | `economics.revenue_coef` = **1657.89** |
| **Status** | 🔴 UNVERIFIED |
| **Derivation** | `1000 × 0.4 × 45/38 × 3.5` |
| **Breakdown** | The individual factors are not documented. Likely candidates: |
| | - `1000` — unit conversion (MMGal → ?) |
| | - `0.4` — possibly HTL oil yield fraction |
| | - `45/38 ≈ 1.184` — possibly energy density ratio or heating value adjustment |
| | - `3.5` — possibly $/gal product price |
| **Notes** | `revenue_exponent = 1.0` means revenue is linear in delivered feed — no diminishing returns from market saturation. |
| **Action needed** | Document each multiplicative factor. Verify yield, price, and conversion assumptions against current HTL literature. |

---

## A4 — Orphan Penalty

| Field | Value |
|---|---|
| **Parameter** | `economics.orphan_penalty` = **50 $/unit** |
| **Status** | 🟡 PLAUSIBLE |
| **Notes** | This is the cost charged when feedstock is *not* assigned to any plant. It represents the social/environmental cost of leaving waste unprocessed — could include landfill costs, missed tipping fees, or regulatory penalties. The value acts as a soft ceiling on how expensive transport can get before the optimiser prefers orphaning. |
| **Action needed** | Calibrate to actual alternative disposal cost for the region of interest. |

---

## A5 — Tipping Fee Range

| Field | Value |
|---|---|
| **Parameter** | `economics.tipping_fee` — Uniform(−75, −25) |
| **Status** | 🟡 PLAUSIBLE |
| **Notes** | Negative tipping fee means the WWTP *pays* the HTL plant to accept sludge. This is realistic — wastewater utilities pay for sludge disposal. The range (−$75 to −$25 per unit) is randomly assigned per source. In practice, tipping fees would correlate with facility size, location, and contractual terms. |
| **Action needed** | If actual tipping fee data is available, use `mode: from_column`. Otherwise, validate range against regional sludge disposal market. |

---

## A6 — Wastewater Conversion Fraction

| Field | Value |
|---|---|
| **Parameter** | `data.convertible_fraction` = **0.01** (1%) |
| **Status** | 🟡 PLAUSIBLE |
| **Notes** | Assumes 1% of wastewater *by mass* is convertible solids/organics suitable for HTL. Typical municipal wastewater has ~0.03–0.05% total suspended solids, but primary + secondary sludge concentrations after thickening can be 2–6%. The 1% figure may represent an intermediate concentration. |
| **Action needed** | Clarify whether this refers to raw influent or concentrated sludge stream. Verify against target WWTP data. |

---

## A7 — Haversine (Great-Circle) Distance

| Field | Value |
|---|---|
| **Parameter** | Distance function in `geo.py` |
| **Status** | 🟡 PLAUSIBLE |
| **Notes** | Uses straight-line (great-circle) distance. Actual road transport distances are typically 1.2–1.4× the straight-line distance (tortuosity factor). This means transport costs are systematically underestimated. |
| **Action needed** | Consider applying a tortuosity multiplier (e.g., 1.3×) or using a road-distance API for higher fidelity. |

---

## A8 — Single Hierarchy

| Field | Value |
|---|---|
| **Parameter** | Model structure |
| **Status** | 🟢 VERIFIED (design choice) |
| **Notes** | The current model uses a single hierarchy: feedstock sources → HTL plants. The original script had a two-hierarchy structure (feedstock → collection plants → processing plants) which has been deferred. Multi-hierarchy support is architecturally straightforward to add. |
| **Action needed** | Re-introduce second hierarchy when needed for depot/hub models. |

---

## How to Add Assumptions

When you introduce a new parameter or modelling choice:

1. Add an entry to this file with the 🔴 status
2. Reference the assumption ID (e.g., `[ASSUMPTION A9]`) as a comment in the relevant code
3. Update the status as validation progresses
