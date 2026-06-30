# Methodology — Mathematical Formulation

This document describes the mathematical structure of the HTL plant location
optimiser.  The core idea is to frame a discrete facility-location problem
(p-median) as a **continuous, differentiable** optimisation that can be
solved with gradient descent on a GPU.

---

## 1. Problem Statement

Given:
- **n** distributed feedstock sources (WWTPs), each with a known location
  (lat, lon) and capacity fᵢ
- **m** candidate plant locations to be determined by the optimiser

Find:
- Optimal plant locations **pⱼ** ∈ ℝ² (latitude, longitude)
- Optimal assignment weights **aᵢⱼ** ∈ [0, 1] (fraction of source i
  sent to plant j)

That minimise total system cost.

---

## 2. Decision Variables

| Variable | Shape | Description |
|---|---|---|
| `plant_coords` | (m, 2) | Continuous lat/lon of each candidate plant |
| `assignment_logits` | (n, m+1) | Pre-softmax assignment weights |

The **softmax** over assignment logits ensures:
- All weights are non-negative
- Each source's weights sum to 1
- Column m+1 is the **orphan** option (source left unassigned)

```
aᵢⱼ = softmax(logits_i)_j     for j = 0, ..., m   (m = orphan)
```

---

## 3. Cost Components

### 3.1 Transport + Tipping Cost

```
C_delivery = Σᵢ Σⱼ aᵢⱼ · fᵢ · (tᵢ + r · dᵢⱼ)
```

Where:
- `fᵢ` = feedstock amount at source i
- `tᵢ` = tipping fee at source i (negative = plant receives payment)
- `r`  = transport cost rate ($/unit/km)
- `dᵢⱼ` = Haversine distance from source i to plant j (km)

### 3.2 Orphan Cost

```
C_orphan = Σᵢ aᵢ,ₘ · fᵢ · p_orphan
```

Where `p_orphan` is the penalty per unit of unprocessed feedstock.
This is included as column m+1 of the cost matrix, so the optimiser
can trade off transport cost against orphaning.

### 3.3 Capital Cost

```
C_capital = c_cap · Σⱼ Lⱼ^α
```

Where:
- `Lⱼ = Σᵢ aᵢⱼ · fᵢ` is the total load arriving at plant j
- `c_cap` = capital cost coefficient
- `α` = capital cost exponent (economies of scale when α < 1)

> ⚠ **See [Assumption A2](assumptions.md#a2--capital-cost-power-law)**:
> the current coefficient is negative, which inverts the usual interpretation.

### 3.4 Revenue

```
R = c_rev · D^β
```

Where:
- `D = Σᵢ Σⱼ aᵢⱼ · fᵢ` (for j < m) is total delivered feedstock
- `c_rev` = revenue coefficient
- `β` = revenue exponent

### 3.5 True Economic Cost

The cost that is **reported** (and used for scenario comparison):

```
C_true = C_delivery + C_orphan + C_capital − R
```

**This does NOT include Lagrangian penalty terms.**

---

## 4. Lagrangian Constraints

Constraints are enforced via penalty terms added to the optimisation
objective but **excluded from the reported cost**.  This is critical
for honest economic comparison across scenarios.

```
C_optimiser = C_true + Σ_k λ_k(t) · P_k
```

Where `P_k` is the violation penalty for constraint k and `λ_k(t)` is a
**scheduled weight** that ramps from soft to hard:

| Schedule | Formula |
|---|---|
| Linear | `λ(t) = λ₀ + (λ_f − λ₀) · t/T` |
| Exponential | `λ(t) = λ₀ · (λ_f/λ₀)^(t/T)` |
| Logarithmic | `λ(t) = 10^(log₁₀(λ₀) + (log₁₀(λ_f) − log₁₀(λ₀)) · t/T)` |
| Sigmoid | `λ(t) = λ₀ + (λ_f − λ₀) · σ(10(t/T − 0.5))` |

Starting soft and hardening gradually helps the optimiser find a good
basin before the constraint becomes binding.

### Built-in Constraint Types

| Type | Violation | Purpose |
|---|---|---|
| `plant_profitability` | `relu(min_npv − NPVⱼ)²` | Every plant must be individually profitable |
| `max_orphan_fraction` | `relu(orphan − f·total)²` | Limit total orphaned feedstock |
| `max_transport_distance` | `aᵢⱼ · relu(dᵢⱼ − d_max)²` | No deliveries beyond d_max km |
| `min_plant_load` | `relu(L_min − Lⱼ)²` | Minimum viable plant size |
| `capital_cost_bound` | box penalty on total capital | Hard capital budget |

---

## 5. Optimisation

The solver uses **Adam** (adaptive moment estimation) to update
`plant_coords` and `assignment_logits` simultaneously.  Key features:

- **Soft assignments** via softmax ensure differentiability everywhere
- **Haversine distance** is fully differentiable (with clamping for
  numerical stability)
- **Heuristic initialisation**: plants start at the m largest feedstock
  sources to avoid poor local minima
- **Convergence check**: stops when `|C(t) − C(t−1)| < ε` for
  `t > min_epochs`

### Relationship to P-Median

The classical p-median problem is NP-hard (discrete assignment, fixed
number of facilities).  This formulation relaxes it to a continuous
problem that can be solved in O(epochs · n · m) time per iteration,
fully parallelisable on GPU.

---

## 6. Per-Plant Metrics

After optimisation, per-plant metrics are computed for reporting:

```
Delivery_j = Σᵢ aᵢⱼ · fᵢ · (tᵢ + r · dᵢⱼ)
Revenue_j  = c_rev · (Σᵢ aᵢⱼ · fᵢ)^β
Capital_j  = c_cap · Lⱼ^α
NPV_j      = Revenue_j − Delivery_j − Capital_j
```

Active plants are those with at least one hard-assigned source
(`argmax(aᵢ) = j` for some i).

---

## 7. Extension Points

The architecture supports future additions:

- **Multi-hierarchy**: feedstock → collection depots → processing plants
  (was in the original script, deferred for v1)
- **Mass-component tracking**: track k material species through the
  network using `feed_compositions` tensors
- **Emissions modelling**: CO₂ from transport, processing, and avoided
  landfill — enables cost-per-tonne-CO₂e-abated metrics
- **Region exclusion**: forbidden zones via smooth `region_penalty()`
  (already implemented in `geo.py`)
