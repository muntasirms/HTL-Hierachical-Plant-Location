# HTL Plant Optimiser — Quick Reference

A single-page guide covering what the solver computes, how to use it, and how to extend it.

---

## What the Solver Does

The optimiser places **m** HTL processing plants across a geography of **n** wastewater treatment plants (WWTPs) and assigns feedstock from each WWTP to the nearest cost-effective plant. It simultaneously optimises **where** plants go and **which sources feed them** using GPU-accelerated gradient descent.

### Decision variables (what the solver adjusts)

| Variable | Shape | Meaning |
|---|---|---|
| `plant_coords` | (m, 2) | Lat/lon of each candidate plant |
| `assignment_logits` | (n, m+1) | How much of each source goes to each plant (column m+1 = orphan — source left unassigned) |

Assignments pass through a **softmax** so they're continuous and differentiable. The solver sharpens them toward hard 0/1 decisions as it converges.

---

## Cost Breakdown

The solver minimises this objective:

```
System Cost  =  Transport Cost  +  Capital Cost  −  Revenue
```

Each component, in plain terms:

### Transport cost
> *"How much does it cost to truck sludge from every WWTP to its assigned plant?"*

```
Σ  (assignment weight) × (feedstock amount) × (tipping fee + rate × distance)
```

- **Distance** is great-circle (Haversine) in km
- **Tipping fee** is what the WWTP pays *you* to take their sludge (negative = income)
- **Rate** is $/MMGal/km for trucking

### Capital cost
> *"How much does it cost to build each plant?"*

```
coef × Σ (plant load)^exponent
```

⚠ *The current coefficient is negative — see [assumptions.md](assumptions.md) A2 for discussion.*

### Revenue
> *"How much money do you make selling HTL products?"*

```
coef × (total delivered feedstock)^exponent
```

With exponent = 1 this is just a fixed price per unit of feed processed.

### Orphan cost
> *"What's the penalty for leaving feedstock unprocessed?"*

Sources not assigned to any plant incur a flat penalty per unit, representing alternative disposal costs or regulatory risk.

### What about constraints?

Constraints (profitability, max orphan %, distance limits) are added as **Lagrangian penalties** that ramp from soft → hard during training. They steer the optimiser but are **excluded from reported costs**, so economic comparisons between scenarios are honest.

---

## Per-Plant Outputs

After the solve, every plant gets its own breakdown:

| Metric | Formula |
|---|---|
| **Load** | Total feedstock arriving at plant j |
| **Delivery cost** | Transport + tipping for all sources assigned to j |
| **Capital cost** | `coef × load^exponent` for plant j |
| **Revenue** | `coef × load^exponent` for plant j |
| **NPV** | Revenue − Delivery − Capital |
| **Active?** | Does at least one source hard-assign here? |

---

## Usage Patterns

### 1. Run a scenario from YAML

```bash
python run.py scenarios/baseline.yaml
```

Outputs land in `outputs/baseline/` — CSVs, a summary JSON, a convergence plot, and an interactive map.

### 2. Run from Python (minimal)

```python
from htl_opt import Scenario, solve

results = solve(Scenario.load("scenarios/baseline.yaml"))
results.summary()          # prints key metrics
results.plants_df()        # pandas DataFrame of all plants
results.plot_map()         # interactive HTML map
results.save("outputs/baseline")
```

### 3. Tweak one parameter

```python
from htl_opt import Scenario, solve

s = Scenario.load("scenarios/baseline.yaml")
s.name = "high_orphan_penalty"
s.economics.orphan_penalty = 200
solve(s).save("outputs/high_orphan_penalty")
```

### 4. One-liner variant

```python
from htl_opt import solve, variant

solve(variant("scenarios/baseline.yaml", "cheap_transport",
              economics__transport_cost_per_unit_km=50)).summary()
```

### 5. Add a constraint without editing the solver

```python
from htl_opt import solve, constrained

s = constrained("scenarios/baseline.yaml",
                plant_profitability={"min_npv": 0},
                max_orphan_fraction={"fraction": 0.05})
solve(s).save("outputs/constrained")
```

Available constraint types: `plant_profitability`, `max_orphan_fraction`, `max_transport_distance`, `min_plant_load`, `capital_cost_bound`.

### 6. Parameter sweep

```python
from htl_opt import sweep

results = sweep("scenarios/baseline.yaml",
                "economics.transport_cost_per_unit_km",
                [50, 100, 200, 400])
# Runs all 4, saves each, prints a comparison table
```

### 7. Batch compare

```python
from htl_opt import batch_solve, variant, Results

results = batch_solve([
    "scenarios/baseline.yaml",
    "scenarios/high_transport.yaml",
    variant("scenarios/baseline.yaml", "many_plants",
            model__num_candidate_plants=100),
])

# Side-by-side table
print(Results.compare(results).to_markdown())
```

### 8. HPC / headless

```bash
python run.py scenarios/baseline.yaml --device cuda --static-map
```

`--static-map` uses matplotlib instead of Folium (no browser needed).
`--device cpu` forces CPU if CUDA isn't available.

### 9. Compare saved runs (no re-solving)

```bash
python run.py --compare outputs/baseline outputs/high_transport outputs/constrained
```

Reads `summary.json` from each directory and prints a markdown table.

---

## Creating a New Scenario

**Option A — Copy YAML:**
```bash
cp scenarios/baseline.yaml scenarios/my_scenario.yaml
# edit, then:
python run.py scenarios/my_scenario.yaml
```

**Option B — Build in Python:**
```python
from htl_opt import Scenario

s = Scenario.load("scenarios/baseline.yaml")
s.name = "my_scenario"
s.description = "Testing higher revenue assumptions"
s.economics.revenue_coef = 2000
s.solver.num_epochs = 50000
s.save("scenarios/my_scenario.yaml")
```

**Option C — Inline (no file):**
```python
from htl_opt import Scenario, solve

s = Scenario()  # all defaults
s.name = "from_scratch"
s.data.feedstock_file = "data/WWTPs.csv"
s.model.num_candidate_plants = 30
solve(s).summary()
```

---

## Output Files (per run)

| File | What's in it |
|---|---|
| `config.yaml` | Exact scenario used (copy this to reproduce) |
| `summary.json` | System cost, active plants, orphan %, wall time |
| `plants.csv` | Plant lat/lon, load, delivery cost, capital, revenue, NPV |
| `assignments.csv` | Each source's assigned plant and delivered amount |
| `orphaned.csv` | Sources with no plant assignment |
| `convergence.csv` | Epoch-by-epoch cost and penalty history |
| `convergence.png` | Training convergence plot |
| `map.html` / `map.png` | Interactive or static map |

---

## Configurable Parameters (YAML reference)

```yaml
data:
  feedstock_file: data/WWTPs.csv       # path to CSV
  feedstock_unit: MMGal/day            # display only
  convertible_fraction: 0.01           # fraction of wastewater that is HTL-convertible
  lat_column: latitude                 # CSV column names
  lon_column: longitude
  scale_column: scale

model:
  num_candidate_plants: 50             # how many plants the solver can place
  initialization: top_feedstock        # "top_feedstock" or "random"

economics:
  transport_cost_per_unit_km: 138.8    # $/MMGal/km
  tipping_fee:
    mode: uniform_random               # "fixed", "uniform_random", or "from_column"
    random_min: -75.0
    random_max: -25.0
  capital_cost_coef: -0.68
  capital_cost_exponent: 2.0
  revenue_coef: 1657.89
  revenue_exponent: 1.0
  orphan_penalty: 50.0

solver:
  num_epochs: 100000
  learning_rate: 0.01
  convergence_tol: 1.0e-7
  min_epochs: 5000
  log_interval: 500
  scheduler_enabled: false

constraints:                           # empty list = no constraints
  - type: plant_profitability
    params: { min_npv: 0 }
    lambda_init: 0.1
    lambda_final: 100000.0
    schedule_type: log                 # linear, exp, log, sigmoid
    ramp_steps: 3000
```

---

## Assumptions

All economic parameters are flagged for verification in [assumptions.md](assumptions.md). The three highest-priority items:

1. **A2 — Capital cost sign** is negative (unusual — needs confirmation)
2. **A3 — Revenue coefficient** factors are undocumented
3. **A1 — Transport rate** source needs verification against cited paper
