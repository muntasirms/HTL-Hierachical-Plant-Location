# htl-plant-opt

GPU-accelerated optimal siting of Hydrothermal Liquefaction (HTL) plants near distributed wastewater feedstock sources.

Uses a continuous, differentiable relaxation of the p-median facility location problem — solvable with gradient descent on a consumer GPU or HPC cluster.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place your data
#    Put WWTPs.csv in data/ with columns: latitude, longitude, scale

# 3. Run baseline scenario
python run.py scenarios/baseline.yaml

# 4. View results
#    outputs/baseline/
#    ├── summary.json
#    ├── plants.csv
#    ├── assignments.csv
#    ├── map.html          ← interactive map (open in browser)
#    └── convergence.png
```

### Python API

```python
from htl_opt import Scenario, solve

scenario = Scenario.load("scenarios/baseline.yaml")
scenario.economics.orphan_penalty = 100  # tweak in Python

results = solve(scenario)
results.summary()
results.plants_df().head(10)
results.save("outputs/my_run")
results.plot_map()
```

### CO₂ Emissions Tracking

```python
from htl_opt import Scenario, solve

scenario = Scenario.load("scenarios/baseline.yaml")
scenario.emissions.enabled = True
scenario.emissions.mode = "combined"  # "post_hoc" | "co2_first" | "combined"
results = solve(scenario)
results.summary()  # includes CO₂ breakdown
```

Or from the CLI:
```bash
python run.py scenarios/baseline.yaml --emissions-mode post_hoc
python run.py scenarios/co2_combined.yaml --co2-weight 0.10
```

---

## Repository Structure

```
├── htl_opt/                  # Core package
│   ├── config.py             #   Scenario config (dataclasses + YAML)
│   ├── data.py               #   CSV loading and preprocessing
│   ├── geo.py                #   Haversine distance, region penalties
│   ├── constraints.py        #   Lagrangian penalty system
│   ├── model.py              #   PyTorch model (plant_coords + assignments + CO₂)
│   ├── solver.py             #   Optimisation loop (3 CO₂ modes)
│   ├── results.py            #   Results container, DataFrames, save/load
│   └── viz.py                #   Maps (Folium/matplotlib) + convergence + CO₂ overlay
│
├── scenarios/                # YAML scenario configs
│   ├── baseline.yaml         #   Unconstrained cost minimisation
│   ├── high_transport.yaml   #   2× transport cost
│   ├── profitable_plants.yaml#   Per-plant profitability constraint
│   ├── co2_post_hoc.yaml     #   Cost-optimised, CO₂ reported post-hoc
│   ├── co2_minimise.yaml     #   CO₂-optimised, cost reported post-hoc
│   └── co2_combined.yaml     #   Combined cost + carbon price × CO₂
│
├── examples/                 # Example scripts
│   ├── quickstart.py         #   Minimal 30-line workflow
│   ├── compare_scenarios.py  #   Run & compare multiple scenarios
│   └── co2_analysis.py       #   All 3 CO₂ modes + Pareto sweep
│
├── data/                     # ← Put WWTPs.csv here
├── outputs/                  # ← Generated results go here
│
├── docs/
│   ├── methodology.md        #   Mathematical formulation (incl. CO₂ §8)
│   ├── guide.md              #   Quick reference guide & usage examples
│   ├── data_pipeline.md      #   Data flow, variables, and CSV requirements
│   └── assumptions.md        #   Parameter assumption register (A1–A13)
│
├── run.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

## Data Format

Place your feedstock CSV in `data/WWTPs.csv` with these columns:

| Column | Type | Description |
|---|---|---|
| `latitude` | float | Latitude in degrees |
| `longitude` | float | Longitude in degrees |
| `scale` | float | Treatment capacity (MMGal/day) |

Column names are configurable in the scenario YAML. Optional additional columns (e.g., `tipping_fee`) can be referenced by the config.

For a complete map of how data flows through the application and internal variables, see [docs/data_pipeline.md](docs/data_pipeline.md).

---

## Scenario Configuration

Every run is defined by a YAML file. Key sections:

| Section | Controls |
|---|---|
| `data` | CSV path, column names, conversion fraction |
| `model` | Number of candidate plants, initialisation strategy |
| `economics` | Transport rate, capital cost, revenue, orphan penalty, tipping fees |
| `emissions` | CO₂ tracking: mode, intensity rates, carbon price (optional) |
| `solver` | Epochs, learning rate, convergence tolerance, LR scheduler |
| `constraints` | List of Lagrangian constraints (type + params + schedule) |

See `scenarios/baseline.yaml` for a fully commented example.

### Creating a New Scenario

```bash
# Copy and edit
cp scenarios/baseline.yaml scenarios/my_scenario.yaml
# Edit the YAML, then run
python run.py scenarios/my_scenario.yaml
```

Or build programmatically:

```python
from htl_opt import Scenario

s = Scenario.load("scenarios/baseline.yaml")
s.name = "low_orphan"
s.economics.orphan_penalty = 200
s.save("scenarios/low_orphan.yaml")
```

---

## Outputs

Each run saves to `outputs/<scenario_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | Exact config used (for reproducibility) |
| `summary.json` | Key metrics (costs, plant count, orphan %, CO₂ when enabled) |
| `plants.csv` | Per-plant: location, load, costs, revenue, NPV, CO₂ |
| `assignments.csv` | Per-source: assigned plant, delivered amount, CO₂ transport |
| `orphaned.csv` | Sources with no plant assignment |
| `convergence.csv` | Cost vs. epoch training history (+ CO₂ when enabled) |
| `convergence.png` | Convergence plot (+ CO₂ panel when enabled) |
| `map.html` | Interactive Folium map (+ CO₂ layer when enabled) |
| `co2_summary.json` | Detailed CO₂ breakdown (when emissions enabled) |

### Scenario Comparison

```bash
# Compare saved runs from the CLI
python run.py --compare outputs/baseline outputs/high_transport
```

```python
# Or in Python
df = Results.compare([res_a, res_b])
print(df.to_markdown())
```

---

## Key Design Decisions

### Lagrangian Penalties ≠ True Cost

Constraints are enforced by adding penalty terms to the optimisation objective. **These penalties are excluded from all reported costs and metrics.** This means:

- The optimiser minimises `C_true + Σ penalties`
- The reported `total_system_cost` is just `C_true`
- Scenario comparisons are economically honest

### Soft Assignments via Softmax

Instead of binary 0/1 assignment (NP-hard), each source distributes its feedstock across plants via softmax weights. The optimiser naturally sharpens these toward hard assignment as it converges.

### Scheduled Constraint Hardening

Lagrangian multipliers ramp from small to large over training. This lets the optimiser explore freely early on, then enforces constraints as it converges — reducing the risk of poor local minima.

---

## HPC Usage

```bash
# SLURM example
srun python run.py scenarios/baseline.yaml --device cuda --static-map --output outputs/baseline_gpu

# With CO₂ tracking on HPC
srun python run.py scenarios/co2_combined.yaml --device cuda --static-map --co2-weight 0.10
```

Use `--static-map` on headless nodes (avoids Folium's JavaScript dependency). Use `--device cpu` to force CPU if needed.

---

## Future Roadmap

- [ ] Multi-hierarchy plant networks (collection depots → processing plants)
- [ ] Mass-component tracking (k species through the network)
- [x] ~~CO₂ emissions modelling and cost-per-tonne-CO₂e-abated~~ **(v0.2)**
- [ ] Zoning / forbidden region constraints
- [ ] Market saturation (diminishing revenue)
- [ ] Sensitivity analysis automation

---

## Assumptions

All economic parameters carry assumption flags. See [`docs/assumptions.md`](docs/assumptions.md) for the full register with verification status and action items.

---

## License

TBD
