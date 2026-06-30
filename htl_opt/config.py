"""
Scenario configuration via nested dataclasses + YAML serialization.

Every tunable knob lives here so scenarios are fully self-describing
and reproducible.  Load with ``Scenario.load("path.yaml")`` or build
in Python and call ``scenario.save("path.yaml")``.
"""

from __future__ import annotations

import copy
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── data ────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Where to find feedstock data and how to interpret it."""
    feedstock_file: str = "data/WWTPs.csv"
    feedstock_unit: str = "MMGal/day"
    convertible_fraction: float = 0.01        # fraction of wastewater that is convertible solids
    lat_column: str = "latitude"
    lon_column: str = "longitude"
    scale_column: str = "scale"


# ── tipping fee ─────────────────────────────────────────────────────
@dataclass
class TippingFeeConfig:
    """
    Tipping fee paid *to* the plant by the feedstock source (negative = plant
    receives payment, positive = plant pays).  Can be fixed, random, or read
    from a CSV column.
    """
    mode: str = "uniform_random"              # "fixed" | "uniform_random" | "from_column"
    fixed_value: float = -50.0                # used when mode == "fixed"
    random_min: float = -75.0                 # used when mode == "uniform_random"
    random_max: float = -25.0
    column_name: str = "tipping_fee"          # used when mode == "from_column"


# ── economics ───────────────────────────────────────────────────────
@dataclass
class EconomicsConfig:
    """All cost / revenue parameters.  See docs/assumptions.md for sources."""
    # Transport
    transport_cost_per_unit_km: float = 138.8  # $/MMGal/km  [ASSUMPTION A1]

    # Tipping fee
    tipping_fee: TippingFeeConfig = field(default_factory=TippingFeeConfig)

    # Capital cost:  C_cap = coef * Σ(load^exponent)              [ASSUMPTION A2]
    capital_cost_coef: float = -0.68
    capital_cost_exponent: float = 2.0

    # Revenue:  R = coef * total_delivered^exponent                [ASSUMPTION A3]
    revenue_coef: float = 1657.89              # 1000 * 0.4 * 45/38 * 3.5
    revenue_exponent: float = 1.0

    # Orphan penalty
    orphan_penalty: float = 50.0               # $/unit for un-assigned feed  [ASSUMPTION A4]


# ── solver ──────────────────────────────────────────────────────────
@dataclass
class SolverConfig:
    """Optimizer hyper-parameters."""
    num_epochs: int = 100_000
    learning_rate: float = 0.01
    convergence_tol: float = 1e-7
    min_epochs: int = 5_000
    log_interval: int = 500                    # print every N epochs
    scheduler_enabled: bool = False
    scheduler_patience: int = 500
    scheduler_factor: float = 0.9


# ── model ───────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Structural model choices."""
    num_candidate_plants: int = 50
    initialization: str = "top_feedstock"       # "top_feedstock" | "random"


# ── constraints ─────────────────────────────────────────────────────
@dataclass
class ConstraintConfig:
    """
    A single Lagrangian constraint added to the objective during
    optimization.  The ``type`` field selects the constraint function;
    ``params`` carries type-specific knobs.

    The penalty is *excluded* from the reported true cost — it only
    steers the optimizer.
    """
    type: str = ""                             # e.g. "plant_profitability", "max_orphan", ...
    params: Dict[str, Any] = field(default_factory=dict)
    lambda_init: float = 0.1
    lambda_final: float = 100_000.0
    schedule_type: str = "log"                 # "linear" | "exp" | "log" | "sigmoid"
    ramp_steps: int = 3_000


# ── top-level scenario ──────────────────────────────────────────────
@dataclass
class Scenario:
    """
    Fully self-describing run configuration.

    Load from YAML::

        scenario = Scenario.load("scenarios/baseline.yaml")

    Modify in Python::

        scenario.economics.orphan_penalty = 100

    Save for reproducibility::

        scenario.save("outputs/my_run/config.yaml")
    """
    name: str = "baseline"
    description: str = "Baseline — minimize system-wide cost, no constraints"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    constraints: List[ConstraintConfig] = field(default_factory=list)

    # ── serialization ───────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        """Deserialize a Scenario from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    def save(self, path: str | Path) -> Path:
        """Serialize this Scenario to a YAML file (creates dirs as needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)
        return path

    def clone(self, **overrides) -> "Scenario":
        """Return a deep copy, optionally overriding top-level fields."""
        s = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    # ── internal helpers ────────────────────────────────────────────
    def _to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def _from_dict(cls, d: dict) -> "Scenario":
        data = DataConfig(**d.get("data", {}))

        # Economics needs nested TippingFeeConfig
        eco_raw = d.get("economics", {})
        tf_raw = eco_raw.pop("tipping_fee", {})
        tipping = TippingFeeConfig(**tf_raw) if tf_raw else TippingFeeConfig()
        eco = EconomicsConfig(tipping_fee=tipping, **eco_raw)

        model = ModelConfig(**d.get("model", {}))
        solver = SolverConfig(**d.get("solver", {}))
        constraints = [ConstraintConfig(**c) for c in d.get("constraints", [])]

        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            data=data,
            model=model,
            economics=eco,
            solver=solver,
            constraints=constraints,
        )
