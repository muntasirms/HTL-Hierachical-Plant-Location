"""
Lagrangian constraint system with scheduled penalty weights.

Constraints are added to the optimisation objective to steer the solver,
but their values are **excluded** from the reported true economic cost.
The ramping schedule starts soft (allowing exploration) and hardens
over training so the final solution respects the constraint.
"""

from __future__ import annotations

import math
import torch
from typing import Callable, List, Optional

from .config import ConstraintConfig


# ────────────────────────────────────────────────────────────────────
# Lambda scheduling
# ────────────────────────────────────────────────────────────────────

def scheduled_lambda(
    step: int,
    total_steps: int,
    lam_init: float,
    lam_final: float,
    schedule: str = "log",
) -> float:
    """
    Compute a monotonically increasing penalty weight.

    Parameters
    ----------
    step, total_steps : int
        Current and final optimisation step.
    lam_init, lam_final : float
        Start and end penalty weight.
    schedule : str
        Interpolation curve — ``"linear"``, ``"exp"``, ``"log"``,
        or ``"sigmoid"``.
    """
    progress = min(1.0, step / max(total_steps, 1))

    if schedule == "linear":
        return lam_init + (lam_final - lam_init) * progress
    elif schedule == "exp":
        return lam_init * (lam_final / max(lam_init, 1e-12)) ** progress
    elif schedule == "log":
        lo = math.log10(max(lam_init, 1e-12))
        hi = math.log10(max(lam_final, 1e-12))
        return 10 ** (lo + (hi - lo) * progress)
    elif schedule == "sigmoid":
        sig = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
        return lam_init + (lam_final - lam_init) * sig
    else:
        return lam_final


# ────────────────────────────────────────────────────────────────────
# Penalty primitives
# ────────────────────────────────────────────────────────────────────

def box_penalty(
    value: torch.Tensor,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Quadratic penalty for *value* outside [lo, hi]."""
    below = torch.relu(lo - value)
    above = torch.relu(value - hi)
    return torch.sum(below ** 2 + above ** 2)


# ────────────────────────────────────────────────────────────────────
# Built-in constraint types
# ────────────────────────────────────────────────────────────────────

def _apply_constraint(
    cfg: ConstraintConfig,
    outputs: dict,
    step: int,
) -> torch.Tensor:
    """
    Dispatch a single constraint config to its penalty function.

    ``outputs`` is the dict returned by ``HTLPlantModel.forward()``.

    Returns a weighted scalar penalty tensor.
    """
    lam = scheduled_lambda(
        step, cfg.ramp_steps,
        cfg.lambda_init, cfg.lambda_final,
        cfg.schedule_type,
    )
    p = cfg.params
    device = outputs["plant_loads"].device

    if cfg.type == "plant_profitability":
        # Every plant's NPV must be ≥ min_npv
        npvs = outputs["plant_npvs"]
        violation = torch.relu(p.get("min_npv", 0.0) - npvs)
        return lam * torch.sum(violation ** 2)

    elif cfg.type == "max_orphan_fraction":
        # Total orphaned feed ≤ fraction * total feed
        total = outputs["total_feed"]
        orphan = outputs["orphan_amount"]
        limit = p.get("fraction", 0.1) * total
        return lam * torch.relu(orphan - limit) ** 2

    elif cfg.type == "max_transport_distance":
        # No assignment beyond max_km (soft)
        dists = outputs["distances"]                       # (n, m)
        assigns = outputs["assignments"][:, :-1]           # (n, m)
        weighted = assigns * torch.relu(dists - p.get("max_km", 500))
        return lam * torch.sum(weighted ** 2)

    elif cfg.type == "min_plant_load":
        loads = outputs["plant_loads"]
        violation = torch.relu(p.get("min_load", 0.0) - loads)
        return lam * torch.sum(violation ** 2)

    elif cfg.type == "capital_cost_bound":
        cost = outputs["capital_cost"]
        lo = p.get("min", -1e18)
        hi = p.get("max", 1e18)
        return lam * box_penalty(cost, lo, hi)

    else:
        raise ValueError(f"Unknown constraint type: '{cfg.type}'")


def compute_total_penalty(
    constraints: List[ConstraintConfig],
    outputs: dict,
    step: int,
) -> torch.Tensor:
    """
    Sum all Lagrangian penalties for the current step.

    Returns a scalar tensor (0 if no constraints are active).
    """
    device = outputs["plant_loads"].device
    total = torch.tensor(0.0, device=device)
    for cfg in constraints:
        total = total + _apply_constraint(cfg, outputs, step)
    return total
