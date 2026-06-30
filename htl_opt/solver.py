"""
Optimisation loop — the main ``solve()`` entry point.

Runs Adam with optional LR scheduling, logs convergence, and returns
a :class:`Results` object containing all outputs.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Callable, Optional

from .config import Scenario
from .data import load_feedstock_data
from .model import HTLPlantModel
from .constraints import compute_total_penalty
from .results import Results


def solve(
    scenario: Scenario,
    *,
    device: torch.device | str | None = None,
    base_dir: str | Path | None = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, dict], None]] = None,
) -> Results:
    """
    Run the HTL plant-location optimisation.

    Parameters
    ----------
    scenario : Scenario
        Fully populated run configuration.
    device : str or torch.device, optional
        Force a device (``"cpu"`` / ``"cuda"``).  Auto-detected if *None*.
    base_dir : Path, optional
        Root directory for resolving relative paths in the scenario.
    verbose : bool
        Print progress to stdout.
    callback : callable, optional
        ``callback(epoch, log_dict)`` called every ``log_interval`` epochs.
        Useful for custom loggers or progress bars.

    Returns
    -------
    Results
        Packaged outputs with ``.summary()``, ``.save()``, ``.plot_map()``
        convenience methods.
    """
    # ── device ──────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print(f"╔══════════════════════════════════════════════════╗")
        print(f"║  HTL Plant Location Optimiser                   ║")
        print(f"║  Scenario: {scenario.name:<38s}║")
        print(f"║  Device:   {str(device):<38s}║")
        print(f"╚══════════════════════════════════════════════════╝")

    # ── data ────────────────────────────────────────────────────────
    feed_coords, feed_amounts, tipping_fees = load_feedstock_data(
        scenario, device, base_dir=base_dir
    )

    # ── model ───────────────────────────────────────────────────────
    model = HTLPlantModel(feed_coords, feed_amounts, tipping_fees, scenario)
    model = model.to(device)

    # ── optimiser ───────────────────────────────────────────────────
    cfg = scenario.solver
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    scheduler = None
    if cfg.scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg.scheduler_patience,
            factor=cfg.scheduler_factor,
        )

    # ── training loop ───────────────────────────────────────────────
    history: list[dict] = []
    prev_cost: float | None = None
    t0 = time.perf_counter()
    final_outputs: dict = {}

    for epoch in range(cfg.num_epochs):
        optimizer.zero_grad()

        outputs = model()

        # True economic cost (what we report)
        true_cost = outputs["total_cost"]

        # Lagrangian penalty (steers optimiser, NOT reported as cost)
        penalty = compute_total_penalty(
            scenario.constraints, outputs, epoch
        )

        # Objective the optimiser sees
        objective = true_cost + penalty

        objective.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(objective.item())

        # ── logging ─────────────────────────────────────────────────
        if epoch % cfg.log_interval == 0 or epoch == cfg.num_epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            log = {
                "epoch":         epoch,
                "true_cost":     true_cost.item(),
                "penalty":       penalty.item(),
                "objective":     objective.item(),
                "delivery_cost": outputs["delivery_cost"].item(),
                "capital_cost":  outputs["capital_cost"].item(),
                "revenue":       outputs["revenue"].item(),
                "orphan_amount": outputs["orphan_amount"].item(),
                "lr":            lr,
            }
            history.append(log)

            if verbose:
                print(
                    f"  epoch {epoch:>7d}  │  cost {true_cost.item():>14,.2f}  │  "
                    f"penalty {penalty.item():>12,.2f}  │  "
                    f"orphan {outputs['orphan_amount'].item():>10,.2f}  │  "
                    f"lr {lr:.2e}"
                )

            if callback is not None:
                callback(epoch, log)

        # ── convergence check ───────────────────────────────────────
        current = objective.item()
        if (
            prev_cost is not None
            and abs(prev_cost - current) < cfg.convergence_tol
            and epoch >= cfg.min_epochs
        ):
            if verbose:
                print(f"\n  ✓ Converged at epoch {epoch} "
                      f"(Δ < {cfg.convergence_tol:.0e})")
            # Run one more forward for final outputs
            with torch.no_grad():
                final_outputs = model()
            break
        prev_cost = current

    else:
        # Loop completed without break
        if verbose:
            print(f"\n  ⚠ Reached max epochs ({cfg.num_epochs})")
        with torch.no_grad():
            final_outputs = model()

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  Wall time: {elapsed:.1f}s\n")

    # ── package results ─────────────────────────────────────────────
    return Results(
        scenario=scenario,
        model=model,
        outputs=final_outputs,
        history=history,
        device=device,
        elapsed_seconds=elapsed,
    )
