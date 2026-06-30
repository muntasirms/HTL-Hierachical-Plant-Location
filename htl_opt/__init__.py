"""
htl_opt — GPU-accelerated HTL plant location optimizer.

A continuous, differentiable p-median solver for siting Hydrothermal
Liquefaction (HTL) facilities near distributed wastewater feedstock sources.

Quickstart
----------
    from htl_opt import Scenario, solve

    scenario = Scenario.load("scenarios/baseline.yaml")
    results  = solve(scenario)
    results.summary()
    results.save("outputs/baseline")
    results.plot_map()

Scenario helpers
----------------
    from htl_opt import variant, sweep, constrained, batch_solve

    # One-liner variant
    results = solve(variant("scenarios/baseline.yaml", "cheap",
                            economics__orphan_penalty=10))

    # Parameter sweep
    all_results = sweep("scenarios/baseline.yaml",
                        "economics.transport_cost_per_unit_km",
                        [100, 200, 300])
"""

from .config import Scenario
from .solver import solve
from .results import Results
from .scenarios import variant, constrained, sweep, batch_solve

__all__ = [
    "Scenario", "solve", "Results",
    "variant", "constrained", "sweep", "batch_solve",
]
__version__ = "0.1.0"
