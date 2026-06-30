"""
Scenario construction helpers — shortcuts for common patterns.

This module exists so that a user script can spin up new scenarios
with minimal boilerplate.  Everything here is a thin wrapper around
:class:`Scenario` and :func:`solve`.

Typical usage in a user script::

    from htl_opt.scenarios import sweep, batch_solve, constrained

    # Parameter sweep
    results = sweep("scenarios/baseline.yaml",
                     "economics.transport_cost_per_unit_km",
                     [100, 150, 200, 250, 300])

    # Quick constrained variant
    s = constrained("scenarios/baseline.yaml",
                     plant_profitability={"min_npv": 0},
                     max_orphan_fraction={"fraction": 0.05})
    results = solve(s)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import Scenario, ConstraintConfig


# ────────────────────────────────────────────────────────────────────
# Deep attribute access helpers
# ────────────────────────────────────────────────────────────────────

def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a nested attribute via dot-notation, e.g. 'economics.orphan_penalty'."""
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _get_nested(obj: Any, dotted_path: str) -> Any:
    """Get a nested attribute via dot-notation."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


# ────────────────────────────────────────────────────────────────────
# Scenario factories
# ────────────────────────────────────────────────────────────────────

def variant(
    base: Union[str, Path, Scenario],
    name: str,
    **overrides: Any,
) -> Scenario:
    """
    Create a named variant of a base scenario with dot-notation overrides.

    Parameters
    ----------
    base : str, Path, or Scenario
        Base scenario (YAML path or object).
    name : str
        Name for the new scenario.
    **overrides
        Dot-notation parameter overrides.

    Example
    -------
    ::

        s = variant("scenarios/baseline.yaml",
                     "cheap_transport",
                     economics__transport_cost_per_unit_km=50,
                     economics__orphan_penalty=100,
                     solver__num_epochs=50000)

    Note: use double-underscore ``__`` as the path separator in kwargs
    (dots aren't valid in Python keyword arguments).
    """
    if isinstance(base, (str, Path)):
        base = Scenario.load(base)
    s = copy.deepcopy(base)
    s.name = name

    for key, val in overrides.items():
        dotted = key.replace("__", ".")
        _set_nested(s, dotted, val)

    return s


def constrained(
    base: Union[str, Path, Scenario],
    name: Optional[str] = None,
    **constraint_kwargs: dict,
) -> Scenario:
    """
    Add constraints to a base scenario by keyword.

    Parameters
    ----------
    base : str, Path, or Scenario
        Base scenario.
    name : str, optional
        Name override (default: base name + '_constrained').
    **constraint_kwargs
        ``constraint_type=param_dict`` pairs.

    Example
    -------
    ::

        s = constrained("scenarios/baseline.yaml",
                         plant_profitability={"min_npv": 0},
                         max_orphan_fraction={"fraction": 0.05})
    """
    if isinstance(base, (str, Path)):
        base = Scenario.load(base)
    s = copy.deepcopy(base)
    s.name = name or f"{base.name}_constrained"

    for ctype, params in constraint_kwargs.items():
        s.constraints.append(ConstraintConfig(
            type=ctype,
            params=params,
        ))

    return s


# ────────────────────────────────────────────────────────────────────
# Batch / sweep runners
# ────────────────────────────────────────────────────────────────────

def sweep(
    base: Union[str, Path, Scenario],
    param_path: str,
    values: Sequence[Any],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    save: bool = True,
    output_root: Union[str, Path] = "outputs",
) -> list:
    """
    Run a parameter sweep over a single variable.

    Parameters
    ----------
    base : str, Path, or Scenario
        Base scenario.
    param_path : str
        Dot-notation path to the parameter, e.g.
        ``"economics.transport_cost_per_unit_km"``.
    values : sequence
        Values to sweep over.
    base_dir : Path, optional
        Root for resolving data paths.
    device : str, optional
        Force device.
    verbose : bool
        Print progress.
    save : bool
        Save each run to ``output_root/<scenario_name>/``.
    output_root : Path
        Parent directory for outputs.

    Returns
    -------
    list of Results
        One per value.

    Example
    -------
    ::

        results = sweep("scenarios/baseline.yaml",
                         "economics.transport_cost_per_unit_km",
                         [100, 150, 200, 250, 300])
        comparison = Results.compare(results)
    """
    from .solver import solve as _solve
    from .results import Results

    if isinstance(base, (str, Path)):
        base = Scenario.load(base)

    results = []
    for val in values:
        s = copy.deepcopy(base)
        s.name = f"{base.name}_{param_path.split('.')[-1]}_{val}"
        _set_nested(s, param_path, val)

        if verbose:
            print(f"\n{'━' * 56}")
            print(f"  Sweep: {param_path} = {val}")
            print(f"{'━' * 56}")

        r = _solve(s, device=device, base_dir=base_dir, verbose=verbose)

        if save:
            out = Path(output_root) / s.name
            r.save(out)

        results.append(r)

    if verbose and len(results) >= 2:
        print("\n" + "=" * 60)
        print("  SWEEP COMPARISON")
        print("=" * 60)
        print(Results.compare(results).to_string())
        print()

    return results


def batch_solve(
    scenarios: List[Union[str, Path, Scenario]],
    *,
    base_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    save: bool = True,
    output_root: Union[str, Path] = "outputs",
) -> list:
    """
    Solve a list of scenarios sequentially and return all Results.

    Parameters
    ----------
    scenarios : list
        Mix of YAML paths and/or Scenario objects.

    Returns
    -------
    list of Results

    Example
    -------
    ::

        results = batch_solve([
            "scenarios/baseline.yaml",
            "scenarios/high_transport.yaml",
            variant("scenarios/baseline.yaml", "big", model__num_candidate_plants=100),
        ])
        print(Results.compare(results).to_markdown())
    """
    from .solver import solve as _solve
    from .results import Results

    results = []
    for s in scenarios:
        if isinstance(s, (str, Path)):
            s = Scenario.load(s)
        r = _solve(s, device=device, base_dir=base_dir, verbose=verbose)
        if save:
            out = Path(output_root) / s.name
            r.save(out)
        results.append(r)

    if verbose and len(results) >= 2:
        print(Results.compare(results).to_string())

    return results
