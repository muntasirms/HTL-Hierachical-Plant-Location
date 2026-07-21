"""
Results container — post-processing, reporting, and persistence.

The :class:`Results` object is the single return value of ``solve()``.
It provides:

- ``.summary()``  — print a human-readable report
- ``.save(dir)``  — write CSVs, JSON, config, and figures to disk
- ``.plot_map()`` — render an interactive Folium map *or* static matplotlib
- ``.compare()``  — class method to build a scenario-comparison table

When CO₂ emissions tracking is enabled, all reporting and persistence
methods automatically include the parallel CO₂ metrics.
"""

from __future__ import annotations

import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import Scenario


@dataclass
class Results:
    """
    Immutable container for a completed optimisation run.

    All tensor data is detached and moved to CPU on construction so the
    GPU memory can be freed immediately.
    """

    scenario: Scenario
    model: object          # HTLPlantModel (kept for viz access)
    outputs: dict          # final forward-pass tensors
    history: list          # per-epoch log dicts
    device: torch.device
    elapsed_seconds: float

    def __post_init__(self):
        # Detach all tensors to CPU for downstream use
        self._cpu = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self.outputs.items()
        }
        self._feed_coords = self.model.feed_coords.detach().cpu()
        self._feed_amounts = self.model.feed_amounts.detach().cpu()
        self._plant_coords = self.model.plant_coords.detach().cpu()
        self._emissions_active = self.scenario.emissions.enabled

    # ── convenience accessors ───────────────────────────────────────
    @property
    def plant_coords_np(self) -> np.ndarray:
        return self._plant_coords.numpy()

    @property
    def feed_coords_np(self) -> np.ndarray:
        return self._feed_coords.numpy()

    @property
    def assignments_np(self) -> np.ndarray:
        return self._cpu["assignments"].numpy()

    @property
    def plant_loads_np(self) -> np.ndarray:
        return self._cpu["plant_loads"].numpy()

    @property
    def hard_assignments(self) -> np.ndarray:
        """Argmax assignment: index of assigned plant (m = orphan)."""
        return torch.argmax(self._cpu["assignments"], dim=1).numpy()

    # ── summary ─────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Print and return a summary dict of key metrics."""
        o = self._cpu
        m = self.scenario.model.num_candidate_plants
        hard = self.hard_assignments

        # Active plants = those with at least one source assigned
        active_plants = len(set(hard[hard < m]))
        orphaned_sources = int((hard == m).sum())
        total_feed = o["total_feed"].item()
        orphan_pct = o["orphan_amount"].item() / max(total_feed, 1e-12) * 100

        metrics = {
            "scenario":              self.scenario.name,
            "device":                str(self.device),
            "wall_time_s":           round(self.elapsed_seconds, 1),
            "epochs":                self.history[-1]["epoch"] if self.history else 0,
            "total_sources":         len(self._feed_amounts),
            "candidate_plants":      m,
            "active_plants":         active_plants,
            "orphaned_sources":      orphaned_sources,
            "orphan_pct":            round(orphan_pct, 2),
            "total_feed":            round(total_feed, 2),
            "total_delivered":       round(o["total_delivered"].item(), 2),
            "delivery_cost":         round(o["delivery_cost"].item(), 2),
            "capital_cost":          round(o["capital_cost"].item(), 2),
            "revenue":               round(o["revenue"].item(), 2),
            "total_system_cost":     round(o["total_cost"].item(), 2),
        }

        # CO₂ metrics (always present in outputs, report when enabled)
        if self._emissions_active:
            metrics.update({
                "co2_transport":     round(o["co2_transport"].item(), 2),
                "co2_orphan":        round(o["co2_orphan"].item(), 2),
                "co2_processing":    round(o["co2_processing"].item(), 2),
                "co2_fuel_credit":   round(o["co2_fuel_credit"].item(), 2),
                "co2_capital":       round(o["co2_capital"].item(), 2),
                "co2_total":         round(o["co2_total"].item(), 2),
                "emissions_mode":    self.scenario.emissions.mode,
            })

        # Pretty-print
        print("\n" + "=" * 56)
        print(f"  RESULTS — {self.scenario.name}")
        print("=" * 56)
        for k, v in metrics.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f"  {label:<28s}  {v:>18,.2f}")
            else:
                print(f"  {label:<28s}  {str(v):>18s}")
        print("=" * 56 + "\n")

        return metrics

    # ── DataFrames ──────────────────────────────────────────────────
    def plants_df(self) -> pd.DataFrame:
        """Per-plant DataFrame: location, load, costs, revenue, NPV, and CO₂."""
        o = self._cpu
        m = self.scenario.model.num_candidate_plants
        coords = self.plant_coords_np

        df = pd.DataFrame({
            "plant_id":      range(m),
            "latitude":      coords[:, 0],
            "longitude":     coords[:, 1],
            "load":          o["plant_loads"].numpy(),
            "delivery_cost": o["plant_delivery_costs"].numpy(),
            "capital_cost":  o["plant_capital_costs"].numpy(),
            "revenue":       o["plant_revenues"].numpy(),
            "npv":           o["plant_npvs"].numpy(),
        })

        # CO₂ columns
        if self._emissions_active:
            df["co2_transport"]   = o["plant_co2_transport"].numpy()
            df["co2_processing"]  = o["plant_co2_processing"].numpy()
            df["co2_fuel_credit"] = o["plant_co2_fuel_credit"].numpy()
            df["co2_capital"]     = o["plant_co2_capital"].numpy()
            df["co2_total"]       = o["plant_co2_total"].numpy()

        # Mark active
        hard = self.hard_assignments
        active_ids = set(hard[hard < m])
        df["active"] = df["plant_id"].isin(active_ids)

        return df.sort_values("load", ascending=False).reset_index(drop=True)

    def assignments_df(self) -> pd.DataFrame:
        """Per-source DataFrame: location, assigned plant, delivered amount."""
        m = self.scenario.model.num_candidate_plants
        hard = self.hard_assignments
        feed = self._feed_amounts.numpy()
        coords = self.feed_coords_np
        assigns = self.assignments_np

        df = pd.DataFrame({
            "source_id":     range(len(feed)),
            "latitude":      coords[:, 0],
            "longitude":     coords[:, 1],
            "feed_amount":   feed,
            "assigned_to":   hard,
            "is_orphaned":   hard == m,
        })
        # Weighted delivery amount to assigned plant
        delivered = np.array([
            feed[i] * assigns[i, hard[i]] if hard[i] < m else 0.0
            for i in range(len(feed))
        ])
        df["delivered"] = delivered

        # Per-source transport CO₂ (to assigned plant)
        if self._emissions_active:
            o = self._cpu
            distances = o["distances"].numpy()  # (n, m)
            emi = self.scenario.emissions
            co2_per_source = np.array([
                feed[i] * assigns[i, hard[i]] * emi.co2_transport_per_unit_km
                * distances[i, hard[i]] if hard[i] < m else
                feed[i] * assigns[i, hard[i]] * emi.co2_orphan_per_unit
                for i in range(len(feed))
            ])
            df["co2_transport"] = co2_per_source

        return df

    def orphaned_df(self) -> pd.DataFrame:
        """Subset of sources that are orphaned (assigned to no plant)."""
        return self.assignments_df().query("is_orphaned").reset_index(drop=True)

    # ── persistence ─────────────────────────────────────────────────
    def save(self, output_dir: str | Path) -> Path:
        """
        Save all results to ``output_dir/``:

        - ``config.yaml``        — scenario for reproducibility
        - ``summary.json``       — key metrics (including CO₂ when enabled)
        - ``plants.csv``         — per-plant table (including CO₂ columns)
        - ``assignments.csv``    — per-source assignments
        - ``orphaned.csv``       — orphaned sources
        - ``convergence.csv``    — training history
        - ``co2_summary.json``   — CO₂ breakdown (when emissions enabled)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Config
        self.scenario.save(out / "config.yaml")

        # Summary
        metrics = self.summary()
        with open(out / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Tables
        self.plants_df().to_csv(out / "plants.csv", index=False)
        self.assignments_df().to_csv(out / "assignments.csv", index=False)
        self.orphaned_df().to_csv(out / "orphaned.csv", index=False)

        # Convergence history
        if self.history:
            pd.DataFrame(self.history).to_csv(
                out / "convergence.csv", index=False
            )

        # CO₂ summary (separate file for easy downstream consumption)
        if self._emissions_active:
            o = self._cpu
            co2_data = {
                "emissions_mode":    self.scenario.emissions.mode,
                "co2_unit":          "kg CO₂",
                "co2_transport":     round(o["co2_transport"].item(), 4),
                "co2_orphan":        round(o["co2_orphan"].item(), 4),
                "co2_processing":    round(o["co2_processing"].item(), 4),
                "co2_fuel_credit":   round(o["co2_fuel_credit"].item(), 4),
                "co2_capital":       round(o["co2_capital"].item(), 4),
                "co2_total":         round(o["co2_total"].item(), 4),
                "co2_per_unit_delivered": round(
                    o["co2_total"].item() / max(o["total_delivered"].item(), 1e-12), 6
                ),
                "parameters": {
                    "co2_transport_per_unit_km":      self.scenario.emissions.co2_transport_per_unit_km,
                    "co2_orphan_per_unit":            self.scenario.emissions.co2_orphan_per_unit,
                    "co2_processing_per_unit":        self.scenario.emissions.co2_processing_per_unit,
                    "co2_fuel_displacement_credit":   self.scenario.emissions.co2_fuel_displacement_credit,
                    "co2_capital_per_unit":           self.scenario.emissions.co2_capital_per_unit,
                    "co2_cost_weight":               self.scenario.emissions.co2_cost_weight,
                },
            }
            with open(out / "co2_summary.json", "w") as f:
                json.dump(co2_data, f, indent=2)

        print(f"  📁 Results saved to {out.resolve()}")
        return out

    # ── visualisation delegates ─────────────────────────────────────
    def plot_map(self, interactive: bool = True, save_path: str | None = None):
        """Render a map of plants and assignments.  See ``viz`` module."""
        from .viz import plot_map
        return plot_map(self, interactive=interactive, save_path=save_path)

    def plot_convergence(self, save_path: str | None = None):
        """Plot cost vs. epoch.  See ``viz`` module."""
        from .viz import plot_convergence
        return plot_convergence(self, save_path=save_path)

    # ── scenario comparison ─────────────────────────────────────────
    @staticmethod
    def compare(results_list: List["Results"]) -> pd.DataFrame:
        """
        Build a side-by-side comparison table from multiple Results.

        CO₂ columns are included when any scenario has emissions enabled.

        Usage::

            df = Results.compare([res_baseline, res_high_transport])
            print(df.to_markdown())
        """
        rows = [r.summary() for r in results_list]
        return pd.DataFrame(rows).set_index("scenario")

