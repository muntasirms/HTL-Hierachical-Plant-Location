"""
PyTorch model wrapping the optimisable parameters and forward-pass
cost computations for single-hierarchy HTL plant siting.

Design note
-----------
All feedstock data is registered as **buffers** (not parameters) so it
automatically follows ``.to(device)`` but is never updated by the
optimiser.  Only ``plant_coords`` and ``assignment_logits`` carry
gradients.

CO₂ emissions are computed in parallel with economic costs using
intensity rates from :class:`EmissionsConfig`.  When emissions are
disabled, CO₂ keys are still present in the output dict but contain
zeros — this keeps downstream code simple and branch-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Scenario
from .geo import haversine_distance


class HTLPlantModel(nn.Module):
    """
    Single-hierarchy HTL facility location model.

    Optimisable parameters
    ----------------------
    plant_coords : (m, 2)
        Candidate plant latitude / longitude.
    assignment_logits : (n, m+1)
        Pre-softmax weights.  Column *m* is the "orphan" option
        (feedstock left unassigned).

    Forward outputs
    ---------------
    A dict of tensors containing every cost component and intermediate
    quantity needed for reporting and constraint evaluation, plus
    parallel CO₂ emissions when enabled.
    """

    def __init__(
        self,
        feed_coords: torch.Tensor,
        feed_amounts: torch.Tensor,
        tipping_fees: torch.Tensor,
        scenario: Scenario,
    ):
        super().__init__()
        m = scenario.model.num_candidate_plants
        n = feed_amounts.shape[0]

        if m > n:
            raise ValueError(
                f"num_candidate_plants ({m}) exceeds number of feedstock "
                f"sources ({n}).  Reduce model.num_candidate_plants."
            )

        # ── fixed data (buffers) ────────────────────────────────────
        self.register_buffer("feed_coords", feed_coords)
        self.register_buffer("feed_amounts", feed_amounts)
        self.register_buffer("tipping_fees", tipping_fees)

        # ── initialise plant locations ──────────────────────────────
        if scenario.model.initialization == "top_feedstock":
            sorted_idx = torch.argsort(feed_amounts, descending=True)
            plant_init = feed_coords[sorted_idx[:m]].clone()
        else:
            # Random initialisation within data bounding box
            lat_min, lat_max = feed_coords[:, 0].min(), feed_coords[:, 0].max()
            lon_min, lon_max = feed_coords[:, 1].min(), feed_coords[:, 1].max()
            plant_init = torch.stack([
                torch.empty(m).uniform_(lat_min.item(), lat_max.item()),
                torch.empty(m).uniform_(lon_min.item(), lon_max.item()),
            ], dim=1)

        self.plant_coords = nn.Parameter(plant_init)
        self.assignment_logits = nn.Parameter(torch.zeros(n, m + 1))

        self._m = m
        self._n = n
        self._eco = scenario.economics
        self._emi = scenario.emissions

    # ----------------------------------------------------------------
    def forward(self) -> dict:
        """
        Compute all cost components, intermediate quantities, and
        CO₂ emissions.

        Returns a dict with keys:

        **Economic outputs:**

        - ``assignments``          (n, m+1) soft assignment weights
        - ``distances``            (n, m)   km from each source to each plant
        - ``plant_loads``          (m,)     total feed arriving at each plant
        - ``delivery_cost``        scalar   total transport + tipping cost
        - ``capital_cost``         scalar   total capital cost
        - ``revenue``              scalar   total system revenue
        - ``total_cost``           scalar   delivery + capital − revenue
        - ``orphan_amount``        scalar   total orphaned feed
        - ``total_feed``           scalar   sum of all feedstock
        - ``total_delivered``      scalar   total feed actually delivered
        - ``plant_delivery_costs`` (m,)     per-plant transport cost
        - ``plant_revenues``       (m,)     per-plant revenue
        - ``plant_capital_costs``  (m,)     per-plant capital cost
        - ``plant_npvs``           (m,)     per-plant net value

        **CO₂ outputs (zeros when emissions disabled):**

        - ``co2_transport``        scalar   total transport CO₂
        - ``co2_orphan``           scalar   total orphan CO₂
        - ``co2_processing``       scalar   total processing CO₂
        - ``co2_fuel_credit``      scalar   total fuel displacement credit
        - ``co2_capital``          scalar   total embodied capital CO₂
        - ``co2_total``            scalar   net system CO₂
        - ``plant_co2_transport``  (m,)     per-plant transport CO₂
        - ``plant_co2_processing`` (m,)     per-plant processing CO₂
        - ``plant_co2_fuel_credit``(m,)     per-plant fuel credit
        - ``plant_co2_capital``    (m,)     per-plant capital CO₂
        - ``plant_co2_total``      (m,)     per-plant net CO₂
        """
        eco = self._eco
        emi = self._emi
        m = self._m

        # ── soft assignments ────────────────────────────────────────
        assignments = F.softmax(self.assignment_logits, dim=1)  # (n, m+1)

        # ── distances ───────────────────────────────────────────────
        distances = haversine_distance(self.feed_coords, self.plant_coords)  # (n, m)

        # ── delivery cost matrix ────────────────────────────────────
        #   cost_ij = feed_i * (tipping_i + transport_rate * dist_ij)
        cost_delivery = self.feed_amounts.unsqueeze(1) * (
            self.tipping_fees.unsqueeze(1)
            + eco.transport_cost_per_unit_km * distances
        )  # (n, m)

        cost_orphan = (
            self.feed_amounts * eco.orphan_penalty
        ).unsqueeze(1)  # (n, 1)

        cost_matrix = torch.cat([cost_delivery, cost_orphan], dim=1)  # (n, m+1)

        delivery_cost = torch.sum(assignments * cost_matrix)

        # ── plant loads ─────────────────────────────────────────────
        plant_loads = torch.sum(
            self.feed_amounts.unsqueeze(1) * assignments[:, :m], dim=0
        )  # (m,)

        # ── capital cost (system-wide) ──────────────────────────────
        capital_cost = eco.capital_cost_coef * torch.sum(
            plant_loads ** eco.capital_cost_exponent
        )

        # ── revenue (system-wide) ──────────────────────────────────
        total_delivered = torch.sum(
            self.feed_amounts * torch.sum(assignments[:, :m], dim=1)
        )
        revenue = eco.revenue_coef * (total_delivered ** eco.revenue_exponent)

        # ── per-plant breakdown ─────────────────────────────────────
        plant_delivery_costs = torch.sum(
            assignments[:, :m] * cost_delivery, dim=0
        )  # (m,)

        per_plant_feed = torch.sum(
            assignments[:, :m] * self.feed_amounts.unsqueeze(1), dim=0
        )  # (m,)

        plant_revenues = eco.revenue_coef * (
            per_plant_feed ** eco.revenue_exponent
        )  # (m,)

        plant_capital_costs = eco.capital_cost_coef * (
            plant_loads ** eco.capital_cost_exponent
        )  # (m,)

        plant_npvs = plant_revenues - plant_delivery_costs - plant_capital_costs

        # ── orphaned feed ───────────────────────────────────────────
        orphan_amount = torch.sum(self.feed_amounts * assignments[:, m])
        total_feed = torch.sum(self.feed_amounts)

        # ── true economic cost (NO Lagrangian penalties) ────────────
        total_cost = delivery_cost + capital_cost - revenue

        # ── CO₂ emissions ───────────────────────────────────────────
        device = self.feed_amounts.device
        zero = torch.tensor(0.0, device=device)

        if emi.enabled:
            # Transport CO₂:  Σ aᵢⱼ · fᵢ · (rate_per_km · dᵢⱼ)
            co2_delivery_matrix = self.feed_amounts.unsqueeze(1) * (
                emi.co2_transport_per_unit_km * distances
            )  # (n, m)
            co2_orphan_matrix = (
                self.feed_amounts * emi.co2_orphan_per_unit
            ).unsqueeze(1)  # (n, 1)

            co2_transport = torch.sum(
                assignments[:, :m] * co2_delivery_matrix
            )
            co2_orphan = torch.sum(
                assignments[:, m:] * co2_orphan_matrix
            )

            # Processing CO₂:  Σ aᵢⱼ · fᵢ · processing_rate  (j < m)
            co2_processing = emi.co2_processing_per_unit * total_delivered

            # Fuel displacement credit (negative):  rate * total_delivered
            co2_fuel_credit = emi.co2_fuel_displacement_credit * total_delivered

            # Capital (embodied) CO₂
            co2_capital = emi.co2_capital_per_unit * torch.sum(
                plant_loads ** emi.co2_capital_exponent
            )

            # Net system CO₂
            co2_total = (
                co2_transport + co2_orphan + co2_processing
                + co2_fuel_credit + co2_capital
            )

            # ── per-plant CO₂ breakdown ─────────────────────────────
            plant_co2_transport = torch.sum(
                assignments[:, :m] * co2_delivery_matrix, dim=0
            )  # (m,)

            plant_co2_processing = emi.co2_processing_per_unit * per_plant_feed
            plant_co2_fuel_credit = emi.co2_fuel_displacement_credit * per_plant_feed

            plant_co2_capital = emi.co2_capital_per_unit * (
                plant_loads ** emi.co2_capital_exponent
            )  # (m,)

            plant_co2_total = (
                plant_co2_transport + plant_co2_processing
                + plant_co2_fuel_credit + plant_co2_capital
            )  # (m,)
        else:
            # Emissions disabled — fill with zeros
            co2_transport = zero
            co2_orphan = zero
            co2_processing = zero
            co2_fuel_credit = zero
            co2_capital = zero
            co2_total = zero

            plant_co2_transport = torch.zeros(m, device=device)
            plant_co2_processing = torch.zeros(m, device=device)
            plant_co2_fuel_credit = torch.zeros(m, device=device)
            plant_co2_capital = torch.zeros(m, device=device)
            plant_co2_total = torch.zeros(m, device=device)

        return {
            # Economic outputs
            "assignments":          assignments,
            "distances":            distances,
            "plant_loads":          plant_loads,
            "delivery_cost":        delivery_cost,
            "capital_cost":         capital_cost,
            "revenue":              revenue,
            "total_cost":           total_cost,
            "orphan_amount":        orphan_amount,
            "total_feed":           total_feed,
            "total_delivered":      total_delivered,
            "plant_delivery_costs": plant_delivery_costs,
            "plant_revenues":       plant_revenues,
            "plant_capital_costs":  plant_capital_costs,
            "plant_npvs":           plant_npvs,
            # CO₂ outputs
            "co2_transport":         co2_transport,
            "co2_orphan":            co2_orphan,
            "co2_processing":        co2_processing,
            "co2_fuel_credit":       co2_fuel_credit,
            "co2_capital":           co2_capital,
            "co2_total":             co2_total,
            "plant_co2_transport":   plant_co2_transport,
            "plant_co2_processing":  plant_co2_processing,
            "plant_co2_fuel_credit": plant_co2_fuel_credit,
            "plant_co2_capital":     plant_co2_capital,
            "plant_co2_total":       plant_co2_total,
        }

