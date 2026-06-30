"""
Geographic utility functions.

Haversine distance on the GPU, region-exclusion penalties, etc.
"""

from __future__ import annotations

import math
import torch


def haversine_distance(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    R: float = 6_371.0,
) -> torch.Tensor:
    """
    Vectorised Haversine distance between two sets of lat/lon points.

    Parameters
    ----------
    coords_a : Tensor (n, 2)
        First set of points as ``[latitude, longitude]`` in **degrees**.
    coords_b : Tensor (m, 2)
        Second set of points as ``[latitude, longitude]`` in **degrees**.
    R : float
        Earth radius in km (default 6 371 km).

    Returns
    -------
    Tensor (n, m)
        Pairwise great-circle distances in kilometres.
    """
    a_rad = coords_a * (math.pi / 180.0)   # (n, 2)
    b_rad = coords_b * (math.pi / 180.0)   # (m, 2)

    a_rad = a_rad.unsqueeze(1)              # (n, 1, 2)
    b_rad = b_rad.unsqueeze(0)              # (1, m, 2)

    dlat = b_rad[..., 0] - a_rad[..., 0]   # (n, m)
    dlon = b_rad[..., 1] - a_rad[..., 1]

    a_val = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(a_rad[..., 0])
        * torch.cos(b_rad[..., 0])
        * torch.sin(dlon / 2) ** 2
    )
    a_val = torch.clamp(a_val, min=1e-7, max=1.0 - 1e-7)
    c = 2 * torch.atan2(torch.sqrt(a_val), torch.sqrt(1.0 - a_val + 1e-7))

    return R * c


def region_penalty(
    plant_coords: torch.Tensor,
    forbidden_regions: list,
    strength: float = 1e7,
) -> torch.Tensor:
    """
    Smooth penalty for plants inside forbidden geographic regions.

    Each region is a dict with ``type`` ("circle" or "rectangle") and
    the corresponding geometry keys.  Returns a scalar penalty tensor.
    """
    device = plant_coords.device
    penalty = torch.tensor(0.0, device=device)

    for region in forbidden_regions:
        if region["type"] == "circle":
            center = torch.tensor(
                region["center"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            dist = haversine_distance(plant_coords, center).squeeze(1)
            penalty = penalty + strength * torch.sum(
                torch.relu(region["radius_km"] - dist) ** 2
            )

        elif region["type"] == "rectangle":
            lat = plant_coords[:, 0]
            lon = plant_coords[:, 1]
            lat_viol = (
                torch.relu(region["min_lat"] - lat)
                + torch.relu(lat - region["max_lat"])
            )
            lon_viol = (
                torch.relu(region["min_lon"] - lon)
                + torch.relu(lon - region["max_lon"])
            )
            penalty = penalty + strength * torch.sum(lat_viol + lon_viol)

    return penalty
