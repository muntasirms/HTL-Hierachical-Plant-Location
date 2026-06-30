"""
Data loading and preprocessing.

Reads feedstock CSVs, generates tipping fees, and returns device-ready tensors.
"""

from __future__ import annotations

import torch
import pandas as pd
from pathlib import Path
from typing import Tuple

from .config import Scenario


def load_feedstock_data(
    scenario: Scenario,
    device: torch.device,
    base_dir: str | Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load feedstock locations and capacities from CSV.

    Parameters
    ----------
    scenario : Scenario
        Run configuration (specifies file path, column names, etc.).
    device : torch.device
        Target device for tensors (cpu / cuda).
    base_dir : Path, optional
        Directory to resolve relative ``feedstock_file`` paths against.
        Defaults to the current working directory.

    Returns
    -------
    feed_coords : Tensor (n, 2)
        Latitude, longitude for each feedstock source.
    feed_amounts : Tensor (n,)
        Feedstock capacity in the configured unit.
    tipping_fees : Tensor (n,)
        Per-source tipping fee.
    """
    cfg = scenario.data

    # ── resolve path ────────────────────────────────────────────────
    csv_path = Path(cfg.feedstock_file)
    if not csv_path.is_absolute() and base_dir is not None:
        csv_path = Path(base_dir) / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feedstock CSV not found: {csv_path}\n"
            f"Place your data file at this path, or update "
            f"scenario.data.feedstock_file."
        )

    # ── read CSV ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = [cfg.lat_column, cfg.lon_column, cfg.scale_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Expected: {required}"
        )

    # Drop rows with NaN in required columns
    n_before = len(df)
    df = df.dropna(subset=required)
    n_after = len(df)
    if n_after < n_before:
        print(f"  ⚠ Dropped {n_before - n_after} rows with missing data "
              f"({n_after} sources remaining)")

    # ── build tensors ───────────────────────────────────────────────
    latitudes  = torch.tensor(df[cfg.lat_column].values, dtype=torch.float32)
    longitudes = torch.tensor(df[cfg.lon_column].values, dtype=torch.float32)
    feed_amounts = torch.tensor(df[cfg.scale_column].values, dtype=torch.float32)

    feed_coords = torch.stack([latitudes, longitudes], dim=1).to(device)
    feed_amounts = feed_amounts.to(device)

    # ── tipping fees ────────────────────────────────────────────────
    n = len(feed_amounts)
    tf_cfg = scenario.economics.tipping_fee
    if tf_cfg.mode == "fixed":
        tipping_fees = torch.full((n,), tf_cfg.fixed_value, device=device)
    elif tf_cfg.mode == "uniform_random":
        tipping_fees = torch.empty(n, device=device).uniform_(
            tf_cfg.random_min, tf_cfg.random_max
        )
    elif tf_cfg.mode == "from_column":
        if tf_cfg.column_name not in df.columns:
            raise ValueError(
                f"Tipping-fee column '{tf_cfg.column_name}' not found in CSV."
            )
        tipping_fees = torch.tensor(
            df[tf_cfg.column_name].values, dtype=torch.float32, device=device
        )
    else:
        raise ValueError(f"Unknown tipping-fee mode: '{tf_cfg.mode}'")

    print(f"  Loaded {n} feedstock sources from {csv_path.name}")
    print(f"  Feed range: {feed_amounts.min().item():.2f} – "
          f"{feed_amounts.max().item():.2f} {cfg.feedstock_unit}")

    return feed_coords, feed_amounts, tipping_fees
