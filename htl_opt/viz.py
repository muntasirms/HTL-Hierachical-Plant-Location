"""
Visualisation — maps, convergence plots, and scenario comparisons.

Two map backends are available:

- **Folium** (default, ``interactive=True``): produces an HTML file with
  pan/zoom, popups, and layer controls.  Requires ``folium``.
- **Matplotlib** (``interactive=False``): produces a static PNG suitable
  for HPC / headless environments.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .results import Results


# ────────────────────────────────────────────────────────────────────
# Convergence plot
# ────────────────────────────────────────────────────────────────────

def plot_convergence(
    results: "Results",
    save_path: str | None = None,
    figsize: tuple = (10, 5),
):
    """
    Plot true economic cost and Lagrangian penalty vs. epoch.
    """
    if not results.history:
        print("  No history to plot.")
        return

    import pandas as pd
    df = pd.DataFrame(results.history)

    fig, ax1 = plt.subplots(figsize=figsize)

    color_cost = "#2563eb"
    color_pen  = "#dc2626"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("True System Cost", color=color_cost)
    ax1.plot(df["epoch"], df["true_cost"], color=color_cost, linewidth=1.2,
             label="True cost")
    ax1.tick_params(axis="y", labelcolor=color_cost)

    if df["penalty"].abs().max() > 0:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Lagrangian Penalty", color=color_pen)
        ax2.plot(df["epoch"], df["penalty"], color=color_pen, linewidth=0.8,
                 alpha=0.7, linestyle="--", label="Penalty")
        ax2.tick_params(axis="y", labelcolor=color_pen)

    ax1.set_title(f"Convergence — {results.scenario.name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Convergence plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Map — interactive (Folium)
# ────────────────────────────────────────────────────────────────────

def _folium_map(results: "Results", save_path: str | None) -> object:
    """Build an interactive Folium map."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        print("  folium not installed — falling back to matplotlib map.")
        return _matplotlib_map(results, save_path)

    plants_df = results.plants_df()
    assigns_df = results.assignments_df()
    m = results.scenario.model.num_candidate_plants

    # Centre on data centroid
    center_lat = results.feed_coords_np[:, 0].mean()
    center_lon = results.feed_coords_np[:, 1].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # ── colour palette for plants ───────────────────────────────────
    active = plants_df[plants_df["active"]]
    n_active = len(active)
    cmap = plt.cm.get_cmap("tab20", max(n_active, 1))
    plant_colors = {}
    for idx, (_, row) in enumerate(active.iterrows()):
        rgba = cmap(idx % 20)
        hex_color = matplotlib.colors.rgb2hex(rgba[:3])
        plant_colors[int(row["plant_id"])] = hex_color

    # ── plant markers ───────────────────────────────────────────────
    plant_group = folium.FeatureGroup(name="Plants", show=True)
    for _, row in active.iterrows():
        pid = int(row["plant_id"])
        radius = max(4, min(20, np.sqrt(row["load"]) * 2))
        popup_html = (
            f"<b>Plant {pid}</b><br>"
            f"Load: {row['load']:,.1f}<br>"
            f"Revenue: ${row['revenue']:,.0f}<br>"
            f"Capital: ${row['capital_cost']:,.0f}<br>"
            f"Transport: ${row['delivery_cost']:,.0f}<br>"
            f"NPV: ${row['npv']:,.0f}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=plant_colors.get(pid, "#333"),
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Plant {pid}",
        ).add_to(plant_group)
    plant_group.add_to(fmap)

    # ── assignment lines ────────────────────────────────────────────
    line_group = folium.FeatureGroup(name="Assignments", show=True)
    plant_coords = results.plant_coords_np
    for _, row in assigns_df.iterrows():
        pid = int(row["assigned_to"])
        if pid >= m or row["delivered"] < 1e-3:
            continue
        color = plant_colors.get(pid, "#999")
        folium.PolyLine(
            locations=[
                [row["latitude"], row["longitude"]],
                [plant_coords[pid, 0], plant_coords[pid, 1]],
            ],
            color=color,
            weight=max(0.5, min(4, row["delivered"] * 0.5)),
            opacity=0.45,
        ).add_to(line_group)
    line_group.add_to(fmap)

    # ── feedstock markers (clustered) ───────────────────────────────
    feed_cluster = MarkerCluster(name="Feedstock Sources", show=False)
    for _, row in assigns_df.iterrows():
        pid = int(row["assigned_to"])
        status = "orphaned" if row["is_orphaned"] else f"→ Plant {pid}"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="#888" if row["is_orphaned"] else plant_colors.get(pid, "#333"),
            fill=True,
            fill_opacity=0.5,
            tooltip=f"Source {int(row['source_id'])}: {row['feed_amount']:.1f} ({status})",
        ).add_to(feed_cluster)
    feed_cluster.add_to(fmap)

    # ── orphaned sources layer ──────────────────────────────────────
    orphan_group = folium.FeatureGroup(name="Orphaned Sources", show=True)
    orphaned = assigns_df[assigns_df["is_orphaned"]]
    for _, row in orphaned.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#ef4444",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"ORPHANED — Source {int(row['source_id'])}: {row['feed_amount']:.1f}",
        ).add_to(orphan_group)
    orphan_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    if save_path:
        fmap.save(save_path)
        print(f"  🗺  Interactive map saved to {save_path}")

    return fmap


# ────────────────────────────────────────────────────────────────────
# Map — static (Matplotlib)
# ────────────────────────────────────────────────────────────────────

def _matplotlib_map(
    results: "Results",
    save_path: str | None,
    figsize: tuple = (12, 9),
):
    """Fallback static map using plain matplotlib."""
    m = results.scenario.model.num_candidate_plants
    plants_df = results.plants_df()
    assigns_df = results.assignments_df()
    feed = results.feed_coords_np
    plant = results.plant_coords_np
    hard = results.hard_assignments

    fig, ax = plt.subplots(figsize=figsize)

    # Feedstock sources
    orphan_mask = hard == m
    ax.scatter(
        feed[~orphan_mask, 1], feed[~orphan_mask, 0],
        c="#94a3b8", s=5, alpha=0.4, label="Assigned sources", zorder=2,
    )
    ax.scatter(
        feed[orphan_mask, 1], feed[orphan_mask, 0],
        c="#ef4444", s=10, alpha=0.6, label="Orphaned sources",
        marker="x", zorder=3,
    )

    # Assignment lines (top-weighted only to reduce clutter)
    assigns = results.assignments_np
    amounts = results._feed_amounts.numpy()
    max_del = 1.0
    for i in range(len(feed)):
        j = hard[i]
        if j >= m:
            continue
        w = amounts[i] * assigns[i, j]
        max_del = max(max_del, w)

    for i in range(len(feed)):
        j = hard[i]
        if j >= m:
            continue
        w = amounts[i] * assigns[i, j]
        if w < max_del * 0.01:
            continue
        lw = 0.3 + 2.5 * (w / max_del)
        ax.plot(
            [feed[i, 1], plant[j, 1]],
            [feed[i, 0], plant[j, 0]],
            color="#3b82f6", linewidth=lw, alpha=0.25, zorder=1,
        )

    # Active plants
    active = plants_df[plants_df["active"]]
    sizes = 30 + 300 * (active["load"].values / max(active["load"].max(), 1))
    ax.scatter(
        active["longitude"], active["latitude"],
        c="#16a34a", s=sizes, edgecolors="white", linewidths=0.8,
        label=f"Active plants ({len(active)})", zorder=4,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"HTL Plant Siting — {results.scenario.name}")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  🗺  Static map saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Public dispatcher
# ────────────────────────────────────────────────────────────────────

def plot_map(
    results: "Results",
    interactive: bool = True,
    save_path: str | None = None,
):
    """
    Render a map showing plant locations, feedstock assignments, and
    orphaned sources.

    Parameters
    ----------
    results : Results
        Completed optimisation results.
    interactive : bool
        ``True`` → Folium HTML map;  ``False`` → matplotlib PNG.
    save_path : str, optional
        File path to save the map.  If *None*, displays inline.
    """
    if interactive:
        return _folium_map(results, save_path)
    else:
        return _matplotlib_map(results, save_path)
