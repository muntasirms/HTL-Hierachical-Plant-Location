"""
quickstart.py — Minimal example of running the HTL plant optimiser.

This script shows the simplest workflow:
  1. Load a scenario from YAML
  2. Solve it
  3. Inspect results
  4. Save outputs

Run from the repo root:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from htl_opt import Scenario, solve, Results


def main():
    # ── 1. Load scenario ────────────────────────────────────────────
    scenario = Scenario.load(ROOT / "scenarios" / "baseline.yaml")

    # Optional: tweak parameters in Python instead of editing YAML
    # scenario.model.num_candidate_plants = 30
    # scenario.economics.orphan_penalty = 100

    # ── 2. Solve ────────────────────────────────────────────────────
    results = solve(scenario, base_dir=ROOT)

    # ── 3. Inspect ──────────────────────────────────────────────────
    results.summary()

    # Per-plant breakdown
    plants = results.plants_df()
    print("\nTop 10 plants by load:")
    print(plants.head(10).to_string(index=False))

    # Orphaned sources
    orphaned = results.orphaned_df()
    print(f"\nOrphaned sources: {len(orphaned)}")

    # ── 4. Save everything ──────────────────────────────────────────
    out_dir = ROOT / "outputs" / "quickstart"
    results.save(out_dir)
    results.plot_convergence(save_path=str(out_dir / "convergence.png"))
    results.plot_map(interactive=True, save_path=str(out_dir / "map.html"))

    print(f"\nDone! Check {out_dir}")


if __name__ == "__main__":
    main()
