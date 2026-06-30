"""
compare_scenarios.py — Run multiple scenarios and produce a comparison table.

Demonstrates the scenario comparison workflow:
  1. Define or load multiple scenarios
  2. Solve each one
  3. Use Results.compare() to get a side-by-side table

Run from the repo root:
    python examples/compare_scenarios.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from htl_opt import Scenario, solve, Results


def main():
    scenario_files = [
        ROOT / "scenarios" / "baseline.yaml",
        ROOT / "scenarios" / "high_transport.yaml",
    ]

    all_results = []

    for sf in scenario_files:
        if not sf.exists():
            print(f"  ⚠ Skipping {sf.name} (not found)")
            continue

        scenario = Scenario.load(sf)
        print(f"\n{'━' * 56}")
        print(f"  Running: {scenario.name}")
        print(f"{'━' * 56}")

        results = solve(scenario, base_dir=ROOT)

        # Save each run
        out_dir = ROOT / "outputs" / scenario.name
        results.save(out_dir)
        results.plot_map(
            interactive=False,
            save_path=str(out_dir / "map.png"),
        )

        all_results.append(results)

    # ── Comparison table ────────────────────────────────────────────
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("  SCENARIO COMPARISON")
        print("=" * 70)
        comparison = Results.compare(all_results)
        print(comparison.to_string())
        print()

        # Save comparison CSV
        comparison.to_csv(ROOT / "outputs" / "comparison.csv")
        print(f"  📊 Comparison table saved to outputs/comparison.csv\n")


if __name__ == "__main__":
    main()
