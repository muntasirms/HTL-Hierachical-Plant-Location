"""
co2_analysis.py — Demonstrate all three CO₂ analysis modes.

This script:
  1. Runs the baseline (no CO₂ tracking) for reference
  2. Runs the same scenario with CO₂ in post_hoc mode
  3. Runs with CO₂ in co2_first mode (minimise emissions)
  4. Runs with CO₂ in combined mode (cost + carbon price × CO₂)
  5. Compares all four results side-by-side
  6. Sweeps co2_cost_weight to show the cost–CO₂ Pareto front

Run from the repo root:
    python examples/co2_analysis.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from htl_opt import Scenario, solve, Results, variant, sweep


def main():
    print("=" * 60)
    print("  CO₂ EMISSIONS ANALYSIS — Three Modes")
    print("=" * 60)

    # ── 1. Baseline (no emissions) ──────────────────────────────────
    print("\n▸ Running baseline (no emissions)...")
    s_base = Scenario.load(ROOT / "scenarios" / "baseline.yaml")
    res_base = solve(s_base, base_dir=ROOT, verbose=False)
    res_base.save(ROOT / "outputs" / "co2_example" / "baseline")

    # ── 2. Post-hoc mode ────────────────────────────────────────────
    print("▸ Running post-hoc CO₂ mode (optimise cost, report CO₂)...")
    s_post = Scenario.load(ROOT / "scenarios" / "co2_post_hoc.yaml")
    res_post = solve(s_post, base_dir=ROOT, verbose=False)
    res_post.save(ROOT / "outputs" / "co2_example" / "post_hoc")

    # ── 3. CO₂-first mode ──────────────────────────────────────────
    print("▸ Running CO₂-first mode (optimise CO₂, report cost)...")
    s_co2 = Scenario.load(ROOT / "scenarios" / "co2_minimise.yaml")
    res_co2 = solve(s_co2, base_dir=ROOT, verbose=False)
    res_co2.save(ROOT / "outputs" / "co2_example" / "co2_first")

    # ── 4. Combined mode ────────────────────────────────────────────
    print("▸ Running combined mode (cost + 0.05 × CO₂)...")
    s_comb = Scenario.load(ROOT / "scenarios" / "co2_combined.yaml")
    res_comb = solve(s_comb, base_dir=ROOT, verbose=False)
    res_comb.save(ROOT / "outputs" / "co2_example" / "combined")

    # ── 5. Comparison table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODE COMPARISON")
    print("=" * 70)
    comparison = Results.compare([res_base, res_post, res_co2, res_comb])
    print(comparison.to_string())
    comparison.to_csv(ROOT / "outputs" / "co2_example" / "comparison.csv")
    print(f"\n  📊 Comparison saved to outputs/co2_example/comparison.csv")

    # ── 6. Carbon price sweep (Pareto front) ────────────────────────
    print("\n" + "=" * 60)
    print("  CARBON PRICE SWEEP (Pareto front)")
    print("=" * 60)

    weights = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    sweep_results = []
    for w in weights:
        s = variant(ROOT / "scenarios" / "co2_combined.yaml",
                    f"carbon_price_{w}",
                    emissions__co2_cost_weight=w)
        r = solve(s, base_dir=ROOT, verbose=False)
        sweep_results.append(r)
        cost = r.summary()["total_system_cost"]
        co2 = r.summary().get("co2_total", "N/A")
        print(f"  α = {w:.2f}  →  cost = {cost:>14,.2f}  │  CO₂ = {co2}")

    # Save Pareto comparison
    pareto = Results.compare(sweep_results)
    pareto.to_csv(ROOT / "outputs" / "co2_example" / "pareto_sweep.csv")
    print(f"\n  📊 Pareto sweep saved to outputs/co2_example/pareto_sweep.csv")

    print(f"\n  ✓ CO₂ analysis complete.  All outputs in outputs/co2_example/\n")


if __name__ == "__main__":
    main()
