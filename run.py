"""
run.py — CLI entry point for htl_opt.

Usage
-----
    # Run a scenario from YAML:
    python run.py scenarios/baseline.yaml

    # Override output directory:
    python run.py scenarios/baseline.yaml --output outputs/my_run

    # Force CPU:
    python run.py scenarios/baseline.yaml --device cpu

    # Enable CO₂ tracking (overrides YAML):
    python run.py scenarios/baseline.yaml --emissions-mode post_hoc

    # Combined mode with custom carbon price:
    python run.py scenarios/baseline.yaml --emissions-mode combined --co2-weight 0.10

    # Compare two saved runs:
    python run.py --compare outputs/baseline outputs/high_transport
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `htl_opt` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from htl_opt import Scenario, solve, Results


def main():
    parser = argparse.ArgumentParser(
        description="HTL Plant Location Optimiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Path to a scenario YAML file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: outputs/<scenario_name>).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip map generation.",
    )
    parser.add_argument(
        "--static-map",
        action="store_true",
        help="Use matplotlib static map instead of Folium.",
    )
    parser.add_argument(
        "--emissions-mode",
        choices=["post_hoc", "co2_first", "combined"],
        default=None,
        help="Enable CO₂ tracking with the specified mode (overrides YAML).",
    )
    parser.add_argument(
        "--co2-weight",
        type=float,
        default=None,
        help="Carbon price $/kg CO₂ for combined mode (overrides YAML).",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="DIR",
        help="Compare saved results from multiple output directories.",
    )
    args = parser.parse_args()

    # ── compare mode ────────────────────────────────────────────────
    if args.compare:
        rows = []
        for d in args.compare:
            p = Path(d) / "summary.json"
            if not p.exists():
                print(f"  ⚠ No summary.json in {d}, skipping.")
                continue
            with open(p) as f:
                rows.append(json.load(f))
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows).set_index("scenario")
            print("\n" + df.to_markdown() + "\n")
        return

    # ── solve mode ──────────────────────────────────────────────────
    if not args.scenario:
        parser.print_help()
        sys.exit(1)

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"  ✗ Scenario file not found: {scenario_path}")
        sys.exit(1)

    scenario = Scenario.load(scenario_path)
    base_dir = scenario_path.resolve().parent.parent  # repo root

    # ── CLI overrides for emissions ─────────────────────────────────
    if args.emissions_mode is not None:
        scenario.emissions.enabled = True
        scenario.emissions.mode = args.emissions_mode
    if args.co2_weight is not None:
        scenario.emissions.enabled = True
        scenario.emissions.co2_cost_weight = args.co2_weight

    results = solve(
        scenario,
        device=args.device,
        base_dir=base_dir,
    )

    # ── save ────────────────────────────────────────────────────────
    out_dir = Path(args.output) if args.output else Path("outputs") / scenario.name
    results.save(out_dir)

    # ── convergence plot ────────────────────────────────────────────
    results.plot_convergence(save_path=str(out_dir / "convergence.png"))

    # ── map ─────────────────────────────────────────────────────────
    if not args.no_map:
        interactive = not args.static_map
        ext = "html" if interactive else "png"
        results.plot_map(
            interactive=interactive,
            save_path=str(out_dir / f"map.{ext}"),
        )

    print(f"\n  ✓ Run complete.  Results in {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()

