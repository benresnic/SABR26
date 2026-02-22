#!/usr/bin/env python3
"""
Generate stat change grid plots for all PVC players.

Usage:
    python Simulations/generate_stat_plots.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Bayes Outcomes"))

from Simulations.simulation_engine import get_pvc_players
from update_stats import plot_stat_change_grid

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt


def main():
    output_dir = Path(__file__).parent / "output" / "stat_update_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    pvc_players = get_pvc_players()
    print(f"Generating plots for {len(pvc_players)} PVC players...")

    success_count = 0
    for idx, row in pvc_players.iterrows():
        player_name = row["player_name"]
        safe_name = player_name.replace(" ", "_").replace(".", "")
        save_path = output_dir / f"{safe_name}_stat_grid.png"

        try:
            fig = plot_stat_change_grid(player=player_name, save_path=str(save_path))
            plt.close(fig)
            success_count += 1
            print(f"  [{success_count}/{len(pvc_players)}] {player_name}")
        except Exception as e:
            print(f"  [SKIP] {player_name}: {e}")

    print(f"\nDone! Generated {success_count} plots in {output_dir}")


if __name__ == "__main__":
    main()
