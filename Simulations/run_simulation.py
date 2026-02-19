#!/usr/bin/env python3
"""
run_simulation.py

CLI entry point for PVC player win probability simulations.

Usage:
    python Simulations/run_simulation.py --help
    python Simulations/run_simulation.py --player "Aaron Judge"
    python Simulations/run_simulation.py --all --reduced
    python Simulations/run_simulation.py --all --workers 4
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Simulations.simulation_engine import (
    run_simulation,
    get_pvc_players,
    get_player_stats,
    generate_state_space,
    load_bayesian_model,
)
from Simulations.win_probability import build_wp_lookup
from Simulations.runner_advancement import build_transition_tables


def main():
    parser = argparse.ArgumentParser(
        description="Run PVC player win probability simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation for a single player
  python Simulations/run_simulation.py --player "Aaron Judge"

  # Run simulation for all PVC players with reduced state space (for testing)
  python Simulations/run_simulation.py --all --reduced

  # Run full simulation for all players
  python Simulations/run_simulation.py --all

  # Build lookup tables only (no simulation)
  python Simulations/run_simulation.py --build-tables

  # List PVC players
  python Simulations/run_simulation.py --list-players
        """
    )

    parser.add_argument(
        "--player",
        type=str,
        help="Player name to simulate (case-insensitive partial match)",
    )

    parser.add_argument(
        "--player-id",
        type=int,
        help="Player MLB AM ID to simulate",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Simulate all PVC players",
    )

    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Use reduced state space for faster testing",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    parser.add_argument(
        "--build-tables",
        action="store_true",
        help="Build lookup tables (WP and transitions) without running simulation",
    )

    parser.add_argument(
        "--list-players",
        action="store_true",
        help="List all PVC-flagged players",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory (default: Simulations/output/)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress (default: True)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # List players
    if args.list_players:
        pvc_players = get_pvc_players()
        print(f"\nPVC-flagged players ({len(pvc_players)}):")
        for _, row in pvc_players.iterrows():
            try:
                stats = get_player_stats(int(row["batter_id"]))
                print(f"  {row['player_name']:25s} (ID: {row['batter_id']}) - BS: {stats['bat_speed']:.1f}, SL: {stats['swing_length']:.2f}")
            except ValueError:
                print(f"  {row['player_name']:25s} (ID: {row['batter_id']}) - No stats available")
        return

    # Build tables only
    if args.build_tables:
        print("Building win probability table...")
        wp_path = Path(__file__).parent / "wp_lookup.npy"
        wp_table = build_wp_lookup(save_path=wp_path)
        print(f"  Saved to {wp_path}")

        print("\nBuilding transition tables...")
        trans_path = Path(__file__).parent / "transition_tables.pkl"
        trans_tables = build_transition_tables(save_path=trans_path)
        print(f"  Saved to {trans_path}")
        print(f"  {len(trans_tables)} transition entries")
        return

    # Determine which players to simulate
    players = None
    if args.player:
        pvc_players = get_pvc_players()
        matches = pvc_players[
            pvc_players["player_name"].str.lower().str.contains(args.player.lower())
        ]
        if len(matches) == 0:
            print(f"No PVC player found matching '{args.player}'")
            return
        elif len(matches) > 1:
            print(f"Multiple players match '{args.player}':")
            for _, row in matches.iterrows():
                print(f"  {row['player_name']} (ID: {row['batter_id']})")
            print("\nPlease be more specific or use --player-id")
            return
        players = [int(matches.iloc[0]["batter_id"])]
        if verbose:
            print(f"Simulating: {matches.iloc[0]['player_name']}")

    elif args.player_id:
        players = [args.player_id]

    elif args.all:
        players = None  # Will use all PVC players

    else:
        parser.print_help()
        return

    # Run simulation
    results = run_simulation(
        players=players,
        reduced_states=args.reduced,
        n_workers=args.workers,
        verbose=verbose,
    )

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Simulation complete!")
        print(f"  Players simulated: {len(results)}")
        if results:
            total_rows = sum(len(df) for df in results.values())
            print(f"  Total rows: {total_rows:,}")
            print(f"\nOutput files:")
            for player_name in results:
                safe_name = player_name.replace(" ", "_").replace(".", "")
                print(f"  Simulations/output/{safe_name}_simulation.parquet")


if __name__ == "__main__":
    main()

