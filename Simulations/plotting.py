"""
plotting.py

Visualization functions for PVC simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Handle both direct execution and module import
try:
    from .config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"


def plot_wp_diff_heatmap(
    player_name: str,
    count: str,
    outs: int,
    base_state: int,
    inning: int,
    inning_topbot: str,
    score_diff: int,
    pitcher_xfip_percentile: int = 50,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot heatmap of win probability difference from baseline approach.

    Parameters
    ----------
    player_name : str
        Player name (must match simulation output file)
    count : str
        Count string (e.g., "0-0", "3-2")
    outs : int
        Number of outs (0, 1, 2)
    base_state : int
        Base state encoding (0-7)
    inning : int
        Inning number (7, 8, 9)
    inning_topbot : str
        "Top" or "Bot"
    score_diff : int
        Score differential from batter's perspective
    pitcher_xfip_percentile : int
        Pitcher xFIP percentile (10, 30, 50, 70, 90)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Load player simulation data
    safe_name = player_name.replace(" ", "_").replace(".", "")
    file_path = OUTPUT_DIR / f"{safe_name}_simulation.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # Filter to specific game state
    mask = (
        (df["count"] == count) &
        (df["outs"] == outs) &
        (df["base_state"] == base_state) &
        (df["inning"] == inning) &
        (df["inning_topbot"] == inning_topbot) &
        (df["score_diff"] == score_diff) &
        (df["pitcher_xfip_percentile"] == pitcher_xfip_percentile)
    )

    state_df = df[mask].copy()

    if len(state_df) == 0:
        raise ValueError(
            f"No data found for state: count={count}, outs={outs}, "
            f"base_state={base_state}, inning={inning}, topbot={inning_topbot}, "
            f"score_diff={score_diff}, xfip_pct={pitcher_xfip_percentile}"
        )

    # Get unique swing lengths and bat speeds for grid
    swing_lengths = sorted(state_df["new_swing_length"].unique())
    bat_speeds = sorted(state_df["new_bat_speed"].unique())

    # Create grid for heatmap
    grid = np.zeros((len(bat_speeds), len(swing_lengths)))

    for i, bs in enumerate(bat_speeds):
        for j, sl in enumerate(swing_lengths):
            val = state_df[
                (state_df["new_bat_speed"] == bs) &
                (state_df["new_swing_length"] == sl)
            ]["wp_diff_from_baseline"].values

            if len(val) > 0:
                grid[i, j] = val[0]

    # Get baseline values for title
    baseline_row = state_df[
        (state_df["bat_speed_change_pct"] == 0) &
        (state_df["swing_length_change_pct"] == 0)
    ].iloc[0]

    baseline_bs = baseline_row["baseline_bat_speed"]
    baseline_sl = baseline_row["baseline_swing_length"]
    wp_before = baseline_row["wp_before_pa"]
    baseline_wp_after = baseline_row["baseline_wp_after_pa"]

    # Base state name mapping
    base_state_names = ["empty", "1B", "2B", "1B_2B", "3B", "1B_3B", "2B_3B", "loaded"]
    base_name = base_state_names[base_state]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to percentage
    grid_pct = grid * 100

    # Calculate cell sizes for proper extent padding
    sl_step = (max(swing_lengths) - min(swing_lengths)) / (len(swing_lengths) - 1) if len(swing_lengths) > 1 else 0.1
    bs_step = (max(bat_speeds) - min(bat_speeds)) / (len(bat_speeds) - 1) if len(bat_speeds) > 1 else 1.0

    # Plot heatmap with fixed range -1% to +1%
    # Note: imshow expects (rows, cols) where rows are y-axis (bat speed) and cols are x-axis (swing length)
    # We want higher bat speed at top, so we flip the grid vertically
    im = ax.imshow(
        grid_pct[::-1],  # Flip so higher bat speed is at top
        extent=[
            min(swing_lengths) - sl_step / 2,
            max(swing_lengths) + sl_step / 2,
            min(bat_speeds) - bs_step / 2,
            max(bat_speeds) + bs_step / 2,
        ],
        aspect="auto",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
    )

    # Add text labels for cells outside the -1% to +1% range
    for i, bs in enumerate(bat_speeds):
        for j, sl in enumerate(swing_lengths):
            val_pct = grid_pct[i, j]
            if abs(val_pct) > 1:
                # Add text label showing actual value
                text_color = "white" if abs(val_pct) > 0.5 else "black"
                ax.text(
                    sl, bs, f"{val_pct:+.1f}%",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color=text_color,
                    clip_on=False
                )

    # Add padding to axis limits so edge labels aren't cut off
    ax.set_xlim(min(swing_lengths) - sl_step * 0.7, max(swing_lengths) + sl_step * 0.7)
    ax.set_ylim(min(bat_speeds) - bs_step * 0.7, max(bat_speeds) + bs_step * 0.7)

    # Mark baseline position
    ax.plot(baseline_sl, baseline_bs, "ko", markersize=12, label="Baseline")
    ax.axhline(baseline_bs, color="black", linestyle="--", alpha=0.3)
    ax.axvline(baseline_sl, color="black", linestyle="--", alpha=0.3)

    # Labels
    ax.set_xlabel("Adjusted Swing Length (ft)", fontsize=12)
    ax.set_ylabel("Adjusted Bat Speed (mph)", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("WP Change from Baseline (%)", fontsize=11)

    # Title with game state info
    # Note: lower xFIP = better pitcher, so 10th pct is elite, 90th pct is poor
    xfip_quality = {10: "elite", 30: "good", 50: "avg", 70: "below avg", 90: "poor"}
    quality_label = xfip_quality.get(pitcher_xfip_percentile, "")
    if score_diff < 0:
        score_label = f"down by {abs(score_diff)}"
    elif score_diff > 0:
        score_label = f"up by {score_diff}"
    else:
        score_label = "tie game"
    title = (
        f"{player_name}\n"
        f"State: {count} count, {outs} out, {base_name}, "
        f"{'Top' if inning_topbot == 'Top' else 'Bot'} {inning}, "
        f"{score_label}\n"
        f"WP before PA: {wp_before:.3f} | Baseline WP after: {baseline_wp_after:.3f} | "
        f"Pitcher: {pitcher_xfip_percentile}th pct xFIP ({quality_label})"
    )
    ax.set_title(title, fontsize=11)

    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_wp_diff_by_xfip(
    player_name: str,
    count: str,
    outs: int,
    base_state: int,
    inning: int,
    inning_topbot: str,
    score_diff: int,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Plot heatmaps for all xFIP percentiles side by side.
    """
    fig, axes = plt.subplots(1, 5, figsize=figsize)

    xfip_percentiles = [10, 30, 50, 70, 90]

    # Load player simulation data
    safe_name = player_name.replace(" ", "_").replace(".", "")
    file_path = OUTPUT_DIR / f"{safe_name}_simulation.parquet"
    df = pd.read_parquet(file_path)

    # Filter to specific game state (all xFIP levels)
    mask = (
        (df["count"] == count) &
        (df["outs"] == outs) &
        (df["base_state"] == base_state) &
        (df["inning"] == inning) &
        (df["inning_topbot"] == inning_topbot) &
        (df["score_diff"] == score_diff)
    )
    state_df = df[mask].copy()

    # Get grid dimensions
    swing_lengths = sorted(state_df["new_swing_length"].unique())
    bat_speeds = sorted(state_df["new_bat_speed"].unique())

    # Calculate cell sizes for proper extent padding
    sl_step = (max(swing_lengths) - min(swing_lengths)) / (len(swing_lengths) - 1) if len(swing_lengths) > 1 else 0.1
    bs_step = (max(bat_speeds) - min(bat_speeds)) / (len(bat_speeds) - 1) if len(bat_speeds) > 1 else 1.0

    for idx, (ax, xfip_pct) in enumerate(zip(axes, xfip_percentiles)):
        xfip_df = state_df[state_df["pitcher_xfip_percentile"] == xfip_pct]

        # Create grid
        grid = np.zeros((len(bat_speeds), len(swing_lengths)))
        for i, bs in enumerate(bat_speeds):
            for j, sl in enumerate(swing_lengths):
                val = xfip_df[
                    (xfip_df["new_bat_speed"] == bs) &
                    (xfip_df["new_swing_length"] == sl)
                ]["wp_diff_from_baseline"].values
                if len(val) > 0:
                    grid[i, j] = val[0]

        # Convert to percentage
        grid_pct = grid * 100

        im = ax.imshow(
            grid_pct[::-1],
            extent=[
                min(swing_lengths) - sl_step / 2,
                max(swing_lengths) + sl_step / 2,
                min(bat_speeds) - bs_step / 2,
                max(bat_speeds) + bs_step / 2,
            ],
            aspect="auto",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
        )

        # Add text labels for cells outside the -1% to +1% range
        for i, bs in enumerate(bat_speeds):
            for j, sl in enumerate(swing_lengths):
                val_pct = grid_pct[i, j]
                if abs(val_pct) > 1:
                    text_color = "white" if abs(val_pct) > 0.5 else "black"
                    ax.text(
                        sl, bs, f"{val_pct:+.1f}%",
                        ha="center", va="center",
                        fontsize=6, fontweight="bold",
                        color=text_color,
                        clip_on=False
                    )

        # Add padding to axis limits so edge labels aren't cut off
        ax.set_xlim(min(swing_lengths) - sl_step * 0.7, max(swing_lengths) + sl_step * 0.7)
        ax.set_ylim(min(bat_speeds) - bs_step * 0.7, max(bat_speeds) + bs_step * 0.7)

        # Mark baseline
        baseline_row = xfip_df[
            (xfip_df["bat_speed_change_pct"] == 0) &
            (xfip_df["swing_length_change_pct"] == 0)
        ].iloc[0]
        ax.plot(baseline_row["baseline_swing_length"], baseline_row["baseline_bat_speed"],
                "ko", markersize=8)

        # Note: lower xFIP = better pitcher
        xfip_quality = {10: "elite", 30: "good", 50: "avg", 70: "below avg", 90: "poor"}
        ax.set_title(f"{xfip_quality[xfip_pct]} pitcher\nxFIP: {baseline_row['pitcher_xfip_value']:.2f}")
        ax.set_xlabel("Adjusted Length (ft)")
        if idx == 0:
            ax.set_ylabel("Bat Speed (mph)")

    # Add colorbar
    fig.colorbar(im, ax=axes, label="WP Change from Baseline (%)", shrink=0.8)

    # Suptitle
    base_state_names = ["empty", "1B", "2B", "1B_2B", "3B", "1B_3B", "2B_3B", "loaded"]
    fig.suptitle(
        f"{player_name}: {count}, {outs} out, {base_state_names[base_state]}, "
        f"{'Top' if inning_topbot == 'Top' else 'Bot'} {inning}, "
        f"{'tied' if score_diff == 0 else f'{score_diff:+d}'}",
        fontsize=13
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Example usage
    fig = plot_wp_diff_heatmap(
        player_name="Kyle Schwarber",
        count="2-0",
        outs=2,
        base_state=6,
        inning=9,
        inning_topbot="Bot",
        score_diff=-1,
        pitcher_xfip_percentile=30,
    )
    plt.show()
