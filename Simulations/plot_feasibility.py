"""
plot_feasibility.py

Generate feasibility mask visualizations for each PVC player.
Shows which bat speed / swing length combinations are considered achievable
based on the player's within-swing covariance structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .config import (
    DATA_DIR,
    BAT_SPEED_CHANGES,
    SWING_LENGTH_CHANGES,
    FEASIBILITY_THRESHOLD,
)
from .simulation_engine import (
    get_pvc_players,
    get_player_stats,
    get_player_covariance,
    get_feasible_grid_mask,
)

OUTPUT_DIR = Path(__file__).parent / "output" / "feasibility_plots"


def plot_player_feasibility(
    player_id: int,
    player_name: str,
    save_path: Path = None,
):
    """
    Create a heatmap showing feasibility densities for a player's approach grid.

    Parameters
    ----------
    player_id : int
        MLB AM ID
    player_name : str
        Player name for title
    save_path : Path, optional
        If provided, save figure to this path
    """
    # Get player stats
    try:
        stats = get_player_stats(player_id)
    except ValueError as e:
        print(f"  Skipping {player_name}: {e}")
        return None

    baseline_bs = stats["bat_speed"]
    baseline_sl = stats["swing_length"]

    # Get covariance and compute mask
    player_cov = get_player_covariance(player_id)
    bs_changes_abs = [baseline_bs * pct for pct in BAT_SPEED_CHANGES]
    sl_changes_abs = [baseline_sl * pct for pct in SWING_LENGTH_CHANGES]

    mask, densities = get_feasible_grid_mask(bs_changes_abs, sl_changes_abs, player_cov)

    # Normalize densities
    normalized = densities / densities.max()

    # Extract stats
    bs_std = np.sqrt(player_cov[0, 0])
    sl_std = np.sqrt(player_cov[1, 1])
    corr = player_cov[0, 1] / (bs_std * sl_std)
    n_feasible = mask.sum()

    # Create figure (matching plot_wp_diff_heatmap style)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute actual swing length and bat speed values for each grid point
    swing_lengths = [baseline_sl + sl_change for sl_change in sl_changes_abs]
    bat_speeds = [baseline_bs + bs_change for bs_change in bs_changes_abs]

    # Calculate cell sizes for extent
    sl_step = abs(swing_lengths[1] - swing_lengths[0]) if len(swing_lengths) > 1 else 0.1
    bs_step = abs(bat_speeds[1] - bat_speeds[0]) if len(bat_speeds) > 1 else 1.0

    # For imshow with default origin='upper': row 0 is at top
    # normalized[0, 0] = baseline (highest BS, highest SL)
    # We want baseline at top-right, so flip columns only
    grid_for_display = normalized[:, ::-1]

    # Custom colormap: full RdYlGn gradient from threshold to 1, solid red below
    from matplotlib.colors import LinearSegmentedColormap
    t = FEASIBILITY_THRESHOLD
    colors = [
        (0.0, "#d73027"),
        (t, "#d73027"),
        (t + 0.001, "#d73027"),
        (t + (1 - t) * 0.25, "#f46d43"),
        (t + (1 - t) * 0.5, "#ffffbf"),
        (t + (1 - t) * 0.75, "#91cf60"),
        (1.0, "#1a9850"),
    ]
    cmap = LinearSegmentedColormap.from_list("feasibility_cmap", [(pos, col) for pos, col in colors])

    # Plot heatmap using imshow (like plot_wp_diff_heatmap)
    im = ax.imshow(
        grid_for_display,
        extent=[
            min(swing_lengths) - sl_step / 2,
            max(swing_lengths) + sl_step / 2,
            min(bat_speeds) - bs_step / 2,
            max(bat_speeds) + bs_step / 2,
        ],
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    # Add text labels - only for feasible cells (density >= threshold)
    for i, bs in enumerate(bat_speeds):
        for j, sl in enumerate(swing_lengths):
            if mask[i, j]:
                # Feasible cell - show density value
                ax.text(
                    sl, bs, f"{normalized[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="black" if normalized[i, j] > 0.5 else "white",
                )
            else:
                # Infeasible cell - show X only (no number)
                ax.text(
                    sl, bs, "X",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="black", alpha=0.8,
                )

    # Add padding to axis limits
    ax.set_xlim(min(swing_lengths) - sl_step * 0.7, max(swing_lengths) + sl_step * 0.7)
    ax.set_ylim(min(bat_speeds) - bs_step * 0.7, max(bat_speeds) + bs_step * 0.7)

    # Labels
    ax.set_xlabel("Swing Length (ft)", fontsize=12)
    ax.set_ylabel("Bat Speed (mph)", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Density", fontsize=11)
    cbar.ax.axhline(y=FEASIBILITY_THRESHOLD, color='black', linewidth=2)

    # Add feasibility threshold explanation in top left
    pct_enclosed = (1 - FEASIBILITY_THRESHOLD) * 100
    ax.text(0.02, 0.98, f"Feasible: density ≥ {FEASIBILITY_THRESHOLD}\n({pct_enclosed:.0f}% of swings enclosed)",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Title
    ax.set_title(
        f"{player_name}\n"
        f"BS: {baseline_bs:.1f} mph (σ={bs_std:.1f})  |  "
        f"SL: {baseline_sl:.2f} ft (σ={sl_std:.2f})  |  "
        f"ρ={corr:.2f}  |  "
        f"Feasible: {n_feasible}/25",
        fontsize=11
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {
        "player_name": player_name,
        "n_feasible": n_feasible,
        "correlation": corr,
        "bs_std": bs_std,
        "sl_std": sl_std,
    }


def plot_player_swing_distribution(
    player_id: int,
    player_name: str,
    season: int = 2025,
    save_path: Path = None,
):
    """
    Plot the bivariate normal density of bat speed vs swing length for a player.
    Uses the same density calculation as the feasibility mask in the simulation.

    Parameters
    ----------
    player_id : int
        MLB AM ID
    player_name : str
        Player name for title
    season : int
        Season to get swing data for
    save_path : Path, optional
        If provided, save figure to this path
    """
    # Load swing data
    pbp = pd.read_parquet(DATA_DIR / f"pbp_{season}.parquet")
    pbp = pbp[pbp["bat_speed"].notna() & pbp["swing_length"].notna()]
    player_swings = pbp[pbp["batter"] == player_id]

    if len(player_swings) < 10:
        print(f"  Skipping {player_name}: insufficient swings ({len(player_swings)})")
        return None

    # Extract swing data
    bs = player_swings["bat_speed"].to_numpy(dtype=np.float64)
    sl = player_swings["swing_length"].to_numpy(dtype=np.float64)

    # Compute stats using only swings within 95% CI (filter out bunts/check swings)
    bs_lo, bs_hi = np.percentile(bs, [2.5, 97.5])
    sl_lo, sl_hi = np.percentile(sl, [2.5, 97.5])
    ci_mask = (bs >= bs_lo) & (bs <= bs_hi) & (sl >= sl_lo) & (sl <= sl_hi)
    bs_ci, sl_ci = bs[ci_mask], sl[ci_mask]

    bs_median, sl_median = np.median(bs), np.median(sl)
    bs_std, sl_std = bs_ci.std(), sl_ci.std()  # std from filtered data
    corr = np.corrcoef(bs_ci, sl_ci)[0, 1]
    n_swings = len(bs)

    # Compute covariance matrix from filtered data
    cov_matrix = np.cov(bs_ci, sl_ci)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot bivariate normal density (same calculation as feasibility mask)
    try:
        # Create grid centered on median, extending 3 std (from filtered data)
        xmin, xmax = sl_median - 3 * sl_std, sl_median + 3 * sl_std
        ymin, ymax = bs_median - 3 * bs_std, bs_median + 3 * bs_std

        # Grid in absolute coordinates
        sl_grid = np.linspace(xmin, xmax, 100)
        bs_grid = np.linspace(ymin, ymax, 100)
        xx, yy = np.meshgrid(sl_grid, bs_grid)

        # Compute bivariate normal density based on CHANGE from median
        # This matches the feasibility calculation which is centered at (0,0) for changes
        cov_inv = np.linalg.inv(cov_matrix)
        cov_det = np.linalg.det(cov_matrix)
        norm_const = 1.0 / (2 * np.pi * np.sqrt(cov_det))

        relative_density = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # Change from median (same as feasibility calculation)
                delta = np.array([yy[i, j] - bs_median, xx[i, j] - sl_median])
                mahal_sq = delta @ cov_inv @ delta
                relative_density[i, j] = np.exp(-0.5 * mahal_sq)

        # Already normalized: max is 1.0 at the median (where delta = 0)

        # Custom colormap: full RdYlGn gradient from threshold to 1, solid red below
        from matplotlib.colors import LinearSegmentedColormap
        t = FEASIBILITY_THRESHOLD
        colors = [
            (0.0, "#d73027"),   # dark red at 0
            (t, "#d73027"),     # dark red up to threshold
            (t + 0.001, "#d73027"),  # dark red at threshold (start of gradient)
            (t + (1 - t) * 0.25, "#f46d43"),  # lighter red
            (t + (1 - t) * 0.5, "#ffffbf"),   # yellow at midpoint
            (t + (1 - t) * 0.75, "#91cf60"),  # light green
            (1.0, "#1a9850"),   # dark green at 1
        ]
        cmap = LinearSegmentedColormap.from_list(
            "feasibility_cmap",
            [(pos, col) for pos, col in colors]
        )

        # Plot filled contours with relative density
        levels = np.linspace(0, 1, 21)
        cf = ax.contourf(xx, yy, relative_density, levels=levels, cmap=cmap, alpha=0.8)
        cbar = plt.colorbar(cf, ax=ax, label="Relative Density (Bivariate Normal)")
        cbar.ax.axhline(y=FEASIBILITY_THRESHOLD, color='black', linewidth=2)  # Mark threshold

        # Plot contour lines
        ax.contour(xx, yy, relative_density, levels=levels[::3], colors="white", alpha=0.3, linewidths=0.5)

    except Exception as e:
        print(f"  Bivariate normal failed for {player_name}: {e}, falling back to scatter")
        ax.scatter(sl, bs, alpha=0.3, s=10, c="steelblue")

    # Plot individual swings as small points
    ax.scatter(sl, bs, alpha=0.15, s=8, c="white", edgecolors="none")

    # Plot median point
    ax.plot(sl_median, bs_median, "r*", markersize=15, markeredgecolor="white",
            markeredgewidth=1.5, label=f"Median ({sl_median:.2f}, {bs_median:.1f})", zorder=10)

    # Overlay the 5x5 approach grid (cell boundaries, not centers)
    # Cell centers are at 0%, -3%, -6%, -9%, -12%
    # Cell boundaries need half-steps: +1.5%, -1.5%, -4.5%, -7.5%, -10.5%, -13.5%
    cell_center_pcts = [0, -0.03, -0.06, -0.09, -0.12]
    step = 0.03
    boundary_pcts = [step/2] + [p - step/2 for p in cell_center_pcts]  # 6 boundaries for 5 cells

    grid_sl_boundaries = [sl_median * (1 + pct) for pct in boundary_pcts]
    grid_bs_boundaries = [bs_median * (1 + pct) for pct in boundary_pcts]
    grid_sl_centers = [sl_median * (1 + pct) for pct in cell_center_pcts]
    grid_bs_centers = [bs_median * (1 + pct) for pct in cell_center_pcts]

    sl_min, sl_max = min(grid_sl_boundaries), max(grid_sl_boundaries)
    bs_min, bs_max = min(grid_bs_boundaries), max(grid_bs_boundaries)

    # Draw vertical lines (swing length boundaries)
    for sl_val in grid_sl_boundaries:
        ax.plot([sl_val, sl_val], [bs_min, bs_max], color="black", linestyle="-", alpha=0.5, linewidth=1, zorder=5)

    # Draw horizontal lines (bat speed boundaries)
    for bs_val in grid_bs_boundaries:
        ax.plot([sl_min, sl_max], [bs_val, bs_val], color="black", linestyle="-", alpha=0.5, linewidth=1, zorder=5)

    # Add percentage labels at cell centers
    pct_labels = ["0%", "-3%", "-6%", "-9%", "-12%"]

    # Labels on bottom edge (swing length percentages) - at cell centers
    for i, sl_val in enumerate(grid_sl_centers):
        ax.text(sl_val, bs_min - (bs_max - bs_min) * 0.03, pct_labels[i],
                fontsize=7, ha="center", va="top", color="black")

    # Labels on left edge (bat speed percentages) - at cell centers
    for i, bs_val in enumerate(grid_bs_centers):
        ax.text(sl_min - (sl_max - sl_min) * 0.03, bs_val, pct_labels[i],
                fontsize=7, ha="right", va="center", color="black")

    # Labels
    ax.set_xlabel("Swing Length (ft)", fontsize=12)
    ax.set_ylabel("Bat Speed (mph)", fontsize=12)
    ax.set_title(
        f"{player_name} - Bivariate Normal Density ({season})\n"
        f"n={n_swings} swings  |  "
        f"BS: {bs_median:.1f} mph (σ={bs_std:.1f})  |  "
        f"SL: {sl_median:.2f} ft (σ={sl_std:.2f})  |  "
        f"R²={corr**2:.2f}",
        fontsize=12
    )

    ax.legend(loc="upper right")

    # Add feasibility threshold explanation in top left
    pct_enclosed = (1 - FEASIBILITY_THRESHOLD) * 100
    ax.text(0.02, 0.98, f"Feasible: density ≥ {FEASIBILITY_THRESHOLD}\n({pct_enclosed:.0f}% of swings enclosed)",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(sl_median - 3 * sl_std, sl_median + 3 * sl_std)
    ax.set_ylim(bs_median - 3 * bs_std, bs_median + 3 * bs_std)
    ax.set_aspect("auto")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {
        "player_name": player_name,
        "n_swings": n_swings,
        "bs_median": bs_median,
        "sl_median": sl_median,
        "bs_std": bs_std,
        "sl_std": sl_std,
        "correlation": corr,
    }


def generate_all_swing_distribution_plots():
    """Generate swing distribution plots for all PVC players."""
    output_dir = Path(__file__).parent / "output" / "swing_distributions"
    output_dir.mkdir(parents=True, exist_ok=True)

    pvc_players = get_pvc_players()
    print(f"Generating swing distribution plots for {len(pvc_players)} players...")

    results = []
    for idx, row in pvc_players.iterrows():
        player_id = int(row["batter_id"])
        player_name = row["player_name"]

        safe_name = player_name.replace(" ", "_").replace(".", "")
        save_path = output_dir / f"{safe_name}_swing_dist.png"

        print(f"  [{idx+1}/{len(pvc_players)}] {player_name}")
        result = plot_player_swing_distribution(player_id, player_name, save_path=save_path)

        if result:
            results.append(result)

    print(f"\nSaved {len(results)} plots to {output_dir}")
    return results


def generate_all_plots():
    """Generate feasibility plots for all PVC players."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pvc_players = get_pvc_players()
    print(f"Generating feasibility plots for {len(pvc_players)} players...")

    results = []
    for idx, row in pvc_players.iterrows():
        player_id = int(row["batter_id"])
        player_name = row["player_name"]

        safe_name = player_name.replace(" ", "_").replace(".", "")
        save_path = OUTPUT_DIR / f"{safe_name}_feasibility.png"

        print(f"  [{idx+1}/{len(pvc_players)}] {player_name}")
        result = plot_player_feasibility(player_id, player_name, save_path)

        if result:
            results.append(result)

    # Summary
    print(f"\nSaved {len(results)} plots to {OUTPUT_DIR}")
    avg_feasible = np.mean([r["n_feasible"] for r in results])
    print(f"Average feasible cells: {avg_feasible:.1f}/25")

    return results


if __name__ == "__main__":
    generate_all_plots()
