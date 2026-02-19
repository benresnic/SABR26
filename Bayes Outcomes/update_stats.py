"""
update_stats.py

Predicts how changes in bat speed and swing length affect zone contact rate and xISO.

Example:
    python "Bayes Outcomes/update_stats.py" \
        --bat-speed 77 --swing-length 7.5 \
        --current-stat 0.79 --stat-type z_contact \
        --bs-change -4 --sl-change -0.3
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COEF_FILE = Path(__file__).parent / "update_stats_coefs.json"
DATA_DIR = Path(__file__).parent.parent / "Data"

# Typical ranges for warnings
BAT_SPEED_RANGE = (55.0, 90.0)
SWING_LENGTH_RANGE = (5.0, 10.0)

# Stat bounds for clipping
STAT_BOUNDS = {
    "z_contact": (0.5, 1.0),
    "xiso": (0.0, 0.6),
}


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def _build_training_data() -> pd.DataFrame:
    """
    Load PBP data, filter to swings, aggregate bat speed and swing length
    to batter-season means, and join with batter stats.

    Returns DataFrame with: mean_bat_speed, mean_swing_length, Z-Contact_pct, xISO
    """
    # Load PBP files for 2023-2025
    pbp_files = [DATA_DIR / f"pbp_{year}.parquet" for year in [2023, 2024, 2025]]
    pbp_list = []
    for f in pbp_files:
        if f.exists():
            pbp_list.append(pd.read_parquet(f))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)

    # Filter to rows where bat_speed is not null (swings only)
    pbp = pbp[pbp["bat_speed"].notna()].copy()

    # Aggregate to batter-season means
    swing_stats = (
        pbp.groupby(["batter", "game_year"])
        .agg({
            "bat_speed": "mean",
            "swing_length": "mean",
        })
        .reset_index()
        .rename(columns={
            "bat_speed": "mean_bat_speed",
            "swing_length": "mean_swing_length",
        })
    )

    # Load batter stats
    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")

    # Join on (batter, game_year) = (xMLBAMID, Season)
    swing_stats["batter"] = swing_stats["batter"].astype("int32")
    swing_stats["game_year"] = swing_stats["game_year"].astype("int32")
    batter_stats["xMLBAMID"] = batter_stats["xMLBAMID"].astype("int32")
    batter_stats["Season"] = batter_stats["Season"].astype("int32")

    merged = swing_stats.merge(
        batter_stats[["xMLBAMID", "Season", "Z-Contact_pct", "xISO"]],
        left_on=["batter", "game_year"],
        right_on=["xMLBAMID", "Season"],
        how="inner",
    )

    # Drop rows with any nulls in the columns we need
    cols_needed = ["mean_bat_speed", "mean_swing_length", "Z-Contact_pct", "xISO"]
    merged = merged.dropna(subset=cols_needed)

    return merged[cols_needed].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_models() -> dict:
    """
    Fit two Ridge regressions with polynomial features:
    - Z-Contact_pct ~ bs + sl + bs² + sl² + bs×sl
    - xISO ~ bs + sl + bs² + sl² + bs×sl

    Uses standardized inputs and Ridge regularization (alpha=0.01) to handle
    collinearity between bat speed and swing length.

    Returns coefficients dictionary.
    """
    df = _build_training_data()

    # Standardize inputs (required for Ridge)
    scaler = StandardScaler()
    X_raw = df[["mean_bat_speed", "mean_swing_length"]].values
    X_scaled = scaler.fit_transform(X_raw)

    # Create polynomial features: [bs, sl, bs², sl², bs×sl]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    results = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "feature_names": poly.get_feature_names_out().tolist(),
        "training_stats": {
            "n_observations": len(df),
            "bat_speed_mean": float(df["mean_bat_speed"].mean()),
            "bat_speed_std": float(df["mean_bat_speed"].std()),
            "swing_length_mean": float(df["mean_swing_length"].mean()),
            "swing_length_std": float(df["mean_swing_length"].std()),
        },
        "fit_date": datetime.now().isoformat(),
    }

    # Fit Z-Contact model
    y_zc = df["Z-Contact_pct"].values
    model_zc = Ridge(alpha=0.01)
    model_zc.fit(X_poly, y_zc)
    results["z_contact"] = {
        "intercept": float(model_zc.intercept_),
        "coefficients": model_zc.coef_.tolist(),
        "r_squared": float(model_zc.score(X_poly, y_zc)),
    }

    # Fit xISO model
    y_xiso = df["xISO"].values
    model_xiso = Ridge(alpha=0.01)
    model_xiso.fit(X_poly, y_xiso)
    results["xiso"] = {
        "intercept": float(model_xiso.intercept_),
        "coefficients": model_xiso.coef_.tolist(),
        "r_squared": float(model_xiso.score(X_poly, y_xiso)),
    }

    return results


# ---------------------------------------------------------------------------
# Coefficient caching
# ---------------------------------------------------------------------------

def _load_or_fit_coefficients(force_refit: bool = False) -> dict:
    """
    Load coefficients from JSON if exists, otherwise fit models and save.
    """
    if not force_refit and COEF_FILE.exists():
        with open(COEF_FILE, "r") as f:
            return json.load(f)

    coeffs = _fit_models()
    with open(COEF_FILE, "w") as f:
        json.dump(coeffs, f, indent=2)

    return coeffs


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def get_player_stats(player_name: str, season: int = None) -> dict:
    """
    Look up a player's stats by name.

    Parameters
    ----------
    player_name : str
        Player name (case-insensitive, partial match supported)
    season : int
        Season to look up. Defaults to most recent available.

    Returns
    -------
    dict
        {
            'player_name': str,
            'season': int,
            'mlbam_id': int,
            'bat_speed': float,
            'swing_length': float,
            'z_contact': float,
            'xiso': float,
        }

    Raises
    ------
    ValueError
        If player not found or multiple matches
    """
    # Load batter stats
    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")

    # Case-insensitive partial match
    mask = batter_stats["PlayerName"].str.lower().str.contains(player_name.lower())
    matches = batter_stats[mask]

    if len(matches) == 0:
        raise ValueError(f"No player found matching '{player_name}'")

    # Get unique player names that match
    unique_names = matches["PlayerName"].unique()
    if len(unique_names) > 1:
        raise ValueError(
            f"Multiple players match '{player_name}': {', '.join(unique_names)}. "
            "Please be more specific."
        )

    player_full_name = unique_names[0]
    player_data = matches[matches["PlayerName"] == player_full_name]

    # Filter to requested season or get most recent
    if season is not None:
        player_data = player_data[player_data["Season"] == season]
        if len(player_data) == 0:
            available = matches["Season"].unique().tolist()
            raise ValueError(
                f"No data for {player_full_name} in {season}. "
                f"Available seasons: {available}"
            )
    else:
        season = player_data["Season"].max()
        player_data = player_data[player_data["Season"] == season]

    row = player_data.iloc[0]

    # Get swing data from PBP
    pbp_files = [DATA_DIR / f"pbp_{year}.parquet" for year in [2023, 2024, 2025]]
    pbp_list = []
    for f in pbp_files:
        if f.exists():
            pbp_list.append(pd.read_parquet(f))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)
    pbp = pbp[pbp["bat_speed"].notna()].copy()

    # Filter to player and season
    player_swings = pbp[
        (pbp["batter"] == int(row["xMLBAMID"])) &
        (pbp["game_year"] == int(season))
    ]

    if len(player_swings) == 0:
        raise ValueError(f"No swing data found for {player_full_name} in {season}")

    return {
        "player_name": player_full_name,
        "season": int(season),
        "mlbam_id": int(row["xMLBAMID"]),
        "bat_speed": float(player_swings["bat_speed"].mean()),
        "swing_length": float(player_swings["swing_length"].mean()),
        "z_contact": float(row["Z-Contact_pct"]),
        "xiso": float(row["xISO"]),
    }


def plot_player_stat_grid(
    player_name: str,
    season: int = None,
    save_path: str = None,
) -> tuple:
    """
    Look up a player's stats and plot their stat change grid.

    Parameters
    ----------
    player_name : str
        Player name (case-insensitive, partial match supported)
    season : int
        Season to look up. Defaults to most recent available.
    save_path : str
        If provided, save figure to this path

    Returns
    -------
    tuple
        (player_stats dict, matplotlib Figure)
    """
    stats = get_player_stats(player_name, season)

    print(f"\nPlayer: {stats['player_name']} ({stats['season']})")
    print(f"  Bat Speed:    {stats['bat_speed']:.1f} mph")
    print(f"  Swing Length: {stats['swing_length']:.2f} ft")
    print(f"  Z-Contact:    {stats['z_contact']:.1%}")
    print(f"  xISO:         {stats['xiso']:.3f}")

    fig = plot_stat_change_grid(
        current_bat_speed=stats["bat_speed"],
        current_swing_length=stats["swing_length"],
        current_z_contact=stats["z_contact"],
        current_xiso=stats["xiso"],
        save_path=save_path,
    )

    # Update title with player name
    fig.suptitle(
        f"{stats['player_name']} ({stats['season']}): "
        f"{stats['bat_speed']:.1f} mph, {stats['swing_length']:.2f} ft",
        fontsize=12,
        y=1.02,
    )

    return stats, fig


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def update_stat(
    current_bat_speed: float,
    current_swing_length: float,
    current_stat_value: float,
    stat_type: str,
    bat_speed_change: float = 0.0,
    swing_length_change: float = 0.0,
) -> dict:
    """
    Predict how changes in bat speed and swing length affect a stat.

    Parameters
    ----------
    current_bat_speed : float
        Current bat speed in mph (e.g., 77.0)
    current_swing_length : float
        Current swing length in feet (e.g., 7.5)
    current_stat_value : float
        Current value of the stat (e.g., 0.79 for 79% zone contact)
    stat_type : str
        Either "z_contact" or "xiso"
    bat_speed_change : float
        Change in bat speed in mph (e.g., -4.0)
    swing_length_change : float
        Change in swing length in feet (e.g., -0.3)

    Returns
    -------
    dict
        {
            'new_stat_value': float,
            'stat_change': float,
            'warnings': list
        }
    """
    # Validate stat_type
    if stat_type not in ["z_contact", "xiso"]:
        raise ValueError(f"stat_type must be 'z_contact' or 'xiso', got '{stat_type}'")

    warnings = []

    # Check for out-of-range inputs
    new_bat_speed = current_bat_speed + bat_speed_change
    new_swing_length = current_swing_length + swing_length_change

    if not (BAT_SPEED_RANGE[0] <= current_bat_speed <= BAT_SPEED_RANGE[1]):
        warnings.append(
            f"Current bat speed {current_bat_speed} mph is outside typical range "
            f"({BAT_SPEED_RANGE[0]}-{BAT_SPEED_RANGE[1]} mph)"
        )

    if not (BAT_SPEED_RANGE[0] <= new_bat_speed <= BAT_SPEED_RANGE[1]):
        warnings.append(
            f"New bat speed {new_bat_speed} mph is outside typical range "
            f"({BAT_SPEED_RANGE[0]}-{BAT_SPEED_RANGE[1]} mph)"
        )

    if not (SWING_LENGTH_RANGE[0] <= current_swing_length <= SWING_LENGTH_RANGE[1]):
        warnings.append(
            f"Current swing length {current_swing_length} ft is outside typical range "
            f"({SWING_LENGTH_RANGE[0]}-{SWING_LENGTH_RANGE[1]} ft)"
        )

    if not (SWING_LENGTH_RANGE[0] <= new_swing_length <= SWING_LENGTH_RANGE[1]):
        warnings.append(
            f"New swing length {new_swing_length} ft is outside typical range "
            f"({SWING_LENGTH_RANGE[0]}-{SWING_LENGTH_RANGE[1]} ft)"
        )

    # Load coefficients
    coeffs = _load_or_fit_coefficients()

    # Helper functions for non-linear model
    def standardize(bs, sl):
        """Standardize inputs using saved scaler parameters."""
        return (
            (bs - coeffs["scaler_mean"][0]) / coeffs["scaler_std"][0],
            (sl - coeffs["scaler_mean"][1]) / coeffs["scaler_std"][1],
        )

    def make_features(bs, sl):
        """Build polynomial features for a point: [bs, sl, bs², sl², bs×sl]."""
        bs_s, sl_s = standardize(bs, sl)
        return [bs_s, sl_s, bs_s**2, sl_s**2, bs_s * sl_s]

    def predict(bs, sl, stat_coeffs):
        """Predict stat value at given bat speed and swing length."""
        features = make_features(bs, sl)
        return stat_coeffs["intercept"] + sum(
            c * f for c, f in zip(stat_coeffs["coefficients"], features)
        )

    # Predict at current and new positions
    stat_coeffs = coeffs[stat_type]
    pred_current = predict(current_bat_speed, current_swing_length, stat_coeffs)
    pred_new = predict(new_bat_speed, new_swing_length, stat_coeffs)

    # Model-based delta captures non-linear effects
    stat_change = pred_new - pred_current
    new_stat_value = current_stat_value + stat_change

    # Clip to valid bounds
    bounds = STAT_BOUNDS[stat_type]
    if new_stat_value < bounds[0]:
        warnings.append(
            f"New {stat_type} value {new_stat_value:.4f} clipped to minimum {bounds[0]}"
        )
        new_stat_value = bounds[0]
    elif new_stat_value > bounds[1]:
        warnings.append(
            f"New {stat_type} value {new_stat_value:.4f} clipped to maximum {bounds[1]}"
        )
        new_stat_value = bounds[1]

    return {
        "new_stat_value": new_stat_value,
        "stat_change": stat_change,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stat_change_grid(
    current_bat_speed: float,
    current_swing_length: float,
    current_z_contact: float = 0.80,
    current_xiso: float = 0.200,
    bat_speed_range: tuple = None,
    swing_length_range: tuple = None,
    grid_size: int = 50,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot heatmaps showing percent change in stats across bat speed / swing length grid.

    Parameters
    ----------
    current_bat_speed : float
        Current bat speed in mph
    current_swing_length : float
        Current swing length in feet
    current_z_contact : float
        Current zone contact rate (e.g., 0.80 for 80%)
    current_xiso : float
        Current xISO value
    bat_speed_range : tuple
        (min, max) for bat speed axis. Defaults to 8% reduction from current.
    swing_length_range : tuple
        (min, max) for swing length axis. Defaults to 8% reduction from current.
    grid_size : int
        Number of points along each axis
    save_path : str
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
    """
    # Default ranges: only decreases, up to 8% reduction from current
    # Current position will be at top-right (max bat speed, max swing length)
    if bat_speed_range is None:
        min_bs = current_bat_speed * 0.92  # 8% reduction
        bat_speed_range = (min_bs, current_bat_speed)
    if swing_length_range is None:
        min_sl = current_swing_length * 0.92  # 8% reduction
        swing_length_range = (min_sl, current_swing_length)

    # Create grid (ascending order so current is at top-right)
    swing_lengths = np.linspace(swing_length_range[0], swing_length_range[1], grid_size)
    bat_speeds = np.linspace(bat_speed_range[0], bat_speed_range[1], grid_size)

    # Load coefficients
    coeffs = _load_or_fit_coefficients()

    # Helper functions for non-linear model
    def standardize(bs, sl):
        return (
            (bs - coeffs["scaler_mean"][0]) / coeffs["scaler_std"][0],
            (sl - coeffs["scaler_mean"][1]) / coeffs["scaler_std"][1],
        )

    def make_features(bs, sl):
        bs_s, sl_s = standardize(bs, sl)
        return [bs_s, sl_s, bs_s**2, sl_s**2, bs_s * sl_s]

    def predict(bs, sl, stat_coeffs):
        features = make_features(bs, sl)
        return stat_coeffs["intercept"] + sum(
            c * f for c, f in zip(stat_coeffs["coefficients"], features)
        )

    # Calculate percent changes for each stat
    stat_configs = [
        ("z_contact", current_z_contact, "Z-Contact % Change", "RdYlGn"),
        ("xiso", current_xiso, "xISO % Change", "RdYlGn"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (stat_type, current_val, title, cmap) in zip(axes, stat_configs):
        stat_coeffs = coeffs[stat_type]

        # Predict at current position
        pred_current = predict(current_bat_speed, current_swing_length, stat_coeffs)

        # Build grid of percent changes
        pct_change_grid = np.zeros((grid_size, grid_size))

        for i, bs in enumerate(bat_speeds):
            for j, sl in enumerate(swing_lengths):
                # Predict at new position and compute model delta
                pred_new = predict(bs, sl, stat_coeffs)
                stat_change = pred_new - pred_current

                pct_change = (stat_change / current_val) * 100
                pct_change_grid[i, j] = pct_change

        # Plot heatmap
        im = ax.imshow(
            pct_change_grid,
            extent=[swing_length_range[0], swing_length_range[1],
                    bat_speed_range[0], bat_speed_range[1]],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-np.abs(pct_change_grid).max(),
            vmax=np.abs(pct_change_grid).max(),
        )

        # Mark current position
        ax.plot(current_swing_length, current_bat_speed, "ko", markersize=10, label="Current")
        ax.axhline(current_bat_speed, color="black", linestyle="--", alpha=0.3)
        ax.axvline(current_swing_length, color="black", linestyle="--", alpha=0.3)

        # Labels
        ax.set_xlabel("Swing Length (ft)")
        ax.set_ylabel("Bat Speed (mph)")
        ax.set_title(f"{title}\n(Current: {current_val:.3f})")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("% Change")

        ax.legend(loc="upper right")

    plt.suptitle(
        f"Stat Changes from Current: {current_bat_speed:.1f} mph, {current_swing_length:.2f} ft",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict how bat speed and swing length changes affect stats"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model coefficients and training stats",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Force refit of models (ignore cached coefficients)",
    )
    parser.add_argument(
        "--bat-speed",
        type=float,
        help="Current bat speed in mph",
    )
    parser.add_argument(
        "--swing-length",
        type=float,
        help="Current swing length in feet",
    )
    parser.add_argument(
        "--current-stat",
        type=float,
        help="Current stat value",
    )
    parser.add_argument(
        "--stat-type",
        choices=["z_contact", "xiso"],
        help="Type of stat to predict",
    )
    parser.add_argument(
        "--bs-change",
        type=float,
        default=0.0,
        help="Change in bat speed (mph)",
    )
    parser.add_argument(
        "--sl-change",
        type=float,
        default=0.0,
        help="Change in swing length (feet)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate heatmap of stat changes across bat speed / swing length grid",
    )
    parser.add_argument(
        "--current-z-contact",
        type=float,
        default=0.80,
        help="Current zone contact rate for plotting (default: 0.80)",
    )
    parser.add_argument(
        "--current-xiso",
        type=float,
        default=0.200,
        help="Current xISO for plotting (default: 0.200)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        help="Path to save the plot (e.g., 'plots/grid.png')",
    )
    parser.add_argument(
        "--player",
        type=str,
        help="Player name to look up and plot (e.g., 'Aaron Judge')",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Season for player lookup (default: most recent)",
    )

    args = parser.parse_args()

    # Show help if no arguments provided
    if not any([args.info, args.refit, args.bat_speed, args.swing_length,
                args.current_stat, args.stat_type, args.plot, args.player]):
        parser.print_help()
        return

    # Handle player lookup and plot
    if args.player:
        try:
            stats, fig = plot_player_stat_grid(
                player_name=args.player,
                season=args.season,
                save_path=args.save_plot,
            )
            if not args.save_plot:
                plt.show()
        except ValueError as e:
            print(f"Error: {e}")
        return

    if args.info or args.refit:
        coeffs = _load_or_fit_coefficients(force_refit=args.refit)
        print("Model Info (Polynomial Ridge Regression)")
        print(f"  Fit date: {coeffs['fit_date']}")
        print(f"\nTraining data:")
        ts = coeffs["training_stats"]
        print(f"  N observations: {ts['n_observations']}")
        print(f"  Bat speed mean: {ts['bat_speed_mean']:.2f} mph (std: {ts['bat_speed_std']:.2f})")
        print(f"  Swing length mean: {ts['swing_length_mean']:.2f} ft (std: {ts['swing_length_std']:.2f})")
        print(f"\nScaler parameters (for standardization):")
        print(f"  Mean: {coeffs['scaler_mean']}")
        print(f"  Std:  {coeffs['scaler_std']}")
        print(f"\nFeatures: {coeffs['feature_names']}")
        print(f"\nZ-Contact model:")
        zc = coeffs["z_contact"]
        print(f"  Intercept: {zc['intercept']:.6f}")
        print(f"  Coefficients: {[f'{c:.6f}' for c in zc['coefficients']]}")
        print(f"  R-squared: {zc['r_squared']:.4f}")
        print(f"\nxISO model:")
        xi = coeffs["xiso"]
        print(f"  Intercept: {xi['intercept']:.6f}")
        print(f"  Coefficients: {[f'{c:.6f}' for c in xi['coefficients']]}")
        print(f"  R-squared: {xi['r_squared']:.4f}")
        return

    # Handle plotting
    if args.plot:
        if args.bat_speed is None or args.swing_length is None:
            parser.error("--bat-speed and --swing-length are required for plotting")

        fig = plot_stat_change_grid(
            current_bat_speed=args.bat_speed,
            current_swing_length=args.swing_length,
            current_z_contact=args.current_z_contact,
            current_xiso=args.current_xiso,
            save_path=args.save_plot,
        )

        if not args.save_plot:
            plt.show()
        return

    # Validate required args for prediction
    if args.bat_speed is None or args.swing_length is None:
        parser.error("--bat-speed and --swing-length are required for prediction")
    if args.current_stat is None or args.stat_type is None:
        parser.error("--current-stat and --stat-type are required for prediction")

    result = update_stat(
        current_bat_speed=args.bat_speed,
        current_swing_length=args.swing_length,
        current_stat_value=args.current_stat,
        stat_type=args.stat_type,
        bat_speed_change=args.bs_change,
        swing_length_change=args.sl_change,
    )

    print(f"\nPrediction for {args.stat_type}:")
    print(f"  Current: {args.current_stat:.4f}")
    print(f"  Change:  {result['stat_change']:+.4f}")
    print(f"  New:     {result['new_stat_value']:.4f}")

    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()



