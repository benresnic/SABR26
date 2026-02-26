"""
update_stats.py

Joint model that maps (bat_speed, swing_length) → (z_contact, xISO).

Uses polynomial features + PLS2 (Partial Least Squares regression), which learns
shared latent components across both outputs (z_contact, xISO), rather than
fitting the outputs independently.

For Z-Contact updates, the model-implied mechanics effect is applied on the
logit (odds) scale and then converted back. This makes changes baseline-sensitive
(e.g., improving from .95 is harder than improving from .80).

Example:
    python "Bayes Outcomes/update_stats.py" \
        --bat-speed 75 --swing-length 7.2 \
        --current-z-contact 0.82 --current-xiso 0.180 \
        --bs-change -3 --sl-change -0.2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COEF_FILE = Path(__file__).parent / "update_stats_coefs.json"
DATA_DIR = Path(__file__).parent.parent / "Data"

# Typical ranges for warnings
BAT_SPEED_RANGE = (55.0, 90.0)
SWING_LENGTH_RANGE = (5.0, 10.0)
MIN_SWINGS_PER_BATTER_SEASON = 500

# Stat bounds for clipping
STAT_BOUNDS = {
    "z_contact": (0.5, 1.0),
    "xiso": (0.0, 0.6),
}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """Numerically stable logit with clipping away from 0/1."""
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


def _inv_logit(x: float) -> float:
    """Inverse logit / logistic function."""
    return float(1.0 / (1.0 + np.exp(-x)))



# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def _build_training_data() -> pd.DataFrame:
    """
    Load PBP data, filter to swings, aggregate bat speed and swing length
    to batter-season medians, and join with batter stats.

    Filters to batter-seasons with at least MIN_SWINGS_PER_BATTER_SEASON tracked swings.

    Returns DataFrame with:
    - median_bat_speed
    - median_swing_length
    - Z-Contact_pct
    - xISO
    - swing_sample_n
    """
    pbp_files = [DATA_DIR / f"pbp_{year}.parquet" for year in [2023, 2024, 2025]]
    pbp_list = []
    for f in pbp_files:
        if f.exists():
            pbp_list.append(pd.read_parquet(f))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)

    # Swings only (based on availability of bat_speed)
    pbp = pbp[pbp["bat_speed"].notna()].copy()

    # Aggregate to batter-season medians (+ sample size for fit filtering)
    swing_stats = (
        pbp.groupby(["batter", "game_year"])
        .agg(
            median_bat_speed=("bat_speed", "median"),
            median_swing_length=("swing_length", "median"),
            swing_sample_n=("bat_speed", "size"),
        )
        .reset_index()
    )

    swing_stats = swing_stats[
        swing_stats["swing_sample_n"] >= MIN_SWINGS_PER_BATTER_SEASON
    ].copy()

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

    cols_needed = ["median_bat_speed", "median_swing_length", "Z-Contact_pct", "xISO"]
    merged = merged.dropna(subset=cols_needed)

    return merged[cols_needed + ["swing_sample_n"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_joint_model() -> dict:
    """
    Fit a joint PLS2 model (multi-output Partial Least Squares):
    X: polynomial features of (bat_speed, swing_length)
    Y: (logit(z_contact), log(xiso))

    Targets are transformed so that diminishing returns at extremes
    are built into the model predictions.

    Returns coefficients dict with:
    - scaler params
    - polynomial feature metadata (including powers_ for exact feature reconstruction)
    - intercept (2 values)
    - coefficients (matrix)
    - r_squared for each output (in transformed space)
    """
    df = _build_training_data()

    # Standardize inputs before polynomial expansion
    scaler = StandardScaler()
    X_raw = df[["median_bat_speed", "median_swing_length"]].values
    X_scaled = scaler.fit_transform(X_raw)

    # Polynomial features (exact ordering saved via powers_)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Transform targets:
    # - z_contact → logit (bounded 0-1, diminishing returns near boundaries)
    # - xiso → log (positive, diminishing returns at high values)
    zc_raw = df["Z-Contact_pct"].values
    xiso_raw = df["xISO"].values

    # Clip to avoid log(0) or logit(0/1)
    zc_clipped = np.clip(zc_raw, 1e-4, 1 - 1e-4)
    xiso_clipped = np.clip(xiso_raw, 1e-4, None)

    zc_logit = np.log(zc_clipped / (1 - zc_clipped))
    xiso_log = np.log(xiso_clipped)

    Y_transformed = np.column_stack([zc_logit, xiso_log])

    # PLS2 shared latent structure
    max_valid_components = min(X_poly.shape[0] - 1, X_poly.shape[1], Y_transformed.shape[1])
    n_components = max(1, max_valid_components)

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_poly, Y_transformed)

    # Training predictions / diagnostics (in transformed space)
    Y_pred = model.predict(X_poly)
    ss_res = np.sum((Y_transformed - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_transformed - Y_transformed.mean(axis=0)) ** 2, axis=0)
    r_squared = 1 - ss_res / ss_tot

    residuals = Y_transformed - Y_pred
    output_cov = np.cov(residuals.T)

    results = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),

        "poly_degree": 2,
        "poly_include_bias": False,
        "feature_names": poly.get_feature_names_out().tolist(),
        "poly_powers": poly.powers_.tolist(),

        # Target transformations (model predicts in transformed space)
        "target_transforms": {
            "z_contact": "logit",  # model predicts logit(z_contact)
            "xiso": "log",         # model predicts log(xiso)
        },

        "training_stats": {
            "n_observations": int(len(df)),
            "min_swings_per_batter_season": int(MIN_SWINGS_PER_BATTER_SEASON),
            "swing_sample_n_min": int(df["swing_sample_n"].min()),
            "swing_sample_n_median": float(df["swing_sample_n"].median()),
            "swing_sample_n_max": int(df["swing_sample_n"].max()),
            "bat_speed_mean": float(df["median_bat_speed"].mean()),
            "bat_speed_std": float(df["median_bat_speed"].std()),
            "swing_length_mean": float(df["median_swing_length"].mean()),
            "swing_length_std": float(df["median_swing_length"].std()),
            "z_contact_mean": float(df["Z-Contact_pct"].mean()),
            "z_contact_std": float(df["Z-Contact_pct"].std()),
            "xiso_mean": float(df["xISO"].mean()),
            "xiso_std": float(df["xISO"].std()),
            "input_correlation": float(np.corrcoef(
                df["median_bat_speed"], df["median_swing_length"]
            )[0, 1]),
            "output_correlation": float(np.corrcoef(
                df["Z-Contact_pct"], df["xISO"]
            )[0, 1]),
        },

        "fit_date": datetime.now().isoformat(),
        "model_type": "joint_pls2_poly_transformed",

        "intercept": np.asarray(model.intercept_).tolist(),
        "coefficients": np.asarray(model.coef_).tolist(),

        "pls_n_components": int(n_components),
        "r_squared": {
            "z_contact": float(r_squared[0]),
            "xiso": float(r_squared[1]),
        },
        "residual_covariance": output_cov.tolist(),
    }

    return results


# ---------------------------------------------------------------------------
# Coefficient caching
# ---------------------------------------------------------------------------

def _cached_coeffs_are_compatible(coeffs: dict) -> bool:
    """Validate cached coefficient JSON schema for this script version."""
    required_top = {
        "scaler_mean", "scaler_std", "feature_names", "poly_powers",
        "intercept", "coefficients", "fit_date", "target_transforms"
    }
    if not isinstance(coeffs, dict):
        return False
    if not required_top.issubset(set(coeffs.keys())):
        return False
    if len(coeffs.get("scaler_mean", [])) != 2 or len(coeffs.get("scaler_std", [])) != 2:
        return False
    if len(coeffs.get("poly_powers", [])) == 0:
        return False
    # Require the transformed model type (fits in logit/log space for diminishing returns)
    if coeffs.get("model_type") != "joint_pls2_poly_transformed":
        return False
    # Verify target transforms are specified
    transforms = coeffs.get("target_transforms", {})
    if transforms.get("z_contact") != "logit" or transforms.get("xiso") != "log":
        return False
    # Require the current minimum swing-sample filter used for fitting
    training_stats = coeffs.get("training_stats", {})
    if training_stats.get("min_swings_per_batter_season") != MIN_SWINGS_PER_BATTER_SEASON:
        return False
    return True


def _load_or_fit_coefficients(force_refit: bool = False) -> dict:
    """
    Load coefficients from JSON if exists and compatible, otherwise fit model and save.
    """
    if not force_refit and COEF_FILE.exists():
        try:
            with open(COEF_FILE, "r") as f:
                coeffs = json.load(f)
            if _cached_coeffs_are_compatible(coeffs):
                return coeffs
            print("Cached coefficients are from an older/incompatible schema. Re-fitting model...")
        except Exception as e:
            print(f"Failed to load cached coefficients ({e}). Re-fitting model...")

    coeffs = _fit_joint_model()
    with open(COEF_FILE, "w") as f:
        json.dump(coeffs, f, indent=2)

    return coeffs


# ---------------------------------------------------------------------------
# Shared prediction helpers
# ---------------------------------------------------------------------------

def _build_joint_predictor(coeffs: dict):
    """
    Build helper functions using cached coeffs so feature construction matches
    the exact PolynomialFeatures ordering used at fit time.
    """

    def standardize(bs: float, sl: float):
        """Standardize inputs using saved scaler parameters."""
        return (
            (bs - coeffs["scaler_mean"][0]) / coeffs["scaler_std"][0],
            (sl - coeffs["scaler_mean"][1]) / coeffs["scaler_std"][1],
        )

    def make_features(bs: float, sl: float) -> np.ndarray:
        """
        Build polynomial features in the EXACT order used during fitting,
        using saved PolynomialFeatures powers_.
        """
        bs_s, sl_s = standardize(bs, sl)
        x = np.array([bs_s, sl_s], dtype=float)  # [bs, sl]
        powers = np.asarray(coeffs["poly_powers"], dtype=int)  # (n_features, 2)
        return np.prod(np.power(x, powers), axis=1)

    def predict_both(bs: float, sl: float):
        """
        Predict both stats at given bat speed and swing length.

        Model predicts in transformed space (logit for z_contact, log for xiso).
        This function transforms predictions back to original scale.

        Returns (z_contact_pred, xiso_pred) in original scale.
        """
        features = make_features(bs, sl)
        coef = np.asarray(coeffs["coefficients"], dtype=float)
        intercept = np.asarray(coeffs["intercept"], dtype=float)

        if coef.ndim != 2:
            raise ValueError(f"Unexpected coefficient shape: {coef.shape}")

        # sklearn versions may store coef as:
        #   (n_features, n_targets) OR (n_targets, n_features)
        if coef.shape[0] == features.shape[0]:
            y_pred = intercept + features @ coef
        elif coef.shape[1] == features.shape[0]:
            y_pred = intercept + coef @ features
        else:
            raise ValueError(
                f"Coefficient shape {coef.shape} does not match feature length {features.shape[0]}"
            )

        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_pred.shape[0] != 2:
            raise ValueError(f"Expected 2 outputs, got {y_pred.shape[0]}")

        # Transform back from model space to original scale
        # y_pred[0] is logit(z_contact) → apply inverse logit
        # y_pred[1] is log(xiso) → apply exp
        zc_pred = _inv_logit(y_pred[0])
        xiso_pred = float(np.exp(y_pred[1]))

        return zc_pred, xiso_pred

    return standardize, make_features, predict_both


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
    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")

    # Case-insensitive partial match
    mask = batter_stats["PlayerName"].str.lower().str.contains(player_name.lower(), na=False)
    matches = batter_stats[mask]

    if len(matches) == 0:
        raise ValueError(f"No player found matching '{player_name}'")

    unique_names = matches["PlayerName"].dropna().unique()
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
            available = sorted(matches["Season"].dropna().unique().tolist())
            raise ValueError(
                f"No data for {player_full_name} in {season}. "
                f"Available seasons: {available}"
            )
    else:
        season = int(player_data["Season"].max())
        player_data = player_data[player_data["Season"] == season]

    row = player_data.iloc[0]

    # Load swing data from PBP
    pbp_files = [DATA_DIR / f"pbp_{year}.parquet" for year in [2023, 2024, 2025]]
    pbp_list = []
    for f in pbp_files:
        if f.exists():
            pbp_list.append(pd.read_parquet(f))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)
    pbp = pbp[pbp["bat_speed"].notna()].copy()

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
        "bat_speed": float(player_swings["bat_speed"].median()),
        "swing_length": float(player_swings["swing_length"].median()),
        "z_contact": float(row["Z-Contact_pct"]),
        "xiso": float(row["xISO"]),
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def update_stats(
    current_bat_speed: float,
    current_swing_length: float,
    current_z_contact: float,
    current_xiso: float,
    bat_speed_change: float = 0.0,
    swing_length_change: float = 0.0,
) -> dict:
    """
    Predict how changes in bat speed and swing length affect BOTH stats simultaneously.

    Uses a joint PLS2 model that learns shared latent structure across z_contact and xISO.

    Z-Contact changes are applied on the logit scale (baseline-sensitive update).
    xISO changes are applied additively.

    Parameters
    ----------
    current_bat_speed : float
        Current bat speed in mph (e.g., 75.0)
    current_swing_length : float
        Current swing length in feet (e.g., 7.2)
    current_z_contact : float
        Current zone contact rate (e.g., 0.82 for 82%)
    current_xiso : float
        Current xISO value (e.g., 0.180)
    bat_speed_change : float
        Change in bat speed in mph (e.g., -3.0)
    swing_length_change : float
        Change in swing length in feet (e.g., -0.2)

    Returns
    -------
    dict
        {
            'new_z_contact': float,
            'new_xiso': float,
            'z_contact_change': float,
            'xiso_change': float,
            'warnings': list
        }
    """
    warnings = []

    original_current_z_contact = float(current_z_contact)
    original_current_xiso = float(current_xiso)

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

    # Load coefficients / build predictor helpers
    coeffs = _load_or_fit_coefficients()
    _, _, predict_both = _build_joint_predictor(coeffs)

    # Predict at current and new mechanics
    # Model is fit in transformed space (logit for zc, log for xiso),
    # so predictions already have diminishing returns at extremes built in
    pred_current_zc, pred_current_xiso = predict_both(current_bat_speed, current_swing_length)
    pred_new_zc, pred_new_xiso = predict_both(new_bat_speed, new_swing_length)

    # Compute deltas from model (these have diminishing returns baked in)
    zc_change = pred_new_zc - pred_current_zc
    xiso_change = pred_new_xiso - pred_current_xiso

    # Apply deltas to player's actual current values
    new_z_contact = original_current_z_contact + zc_change
    new_xiso = original_current_xiso + xiso_change

    # Clip to valid bounds
    zc_bounds = STAT_BOUNDS["z_contact"]
    if new_z_contact < zc_bounds[0]:
        warnings.append(
            f"New z_contact value {new_z_contact:.4f} clipped to minimum {zc_bounds[0]}"
        )
        new_z_contact = zc_bounds[0]
    elif new_z_contact > zc_bounds[1]:
        warnings.append(
            f"New z_contact value {new_z_contact:.4f} clipped to maximum {zc_bounds[1]}"
        )
        new_z_contact = zc_bounds[1]

    xiso_bounds = STAT_BOUNDS["xiso"]
    if new_xiso < xiso_bounds[0]:
        warnings.append(
            f"New xiso value {new_xiso:.4f} clipped to minimum {xiso_bounds[0]}"
        )
        new_xiso = xiso_bounds[0]
    elif new_xiso > xiso_bounds[1]:
        warnings.append(
            f"New xiso value {new_xiso:.4f} clipped to maximum {xiso_bounds[1]}"
        )
        new_xiso = xiso_bounds[1]

    # Final reported changes should match clipped outputs
    z_contact_change = new_z_contact - original_current_z_contact
    xiso_change = new_xiso - original_current_xiso

    return {
        "new_z_contact": float(new_z_contact),
        "new_xiso": float(new_xiso),
        "z_contact_change": float(z_contact_change),
        "xiso_change": float(xiso_change),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stat_change_grid(
    player: str = None,
    season: int = None,
    current_bat_speed: float = None,
    current_swing_length: float = None,
    current_z_contact: float = None,
    current_xiso: float = None,
    bat_speed_range: tuple = None,
    swing_length_range: tuple = None,
    grid_size: int = 50,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot heatmaps showing percent change in stats across bat speed / swing length grid.

    Uses the joint model for predictions. Can lookup player stats automatically.

    Parameters
    ----------
    player : str
        Player name to look up (case-insensitive, partial match). If provided,
        stats are looked up automatically and other stat parameters are ignored.
    season : int
        Season for player lookup. Defaults to most recent.
    current_bat_speed : float
        Current bat speed in mph (ignored if player is provided)
    current_swing_length : float
        Current swing length in feet (ignored if player is provided)
    current_z_contact : float
        Current zone contact rate (e.g., 0.80 for 80%) (ignored if player is provided)
    current_xiso : float
        Current xISO value (ignored if player is provided)
    bat_speed_range : tuple
        (min, max) for bat speed axis. Defaults to -12% to +3% from current.
    swing_length_range : tuple
        (min, max) for swing length axis. Defaults to -12% to +3% from current.
    grid_size : int
        Number of points along each axis
    save_path : str
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
    """
    # Look up player stats if player name provided
    player_name_display = None
    if player is not None:
        stats = get_player_stats(player, season)
        current_bat_speed = stats["bat_speed"]
        current_swing_length = stats["swing_length"]
        current_z_contact = stats["z_contact"]
        current_xiso = stats["xiso"]
        player_name_display = f"{stats['player_name']} ({stats['season']})"
        print(f"\nPlayer: {stats['player_name']} ({stats['season']})")
        print(f"  Bat Speed:    {stats['bat_speed']:.1f} mph")
        print(f"  Swing Length: {stats['swing_length']:.2f} ft")
        print(f"  Z-Contact:    {stats['z_contact']:.1%}")
        print(f"  xISO:         {stats['xiso']:.3f}")
    else:
        if current_bat_speed is None or current_swing_length is None:
            raise ValueError(
                "Must provide either 'player' or both 'current_bat_speed' and 'current_swing_length'"
            )
        if current_z_contact is None:
            current_z_contact = 0.80
        if current_xiso is None:
            current_xiso = 0.200

    # Default ranges: -12% to +3% from current
    if bat_speed_range is None:
        bat_speed_range = (current_bat_speed * 0.88, current_bat_speed * 1.03)
    if swing_length_range is None:
        swing_length_range = (current_swing_length * 0.88, current_swing_length * 1.03)

    swing_lengths = np.linspace(swing_length_range[0], swing_length_range[1], grid_size)
    bat_speeds = np.linspace(bat_speed_range[0], bat_speed_range[1], grid_size)

    coeffs = _load_or_fit_coefficients()
    _, _, predict_both = _build_joint_predictor(coeffs)

    # Model baseline at current mechanics
    # Model is fit in transformed space, so predictions have diminishing returns built in
    pred_current_zc, pred_current_xiso = predict_both(current_bat_speed, current_swing_length)

    zc_pct_change_grid = np.zeros((grid_size, grid_size))
    xiso_pct_change_grid = np.zeros((grid_size, grid_size))

    for i, bs in enumerate(bat_speeds):
        for j, sl in enumerate(swing_lengths):
            pred_new_zc, pred_new_xiso = predict_both(bs, sl)

            # Deltas have diminishing returns baked in from model's transformed space
            zc_change = pred_new_zc - pred_current_zc
            xiso_change = pred_new_xiso - pred_current_xiso

            zc_pct_change_grid[i, j] = (
                (zc_change / current_z_contact) * 100 if current_z_contact != 0 else np.nan
            )
            xiso_pct_change_grid[i, j] = (
                (xiso_change / current_xiso) * 100 if current_xiso != 0 else np.nan
            )

    stat_configs = [
        (zc_pct_change_grid, current_z_contact, "Z-Contact % Change", "RdYlGn", 15),
        (xiso_pct_change_grid, current_xiso, "xISO % Change", "RdYlGn", 30),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (pct_change_grid, current_val, title, cmap, vmax_limit) in zip(axes, stat_configs):
        im = ax.imshow(
            pct_change_grid,
            extent=[
                swing_length_range[0], swing_length_range[1],
                bat_speed_range[0], bat_speed_range[1]
            ],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-vmax_limit,
            vmax=vmax_limit,
        )

        ax.plot(current_swing_length, current_bat_speed, "ko", markersize=10, label="Current")
        ax.axhline(current_bat_speed, color="black", linestyle="--", alpha=0.3)
        ax.axvline(current_swing_length, color="black", linestyle="--", alpha=0.3)

        ax.set_xlabel("Swing Length (ft)")
        ax.set_ylabel("Bat Speed (mph)")
        ax.set_title(f"{title}\n(Current: {current_val:.3f})")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("% Change")

        ax.legend(loc="upper right")

    if player_name_display:
        suptitle = f"{player_name_display}: {current_bat_speed:.1f} mph, {current_swing_length:.2f} ft"
    else:
        suptitle = f"Stat Changes: {current_bat_speed:.1f} mph, {current_swing_length:.2f} ft"
    plt.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_player_stat_grid(
    player_name: str,
    season: int = None,
    save_path: str = None,
) -> tuple:
    """
    Look up a player's stats and plot their stat change grid.

    This is a convenience wrapper around plot_stat_change_grid.

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
    fig = plot_stat_change_grid(
        player=player_name,
        season=season,
        save_path=save_path,
    )
    return stats, fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict how bat speed and swing length changes affect both z_contact and xISO"
    )

    parser.add_argument("--info", action="store_true",
                        help="Show model coefficients and training stats")
    parser.add_argument("--refit", action="store_true",
                        help="Force refit of model (ignore cached coefficients)")
    parser.add_argument("--bat-speed", type=float, help="Current bat speed in mph")
    parser.add_argument("--swing-length", type=float, help="Current swing length in feet")
    parser.add_argument("--current-z-contact", type=float,
                        help="Current zone contact rate (e.g., 0.82)")
    parser.add_argument("--current-xiso", type=float,
                        help="Current xISO value (e.g., 0.180)")
    parser.add_argument("--bs-change", type=float, default=0.0,
                        help="Change in bat speed (mph)")
    parser.add_argument("--sl-change", type=float, default=0.0,
                        help="Change in swing length (feet)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate heatmap of stat changes across bat speed / swing length grid")
    parser.add_argument("--save-plot", type=str,
                        help="Path to save the plot (e.g., 'plots/grid.png')")
    parser.add_argument("--player", type=str,
                        help="Player name to look up and plot (e.g., 'Aaron Judge')")
    parser.add_argument("--season", type=int,
                        help="Season for player lookup (default: most recent)")

    args = parser.parse_args()

    # Show help if no arguments provided
    if not any([
        args.info, args.refit, args.bat_speed, args.swing_length,
        args.current_z_contact, args.current_xiso, args.plot, args.player
    ]):
        parser.print_help()
        return

    # Player lookup + plot mode
    if args.player:
        try:
            plot_stat_change_grid(
                player=args.player,
                season=args.season,
                save_path=args.save_plot,
            )
            if not args.save_plot:
                plt.show()
        except ValueError as e:
            print(f"Error: {e}")
        return

    # Info / refit mode
    if args.info or args.refit:
        coeffs = _load_or_fit_coefficients(force_refit=args.refit)
        print("Model Info (Joint PLS2 Regression with Polynomial Features)")
        print(f"  Model type: {coeffs.get('model_type', 'joint_pls2_poly')}")
        if "pls_n_components" in coeffs:
            print(f"  PLS components: {coeffs['pls_n_components']}")
        print(f"  Fit date: {coeffs['fit_date']}")

        print("\nTraining data:")
        ts = coeffs["training_stats"]
        print(f"  N observations: {ts['n_observations']}")
        print(f"  Bat speed mean: {ts['bat_speed_mean']:.2f} mph (std: {ts['bat_speed_std']:.2f})")
        print(f"  Swing length mean: {ts['swing_length_mean']:.2f} ft (std: {ts['swing_length_std']:.2f})")
        print(f"  Z-contact mean: {ts['z_contact_mean']:.4f} (std: {ts['z_contact_std']:.4f})")
        print(f"  xISO mean: {ts['xiso_mean']:.4f} (std: {ts['xiso_std']:.4f})")
        print(f"  Input correlation (bs, sl): {ts['input_correlation']:.4f}")
        print(f"  Output correlation (z_contact, xiso): {ts['output_correlation']:.4f}")

        print("\nScaler parameters (for standardization):")
        print(f"  Mean: {coeffs['scaler_mean']}")
        print(f"  Std:  {coeffs['scaler_std']}")

        print(f"\nFeatures: {coeffs['feature_names']}")
        print("Polynomial powers (rows align with feature order):")
        for name, powers in zip(coeffs["feature_names"], coeffs["poly_powers"]):
            print(f"  {name}: {powers}")

        intercept = np.asarray(coeffs["intercept"], dtype=float).reshape(-1)
        coef = np.asarray(coeffs["coefficients"], dtype=float)

        print("\nModel coefficients:")
        print(f"  Intercepts: z_contact={intercept[0]:.6f}, xiso={intercept[1]:.6f}")
        print(f"  Coefficient shape: {coef.shape}")
        print("  Coefficients matrix:")
        print(coef)

        print("\nR-squared:")
        print(f"  Z-Contact: {coeffs['r_squared']['z_contact']:.4f}")
        print(f"  xISO: {coeffs['r_squared']['xiso']:.4f}")

        print("\nResidual covariance matrix:")
        cov = coeffs["residual_covariance"]
        print(f"  [[{cov[0][0]:.6f}, {cov[0][1]:.6f}],")
        print(f"   [{cov[1][0]:.6f}, {cov[1][1]:.6f}]]")
        return

    # Plot mode with manual inputs
    if args.plot:
        if args.bat_speed is None or args.swing_length is None:
            parser.error("--bat-speed and --swing-length are required for plotting")

        current_z_contact = args.current_z_contact if args.current_z_contact is not None else 0.80
        current_xiso = args.current_xiso if args.current_xiso is not None else 0.200

        plot_stat_change_grid(
            current_bat_speed=args.bat_speed,
            current_swing_length=args.swing_length,
            current_z_contact=current_z_contact,
            current_xiso=current_xiso,
            save_path=args.save_plot,
        )

        if not args.save_plot:
            plt.show()
        return

    # Prediction mode
    if args.bat_speed is None or args.swing_length is None:
        parser.error("--bat-speed and --swing-length are required for prediction")
    if args.current_z_contact is None or args.current_xiso is None:
        parser.error("--current-z-contact and --current-xiso are required for prediction")

    result = update_stats(
        current_bat_speed=args.bat_speed,
        current_swing_length=args.swing_length,
        current_z_contact=args.current_z_contact,
        current_xiso=args.current_xiso,
        bat_speed_change=args.bs_change,
        swing_length_change=args.sl_change,
    )

    print("\nPrediction:")
    print("\nZ-Contact:")
    print(f"  Current: {args.current_z_contact:.4f}")
    print(f"  Change:  {result['z_contact_change']:+.4f}")
    print(f"  New:     {result['new_z_contact']:.4f}")

    print("\nxISO:")
    print(f"  Current: {args.current_xiso:.4f}")
    print(f"  Change:  {result['xiso_change']:+.4f}")
    print(f"  New:     {result['new_xiso']:.4f}")

    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
