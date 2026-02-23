"""
simulation_engine.py

Main orchestration module for PVC player win probability simulations.
"""

import json
import sys
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from .config import (
    DATA_DIR,
    BAYES_DIR,
    OUTPUT_DIR,
    COUNTS,
    OUTS,
    INNINGS,
    INNING_TOPBOT,
    SCORE_DIFFS,
    XFIP_PERCENTILES,
    BAT_SPEED_CHANGES,
    SWING_LENGTH_CHANGES,
    OUTCOMES,
    N_POSTERIOR_SAMPLES,
    BASE_STATE_NAMES,
    POPULATION_COV_MATRIX,
    MIN_SWINGS_FOR_PLAYER_COV,
    FEASIBILITY_THRESHOLD,
)
from .win_probability import build_wp_lookup, lookup_wp
from .runner_advancement import build_transition_tables, get_transitions, load_transition_tables
from .outcome_simulator import compute_expected_delta_wp

# Add parent directory to path for importing Bayes Outcomes modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from importlib import import_module


def get_pvc_players(year: int = 2025) -> pd.DataFrame:
    """
    Get list of PVC-flagged players from PBP data.

    Parameters
    ----------
    year : int
        Season to check for PVC flag.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: batter_id, player_name
    """
    pbp = pd.read_parquet(DATA_DIR / f"pbp_{year}.parquet")

    # Filter to PVC=1
    pvc_batters = pbp[pbp["PVC"] == 1][["batter", "batter_name"]].drop_duplicates()
    pvc_batters = pvc_batters.rename(columns={"batter": "batter_id", "batter_name": "player_name"})

    return pvc_batters.reset_index(drop=True)


def get_player_stats(player_id: int, season: int = 2025) -> Dict:
    """
    Get a player's baseline stats for simulation.

    Parameters
    ----------
    player_id : int
        MLB AM ID
    season : int
        Season to get stats for

    Returns
    -------
    dict
        Player stats including bat_speed, swing_length, z_contact, xiso, o_swing
    """
    # Load batter stats
    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")
    batter_stats["xMLBAMID"] = batter_stats["xMLBAMID"].astype(int)
    batter_stats["Season"] = batter_stats["Season"].astype(int)

    player_row = batter_stats[
        (batter_stats["xMLBAMID"] == player_id) &
        (batter_stats["Season"] == season)
    ]

    if len(player_row) == 0:
        raise ValueError(f"No stats found for player {player_id} in {season}")

    row = player_row.iloc[0]

    # Get swing data from PBP
    pbp = pd.read_parquet(DATA_DIR / f"pbp_{season}.parquet")
    pbp = pbp[pbp["bat_speed"].notna()]
    player_swings = pbp[pbp["batter"] == player_id]

    if len(player_swings) == 0:
        raise ValueError(f"No swing data found for player {player_id} in {season}")

    return {
        "player_id": player_id,
        "player_name": row["PlayerName"],
        "season": season,
        "bat_speed": float(player_swings["bat_speed"].median()),
        "swing_length": float(player_swings["swing_length"].median()),
        "z_contact": float(row["Z-Contact_pct"]),
        "xiso": float(row["xISO"]),
        "o_swing": float(row["O-Swing_pct"]),
    }


def get_player_covariance(player_id: int, season: int = 2025) -> np.ndarray:
    """
    Compute within-player covariance matrix from their swing data.

    Uses only swings within 95% CI to filter out bunts/check swings.

    Parameters
    ----------
    player_id : int
        MLB AM ID
    season : int
        Season to get swing data for

    Returns
    -------
    np.ndarray
        2x2 covariance matrix [[var_bs, cov], [cov, var_sl]].
        Falls back to population average if player has <30 swings.
    """
    pbp = pd.read_parquet(DATA_DIR / f"pbp_{season}.parquet")
    pbp = pbp[pbp["bat_speed"].notna() & pbp["swing_length"].notna()]
    player_swings = pbp[pbp["batter"] == player_id]

    if len(player_swings) < MIN_SWINGS_FOR_PLAYER_COV:
        return POPULATION_COV_MATRIX.copy()

    # Convert to float64 to handle pandas nullable Float64 dtype
    bs = player_swings["bat_speed"].to_numpy(dtype=np.float64)
    sl = player_swings["swing_length"].to_numpy(dtype=np.float64)

    # Filter to 95% CI to exclude bunts/check swings
    bs_lo, bs_hi = np.percentile(bs, [2.5, 97.5])
    sl_lo, sl_hi = np.percentile(sl, [2.5, 97.5])
    ci_mask = (bs >= bs_lo) & (bs <= bs_hi) & (sl >= sl_lo) & (sl <= sl_hi)
    bs_ci, sl_ci = bs[ci_mask], sl[ci_mask]

    if len(bs_ci) < MIN_SWINGS_FOR_PLAYER_COV:
        return POPULATION_COV_MATRIX.copy()

    # Compute covariance matrix from filtered swing data
    cov_result = np.cov(bs_ci, sl_ci)

    # Ensure we have a 2x2 matrix
    if cov_result.ndim == 0 or cov_result.shape != (2, 2):
        return POPULATION_COV_MATRIX.copy()

    return cov_result


def get_feasible_grid_mask(
    bs_changes: List[float],
    sl_changes: List[float],
    cov_matrix: np.ndarray,
    threshold: float = FEASIBILITY_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feasibility mask for grid cells based on bivariate normal density.

    Parameters
    ----------
    bs_changes : list
        Absolute changes in bat speed (mph)
    sl_changes : list
        Absolute changes in swing length (feet)
    cov_matrix : np.ndarray
        2x2 covariance matrix [[var_bs, cov], [cov, var_sl]]
    threshold : float
        Minimum density (as fraction of max) to consider feasible

    Returns
    -------
    tuple
        (mask, densities) where:
        - mask: boolean (n_bs, n_sl), True = feasible
        - densities: float (n_bs, n_sl), bivariate normal PDF values
    """
    n_bs = len(bs_changes)
    n_sl = len(sl_changes)

    # Compute bivariate normal density at each grid point
    # Mean is (0, 0) since we're measuring changes from baseline
    densities = np.zeros((n_bs, n_sl))

    # Inverse of covariance matrix for Mahalanobis distance
    cov_inv = np.linalg.inv(cov_matrix)
    cov_det = np.linalg.det(cov_matrix)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(cov_det))

    for i, bs_change in enumerate(bs_changes):
        for j, sl_change in enumerate(sl_changes):
            x = np.array([bs_change, sl_change])
            # Mahalanobis distance squared
            mahal_sq = x @ cov_inv @ x
            densities[i, j] = norm_const * np.exp(-0.5 * mahal_sq)

    # Normalize relative to maximum (at origin 0,0)
    max_density = densities.max()
    if max_density > 0:
        relative_densities = densities / max_density
    else:
        relative_densities = densities

    # Create mask: True if density >= threshold * max_density
    mask = relative_densities >= threshold

    return mask, densities


def compute_xfip_percentiles(years: List[int] = None) -> Dict[int, float]:
    """
    Compute xFIP values at various percentiles from pitcher stats.

    Parameters
    ----------
    years : list
        Seasons to include

    Returns
    -------
    dict
        Maps percentile (10, 30, 50, 70, 90) to xFIP value
    """
    if years is None:
        years = [2023, 2024, 2025]

    pitcher_stats = pd.read_parquet(DATA_DIR / "pitcher_stats.parquet")
    pitcher_stats = pitcher_stats[pitcher_stats["Season"].isin(years)]

    xfip_values = pitcher_stats["xFIP"].dropna()

    return {
        pct: float(np.percentile(xfip_values, pct))
        for pct in XFIP_PERCENTILES
    }


def generate_state_space(
    reduced: bool = False,
) -> pd.DataFrame:
    """
    Generate all game states for simulation.

    Parameters
    ----------
    reduced : bool
        If True, use reduced state space for faster testing

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: count, balls, strikes, outs, base_state, inning, inning_topbot, score_diff
    """
    if reduced:
        # Reduced state space for testing
        counts = ["0-0", "1-1", "3-2"]
        outs_list = [0, 2]
        bases = [0, 1, 7]
        innings = [1, 5, 9]
        topbots = ["Top", "Bot"]
        diffs = [-2, 0, 2]
    else:
        counts = COUNTS
        outs_list = OUTS
        bases = list(range(8))
        innings = INNINGS
        topbots = INNING_TOPBOT
        diffs = SCORE_DIFFS

    states = []
    for count, outs, base, inning, topbot, diff in product(
        counts, outs_list, bases, innings, topbots, diffs
    ):
        balls, strikes = map(int, count.split("-"))
        states.append({
            "count": count,
            "balls": balls,
            "strikes": strikes,
            "outs": outs,
            "base_state": base,
            "inning": inning,
            "inning_topbot": topbot,
            "score_diff": diff,
        })

    return pd.DataFrame(states)


def load_bayesian_model():
    """
    Load the fitted Bayesian outcome model and scalers.

    Returns
    -------
    tuple
        (idata, scalers, batter_stats_df, pitcher_stats_df)
    """
    model_path = BAYES_DIR / "bayes_outcome_model.nc"
    scalers_path = BAYES_DIR / "scalers.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    idata = az.from_netcdf(model_path)

    with open(scalers_path, "r") as f:
        scalers = json.load(f)

    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")
    pitcher_stats = pd.read_parquet(DATA_DIR / "pitcher_stats.parquet")

    return idata, scalers, batter_stats, pitcher_stats


def predict_outcome_probs_direct(
    states_df: pd.DataFrame,
    o_swing: float,
    z_contact: float,
    xiso: float,
    xfip: float,
    idata,
    scalers: Dict,
    n_samples: int = N_POSTERIOR_SAMPLES,
) -> np.ndarray:
    """
    Predict outcome probabilities for given states and stats.

    This is a simplified version of predict_outcome_probs from outcome_probs.py
    that works directly with pre-computed stats.

    Parameters
    ----------
    states_df : pd.DataFrame
        DataFrame with columns: balls, strikes
    o_swing, z_contact, xiso, xfip : float
        Player/pitcher stats (raw, not standardized)
    idata : az.InferenceData
        Fitted model
    scalers : dict
        Standardization parameters
    n_samples : int
        Number of posterior samples to average over

    Returns
    -------
    np.ndarray
        Shape (N, 7) - outcome probabilities for each state
    """
    # Count dummies (same as outcome_probs.py)
    ALL_COUNT_COLS = [
        f"count_{b}-{s}"
        for b in range(4)
        for s in range(3)
        if not (b == 0 and s == 0)
    ]

    states_df = states_df.copy()
    states_df["count_str"] = states_df["balls"].astype(str) + "-" + states_df["strikes"].astype(str)
    count_dummies = pd.get_dummies(states_df["count_str"], prefix="count").astype(float)

    for col in ALL_COUNT_COLS:
        if col not in count_dummies.columns:
            count_dummies[col] = 0.0
    count_dummies = count_dummies[ALL_COUNT_COLS].reset_index(drop=True)

    M = len(states_df)
    count_X = count_dummies.values.astype("float32")

    # Standardize stats
    o_swing_s = (o_swing - scalers["O-Swing_pct"]["mean"]) / scalers["O-Swing_pct"]["std"]
    z_contact_s = (z_contact - scalers["Z-Contact_pct"]["mean"]) / scalers["Z-Contact_pct"]["std"]
    xiso_s = (xiso - scalers["xISO"]["mean"]) / scalers["xISO"]["std"]
    xfip_s = (xfip - scalers["xFIP"]["mean"]) / scalers["xFIP"]["std"]

    # Extract posterior samples
    post = idata.posterior

    def get_flat(name):
        arr = post[name].values
        return arr.reshape(-1, *arr.shape[2:])

    total_draws = get_flat("alpha").shape[0]
    rng = np.random.default_rng(42)
    draw_idx = rng.choice(total_draws, size=min(n_samples, total_draws), replace=False)
    S = len(draw_idx)

    alpha_s = get_flat("alpha")[draw_idx]
    beta_count_s = get_flat("beta_count")[draw_idx]
    beta_o_swing_s = get_flat("beta_o_swing")[draw_idx]
    beta_zc_s = get_flat("beta_z_contact")[draw_idx]
    beta_xi_s = get_flat("beta_xiso")[draw_idx]
    beta_xf_s = get_flat("beta_xfip")[draw_idx]

    # Compute probabilities for each posterior draw
    all_probs = np.empty((S, M, 7), dtype=np.float32)

    for s in range(S):
        logit_mat = (
            alpha_s[s][None, :]
            + count_X @ beta_count_s[s]
            + o_swing_s * beta_o_swing_s[s]
            + z_contact_s * beta_zc_s[s]
            + xiso_s * beta_xi_s[s]
            + xfip_s * beta_xf_s[s]
        )

        # Insert reference column (field_out) at position 2
        logit_full = np.concatenate([
            logit_mat[:, 0:2],
            np.zeros((M, 1), dtype=np.float32),
            logit_mat[:, 2:6],
        ], axis=1)

        # Stable softmax
        logit_full -= logit_full.max(axis=1, keepdims=True)
        exp_l = np.exp(logit_full)
        all_probs[s] = exp_l / exp_l.sum(axis=1, keepdims=True)

    return all_probs.mean(axis=0)


def simulate_player(
    player_id: int,
    player_name: str,
    idata,
    scalers: Dict,
    wp_table: np.ndarray,
    transition_tables: Dict,
    xfip_percentile_values: Dict[int, float],
    states_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run simulation for a single player across all approach changes and states.

    Parameters
    ----------
    player_id : int
        MLB AM ID
    player_name : str
        Player name
    idata : az.InferenceData
        Fitted Bayesian model
    scalers : dict
        Feature standardization parameters
    wp_table : np.ndarray
        Win probability lookup table
    transition_tables : dict
        State transition probabilities
    xfip_percentile_values : dict
        xFIP values at each percentile
    states_df : pd.DataFrame
        All game states to simulate
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Simulation results for this player
    """
    # Import update_stats function (joint model)
    sys.path.insert(0, str(BAYES_DIR))
    from update_stats import update_stats, get_player_stats as get_player_stats_swing

    try:
        # Get player baseline stats
        player_stats = get_player_stats(player_id, season=2025)
    except ValueError as e:
        if verbose:
            print(f"  Skipping {player_name}: {e}")
        return pd.DataFrame()

    baseline_bat_speed = player_stats["bat_speed"]
    baseline_swing_length = player_stats["swing_length"]
    baseline_z_contact = player_stats["z_contact"]
    baseline_xiso = player_stats["xiso"]
    baseline_o_swing = player_stats["o_swing"]

    if verbose:
        print(f"  {player_name}: BS={baseline_bat_speed:.1f}, SL={baseline_swing_length:.2f}")

    # Compute player-specific feasibility mask
    player_cov = get_player_covariance(player_id)
    bs_changes_abs = [baseline_bat_speed * pct for pct in BAT_SPEED_CHANGES]
    sl_changes_abs = [baseline_swing_length * pct for pct in SWING_LENGTH_CHANGES]
    feasibility_mask, feasibility_densities = get_feasible_grid_mask(
        bs_changes_abs, sl_changes_abs, player_cov
    )

    # Extract player swing stats for output
    player_bs_std = np.sqrt(player_cov[0, 0])
    player_sl_std = np.sqrt(player_cov[1, 1])
    player_bs_sl_corr = player_cov[0, 1] / (player_bs_std * player_sl_std)

    n_feasible = feasibility_mask.sum()
    if verbose:
        print(f"    Feasibility: {n_feasible}/25 cells (corr={player_bs_sl_corr:.2f})")

    results = []
    n_states = len(states_df)

    # Precompute wp_before_PA for all states (same regardless of approach)
    wp_before_pa_list = []
    for i in range(n_states):
        row = states_df.iloc[i]
        wp_before = lookup_wp(
            outs=int(row["outs"]),
            base_state=int(row["base_state"]),
            inning=int(row["inning"]),
            inning_topbot=str(row["inning_topbot"]),
            score_diff=int(row["score_diff"]),
            wp_table=wp_table,
        )
        wp_before_pa_list.append(wp_before)

    # Precompute baseline (0,0) outcome probs and expected WP after PA for each xFIP level
    # This is the reference point for comparing approach changes
    baseline_wp_after_pa = {}  # {xfip_pct: [wp_after for each state]}
    baseline_outcome_probs = {}  # {xfip_pct: (n_states, 7) array}

    for xfip_pct, xfip_value in xfip_percentile_values.items():
        # Baseline outcome probs (no approach change)
        probs = predict_outcome_probs_direct(
            states_df,
            o_swing=baseline_o_swing,
            z_contact=baseline_z_contact,
            xiso=baseline_xiso,
            xfip=xfip_value,
            idata=idata,
            scalers=scalers,
        )
        baseline_outcome_probs[xfip_pct] = probs

        # Compute baseline expected WP after PA for each state
        wp_after_list = []
        for i in range(n_states):
            row = states_df.iloc[i]
            _, expected_wp, _ = compute_expected_delta_wp(
                outcome_probs=probs[i],
                outs=int(row["outs"]),
                base_state=int(row["base_state"]),
                inning=int(row["inning"]),
                inning_topbot=str(row["inning_topbot"]),
                score_diff=int(row["score_diff"]),
                wp_table=wp_table,
                transition_tables=transition_tables,
            )
            wp_after_list.append(expected_wp)
        baseline_wp_after_pa[xfip_pct] = wp_after_list

    # Now iterate through all approach changes
    for bs_idx, bs_pct in enumerate(BAT_SPEED_CHANGES):
        for sl_idx, sl_pct in enumerate(SWING_LENGTH_CHANGES):
            feasibility_density = feasibility_densities[bs_idx, sl_idx]
            is_feasible = int(feasibility_mask[bs_idx, sl_idx])

            # Compute adjusted stats using update_stat
            bs_change = baseline_bat_speed * bs_pct
            sl_change = baseline_swing_length * sl_pct

            # Compute new bat speed and swing length values
            new_bat_speed = baseline_bat_speed + bs_change
            new_swing_length = baseline_swing_length + sl_change

            # Use joint model to get both adjusted stats at once
            stats_result = update_stats(
                current_bat_speed=baseline_bat_speed,
                current_swing_length=baseline_swing_length,
                current_z_contact=baseline_z_contact,
                current_xiso=baseline_xiso,
                bat_speed_change=bs_change,
                swing_length_change=sl_change,
            )
            adjusted_z_contact = stats_result["new_z_contact"]
            adjusted_xiso = stats_result["new_xiso"]

            # For each pitcher xFIP percentile
            for xfip_pct, xfip_value in xfip_percentile_values.items():
                # For baseline (0,0) approach, use precomputed values
                if bs_pct == 0.0 and sl_pct == 0.0:
                    outcome_probs = baseline_outcome_probs[xfip_pct]
                else:
                    # Predict outcome probabilities for adjusted approach
                    outcome_probs = predict_outcome_probs_direct(
                        states_df,
                        o_swing=baseline_o_swing,
                        z_contact=adjusted_z_contact,
                        xiso=adjusted_xiso,
                        xfip=xfip_value,
                        idata=idata,
                        scalers=scalers,
                    )

                # Compute delta WP for each state
                for state_idx in range(n_states):
                    row = states_df.iloc[state_idx]

                    wp_before = wp_before_pa_list[state_idx]
                    baseline_wp_after = baseline_wp_after_pa[xfip_pct][state_idx]

                    if bs_pct == 0.0 and sl_pct == 0.0:
                        # Baseline approach - adjusted equals baseline
                        adjusted_wp_after = baseline_wp_after
                    else:
                        # Compute adjusted expected WP after PA
                        _, adjusted_wp_after, _ = compute_expected_delta_wp(
                            outcome_probs=outcome_probs[state_idx],
                            outs=int(row["outs"]),
                            base_state=int(row["base_state"]),
                            inning=int(row["inning"]),
                            inning_topbot=str(row["inning_topbot"]),
                            score_diff=int(row["score_diff"]),
                            wp_table=wp_table,
                            transition_tables=transition_tables,
                        )

                    # Calculate the difference from baseline approach
                    wp_diff_from_baseline = adjusted_wp_after - baseline_wp_after

                    results.append({
                        "batter_id": player_id,
                        "player_name": player_name,
                        "baseline_bat_speed": baseline_bat_speed,
                        "baseline_swing_length": baseline_swing_length,
                        "baseline_z_contact": baseline_z_contact,
                        "baseline_xiso": baseline_xiso,
                        "baseline_o_swing_pct": baseline_o_swing,
                        "bat_speed_change_pct": bs_pct,
                        "swing_length_change_pct": sl_pct,
                        "new_bat_speed": new_bat_speed,
                        "new_swing_length": new_swing_length,
                        "adjusted_z_contact": adjusted_z_contact,
                        "adjusted_xiso": adjusted_xiso,
                        "pitcher_xfip_percentile": xfip_pct,
                        "pitcher_xfip_value": xfip_value,
                        "count": row["count"],
                        "outs": row["outs"],
                        "base_state": row["base_state"],
                        "base_state_name": BASE_STATE_NAMES[row["base_state"]],
                        "inning": row["inning"],
                        "inning_topbot": row["inning_topbot"],
                        "score_diff": row["score_diff"],
                        "wp_before_pa": wp_before,
                        "baseline_wp_after_pa": baseline_wp_after,
                        "adjusted_wp_after_pa": adjusted_wp_after,
                        "wp_diff_from_baseline": wp_diff_from_baseline,
                        "feasibility_density": feasibility_density,
                        "is_feasible": is_feasible,
                        "player_bs_std": player_bs_std,
                        "player_sl_std": player_sl_std,
                        "player_bs_sl_corr": player_bs_sl_corr,
                        "prob_K": outcome_probs[state_idx, 0],
                        "prob_BB_HBP": outcome_probs[state_idx, 1],
                        "prob_field_out": outcome_probs[state_idx, 2],
                        "prob_1B": outcome_probs[state_idx, 3],
                        "prob_2B": outcome_probs[state_idx, 4],
                        "prob_3B": outcome_probs[state_idx, 5],
                        "prob_HR": outcome_probs[state_idx, 6],
                    })

    return pd.DataFrame(results)


def _simulate_player_worker(args: Tuple) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Worker function for parallel simulation.

    Loads all necessary data within the worker process to avoid pickling issues.

    Parameters
    ----------
    args : tuple
        (player_id, player_name, reduced_states, worker_id, total_players)

    Returns
    -------
    tuple
        (player_name, results_df or None)
    """
    player_id, player_name, reduced_states, worker_id, total_players = args

    try:
        # Load all data within worker process (avoids pickling issues with idata)
        idata, scalers, _, _ = load_bayesian_model()
        wp_table = build_wp_lookup()
        transition_tables = load_transition_tables()
        xfip_percentile_values = compute_xfip_percentiles()
        states_df = generate_state_space(reduced=reduced_states)

        print(f"[Worker] Simulating {player_name} ({worker_id}/{total_players})...")

        player_results = simulate_player(
            player_id=player_id,
            player_name=player_name,
            idata=idata,
            scalers=scalers,
            wp_table=wp_table,
            transition_tables=transition_tables,
            xfip_percentile_values=xfip_percentile_values,
            states_df=states_df,
            verbose=False,  # Reduce noise in parallel mode
        )

        if len(player_results) > 0:
            # Save to output directory
            safe_name = player_name.replace(" ", "_").replace(".", "")
            output_path = OUTPUT_DIR / f"{safe_name}_simulation.parquet"
            player_results.to_parquet(output_path, index=False)
            print(f"[Worker] Completed {player_name}: {len(player_results)} rows saved")
            return (player_name, player_results)
        else:
            print(f"[Worker] Skipped {player_name}: no results")
            return (player_name, None)

    except Exception as e:
        print(f"[Worker] Error simulating {player_name}: {e}")
        return (player_name, None)


def run_simulation(
    players: List[int] = None,
    reduced_states: bool = False,
    n_workers: int = 1,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run full simulation for PVC players.

    Parameters
    ----------
    players : list
        List of player IDs to simulate. If None, uses all PVC players.
    reduced_states : bool
        If True, use reduced state space for faster testing.
    n_workers : int
        Number of parallel workers. If > 1, runs simulations in parallel.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Maps player name to results DataFrame
    """
    if verbose:
        print("Loading models and data...")

    # Get PVC players
    pvc_players = get_pvc_players()
    if verbose:
        print(f"Found {len(pvc_players)} PVC players")

    if players is not None:
        pvc_players = pvc_players[pvc_players["batter_id"].isin(players)]

    player_list = [
        (int(row["batter_id"]), row["player_name"])
        for _, row in pvc_players.iterrows()
    ]
    total_players = len(player_list)

    if verbose:
        print(f"Will simulate {total_players} players with {n_workers} worker(s)")

    results = {}

    if n_workers > 1:
        # Parallel execution
        if verbose:
            print(f"Starting parallel simulation with {n_workers} workers...")

        # Prepare worker arguments
        worker_args = [
            (player_id, player_name, reduced_states, idx + 1, total_players)
            for idx, (player_id, player_name) in enumerate(player_list)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_simulate_player_worker, args): args[1]
                for args in worker_args
            }

            for future in as_completed(futures):
                player_name = futures[future]
                try:
                    name, player_results = future.result()
                    if player_results is not None:
                        results[name] = player_results
                except Exception as e:
                    print(f"Error processing {player_name}: {e}")
    else:
        # Sequential execution (original behavior)
        # Load shared data once for efficiency
        idata, scalers, batter_stats, pitcher_stats = load_bayesian_model()

        if verbose:
            print("Building win probability table...")
        wp_table = build_wp_lookup()

        if verbose:
            print("Building transition tables...")
        transition_tables = load_transition_tables()

        xfip_percentile_values = compute_xfip_percentiles()
        if verbose:
            print(f"xFIP percentiles: {xfip_percentile_values}")

        states_df = generate_state_space(reduced=reduced_states)
        if verbose:
            print(f"State space size: {len(states_df)}")

        for idx, (player_id, player_name) in enumerate(player_list):
            if verbose:
                print(f"\nSimulating {player_name} ({idx + 1}/{total_players})...")

            player_results = simulate_player(
                player_id=player_id,
                player_name=player_name,
                idata=idata,
                scalers=scalers,
                wp_table=wp_table,
                transition_tables=transition_tables,
                xfip_percentile_values=xfip_percentile_values,
                states_df=states_df,
                verbose=verbose,
            )

            if len(player_results) > 0:
                safe_name = player_name.replace(" ", "_").replace(".", "")
                output_path = OUTPUT_DIR / f"{safe_name}_simulation.parquet"
                player_results.to_parquet(output_path, index=False)

                if verbose:
                    print(f"  Saved {len(player_results)} rows to {output_path}")

                results[player_name] = player_results

    return results


if __name__ == "__main__":
    # Quick test with reduced state space
    results = run_simulation(reduced_states=True, verbose=True)
    print(f"\nSimulated {len(results)} players")
