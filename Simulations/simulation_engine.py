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
        "bat_speed": float(player_swings["bat_speed"].mean()),
        "swing_length": float(player_swings["swing_length"].mean()),
        "z_contact": float(row["Z-Contact_pct"]),
        "xiso": float(row["xISO"]),
        "o_swing": float(row["O-Swing_pct"]),
    }


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
    # Import update_stat function
    sys.path.insert(0, str(BAYES_DIR))
    from update_stats import update_stat, get_player_stats as get_player_stats_swing

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
    for bs_pct in BAT_SPEED_CHANGES:
        for sl_pct in SWING_LENGTH_CHANGES:
            # Compute adjusted stats using update_stat
            bs_change = baseline_bat_speed * bs_pct
            sl_change = baseline_swing_length * sl_pct

            # Compute new bat speed and swing length values
            new_bat_speed = baseline_bat_speed + bs_change
            new_swing_length = baseline_swing_length + sl_change

            zc_result = update_stat(
                current_bat_speed=baseline_bat_speed,
                current_swing_length=baseline_swing_length,
                current_stat_value=baseline_z_contact,
                stat_type="z_contact",
                bat_speed_change=bs_change,
                swing_length_change=sl_change,
            )
            adjusted_z_contact = zc_result["new_stat_value"]

            xiso_result = update_stat(
                current_bat_speed=baseline_bat_speed,
                current_swing_length=baseline_swing_length,
                current_stat_value=baseline_xiso,
                stat_type="xiso",
                bat_speed_change=bs_change,
                swing_length_change=sl_change,
            )
            adjusted_xiso = xiso_result["new_stat_value"]

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
                for i in range(n_states):
                    row = states_df.iloc[i]

                    wp_before = wp_before_pa_list[i]
                    baseline_wp_after = baseline_wp_after_pa[xfip_pct][i]

                    if bs_pct == 0.0 and sl_pct == 0.0:
                        # Baseline approach - adjusted equals baseline
                        adjusted_wp_after = baseline_wp_after
                    else:
                        # Compute adjusted expected WP after PA
                        _, adjusted_wp_after, _ = compute_expected_delta_wp(
                            outcome_probs=outcome_probs[i],
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
                        "prob_K": outcome_probs[i, 0],
                        "prob_BB_HBP": outcome_probs[i, 1],
                        "prob_field_out": outcome_probs[i, 2],
                        "prob_1B": outcome_probs[i, 3],
                        "prob_2B": outcome_probs[i, 4],
                        "prob_3B": outcome_probs[i, 5],
                        "prob_HR": outcome_probs[i, 6],
                    })

    return pd.DataFrame(results)


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
        Number of parallel workers (not yet implemented for parallel execution).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Maps player name to results DataFrame
    """
    if verbose:
        print("Loading models and data...")

    # Load Bayesian model
    idata, scalers, batter_stats, pitcher_stats = load_bayesian_model()

    # Build lookup tables
    if verbose:
        print("Building win probability table...")
    wp_table = build_wp_lookup()

    if verbose:
        print("Building transition tables...")
    transition_tables = load_transition_tables()

    # Get xFIP percentiles
    xfip_percentile_values = compute_xfip_percentiles()
    if verbose:
        print(f"xFIP percentiles: {xfip_percentile_values}")

    # Generate state space
    states_df = generate_state_space(reduced=reduced_states)
    if verbose:
        print(f"State space size: {len(states_df)}")

    # Get PVC players
    pvc_players = get_pvc_players()
    if verbose:
        print(f"Found {len(pvc_players)} PVC players")

    if players is not None:
        pvc_players = pvc_players[pvc_players["batter_id"].isin(players)]

    # Run simulations
    results = {}

    for idx, row in pvc_players.iterrows():
        player_id = int(row["batter_id"])
        player_name = row["player_name"]

        if verbose:
            print(f"\nSimulating {player_name} ({idx + 1}/{len(pvc_players)})...")

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
            # Save to output directory
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
