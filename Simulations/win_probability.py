"""
win_probability.py

Build and query a win probability lookup table from historical PBP data.

The table is indexed by (outs, base_state, inning, inning_topbot, score_diff)
and returns the mean bat_win_exp (batting team's win expectancy) for that state.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

from .config import (
    DATA_DIR,
    OUTS,
    INNINGS,
    INNING_TOPBOT,
    SCORE_DIFFS,
    WP_LOOKUP_YEARS,
)


def encode_base_state(on_1b, on_2b, on_3b) -> int:
    """
    Encode base occupancy as a single integer 0-7.

    Uses binary representation: bit 0 = 1B, bit 1 = 2B, bit 2 = 3B
    """
    state = 0
    if pd.notna(on_1b):
        state |= 1  # bit 0
    if pd.notna(on_2b):
        state |= 2  # bit 1
    if pd.notna(on_3b):
        state |= 4  # bit 2
    return state


def build_wp_lookup(
    years: list = None,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Build win probability lookup table from PBP data.

    Returns NumPy array of shape (3, 8, 9, 2, 11) indexed by:
        [outs, base_state, inning-1, topbot_idx, score_diff+5]

    Parameters
    ----------
    years : list
        Years of PBP data to use. Defaults to WP_LOOKUP_YEARS.
    save_path : Path
        If provided, save the lookup table as .npy file.

    Returns
    -------
    np.ndarray
        Win probability lookup table, shape (3, 8, 9, 2, 11)
    """
    if years is None:
        years = WP_LOOKUP_YEARS

    # Load PBP data
    pbp_list = []
    for year in years:
        pbp_file = DATA_DIR / f"pbp_{year}.parquet"
        if pbp_file.exists():
            pbp_list.append(pd.read_parquet(pbp_file))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)

    # Filter to rows with valid bat_win_exp
    pbp = pbp[pbp["bat_win_exp"].notna()].copy()

    # Encode base state
    pbp["base_state"] = pbp.apply(
        lambda row: encode_base_state(row["on_1b"], row["on_2b"], row["on_3b"]),
        axis=1
    )

    # Cap innings at 9 (extra innings treated as 9th)
    pbp["inning_capped"] = pbp["inning"].clip(upper=9)

    # Cap score differential at +/- 5
    pbp["score_diff_capped"] = pbp["bat_score_diff"].clip(lower=-5, upper=5)

    # Encode inning_topbot as 0=Top, 1=Bot
    pbp["topbot_idx"] = (pbp["inning_topbot"] == "Bot").astype(int)

    # Group by state and compute mean win expectancy
    grouped = pbp.groupby([
        "outs_when_up",
        "base_state",
        "inning_capped",
        "topbot_idx",
        "score_diff_capped"
    ])["bat_win_exp"].mean()

    # Initialize lookup table with NaN
    # Shape: (3 outs, 8 base states, 9 innings, 2 top/bot, 11 score diffs)
    wp_table = np.full((3, 8, 9, 2, 11), np.nan, dtype=np.float32)

    # Fill in observed values
    for (outs, base_state, inning, topbot, score_diff), wp in grouped.items():
        outs_idx = int(outs)
        base_idx = int(base_state)
        inning_idx = int(inning) - 1  # Convert 1-9 to 0-8
        topbot_idx = int(topbot)
        score_idx = int(score_diff) + 5  # Convert -5..5 to 0..10

        if 0 <= outs_idx < 3 and 0 <= base_idx < 8 and 0 <= inning_idx < 9:
            wp_table[outs_idx, base_idx, inning_idx, topbot_idx, score_idx] = wp

    # Fill missing values with interpolation / nearest neighbor
    wp_table = _fill_missing_wp(wp_table)

    if save_path:
        np.save(save_path, wp_table)
        print(f"Saved WP lookup table to {save_path}")

    return wp_table


def _fill_missing_wp(wp_table: np.ndarray) -> np.ndarray:
    """
    Fill missing values in the WP table using interpolation.

    Strategy:
    1. For each missing cell, find nearest neighbors in score_diff dimension
    2. Fall back to mean of non-missing values in same (outs, inning, topbot) slice
    3. Final fallback to 0.5 (neutral expectancy)
    """
    filled = wp_table.copy()

    # Iterate over all positions
    for outs in range(3):
        for base in range(8):
            for inning in range(9):
                for topbot in range(2):
                    slice_data = filled[outs, base, inning, topbot, :]

                    # Find missing indices
                    missing_mask = np.isnan(slice_data)
                    if not missing_mask.any():
                        continue

                    # Try to interpolate from neighbors in score_diff
                    valid_mask = ~np.isnan(slice_data)
                    if valid_mask.any():
                        valid_indices = np.where(valid_mask)[0]
                        valid_values = slice_data[valid_mask]

                        for idx in np.where(missing_mask)[0]:
                            # Find nearest valid neighbor
                            distances = np.abs(valid_indices - idx)
                            nearest_idx = valid_indices[np.argmin(distances)]
                            filled[outs, base, inning, topbot, idx] = slice_data[nearest_idx]
                    else:
                        # Fall back to mean of same outs, inning, topbot across all bases
                        broader_slice = filled[outs, :, inning, topbot, :]
                        valid_in_broader = broader_slice[~np.isnan(broader_slice)]
                        if len(valid_in_broader) > 0:
                            fill_value = valid_in_broader.mean()
                        else:
                            # Ultimate fallback
                            fill_value = 0.5

                        for idx in np.where(missing_mask)[0]:
                            filled[outs, base, inning, topbot, idx] = fill_value

    # Final pass: fill any remaining NaNs with 0.5
    filled = np.nan_to_num(filled, nan=0.5)

    return filled


def lookup_wp(
    outs: int,
    base_state: int,
    inning: int,
    inning_topbot: str,
    score_diff: int,
    wp_table: np.ndarray,
) -> float:
    """
    Look up win probability for a given game state.

    Parameters
    ----------
    outs : int
        Number of outs (0, 1, 2)
    base_state : int
        Encoded base state (0-7)
    inning : int
        Inning number (1-9+, capped at 9)
    inning_topbot : str
        "Top" or "Bot"
    score_diff : int
        Score differential from batting team's perspective (-5 to +5, capped)
    wp_table : np.ndarray
        Win probability lookup table from build_wp_lookup()

    Returns
    -------
    float
        Win probability for the batting team
    """
    # Clamp indices
    outs_idx = max(0, min(2, outs))
    base_idx = max(0, min(7, base_state))
    inning_idx = max(0, min(8, min(inning, 9) - 1))
    topbot_idx = 1 if inning_topbot == "Bot" else 0
    score_idx = max(0, min(10, score_diff + 5))

    return float(wp_table[outs_idx, base_idx, inning_idx, topbot_idx, score_idx])


def get_state_wp_vectorized(
    states_df: pd.DataFrame,
    wp_table: np.ndarray,
) -> np.ndarray:
    """
    Vectorized win probability lookup for multiple states.

    Parameters
    ----------
    states_df : pd.DataFrame
        DataFrame with columns: outs, base_state, inning, inning_topbot, score_diff
    wp_table : np.ndarray
        Win probability lookup table

    Returns
    -------
    np.ndarray
        Win probabilities for each row
    """
    # Clamp indices
    outs_idx = states_df["outs"].clip(0, 2).astype(int).values
    base_idx = states_df["base_state"].clip(0, 7).astype(int).values
    inning_idx = (states_df["inning"].clip(1, 9) - 1).astype(int).values
    topbot_idx = (states_df["inning_topbot"] == "Bot").astype(int).values
    score_idx = (states_df["score_diff"].clip(-5, 5) + 5).astype(int).values

    return wp_table[outs_idx, base_idx, inning_idx, topbot_idx, score_idx]


if __name__ == "__main__":
    # Build and save WP table
    output_path = Path(__file__).parent / "wp_lookup.npy"
    wp_table = build_wp_lookup(save_path=output_path)

    # Print some example lookups
    print("\nExample WP lookups:")
    test_cases = [
        (0, 0, 1, "Top", 0),   # Start of game
        (0, 7, 9, "Bot", -1),  # Bases loaded, bottom 9, down 1
        (2, 0, 9, "Bot", 1),   # 2 outs, nobody on, bottom 9, up 1
    ]

    for outs, base, inning, topbot, diff in test_cases:
        wp = lookup_wp(outs, base, inning, topbot, diff, wp_table)
        print(f"  {outs} outs, base={base}, inning={inning} {topbot}, diff={diff:+d}: WP={wp:.3f}")
