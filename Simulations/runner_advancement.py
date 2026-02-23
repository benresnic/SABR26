"""
runner_advancement.py

Build empirical transition probability tables from PBP data.

For each (outcome, outs, base_state), compute the distribution of:
- post_base_state
- post_outs
- runs_scored

This captures real-world transitions including sac flies, double plays,
runner advancement rates, and rare events.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle

from .config import DATA_DIR, DEFAULT_YEARS, OUTCOMES, OUTCOME_INDICES


# Event mapping to outcome indices (same as outcome_probs.py)
EVENT_TO_OUTCOME = {
    "strikeout": 0, "strikeout_double_play": 0,
    "walk": 1, "intent_walk": 1, "hit_by_pitch": 1,
    "field_out": 2, "force_out": 2, "double_play": 2,
    "fielders_choice": 2, "fielders_choice_out": 2,
    "grounded_into_double_play": 2, "sac_fly": 2,
    "sac_bunt": 2, "sac_fly_double_play": 2, "triple_play": 2,
    "single": 3, "double": 4, "triple": 5, "home_run": 6,
}

EXCLUDE_EVENTS = {"catcher_interf", "truncated_pa", "field_error"}


def encode_base_state(on_1b, on_2b, on_3b) -> int:
    """Encode base occupancy as integer 0-7 using binary representation."""
    state = 0
    if pd.notna(on_1b):
        state |= 1
    if pd.notna(on_2b):
        state |= 2
    if pd.notna(on_3b):
        state |= 4
    return state


def build_transition_tables(
    years: list = None,
    save_path: Optional[Path] = None,
    min_observations: int = 5,
) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int, float]]]:
    """
    Build empirical transition probability tables from PBP data.

    Groups PBP data by (outcome, outs, base_state) and computes the distribution
    of (post_base_state, post_outs, runs_scored).

    Parameters
    ----------
    years : list
        Years of PBP data to use. Defaults to DEFAULT_YEARS.
    save_path : Path
        If provided, save the transition tables as pickle file.
    min_observations : int
        Minimum number of observations required for a transition to be included.

    Returns
    -------
    dict
        Maps (outcome_idx, outs, base_state) -> list of (new_base_state, new_outs, runs_scored, probability)
    """
    if years is None:
        years = DEFAULT_YEARS

    # Load PBP data
    pbp_list = []
    for year in years:
        pbp_file = DATA_DIR / f"pbp_{year}.parquet"
        if pbp_file.exists():
            pbp_list.append(pd.read_parquet(pbp_file))

    if not pbp_list:
        raise FileNotFoundError(f"No PBP files found in {DATA_DIR}")

    pbp = pd.concat(pbp_list, ignore_index=True)

    # Filter to PA-ending events
    pbp = pbp[pbp["events"].notna()].copy()
    pbp = pbp[~pbp["events"].isin(EXCLUDE_EVENTS)].copy()

    # Map events to outcome indices
    pbp["outcome_idx"] = pbp["events"].map(EVENT_TO_OUTCOME)
    pbp = pbp[pbp["outcome_idx"].notna()].copy()
    pbp["outcome_idx"] = pbp["outcome_idx"].astype(int)

    # Encode pre-PA base state
    pbp["pre_base_state"] = pbp.apply(
        lambda row: encode_base_state(row["on_1b"], row["on_2b"], row["on_3b"]),
        axis=1
    )

    # For post-PA base state, we need to look at the next pitch in the same half-inning
    # or infer from score changes. We'll use a simpler approach:
    # look for next row within same game and at_bat_number + 1

    # Sort by game and at_bat for proper sequencing
    pbp = pbp.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    # Get the last pitch of each PA (where events is not null)
    pa_endings = pbp[pbp["events"].notna()].copy()

    # Calculate runs scored
    pa_endings["runs_scored"] = (
        pa_endings["post_bat_score"].fillna(pa_endings["bat_score"]) -
        pa_endings["bat_score"]
    ).fillna(0).astype(int)

    # For post-PA state, we need to look at the start of the next PA
    # Create a mapping from (game_pk, at_bat_number) to the first pitch of next PA

    # Get first pitch of each PA
    first_pitches = pbp.groupby(["game_pk", "at_bat_number"]).first().reset_index()
    first_pitches["next_at_bat"] = first_pitches["at_bat_number"] - 1  # Will merge with previous PA

    # Compute post-base state from the next PA's starting state
    first_pitches["post_base_state_check"] = first_pitches.apply(
        lambda row: encode_base_state(row["on_1b"], row["on_2b"], row["on_3b"]),
        axis=1
    )
    first_pitches["post_outs_check"] = first_pitches["outs_when_up"]

    # Merge to get post-PA state
    next_pa_state = first_pitches[["game_pk", "next_at_bat", "post_base_state_check", "post_outs_check", "inning", "inning_topbot"]].rename(
        columns={"next_at_bat": "at_bat_number", "inning": "next_inning", "inning_topbot": "next_topbot"}
    )

    pa_endings = pa_endings.merge(
        next_pa_state,
        on=["game_pk", "at_bat_number"],
        how="left"
    )

    # Only use transitions where we stay in the same half-inning
    same_half_inning = (
        (pa_endings["inning"] == pa_endings["next_inning"]) &
        (pa_endings["inning_topbot"] == pa_endings["next_topbot"])
    )

    # For end-of-inning transitions (3 outs), the inning changes
    # Handle this separately
    pa_endings["post_base_state"] = pa_endings["post_base_state_check"]
    pa_endings["post_outs"] = pa_endings["post_outs_check"]

    # If inning changed, the play resulted in 3 outs (inning over)
    inning_ended = ~same_half_inning & pa_endings["next_inning"].notna()
    pa_endings.loc[inning_ended, "post_base_state"] = 0
    pa_endings.loc[inning_ended, "post_outs"] = 3  # 3 outs = inning over

    # If this is the last PA of the game, set post state to current outs + delta
    # We'll use a heuristic: if post_outs is missing, estimate from outcome
    missing_post = pa_endings["post_outs"].isna()

    # For missing transitions, use logical defaults based on outcome
    for idx in pa_endings[missing_post].index:
        outcome = pa_endings.loc[idx, "outcome_idx"]
        pre_outs = pa_endings.loc[idx, "outs_when_up"]

        if outcome == 0:  # Strikeout
            post_outs = min(pre_outs + 1, 3)
        elif outcome == 1:  # Walk/HBP
            post_outs = pre_outs
        elif outcome == 2:  # Field out (could be DP)
            # Check if it's a double play event
            event = pa_endings.loc[idx, "events"]
            if "double_play" in str(event).lower():
                post_outs = min(pre_outs + 2, 3)
            else:
                post_outs = min(pre_outs + 1, 3)
        elif outcome == 6:  # Home run
            post_outs = pre_outs
        else:  # Singles, doubles, triples
            post_outs = pre_outs

        pa_endings.loc[idx, "post_outs"] = post_outs

    # Fill missing post_base_state with logical defaults
    for idx in pa_endings[pa_endings["post_base_state"].isna()].index:
        outcome = pa_endings.loc[idx, "outcome_idx"]
        pre_base = pa_endings.loc[idx, "pre_base_state"]

        if outcome == 6:  # Home run - bases empty after
            post_base = 0
        elif outcome == 1:  # Walk/HBP - batter on first
            # Runners advance if forced
            if pre_base == 0:  # Empty -> 1B
                post_base = 1
            elif pre_base == 1:  # 1B -> 1B+2B
                post_base = 3
            elif pre_base == 3:  # 1B+2B -> loaded
                post_base = 7
            elif pre_base == 7:  # Loaded stays loaded
                post_base = 7
            else:
                post_base = pre_base | 1  # Add batter to first
        elif outcome in [3, 4, 5]:  # Hit - batter on base
            if outcome == 3:  # Single
                post_base = 1  # Simplification: batter on first
            elif outcome == 4:  # Double
                post_base = 2  # Batter on second
            else:  # Triple
                post_base = 4  # Batter on third
        else:  # Out
            if pa_endings.loc[idx, "post_outs"] >= 3:
                post_base = 0  # Inning over
            else:
                post_base = pre_base  # Simplification

        pa_endings.loc[idx, "post_base_state"] = post_base

    # Convert to int
    pa_endings["post_base_state"] = pa_endings["post_base_state"].astype(int)
    pa_endings["post_outs"] = pa_endings["post_outs"].astype(int)

    # Cap post_outs at 3 (for inning-ending plays)
    pa_endings["post_outs"] = pa_endings["post_outs"].clip(0, 3)

    # Now build transition tables
    transitions = defaultdict(lambda: defaultdict(int))

    for _, row in pa_endings.iterrows():
        key = (int(row["outcome_idx"]), int(row["outs_when_up"]), int(row["pre_base_state"]))
        result = (int(row["post_base_state"]), int(row["post_outs"]), int(row["runs_scored"]))
        transitions[key][result] += 1

    # Convert counts to probabilities
    transition_tables = {}

    for key, result_counts in transitions.items():
        total = sum(result_counts.values())
        if total < min_observations:
            continue

        probs = []
        for (post_base, post_outs, runs), count in result_counts.items():
            prob = count / total
            probs.append((post_base, post_outs, runs, prob))

        transition_tables[key] = probs

    # Fill missing transitions with reasonable defaults
    transition_tables = _fill_missing_transitions(transition_tables)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(transition_tables, f)
        print(f"Saved transition tables to {save_path}")

    return transition_tables


def _fill_missing_transitions(
    transitions: Dict[Tuple[int, int, int], List[Tuple[int, int, int, float]]]
) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int, float]]]:
    """
    Fill in missing transitions with default logic.

    Creates transitions for all (outcome, outs, base_state) combinations
    using logical rules when empirical data is unavailable.
    """
    filled = dict(transitions)

    for outcome_idx in range(7):
        for outs in range(3):
            for base_state in range(8):
                key = (outcome_idx, outs, base_state)
                if key in filled:
                    continue

                # Generate default transition
                filled[key] = _default_transition(outcome_idx, outs, base_state)

    return filled


def _default_transition(outcome_idx: int, outs: int, base_state: int) -> List[Tuple[int, int, int, float]]:
    """
    Generate default transition for a given (outcome, outs, base_state).

    Returns list of (post_base_state, post_outs, runs_scored, probability).
    """
    # Decode base state
    on_1b = bool(base_state & 1)
    on_2b = bool(base_state & 2)
    on_3b = bool(base_state & 4)
    num_runners = on_1b + on_2b + on_3b

    if outcome_idx == 0:  # Strikeout
        new_outs = min(outs + 1, 3)
        if new_outs == 3:
            return [(0, 3, 0, 1.0)]
        return [(base_state, new_outs, 0, 1.0)]

    elif outcome_idx == 1:  # Walk/HBP
        runs = 0
        new_base = base_state

        if base_state == 7:  # Loaded
            runs = 1  # Runner scores from third
            new_base = 7  # Still loaded

        elif on_1b and on_2b:  # 1B + 2B
            new_base = 7  # Now loaded

        elif on_1b:  # Just 1B
            new_base = 3  # 1B + 2B

        else:
            # Put batter on first
            new_base = base_state | 1

        return [(new_base, outs, runs, 1.0)]

    elif outcome_idx == 2:  # Field out
        new_outs = min(outs + 1, 3)

        if new_outs == 3:
            return [(0, 3, 0, 1.0)]

        # Simple model: runners stay, unless sac fly potential
        if on_3b and outs < 2:
            # Possible sac fly - runner scores
            new_base = base_state & ~4  # Remove runner from third
            return [(new_base, new_outs, 1, 0.5), (base_state, new_outs, 0, 0.5)]

        return [(base_state, new_outs, 0, 1.0)]

    elif outcome_idx == 3:  # Single
        runs = 0
        new_base = 0

        if on_3b:
            runs += 1
        if on_2b:
            runs += 1  # Runner from 2B usually scores on single
        if on_1b:
            new_base |= 4  # Runner from 1B to 3B (conservative)

        new_base |= 1  # Batter on first

        return [(new_base, outs, runs, 1.0)]

    elif outcome_idx == 4:  # Double
        runs = 0

        if on_3b:
            runs += 1
        if on_2b:
            runs += 1
        if on_1b:
            runs += 1  # Runner from 1B usually scores on double

        new_base = 2  # Batter on second

        return [(new_base, outs, runs, 1.0)]

    elif outcome_idx == 5:  # Triple
        runs = num_runners  # All runners score
        new_base = 4  # Batter on third

        return [(new_base, outs, runs, 1.0)]

    elif outcome_idx == 6:  # Home run
        runs = num_runners + 1  # All runners + batter score
        return [(0, outs, runs, 1.0)]

    return [(base_state, outs, 0, 1.0)]


def get_transitions(
    outcome_idx: int,
    outs: int,
    base_state: int,
    transition_tables: Dict,
) -> List[Tuple[int, int, int, float]]:
    """
    Get possible transitions for a given outcome and state.

    Parameters
    ----------
    outcome_idx : int
        Outcome index (0=K, 1=BB_HBP, 2=field_out, 3=1B, 4=2B, 5=3B, 6=HR)
    outs : int
        Current number of outs (0, 1, 2)
    base_state : int
        Current base state (0-7)
    transition_tables : dict
        Transition probability tables from build_transition_tables()

    Returns
    -------
    list
        List of (post_base_state, post_outs, runs_scored, probability)
    """
    key = (outcome_idx, outs, base_state)

    if key in transition_tables:
        return transition_tables[key]

    # Fall back to default
    return _default_transition(outcome_idx, outs, base_state)


# Cache for loaded transition tables
_transition_cache = None


def load_transition_tables(cache_path: Optional[Path] = None) -> Dict:
    """
    Load or build transition tables with caching.
    """
    global _transition_cache

    if _transition_cache is not None:
        return _transition_cache

    default_path = Path(__file__).parent / "transition_tables.pkl"
    cache_path = cache_path or default_path

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _transition_cache = pickle.load(f)
    else:
        _transition_cache = build_transition_tables(save_path=cache_path)

    return _transition_cache


if __name__ == "__main__":
    # Build and save transition tables
    output_path = Path(__file__).parent / "transition_tables.pkl"
    tables = build_transition_tables(save_path=output_path)

    print(f"\nBuilt {len(tables)} transition entries")

    # Print some examples
    print("\nExample transitions:")
    test_cases = [
        (3, 0, 1, "Single with runner on 1B"),
        (6, 1, 7, "Home run, bases loaded, 1 out"),
        (2, 0, 4, "Field out with runner on 3B, 0 out"),
    ]

    for outcome, outs, base, desc in test_cases:
        trans = get_transitions(outcome, outs, base, tables)
        print(f"\n  {desc}:")
        for post_base, post_outs, runs, prob in trans[:3]:
            print(f"    -> base={post_base}, outs={post_outs}, runs={runs}, p={prob:.3f}")
