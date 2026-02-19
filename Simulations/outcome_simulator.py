"""
outcome_simulator.py

Compute expected delta win probability using closed-form expected value calculation.
No Monte Carlo simulation needed - we use exact probability-weighted sums.
"""

import numpy as np
from typing import Dict, List, Tuple

from .win_probability import lookup_wp
from .runner_advancement import get_transitions


def compute_expected_delta_wp(
    outcome_probs: np.ndarray,
    outs: int,
    base_state: int,
    inning: int,
    inning_topbot: str,
    score_diff: int,
    wp_table: np.ndarray,
    transition_tables: Dict,
) -> Tuple[float, float, float]:
    """
    Compute expected win probability delta for given outcome probabilities and state.

    Uses exact expected value calculation:
    delta_wp = sum(p_outcome * E[delta_wp | outcome, state])

    where E[delta_wp | outcome, state] = sum(p_transition * (new_wp - current_wp))

    Parameters
    ----------
    outcome_probs : np.ndarray
        Probability of each outcome [K, BB_HBP, field_out, 1B, 2B, 3B, HR], shape (7,)
    outs : int
        Current number of outs (0, 1, 2)
    base_state : int
        Current base state (0-7)
    inning : int
        Current inning (1-9)
    inning_topbot : str
        "Top" or "Bot"
    score_diff : int
        Score differential from batting team's perspective
    wp_table : np.ndarray
        Win probability lookup table
    transition_tables : dict
        State transition tables from runner_advancement

    Returns
    -------
    tuple
        (current_wp, expected_new_wp, delta_wp)
    """
    # Get current win probability
    current_wp = lookup_wp(outs, base_state, inning, inning_topbot, score_diff, wp_table)

    # Compute expected new WP
    expected_new_wp = 0.0

    for outcome_idx, p_outcome in enumerate(outcome_probs):
        if p_outcome < 1e-10:
            continue

        # Get possible transitions for this outcome
        transitions = get_transitions(outcome_idx, outs, base_state, transition_tables)

        for post_base, post_outs, runs_scored, p_trans in transitions:
            # Update score differential
            new_score_diff = score_diff + runs_scored

            if post_outs >= 3:
                # Inning is over - switch perspective
                # At end of top of inning, batting team becomes fielding team
                # At end of bottom, vice versa
                if inning_topbot == "Top":
                    # Was batting in top, now fielding in bottom
                    # Score diff stays same (from original batting team's perspective)
                    new_topbot = "Bot"
                    new_inning = inning
                else:
                    # Was batting in bottom, now fielding in top of next inning
                    new_topbot = "Top"
                    new_inning = min(inning + 1, 9)

                # When team switches from batting to fielding, WP lookup uses
                # the opponent's perspective. We need to flip.
                # Actually, bat_win_exp is always from batting team's view.
                # When we become the fielding team, our WP is 1 - batting_team_wp
                # But simpler: after 3 outs, we look up WP for the new batting team
                # which is our opponent, so our WP = 1 - that value

                opponent_wp = lookup_wp(0, 0, new_inning, new_topbot, -new_score_diff, wp_table)
                new_wp = 1.0 - opponent_wp
            else:
                # Same half-inning continues
                new_wp = lookup_wp(post_outs, post_base, inning, inning_topbot, new_score_diff, wp_table)

            expected_new_wp += p_outcome * p_trans * new_wp

    delta_wp = expected_new_wp - current_wp

    return current_wp, expected_new_wp, delta_wp


def compute_delta_wp_batch(
    outcome_probs_batch: np.ndarray,
    states_df,
    wp_table: np.ndarray,
    transition_tables: Dict,
) -> np.ndarray:
    """
    Compute expected delta WP for a batch of states.

    Parameters
    ----------
    outcome_probs_batch : np.ndarray
        Shape (N, 7) - outcome probabilities for each state
    states_df : pd.DataFrame
        DataFrame with columns: outs, base_state, inning, inning_topbot, score_diff
    wp_table : np.ndarray
        Win probability lookup table
    transition_tables : dict
        State transition tables

    Returns
    -------
    np.ndarray
        Shape (N, 3) - columns: current_wp, expected_new_wp, delta_wp
    """
    n_states = len(states_df)
    results = np.zeros((n_states, 3), dtype=np.float32)

    for i in range(n_states):
        row = states_df.iloc[i]
        current_wp, expected_new_wp, delta_wp = compute_expected_delta_wp(
            outcome_probs=outcome_probs_batch[i],
            outs=int(row["outs"]),
            base_state=int(row["base_state"]),
            inning=int(row["inning"]),
            inning_topbot=str(row["inning_topbot"]),
            score_diff=int(row["score_diff"]),
            wp_table=wp_table,
            transition_tables=transition_tables,
        )
        results[i, 0] = current_wp
        results[i, 1] = expected_new_wp
        results[i, 2] = delta_wp

    return results


def compute_outcome_delta_wp(
    outcome_idx: int,
    outs: int,
    base_state: int,
    inning: int,
    inning_topbot: str,
    score_diff: int,
    wp_table: np.ndarray,
    transition_tables: Dict,
) -> float:
    """
    Compute expected delta WP for a specific outcome.

    This gives E[delta_wp | outcome, state], which is the expected change
    in win probability given that a specific outcome occurs.

    Parameters
    ----------
    outcome_idx : int
        Outcome index (0=K, 1=BB_HBP, 2=field_out, 3=1B, 4=2B, 5=3B, 6=HR)
    outs : int
        Current number of outs
    base_state : int
        Current base state (0-7)
    inning : int
        Current inning
    inning_topbot : str
        "Top" or "Bot"
    score_diff : int
        Score differential from batting team's perspective
    wp_table : np.ndarray
        Win probability lookup table
    transition_tables : dict
        State transition tables

    Returns
    -------
    float
        Expected delta WP for this specific outcome
    """
    current_wp = lookup_wp(outs, base_state, inning, inning_topbot, score_diff, wp_table)

    transitions = get_transitions(outcome_idx, outs, base_state, transition_tables)

    expected_new_wp = 0.0

    for post_base, post_outs, runs_scored, p_trans in transitions:
        new_score_diff = score_diff + runs_scored

        if post_outs >= 3:
            if inning_topbot == "Top":
                new_topbot = "Bot"
                new_inning = inning
            else:
                new_topbot = "Top"
                new_inning = min(inning + 1, 9)

            opponent_wp = lookup_wp(0, 0, new_inning, new_topbot, -new_score_diff, wp_table)
            new_wp = 1.0 - opponent_wp
        else:
            new_wp = lookup_wp(post_outs, post_base, inning, inning_topbot, new_score_diff, wp_table)

        expected_new_wp += p_trans * new_wp

    return expected_new_wp - current_wp


if __name__ == "__main__":
    from .win_probability import build_wp_lookup
    from .runner_advancement import build_transition_tables

    # Build lookup tables
    print("Building WP table...")
    wp_table = build_wp_lookup()

    print("Building transition tables...")
    trans_tables = build_transition_tables()

    # Test with some example states
    print("\nExample delta WP calculations:")

    # Baseline probabilities (roughly league average)
    baseline_probs = np.array([0.22, 0.09, 0.40, 0.15, 0.05, 0.01, 0.08])

    test_states = [
        (0, 0, 1, "Top", 0, "Start of game"),
        (0, 7, 9, "Bot", -1, "Bases loaded, bottom 9, down 1"),
        (2, 0, 9, "Bot", 1, "2 outs, nobody on, bottom 9, up 1"),
        (1, 5, 7, "Top", 0, "1 out, 1B+3B, 7th inning, tied"),
    ]

    for outs, base, inning, topbot, diff, desc in test_states:
        current_wp, expected_wp, delta_wp = compute_expected_delta_wp(
            baseline_probs, outs, base, inning, topbot, diff, wp_table, trans_tables
        )
        print(f"\n  {desc}:")
        print(f"    Current WP: {current_wp:.3f}")
        print(f"    Expected new WP: {expected_wp:.3f}")
        print(f"    Delta WP: {delta_wp:+.4f}")
