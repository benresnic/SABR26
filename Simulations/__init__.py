"""
PVC Player Win Probability Simulation Framework

Simulates plate appearance outcomes across all game states and approach changes
to calculate expected win probability deltas for PVC-flagged players.
"""

from .config import (
    COUNTS,
    BASE_STATES,
    XFIP_PERCENTILES,
    BAT_SPEED_CHANGES,
    SWING_LENGTH_CHANGES,
    OUTCOMES,
)
from .win_probability import build_wp_lookup, lookup_wp
from .runner_advancement import build_transition_tables, get_transitions
from .outcome_simulator import compute_expected_delta_wp
from .simulation_engine import run_simulation, get_pvc_players

__all__ = [
    "COUNTS",
    "BASE_STATES",
    "XFIP_PERCENTILES",
    "BAT_SPEED_CHANGES",
    "SWING_LENGTH_CHANGES",
    "OUTCOMES",
    "build_wp_lookup",
    "lookup_wp",
    "build_transition_tables",
    "get_transitions",
    "compute_expected_delta_wp",
    "run_simulation",
    "get_pvc_players",
]
