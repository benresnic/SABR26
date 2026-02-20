"""
config.py

Constants and configuration for the PVC simulation framework.
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
BAYES_DIR = PROJECT_ROOT / "Bayes Outcomes"
OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# State Space Constants
# ---------------------------------------------------------------------------

# 12 counts (balls-strikes)
COUNTS = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2"
]

# 8 base states encoded as integers
# Binary representation: bit 0 = 1B, bit 1 = 2B, bit 2 = 3B
BASE_STATES = {
    0: "empty",   # 000 - no runners
    1: "1B",      # 001 - runner on 1st
    2: "2B",      # 010 - runner on 2nd
    3: "1B_2B",   # 011 - runners on 1st and 2nd
    4: "3B",      # 100 - runner on 3rd
    5: "1B_3B",   # 101 - runners on 1st and 3rd
    6: "2B_3B",   # 110 - runners on 2nd and 3rd
    7: "loaded",  # 111 - bases loaded
}

# List version for indexing
BASE_STATE_NAMES = ["empty", "1B", "2B", "1B_2B", "3B", "1B_3B", "2B_3B", "loaded"]

# Outs: 0, 1, 2
OUTS = [0, 1, 2]

# Innings 7-9 (late-game high-leverage situations)
INNINGS = [7, 8, 9]

# Top/Bottom inning
INNING_TOPBOT = ["Top", "Bot"]

# Score differential from batter's perspective: -4 to +2
SCORE_DIFFS = list(range(-4, 3))  # [-4, -3, -2, -1, 0, 1, 2]

# Pitcher xFIP percentiles to simulate against
XFIP_PERCENTILES = [10, 30, 50, 70, 90]

# ---------------------------------------------------------------------------
# Approach Grid
# ---------------------------------------------------------------------------

# 5x5 grid: bat speed change (%) x swing length change (%)
# Values: 0%, -3%, -6%, -9%, -12% on each axis
# (0, 0) = baseline, 25 grid points total
BAT_SPEED_CHANGES = [-0.03 * i for i in range(5)]    # [0.0, -0.03, -0.06, -0.09, -0.12]
SWING_LENGTH_CHANGES = [-0.03 * i for i in range(5)] # [0.0, -0.03, -0.06, -0.09, -0.12]

# ---------------------------------------------------------------------------
# Outcome Categories
# ---------------------------------------------------------------------------

# Matches EVENT_MAP from outcome_probs.py
OUTCOMES = ["K", "BB_HBP", "field_out", "1B", "2B", "3B", "HR"]
OUTCOME_INDICES = {outcome: i for i, outcome in enumerate(OUTCOMES)}

# ---------------------------------------------------------------------------
# Simulation Parameters
# ---------------------------------------------------------------------------

# Number of posterior samples to use for outcome probability prediction
N_POSTERIOR_SAMPLES = 500

# Default years for building transition tables and WP lookup
DEFAULT_YEARS = [2023, 2024, 2025]

# Years for WP lookup (more historical data for stability)
WP_LOOKUP_YEARS = [2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Feasibility Masking
# ---------------------------------------------------------------------------

# Fallback covariance matrix for players with <30 swings
# Computed from 791,927 swings across 865 batters (2023-2025)
# [[var_bat_speed, cov], [cov, var_swing_length]]
POPULATION_COV_MATRIX = np.array([
    [83.7, 4.86],
    [4.86, 0.91]
])

# Minimum swings required for player-specific covariance
MIN_SWINGS_FOR_PLAYER_COV = 30

# Mask out grid cells below this fraction of max bivariate normal density
# threshold = 0.25 means feasible if within region containing 75% of swings
FEASIBILITY_THRESHOLD = 0.25
