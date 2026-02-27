import json
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_MAP = {
    "strikeout": 0, "strikeout_double_play": 0,
    "walk": 1, "intent_walk": 1, "hit_by_pitch": 1,
    "field_out": 2, "force_out": 2, "double_play": 2,
    "fielders_choice": 2, "fielders_choice_out": 2,
    "grounded_into_double_play": 2, "sac_fly": 2,
    "sac_bunt": 2, "sac_fly_double_play": 2, "triple_play": 2,
    "single": 3, "double": 4, "triple": 5, "home_run": 6,
}

EXCLUDE_EVENTS = {"catcher_interf", "truncated_pa", "field_error"}

# 11 count dummy columns (reference = 0-0)
ALL_COUNT_COLS = [
    f"count_{b}-{s}"
    for b in range(4)
    for s in range(3)
    if not (b == 0 and s == 0)
]


# ---------------------------------------------------------------------------
# Step 1: prepare_model_data
# ---------------------------------------------------------------------------

def prepare_model_data(data_dir="Data", years=None, subsample_frac=1.0):
    """Lo PBP + stats, filter, featurize, and return arrays for model."""
    if years is None:
        years = [2025]

    # 1. Load and concat PBP
    pbp = pd.concat(
        [pd.read_parquet(f"{data_dir}/pbp_{year}.parquet") for year in years],
        ignore_index=True,
    )

    # 2. Propagate each PA's final event to all pitch rows of that PA
    pa_outcomes = (
        pbp.dropna(subset=["events"])
        [["game_pk", "at_bat_number", "events"]]
        .rename(columns={"events": "pa_outcome"})
    )
    pa = pbp.merge(pa_outcomes, on=["game_pk", "at_bat_number"], how="left")
    pa = pa[pa["pa_outcome"].notna()].copy()
    pa = pa[~pa["pa_outcome"].isin(EXCLUDE_EVENTS)].copy()

    # 4. Map events to integer outcome y
    pa["y"] = pa["pa_outcome"].map(EVENT_MAP)
    pa = pa.dropna(subset=["y"])
    pa["y"] = pa["y"].astype(int)

    # 5. Count encoding: one-hot with 0-0 as reference (11 dummy columns)
    pa["count_str"] = pa["balls"].astype(str) + "-" + pa["strikes"].astype(str)
    count_dummies = pd.get_dummies(pa["count_str"], prefix="count").astype(float)
    if "count_0-0" in count_dummies.columns:
        count_dummies = count_dummies.drop(columns=["count_0-0"])
    for col in ALL_COUNT_COLS:
        if col not in count_dummies.columns:
            count_dummies[col] = 0.0
    count_dummies = count_dummies[ALL_COUNT_COLS]

    # Attach count dummies directly onto pa so they survive all merges/dropna
    pa = pa.reset_index(drop=True)
    for col in ALL_COUNT_COLS:
        pa[col] = count_dummies[col].values

    # 6. Load stats tables
    batter_stats = pd.read_parquet(f"{data_dir}/batter_stats.parquet")
    pitcher_stats = pd.read_parquet(f"{data_dir}/pitcher_stats.parquet")

    batter_stats["xMLBAMID"] = batter_stats["xMLBAMID"].astype("int32")
    batter_stats["Season"] = batter_stats["Season"].astype("int32")

    # 7. Join batter stats onto PA rows
    pa["batter_id"] = pa["batter"].astype("int32")
    pa["game_year_32"] = pa["game_year"].astype("int32")
    pa = pa.merge(
        batter_stats[
            ["xMLBAMID", "Season", "O-Swing_pct", "Z-Contact_pct", "xISO"]
        ].rename(columns={"xMLBAMID": "batter_id", "Season": "game_year_32"}),
        on=["batter_id", "game_year_32"],
        how="left",
    )

    # 8. Join pitcher stats
    pa["pitcher_id"] = pa["pitcher"].astype("int32")
    pitcher_stats["xMLBAMID"] = pitcher_stats["xMLBAMID"].astype("int32")
    pitcher_stats["Season"] = pitcher_stats["Season"].astype("int32")
    pa = pa.merge(
        pitcher_stats[["xMLBAMID", "Season", "xFIP"]].rename(
            columns={"xMLBAMID": "pitcher_id", "Season": "game_year_32"}
        ),
        on=["pitcher_id", "game_year_32"],
        how="left",
    )

    # 9. Drop rows with nulls in continuous predictors
    cont_cols = ["O-Swing_pct", "Z-Contact_pct", "xISO", "xFIP"]
    pa = pa.dropna(subset=cont_cols)
    pa = pa.reset_index(drop=True)

    # 10b. Stratified subsample (optional) — preserves count × outcome distribution
    if subsample_frac < 1.0:
        pa = (
            pa.groupby(["count_str", "y"], group_keys=False)
            .sample(frac=subsample_frac, random_state=42)
            .reset_index(drop=True)
        )

    # 11. Standardize continuous features; store scalers
    scalers = {}
    feat_cols = ["O-Swing_pct", "Z-Contact_pct", "xISO", "xFIP"]
    for feat in feat_cols:
        mu = float(pa[feat].mean())
        sigma = float(pa[feat].std())
        scalers[feat] = {"mean": mu, "std": sigma}
        pa[feat] = (pa[feat] - mu) / sigma

    return {
        "y": pa["y"].values.astype("int32"),
        "count_X": pa[ALL_COUNT_COLS].values.astype("float32"),
        "o_swing": pa["O-Swing_pct"].values.astype("float32"),
        "z_contact": pa["Z-Contact_pct"].values.astype("float32"),
        "xiso": pa["xISO"].values.astype("float32"),
        "xfip": pa["xFIP"].values.astype("float32"),
        "N": len(pa),
        "K": 7,
        "J": 4,
        "scalers": scalers,
    }


# ---------------------------------------------------------------------------
# Step 2: build_model
# ---------------------------------------------------------------------------

def build_model(data):
    """Construct PyMC multinomial logistic regression model."""
    N = data["N"]

    with pm.Model() as model:
        # --- Data ---
        count_X = pm.Data("count_X", data["count_X"])
        o_swing = pm.Data("o_swing", data["o_swing"])
        z_contact = pm.Data("z_contact", data["z_contact"])
        xiso = pm.Data("xiso", data["xiso"])
        xfip = pm.Data("xfip", data["xfip"])
        y_obs = pm.Data("y_obs", data["y"])

        # --- Global priors ---
        alpha = pm.Normal("alpha", 0, 2, shape=(6,))
        beta_count = pm.Normal("beta_count", 0, 2, shape=(11, 6))
        beta_o_swing = pm.Normal("beta_o_swing", 0, 2, shape=(6,))
        beta_z_contact = pm.Normal("beta_z_contact", 0, 2, shape=(6,))
        beta_xiso = pm.Normal("beta_xiso", 0, 2, shape=(6,))
        beta_xfip = pm.Normal("beta_xfip", 0, 2, shape=(6,))

        # --- Linear predictor: (N, 6) for non-reference outcomes ---
        logit_mat = (
            alpha[None, :]                               # (1, 6) broadcast
            + pt.dot(count_X, beta_count)               # (N, 11) @ (11, 6)
            + o_swing[:, None] * beta_o_swing
            + z_contact[:, None] * beta_z_contact
            + xiso[:, None] * beta_xiso
            + xfip[:, None] * beta_xfip
        )

        # --- Insert field_out zero column at position 2 (y=2) ---
        ref_col = pt.zeros_like(logit_mat[:, 0:1])     # (N, 1)
        logit_full = pt.concatenate([
            logit_mat[:, 0:2],   # K, BB_HBP
            ref_col,             # field_out (reference)
            logit_mat[:, 2:6],   # 1B, 2B, 3B, HR
        ], axis=1)               # (N, 7)

        # --- Likelihood ---
        pm.Categorical("obs", p=pm.math.softmax(logit_full, axis=1), observed=y_obs)

    return model


# ---------------------------------------------------------------------------
# Step 3: fit_model
# ---------------------------------------------------------------------------

def fit_model(
    model,
    draws=2000,
    tune=1000,
    target_accept=0.99,
    chains=4,
    cores=4,
    random_seed=1738,
    nuts_sampler="pymc",
):
    """Sample from the model and return InferenceData."""
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
            idata_kwargs={"log_likelihood": False},
            nuts_sampler=nuts_sampler,
        )

    summary = az.summary(
        idata,
        var_names=["alpha", "beta_count"],
    )
    print(summary)
    return idata


# ---------------------------------------------------------------------------
# Step 4: predict_outcome_probs
# ---------------------------------------------------------------------------

def predict_outcome_probs(
    new_df,
    idata,
    data_dict,
    batter_stats_df,
    pitcher_stats_df,
    n_samples=500):
    """
    Compute posterior mean outcome probabilities for rows in new_df.

    Parameters
    ----------
    new_df : pd.DataFrame
        Must contain columns: batter, pitcher, game_year, balls, strikes.
        If swing_length is present, a per-batter-season mean is computed.
    idata : az.InferenceData
        Posterior samples from fit_model.
    data_dict : dict
        Output of prepare_model_data (for scalers).
    batter_stats_df : pd.DataFrame
        Raw batter stats (xMLBAMID, Season, O-Swing_pct, Z-Contact_pct, ...).
    pitcher_stats_df : pd.DataFrame
        Raw pitcher stats (xMLBAMID, Season, xFIP).
    n_samples : int
        Number of posterior draws to average over.

    Returns
    -------
    np.ndarray, shape (M, 7)
        Posterior mean probabilities for [K, BB_HBP, field_out, 1B, 2B, 3B, HR].
    """
    new_df = new_df.copy()
    scalers = data_dict["scalers"]

    # 1. Count dummies
    new_df["count_str"] = new_df["balls"].astype(str) + "-" + new_df["strikes"].astype(str)
    count_dummies = pd.get_dummies(new_df["count_str"], prefix="count").astype(float)
    for col in ALL_COUNT_COLS:
        if col not in count_dummies.columns:
            count_dummies[col] = 0.0
    count_dummies = count_dummies[ALL_COUNT_COLS].reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)

    # 2. Prepare batter stats
    batter_stats_df = batter_stats_df.copy()
    batter_stats_df["xMLBAMID"] = batter_stats_df["xMLBAMID"].astype("int32")
    batter_stats_df["Season"] = batter_stats_df["Season"].astype("int32")

    # 3. Join batter stats
    new_df["batter_id"] = new_df["batter"].astype("int32")
    new_df["game_year_32"] = new_df["game_year"].astype("int32")
    new_df = new_df.merge(
        batter_stats_df[
            ["xMLBAMID", "Season", "O-Swing_pct", "Z-Contact_pct", "xISO"]
        ].rename(columns={"xMLBAMID": "batter_id", "Season": "game_year_32"}),
        on=["batter_id", "game_year_32"],
        how="left",
    )

    # 4. Join pitcher stats
    pitcher_stats_df = pitcher_stats_df.copy()
    pitcher_stats_df["xMLBAMID"] = pitcher_stats_df["xMLBAMID"].astype("int32")
    pitcher_stats_df["Season"] = pitcher_stats_df["Season"].astype("int32")
    new_df["pitcher_id"] = new_df["pitcher"].astype("int32")
    new_df = new_df.merge(
        pitcher_stats_df[["xMLBAMID", "Season", "xFIP"]].rename(
            columns={"xMLBAMID": "pitcher_id", "Season": "game_year_32"}
        ),
        on=["pitcher_id", "game_year_32"],
        how="left",
    )

    # 5. Apply stored scalers (fill missing with training mean → 0 after standardization)
    feat_map = {
        "O-Swing_pct": "o_swing_s",
        "Z-Contact_pct": "zc_s",
        "xISO": "xiso_s",
        "xFIP": "xfip_s",
    }
    for feat, col in feat_map.items():
        mu = scalers[feat]["mean"]
        sigma = scalers[feat]["std"]
        new_df[col] = (new_df[feat].fillna(mu) - mu) / sigma

    M = len(new_df)

    count_X = count_dummies.values.astype("float32")
    o_swing = new_df["o_swing_s"].values.astype("float32")
    z_contact = new_df["zc_s"].values.astype("float32")
    xiso = new_df["xiso_s"].values.astype("float32")
    xfip = new_df["xfip_s"].values.astype("float32")

    # 7. Extract and subsample posterior draws
    post = idata.posterior

    def get_flat(name):
        arr = post[name].values  # (chains, draws, ...)
        return arr.reshape(-1, *arr.shape[2:])

    total_draws = get_flat("alpha").shape[0]
    rng = np.random.default_rng(42)
    draw_idx = rng.choice(total_draws, size=min(n_samples, total_draws), replace=False)
    S = len(draw_idx)

    alpha_s = get_flat("alpha")[draw_idx]            # (S, 6)
    beta_count_s = get_flat("beta_count")[draw_idx]  # (S, 11, 6)
    beta_o_swing_s = get_flat("beta_o_swing")[draw_idx]
    beta_zc_s = get_flat("beta_z_contact")[draw_idx]
    beta_xi_s = get_flat("beta_xiso")[draw_idx]
    beta_xf_s = get_flat("beta_xfip")[draw_idx]

    # 8. Compute probabilities for each posterior draw
    all_probs = np.empty((S, M, 7), dtype=np.float32)

    for s in range(S):
        logit_mat = (
            alpha_s[s][None, :]
            + count_X @ beta_count_s[s]
            + o_swing[:, None] * beta_o_swing_s[s]
            + z_contact[:, None] * beta_zc_s[s]
            + xiso[:, None] * beta_xi_s[s]
            + xfip[:, None] * beta_xf_s[s]
        )  # (M, 6)

        # Insert field_out zero column at position 2
        logit_full = np.concatenate([
            logit_mat[:, 0:2],
            np.zeros((M, 1), dtype=np.float32),
            logit_mat[:, 2:6],
        ], axis=1)  # (M, 7)

        # Numerically stable softmax
        logit_full -= logit_full.max(axis=1, keepdims=True)
        exp_l = np.exp(logit_full)
        all_probs[s] = exp_l / exp_l.sum(axis=1, keepdims=True)

    return all_probs.mean(axis=0)  # (M, 7)



if __name__ == "__main__":
    data = prepare_model_data(data_dir="Data", years=[2024], subsample_frac=0.5)
    with open("Bayes Outcomes/scalers.json", "w") as f:
        json.dump(data["scalers"], f)
    model = build_model(data)
    idata = fit_model(model, draws=2000, tune=1000,nuts_sampler="numpyro")
    az.to_netcdf(idata, "Bayes Outcomes/bayes_outcome_model.nc")

