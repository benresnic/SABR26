"""
validate_predictions.py

Validate that predicted outcome probabilities match observed frequencies.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from outcome_probs import (
    prepare_model_data,
    predict_outcome_probs,
    EVENT_MAP,
    EXCLUDE_EVENTS,
    ALL_COUNT_COLS,
)

DATA_DIR = Path(__file__).parent.parent / "Data"
BAYES_DIR = Path(__file__).parent

OUTCOMES = ["K", "BB_HBP", "field_out", "1B", "2B", "3B", "HR"]


def load_validation_data(years=[2025], subsample_frac=1.0):
    """
    Load PBP data for validation.
    Returns DataFrame with actual outcomes and features needed for prediction.
    """
    pbp = pd.concat(
        [pd.read_parquet(DATA_DIR / f"pbp_{year}.parquet") for year in years],
        ignore_index=True,
    )

    # Get PA-ending events
    pa_outcomes = (
        pbp.dropna(subset=["events"])
        [["game_pk", "at_bat_number", "events"]]
        .rename(columns={"events": "pa_outcome"})
    )
    pa = pbp.merge(pa_outcomes, on=["game_pk", "at_bat_number"], how="left")
    pa = pa[pa["pa_outcome"].notna()].copy()
    pa = pa[~pa["pa_outcome"].isin(EXCLUDE_EVENTS)].copy()

    # Map events to outcome index
    pa["y"] = pa["pa_outcome"].map(EVENT_MAP)
    pa = pa.dropna(subset=["y"])
    pa["y"] = pa["y"].astype(int)

    # Keep only the final pitch of each PA
    pa = pa.groupby(["game_pk", "at_bat_number"]).last().reset_index()

    # Subsample if requested
    if subsample_frac < 1.0:
        pa = pa.sample(frac=subsample_frac, random_state=42).reset_index(drop=True)

    return pa


def get_predictions_for_validation(pa_df, idata, scalers):
    """
    Get predicted probabilities for each PA in validation set.
    """
    batter_stats = pd.read_parquet(DATA_DIR / "batter_stats.parquet")
    pitcher_stats = pd.read_parquet(DATA_DIR / "pitcher_stats.parquet")

    # Prepare input DataFrame
    input_df = pa_df[["batter", "pitcher", "game_year", "balls", "strikes"]].copy()

    # Create dummy data_dict with scalers
    data_dict = {"scalers": scalers}

    probs = predict_outcome_probs(
        input_df,
        idata,
        data_dict,
        batter_stats,
        pitcher_stats,
        n_samples=200,
    )

    return probs


def plot_calibration(y_true, y_pred_probs, outcome_names, n_bins=50, save_path=None):
    """
    Plot calibration curves for each outcome.

    For well-calibrated predictions, the curve should follow the diagonal.
    """
    n_outcomes = len(outcome_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (outcome, ax) in enumerate(zip(outcome_names, axes[:n_outcomes])):
        # Binary indicator for this outcome
        y_binary = (y_true == i).astype(int)
        y_prob = y_pred_probs[:, i]

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_binary, y_prob, n_bins=n_bins, strategy='quantile')

            ax.plot(prob_pred, prob_true, "s-", label="Model", color="#F76900")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect")

            # Brier score
            brier = brier_score_loss(y_binary, y_prob)
            ax.set_title(f"{outcome}\nBrier: {brier:.4f}")
        except ValueError as e:
            ax.set_title(f"{outcome}\n(insufficient data)")

        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    axes[-1].axis("off")

    plt.suptitle("Calibration Curves: Predicted vs Observed Probability", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_predicted_vs_observed_by_bin(y_true, y_pred_probs, outcome_names, n_bins=250, save_path=None):
    """
    Bin predictions and compare mean predicted prob to observed frequency.
    """
    n_outcomes = len(outcome_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (outcome, ax) in enumerate(zip(outcome_names, axes[:n_outcomes])):
        y_binary = (y_true == i).astype(int)
        y_prob = y_pred_probs[:, i]

        # Bin by predicted probability (quantile-based for equal sample sizes)
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins[-1] += 0.001  # Ensure max value included
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        pred_means = []
        obs_means = []
        counts = []

        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                pred_means.append(y_prob[mask].mean())
                obs_means.append(y_binary[mask].mean())
                counts.append(mask.sum())

        pred_means = np.array(pred_means)
        obs_means = np.array(obs_means)

        ax.scatter(pred_means, obs_means, s=50, alpha=0.7)
        ax.plot([0, max(pred_means.max(), obs_means.max())],
                [0, max(pred_means.max(), obs_means.max())], "k--", label="Perfect")

        # Correlation
        if len(pred_means) > 1:
            corr = np.corrcoef(pred_means, obs_means)[0, 1]
            ax.set_title(f"{outcome}\nr = {corr:.3f}")
        else:
            ax.set_title(outcome)

        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Observed frequency")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")

    plt.suptitle("Predicted vs Observed by Probability Bin", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_outcome_distribution(y_true, y_pred_probs, outcome_names, save_path=None):
    """
    Compare overall predicted vs observed outcome distributions.
    """
    n_outcomes = len(outcome_names)

    # Observed frequencies
    obs_counts = np.bincount(y_true, minlength=n_outcomes)
    obs_freq = obs_counts / obs_counts.sum()

    # Mean predicted probabilities
    pred_freq = y_pred_probs.mean(axis=0)

    x = np.arange(n_outcomes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, obs_freq, width, label="Observed", color="#000E54")
    bars2 = ax.bar(x + width/2, pred_freq, width, label="Predicted", color="#F76900")

    ax.set_xlabel("Outcome")
    ax.set_ylabel("Frequency")
    ax.set_title("Observed vs Predicted Outcome Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, obs_freq):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, pred_freq):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compute_validation_metrics(y_true, y_pred_probs, outcome_names):
    """
    Compute validation metrics for each outcome.
    """
    metrics = []

    for i, outcome in enumerate(outcome_names):
        y_binary = (y_true == i).astype(int)
        y_prob = y_pred_probs[:, i]

        # Brier score
        brier = brier_score_loss(y_binary, y_prob)

        # Observed and predicted frequency
        obs_freq = y_binary.mean()
        pred_freq = y_prob.mean()

        # Log loss for this outcome
        eps = 1e-10
        log_loss = -np.mean(
            y_binary * np.log(y_prob + eps) +
            (1 - y_binary) * np.log(1 - y_prob + eps)
        )

        metrics.append({
            "outcome": outcome,
            "observed_freq": obs_freq,
            "predicted_freq": pred_freq,
            "freq_diff": pred_freq - obs_freq,
            "brier_score": brier,
            "log_loss": log_loss,
        })

    return pd.DataFrame(metrics)


def run_validation(years=[2025], subsample_frac=0.2, save_plots=True):
    """
    Run full validation pipeline.
    """
    print("Loading model and scalers...")
    idata = az.from_netcdf(BAYES_DIR / "bayes_outcome_model.nc")
    with open(BAYES_DIR / "scalers.json") as f:
        scalers = json.load(f)

    print(f"Loading validation data (years={years}, subsample={subsample_frac})...")
    pa_df = load_validation_data(years=years, subsample_frac=subsample_frac)
    print(f"  {len(pa_df)} plate appearances")

    print("Getting predictions...")
    pred_probs = get_predictions_for_validation(pa_df, idata, scalers)

    # Filter to rows that got predictions (some may be dropped due to missing stats)
    valid_mask = ~np.isnan(pred_probs).any(axis=1)
    pred_probs = pred_probs[valid_mask]
    y_true = pa_df.loc[valid_mask, "y"].values

    print(f"  {len(y_true)} PAs with valid predictions")

    # Compute metrics
    print("\nValidation Metrics:")
    metrics_df = compute_validation_metrics(y_true, pred_probs, OUTCOMES)
    print(metrics_df.to_string(index=False))

    # Overall Brier score
    overall_brier = sum(
        brier_score_loss((y_true == i).astype(int), pred_probs[:, i])
        for i in range(len(OUTCOMES))
    ) / len(OUTCOMES)
    print(f"\nOverall mean Brier score: {overall_brier:.4f}")

    # Plots
    if save_plots:
        plot_dir = BAYES_DIR / "validation_plots"
        plot_dir.mkdir(exist_ok=True)

        print("\nGenerating plots...")
        plot_calibration(y_true, pred_probs, OUTCOMES,
                        save_path=plot_dir / "calibration_curves.png")
        plot_predicted_vs_observed_by_bin(y_true, pred_probs, OUTCOMES,
                                          save_path=plot_dir / "pred_vs_obs_bins.png")
        plot_outcome_distribution(y_true, pred_probs, OUTCOMES,
                                  save_path=plot_dir / "outcome_distribution.png")

    return metrics_df, y_true, pred_probs


if __name__ == "__main__":
    metrics, y_true, pred_probs = run_validation(
        years=[2025],
        subsample_frac=.2,
        save_plots=True,
    )
    plt.show()
