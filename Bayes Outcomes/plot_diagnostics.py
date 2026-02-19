"""
Model validation plots for the multinomial PA outcome model.

Usage
-----
    import arviz as az
    from plot_diagnostics import (
        plot_trace,
        plot_rhat,
        plot_energy,
        plot_alpha_forest,
        plot_count_effects,
        plot_coef_effects,
        plot_baseline_probs,
        plot_all,
    )

    idata = az.from_netcdf("Bayes Outcomes/bayes_outcome_model.nc")
    plot_all(idata, save_dir="Bayes Outcomes/plots")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import arviz as az
from pathlib import Path

# ---------------------------------------------------------------------------
# Label constants (match model structure)
# ---------------------------------------------------------------------------

# 6 non-reference outcome labels (field_out is reference → zero column)
OUTCOME_LABELS_6 = ["K", "BB/HBP", "1B", "2B", "3B", "HR"]

# Full 7-outcome labels in model order
OUTCOME_LABELS_7 = ["K", "BB/HBP", "field_out", "1B", "2B", "3B", "HR"]

# 11 count dummy labels (reference = 0-0)
COUNT_LABELS = [
    f"{b}-{s}"
    for b in range(4)
    for s in range(3)
    if not (b == 0 and s == 0)
]

# Continuous predictor labels
CONT_BETAS = {
    "beta_o_swing":   "O-Swing %",
    "beta_z_contact": "Z-Contact %",
    "beta_xiso":      "xISO",
    "beta_xfip":      "xFIP",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _logodds_to_pct_odds_shift(x):
    """
    Convert log-odds coefficient(s) to percent odds shift:
        100 * (exp(beta) - 1)
    """
    x = np.clip(x, -20, 20)  # avoid numerical overflow in exp
    return 100.0 * (np.exp(x) - 1.0)


def _maybe_save(fig, save_dir, fname):
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / fname, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Trace plots
# ---------------------------------------------------------------------------

def plot_trace(idata, save_dir=None):
    """
    Trace + marginal density for alpha and all continuous betas.
    beta_count is excluded — it has 66 scalar parameters and would overflow
    the figure; its convergence is fully captured by the R-hat plot.
    """
    var_names = ["alpha"] + list(CONT_BETAS.keys())
    n_vars = sum(
        int(np.prod(idata.posterior[v].shape[2:]))
        for v in var_names
    )
    axes = az.plot_trace(
        idata,
        var_names=var_names,
        figsize=(14, 2.5 * n_vars),
        compact=True,
    )
    fig = axes.ravel()[0].get_figure()
    fig.suptitle("Trace plots — alpha & continuous betas", y=1.005, fontsize=13)
    fig.subplots_adjust(hspace=0.4)
    _maybe_save(fig, save_dir, "trace.png")
    return fig


# ---------------------------------------------------------------------------
# 2. R-hat bar chart
# ---------------------------------------------------------------------------

def plot_rhat(idata, threshold=1.01, save_dir=None):
    """
    Horizontal bar chart of R-hat for every scalar parameter.
    Bars exceeding `threshold` are highlighted in red.
    """
    summary = az.summary(idata, kind="diagnostics")
    r_hat = summary["r_hat"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(r_hat) * 0.25)))
    colors = ["#d62728" if v > threshold else "#1f77b4" for v in r_hat]
    ax.barh(range(len(r_hat)), r_hat.values, color=colors, height=0.7)
    ax.set_yticks(range(len(r_hat)))
    ax.set_yticklabels(r_hat.index, fontsize=7)
    ax.axvline(threshold, color="red", linewidth=1.2, linestyle="--",
               label=f"threshold = {threshold}")
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("R-hat")
    ax.set_title("R-hat — all parameters")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _maybe_save(fig, save_dir, "rhat.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Energy plot (BFMI / sampler health)
# ---------------------------------------------------------------------------

def plot_energy(idata, save_dir=None):
    """
    Energy transition distribution vs. marginal energy distribution.
    BFMI < 0.3 indicates sampler trouble.
    """
    axes = az.plot_energy(idata, figsize=(8, 4))
    fig = axes.get_figure()
    fig.suptitle("Energy plot — NUTS sampler health", fontsize=12)
    fig.tight_layout()
    _maybe_save(fig, save_dir, "energy.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Forest plot — alpha (baseline odds shift %)
# ---------------------------------------------------------------------------

def plot_alpha_forest(idata, save_dir=None):
    """
    Forest-style plot of alpha[0..5] as percent odds shifts with 94% HDI.
    Each alpha is the baseline odds shift of that outcome vs. field_out at 0-0,
    with all continuous features at their training mean.
    """
    alpha_draws = idata.posterior["alpha"].values.reshape(-1, 6)  # (S, 6)
    alpha_pct = _logodds_to_pct_odds_shift(alpha_draws)

    med = np.median(alpha_pct, axis=0)
    lo = np.percentile(alpha_pct, 3, axis=0)
    hi = np.percentile(alpha_pct, 97, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(6)
    ax.barh(
        y,
        med,
        xerr=[med - lo, hi - med],
        color=["#d62728" if m > 0 else "#1f77b4" for m in med],
        error_kw=dict(ecolor="black", lw=1.2, capsize=3),
        height=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(OUTCOME_LABELS_6, fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Baseline odds shift (%) vs. field_out")
    ax.set_title("alpha — baseline odds shift vs. field_out\n(0-0 count, avg batter & pitcher)")
    fig.tight_layout()
    _maybe_save(fig, save_dir, "alpha_forest.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Count-effect heatmap (beta_count)
# ---------------------------------------------------------------------------

def plot_count_effects(idata, save_dir=None):
    """
    Heatmap of posterior median count effects as percent odds shifts.
    Rows = 11 count states, columns = 6 non-reference outcomes.
    """
    post = idata.posterior
    beta_count = post["beta_count"].values.reshape(-1, 11, 6)  # (S, 11, 6)
    beta_count_pct = _logodds_to_pct_odds_shift(beta_count)
    med_bc = np.median(beta_count_pct, axis=0)  # (11, 6)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(med_bc).max()
    im = ax.imshow(med_bc, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Posterior median odds shift (%)")

    ax.set_xticks(range(6))
    ax.set_xticklabels(OUTCOME_LABELS_6, fontsize=10)
    ax.set_yticks(range(11))
    ax.set_yticklabels(COUNT_LABELS, fontsize=9)
    ax.set_xlabel("Outcome (vs. field_out)")
    ax.set_ylabel("Count (vs. 0-0)")
    ax.set_title("beta_count — posterior median\ncount-state odds shift (%)")

    # Annotate cells
    for i in range(11):
        for j in range(6):
            ax.text(j, i, f"{med_bc[i, j]:.1f}%",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(med_bc[i, j]) > vmax * 0.6 else "black")
    fig.tight_layout()
    _maybe_save(fig, save_dir, "count_effects.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Continuous predictor coefficient plot
# ---------------------------------------------------------------------------

def plot_coef_effects(idata, save_dir=None):
    """
    One panel per continuous predictor: 94% HDI for each outcome.
    Gives a quick read on which predictors push which outcomes.
    """
    n_betas = len(CONT_BETAS)
    fig, axes = plt.subplots(1, n_betas, figsize=(4 * n_betas, 4), sharey=True)

    post = idata.posterior
    for ax, (var, label) in zip(axes, CONT_BETAS.items()):
        draws = post[var].values.reshape(-1, 6)  # (S, 6) in log-odds
        draws_pct = _logodds_to_pct_odds_shift(draws)  # (S, 6) in %
        med = np.median(draws_pct, axis=0)
        lo = np.percentile(draws_pct, 3, axis=0)
        hi = np.percentile(draws_pct, 97, axis=0)

        y = np.arange(6)
        ax.barh(y, med, xerr=[med - lo, hi - med],
                color=["#d62728" if m > 0 else "#1f77b4" for m in med],
                error_kw=dict(ecolor="black", lw=1.2, capsize=3),
                height=0.6)
        ax.axvline(0, color="gray", linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(OUTCOME_LABELS_6, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Odds shift (%)", fontsize=8)

    fig.suptitle("Continuous predictor effects — posterior median odds shift (%) ± 94% HDI",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _maybe_save(fig, save_dir, "coef_effects.png")
    return fig


# ---------------------------------------------------------------------------
# 7. Implied baseline outcome probabilities (0-0 count, avg batter/pitcher)
# ---------------------------------------------------------------------------

def plot_baseline_probs(idata, count_X_train=None, save_dir=None):
    """
    Posterior predicted outcome probabilities, marginalized over the actual
    count distribution (or a supplied count_X matrix).

    NOTE: The 0-0 reference count is the count BEFORE the PA-ending pitch,
    so at 0-0 K and BB are literally impossible (need 3 strikes / 4 balls).
    Marginalizing over counts gives realistic, interpretable rates.

    Parameters
    ----------
    idata        : az.InferenceData
    count_X_train: np.ndarray, shape (N, 11) or None
        Count dummy matrix from prepare_model_data. If None, the plot uses
        equal weights across the 11 non-reference count states (approximate).
    save_dir     : str or None
    """
    post = idata.posterior
    alpha_draws    = post["alpha"].values.reshape(-1, 6)     # (S, 6)
    beta_count_draws = post["beta_count"].values.reshape(-1, 11, 6)  # (S, 11, 6)
    S = alpha_draws.shape[0]

    if count_X_train is not None:
        # Sample 2000 rows from training count distribution
        rng = np.random.default_rng(42)
        idx = rng.choice(len(count_X_train), size=min(2000, len(count_X_train)), replace=False)
        count_sample = count_X_train[idx]  # (M, 11)
    else:
        # Uniform over all 11 non-reference counts + reference (12 rows)
        count_sample = np.vstack([np.eye(11), np.zeros((1, 11))])  # (12, 11)

    M = len(count_sample)

    # For each posterior draw, compute mean prob across count_sample rows
    # (all continuous features = 0 → average batter & pitcher)
    draw_idx = np.random.default_rng(0).choice(S, size=min(500, S), replace=False)
    per_draw_mean = np.empty((len(draw_idx), 7))

    for i, s in enumerate(draw_idx):
        lm = (alpha_draws[s][None, :]           # (1, 6)
              + count_sample @ beta_count_draws[s])  # (M, 11) @ (11, 6) → (M, 6)
        lf = np.concatenate([lm[:, 0:2], np.zeros((M, 1)), lm[:, 2:6]], axis=1)
        lf -= lf.max(axis=1, keepdims=True)
        e = np.exp(lf)
        per_draw_mean[i] = (e / e.sum(axis=1, keepdims=True)).mean(axis=0)

    medians = np.median(per_draw_mean, axis=0)
    lo = np.percentile(per_draw_mean, 3, axis=0)
    hi = np.percentile(per_draw_mean, 97, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(7)
    ax.bar(x, medians, color="#4878d0", alpha=0.8)
    ax.errorbar(x, medians, yerr=[medians - lo, hi - medians],
                fmt="none", color="black", capsize=4, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOME_LABELS_7, fontsize=10)
    ax.set_ylabel("Probability (%)")
    src = "training count distribution" if count_X_train is not None else "uniform count weights"
    ax.set_title(f"Predicted outcome probabilities\n(average batter & pitcher, {src}, posterior median + 94% HDI)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    fig.tight_layout()
    _maybe_save(fig, save_dir, "baseline_probs.png")
    return fig


# ---------------------------------------------------------------------------
# 8. Divergence count summary
# ---------------------------------------------------------------------------

def plot_divergences(idata, save_dir=None):
    """
    Bar chart of divergent transitions per chain.
    Any nonzero divergences warrant investigation.
    """
    divs = idata.sample_stats["diverging"].values  # (chains, draws)
    n_chains = divs.shape[0]
    div_per_chain = divs.sum(axis=1)
    total = div_per_chain.sum()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(n_chains), div_per_chain,
           color=["#d62728" if d > 0 else "#2ca02c" for d in div_per_chain])
    ax.set_xticks(range(n_chains))
    ax.set_xticklabels([f"Chain {i}" for i in range(n_chains)])
    ax.set_ylabel("Divergent transitions")
    ax.set_title(f"Divergences per chain  (total = {total})")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _maybe_save(fig, save_dir, "divergences.png")
    return fig


# ---------------------------------------------------------------------------
# Convenience: run all plots
# ---------------------------------------------------------------------------

def plot_all(idata, count_X_train=None, save_dir=None, show=True):
    """
    Run every diagnostic plot. Pass save_dir to write PNGs to disk.

    Parameters
    ----------
    idata        : az.InferenceData
    count_X_train: np.ndarray or None
        count_X array from prepare_model_data, used to marginalize baseline_probs
        over the real count distribution. Pass data["count_X"] for best results.
    save_dir     : str or None  — directory to save plots (created if absent)
    show         : bool         — call plt.show() after each figure

    Example
    -------
        data  = prepare_model_data(...)
        idata = az.from_netcdf("Bayes Outcomes/bayes_outcome_model.nc")
        plot_all(idata, count_X_train=data["count_X"], save_dir="Bayes Outcomes/plots")
    """
    figs = {}
    for name, fn in [
        ("trace",         plot_trace),
        ("rhat",          plot_rhat),
        ("energy",        plot_energy),
        ("alpha_forest",  plot_alpha_forest),
        ("count_effects", plot_count_effects),
        ("coef_effects",  plot_coef_effects),
        ("divergences",   plot_divergences),
    ]:
        print(f"Generating: {name}...")
        figs[name] = fn(idata, save_dir=save_dir)
        if show:
            plt.show()

    print("Generating: baseline_probs...")
    figs["baseline_probs"] = plot_baseline_probs(
        idata, count_X_train=count_X_train, save_dir=save_dir
    )
    if show:
        plt.show()

    return figs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "Bayes Outcomes")
    from outcome_probs import prepare_model_data

    idata = az.from_netcdf("Bayes Outcomes/bayes_outcome_model.nc")
    data  = prepare_model_data(data_dir="Data", years=[2025], subsample_frac=0.6)
    plot_all(idata, count_X_train=data["count_X"], save_dir="Bayes Outcomes/plots", show=True)
