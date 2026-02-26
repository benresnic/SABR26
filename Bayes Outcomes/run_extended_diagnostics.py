"""
Extended diagnostics for the Bayes outcome model.

Writes plots to `Bayes Outcomes/plots/` and a summary JSON with metrics.

Notes
-----
- Uses subsets for some diagnostics (PPC, subgroup calibration, approx LOO/WAIC)
  to keep runtime manageable.
- This script complements (does not replace) `plot_diagnostics.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Avoid sandbox write errors from ArviZ/Matplotlib caches.
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("ARVIZ_DATA", "/tmp/arviz_data")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))

import plot_diagnostics as base_diag
import validate_predictions as vp

PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME_NAMES = ["K", "BB/HBP", "field_out", "1B", "2B", "3B", "HR"]


def _save(fig, name: str):
    fig.savefig(PLOTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _flatten_posterior_scalar_params(idata):
    """Return matrix (samples x params) and names for scalar posterior parameters."""
    post = idata.posterior
    names = []
    cols = []
    for var in post.data_vars:
        arr = np.asarray(post[var].values)  # chain, draw, ...
        base_shape = arr.shape[:2]
        flat = arr.reshape(base_shape[0] * base_shape[1], -1)
        if flat.shape[1] == 1:
            names.append(var)
        else:
            for j in range(flat.shape[1]):
                names.append(f"{var}[{j}]")
        cols.append(flat)
    X = np.concatenate(cols, axis=1) if cols else np.empty((0, 0))
    return X, names


def _plot_ess(idata):
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")

    def _to_df(ds, label):
        records = []
        for var in ds.data_vars:
            arr = np.asarray(ds[var].values)
            flat = arr.ravel()
            if flat.size == 1:
                records.append({"param": var, label: float(flat[0])})
            else:
                for i, v in enumerate(flat):
                    records.append({"param": f"{var}[{i}]", label: float(v)})
        return pd.DataFrame(records)

    df = _to_df(ess_bulk, "ess_bulk").merge(_to_df(ess_tail, "ess_tail"), on="param", how="outer")
    df = df.sort_values("ess_bulk", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.18 * len(df))))
    for ax, col, title in zip(axes, ["ess_bulk", "ess_tail"], ["ESS (bulk)", "ESS (tail)"]):
        vals = df[col].fillna(0).to_numpy()
        y = np.arange(len(df))
        ax.barh(y, vals, color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(df["param"], fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Effective sample size")
        ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    _save(fig, "ess_bulk_tail.png")
    return df


def _plot_rank_and_autocorr(idata):
    var_names = ["alpha", "beta_o_swing", "beta_z_contact", "beta_xiso", "beta_xfip"]
    fig1 = az.plot_rank(idata, var_names=var_names, figsize=(16, 8))
    fig1 = np.asarray(fig1).ravel()[0].get_figure()
    fig1.suptitle("Rank plots — alpha & continuous betas", fontsize=12, y=1.02)
    _save(fig1, "rank_plots.png")

    fig2 = az.plot_autocorr(idata, var_names=var_names, figsize=(16, 8))
    fig2 = np.asarray(fig2).ravel()[0].get_figure()
    fig2.suptitle("Autocorrelation — alpha & continuous betas", fontsize=12, y=1.02)
    _save(fig2, "autocorr_plots.png")


def _plot_bfmi_and_sampler_health(idata):
    bfmi_vals = np.asarray(az.bfmi(idata), dtype=float).ravel()
    ss = idata.sample_stats
    energy = np.asarray(ss["energy"].values).ravel()
    tree_depth = np.asarray(ss["tree_depth"].values).ravel() if "tree_depth" in ss else None
    n_steps = np.asarray(ss["n_steps"].values).ravel() if "n_steps" in ss else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.bar(np.arange(len(bfmi_vals)), bfmi_vals, color=["#2ca02c" if x >= 0.3 else "#d62728" for x in bfmi_vals])
    ax.axhline(0.3, color="red", linestyle="--", lw=1)
    ax.set_title("BFMI by chain")
    ax.set_xlabel("Chain")
    ax.set_ylabel("BFMI")

    axes[0, 1].hist(energy, bins=40, color="#4C78A8", alpha=0.8)
    axes[0, 1].set_title("Energy distribution")
    axes[0, 1].grid(alpha=0.2)

    if tree_depth is not None:
        axes[1, 0].hist(tree_depth, bins=np.arange(tree_depth.min(), tree_depth.max() + 2) - 0.5,
                        color="#F58518", alpha=0.8)
        axes[1, 0].set_title("Tree depth distribution")
        axes[1, 0].grid(alpha=0.2)
    else:
        axes[1, 0].axis("off")

    if n_steps is not None:
        axes[1, 1].hist(n_steps, bins=40, color="#54A24B", alpha=0.8)
        axes[1, 1].set_title("NUTS steps distribution")
        axes[1, 1].grid(alpha=0.2)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    _save(fig, "bfmi_sampler_health.png")
    return {"bfmi_by_chain": bfmi_vals.tolist()}


def _plot_posterior_corr_heatmap(idata):
    X, names = _flatten_posterior_scalar_params(idata)
    if X.size == 0:
        return None
    corr = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = float(np.nanmax(np.abs(corr)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # Label only every kth tick to keep readable
    step = max(1, len(names) // 20)
    ticks = np.arange(0, len(names), step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([names[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([names[i] for i in ticks], fontsize=6)
    ax.set_title("Posterior parameter correlation heatmap")
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    _save(fig, "posterior_corr_heatmap.png")
    return corr


def _plot_divergence_localization(idata):
    ss = idata.sample_stats
    div = np.asarray(ss["diverging"].values).astype(bool).ravel()
    energy = np.asarray(ss["energy"].values).ravel()
    lp = np.asarray(ss["lp"].values).ravel() if "lp" in ss else np.zeros_like(energy)
    tree_depth = np.asarray(ss["tree_depth"].values).ravel() if "tree_depth" in ss else None
    n_steps = np.asarray(ss["n_steps"].values).ravel() if "n_steps" in ss else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.scatter(energy[~div], lp[~div], s=8, alpha=0.25, color="#1f77b4", edgecolors="none", label="non-div")
    if div.any():
        ax.scatter(energy[div], lp[div], s=16, alpha=0.9, color="#d62728", edgecolors="none", label="div")
    ax.set_xlabel("Energy")
    ax.set_ylabel("lp")
    ax.set_title("Divergence localization: energy vs lp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    if tree_depth is not None:
        ax.scatter(tree_depth[~div], energy[~div], s=8, alpha=0.25, color="#1f77b4", edgecolors="none")
        if div.any():
            ax.scatter(tree_depth[div], energy[div], s=16, alpha=0.9, color="#d62728", edgecolors="none")
        ax.set_xlabel("Tree depth")
        ax.set_ylabel("Energy")
        ax.set_title("Tree depth vs energy")
        ax.grid(alpha=0.2)
    else:
        ax.axis("off")

    ax = axes[1, 0]
    if n_steps is not None:
        ax.scatter(n_steps[~div], energy[~div], s=8, alpha=0.25, color="#1f77b4", edgecolors="none")
        if div.any():
            ax.scatter(n_steps[div], energy[div], s=16, alpha=0.9, color="#d62728", edgecolors="none")
        ax.set_xlabel("NUTS steps")
        ax.set_ylabel("Energy")
        ax.set_title("NUTS steps vs energy")
        ax.grid(alpha=0.2)
    else:
        ax.axis("off")

    axes[1, 1].bar(["Non-div", "Div"], [int((~div).sum()), int(div.sum())], color=["#2ca02c", "#d62728"])
    axes[1, 1].set_title("Divergence counts")
    axes[1, 1].grid(alpha=0.2, axis="y")

    plt.tight_layout()
    _save(fig, "divergence_localization.png")
    return {"n_divergent": int(div.sum()), "n_total": int(div.size)}


def _softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def _posterior_arrays(idata):
    post = idata.posterior
    return {
        "alpha": np.asarray(post["alpha"].values),  # C,D,6
        "beta_count": np.asarray(post["beta_count"].values),  # C,D,11,6
        "beta_o_swing": np.asarray(post["beta_o_swing"].values),
        "beta_z_contact": np.asarray(post["beta_z_contact"].values),
        "beta_xiso": np.asarray(post["beta_xiso"].values),
        "beta_xfip": np.asarray(post["beta_xfip"].values),
    }


def _compute_loglik_and_ppc_subset(idata, n_obs=4000, draws_per_chain=100, seed=42):
    """Approximate log-likelihood and PPC on a training subset using constant_data."""
    rng = np.random.default_rng(seed)
    cd = idata.constant_data
    y = np.asarray(cd["y_obs"].values).astype(int)
    count_X = np.asarray(cd["count_X"].values, dtype=float)
    o = np.asarray(cd["o_swing"].values, dtype=float)
    zc = np.asarray(cd["z_contact"].values, dtype=float)
    xi = np.asarray(cd["xiso"].values, dtype=float)
    xf = np.asarray(cd["xfip"].values, dtype=float)

    N = len(y)
    idx = rng.choice(N, size=min(n_obs, N), replace=False)
    y_s = y[idx]
    Xc = count_X[idx]
    o_s = o[idx]
    zc_s = zc[idx]
    xi_s = xi[idx]
    xf_s = xf[idx]

    arr = _posterior_arrays(idata)
    C, D = arr["alpha"].shape[:2]
    d_idx = np.linspace(0, D - 1, num=min(draws_per_chain, D), dtype=int)
    Dsub = len(d_idx)

    loglik = np.zeros((C, Dsub, len(idx)), dtype=np.float32)
    sim_counts = np.zeros((C, Dsub, 7), dtype=np.int32)
    pred_mean_freq = np.zeros((C, Dsub, 7), dtype=np.float64)

    for c in range(C):
        for j, d in enumerate(d_idx):
            lm = (
                arr["alpha"][c, d][None, :]
                + Xc @ arr["beta_count"][c, d]
                + o_s[:, None] * arr["beta_o_swing"][c, d]
                + zc_s[:, None] * arr["beta_z_contact"][c, d]
                + xi_s[:, None] * arr["beta_xiso"][c, d]
                + xf_s[:, None] * arr["beta_xfip"][c, d]
            )  # (M, 6)
            ref = np.zeros((lm.shape[0], 1), dtype=lm.dtype)
            logits = np.concatenate([lm[:, 0:2], ref, lm[:, 2:6]], axis=1)
            p = _softmax_rows(logits)
            pred_mean_freq[c, j] = p.mean(axis=0)
            loglik[c, j] = np.log(np.clip(p[np.arange(len(idx)), y_s], 1e-12, 1.0))
            y_sim = np.array([rng.choice(7, p=row) for row in p], dtype=int)
            sim_counts[c, j] = np.bincount(y_sim, minlength=7)

    obs_counts = np.bincount(y_s, minlength=7)
    return {
        "subset_idx": idx,
        "y_true": y_s,
        "loglik": loglik,
        "obs_counts": obs_counts,
        "sim_counts": sim_counts,
        "pred_mean_freq": pred_mean_freq,
    }


def _plot_ppc(ppc):
    obs_freq = ppc["obs_counts"] / ppc["obs_counts"].sum()
    sim_freq = ppc["sim_counts"] / ppc["sim_counts"].sum(axis=-1, keepdims=True)
    sim_flat = sim_freq.reshape(-1, 7)
    lo = np.percentile(sim_flat, 5, axis=0)
    med = np.percentile(sim_flat, 50, axis=0)
    hi = np.percentile(sim_flat, 95, axis=0)
    pred_med = np.percentile(ppc["pred_mean_freq"].reshape(-1, 7), 50, axis=0)

    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, obs_freq, width=0.35, color="#000E54", label="Observed")
    ax.bar(x + 0.2, pred_med, width=0.35, color="#F76900", label="Posterior mean (median draw)")
    ax.errorbar(x + 0.2, med, yerr=[med - lo, hi - med], fmt="none", ecolor="black", capsize=4, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOME_NAMES)
    ax.set_ylabel("Frequency")
    ax.set_title("Posterior Predictive Check (subset): outcome frequencies with 90% PPC band")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "ppc_outcome_frequencies.png")


def _loo_waic_from_loglik(loglik_arr):
    # Build minimal InferenceData with posterior + log_likelihood groups
    # (ArviZ loo/waic expects a posterior group to exist.)
    C, D, M = loglik_arr.shape
    dummy = np.zeros((C, D), dtype=np.float32)
    id_ll = az.from_dict(
        posterior={"dummy": dummy},
        log_likelihood={"obs": loglik_arr},
        coords={"obs_id": np.arange(M)},
        dims={"obs": ["obs_id"]},
    )
    loo = az.loo(id_ll, pointwise=True)
    waic = az.waic(id_ll, pointwise=True)
    return loo, waic


def _plot_loo_waic_pareto(loo, waic):
    pareto_k = np.asarray(getattr(loo, "pareto_k", np.array([])), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if pareto_k.size:
        axes[0].hist(pareto_k, bins=30, color="#4C78A8", alpha=0.8)
        axes[0].axvline(0.5, color="orange", linestyle="--", lw=1)
        axes[0].axvline(0.7, color="red", linestyle="--", lw=1)
        axes[0].set_title("Pareto-k distribution (approx subset LOO)")
        axes[0].set_xlabel("Pareto k")
        axes[0].grid(alpha=0.2)
    else:
        axes[0].axis("off")

    rows = [
        ("LOO elpd", float(loo.elpd_loo)),
        ("LOO p_loo", float(loo.p_loo)),
        ("WAIC elpd", float(waic.elpd_waic)),
        ("WAIC p_waic", float(waic.p_waic)),
    ]
    axes[1].axis("off")
    txt = "\n".join([f"{k}: {v:.3f}" for k, v in rows])
    axes[1].text(0.05, 0.95, txt, va="top", ha="left", fontsize=11, family="monospace")
    axes[1].set_title("Approx LOO / WAIC summary (subset)")

    plt.tight_layout()
    _save(fig, "loo_waic_pareto_k.png")


def _multiclass_log_loss(y_true, probs):
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _mean_one_vs_rest_brier(y_true, probs):
    return float(np.mean([
        np.mean(((y_true == k).astype(float) - probs[:, k]) ** 2)
        for k in range(probs.shape[1])
    ]))


def _freq_l1_calibration(y_true, probs):
    obs = np.bincount(y_true, minlength=probs.shape[1]) / len(y_true)
    pred = probs.mean(axis=0)
    return float(np.sum(np.abs(obs - pred)))


def _bucket_score_diff(s):
    if s <= -4:
        return "<=-4"
    if s >= 4:
        return ">=+4"
    return f"{int(s):+d}"


def _compute_subgroup_metrics(pa_df, pred_probs):
    df = pa_df.copy().reset_index(drop=True)
    df["y_true"] = df["y"].astype(int)
    count_labels = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["count_group"] = count_labels
    df["inning_bucket"] = pd.cut(df["inning"], bins=[0, 3, 6, 20], labels=["1-3", "4-6", "7+"])
    df["score_diff_bucket"] = df["bat_score_diff"].fillna(0).apply(_bucket_score_diff)
    if "xFIP" in df.columns:
        df["pitcher_quality_bucket"] = pd.qcut(df["xFIP"], q=4, labels=["Q1 best", "Q2", "Q3", "Q4 worst"], duplicates="drop")
    else:
        df["pitcher_quality_bucket"] = "missing"

    valid = ~np.isnan(pred_probs).any(axis=1)
    df = df.loc[valid].copy()
    probs = pred_probs[valid]

    group_specs = [
        ("count_group", "By Count"),
        ("inning_bucket", "By Inning Bucket"),
        ("score_diff_bucket", "By Score Diff Bucket"),
        ("pitcher_quality_bucket", "By Pitcher xFIP Quartile"),
    ]

    metrics_by_group = {}
    cal_diff_heatmaps = {}
    for col, label in group_specs:
        records = []
        heat_rows = []
        for gval, gidx in df.groupby(col).groups.items():
            idx = np.asarray(list(gidx), dtype=int)
            y_g = df.loc[idx, "y_true"].to_numpy()
            p_g = probs[idx]
            if len(y_g) < 50:
                continue
            obs_freq = np.bincount(y_g, minlength=7) / len(y_g)
            pred_freq = p_g.mean(axis=0)
            heat_rows.append(pd.Series(pred_freq - obs_freq, index=OUTCOME_NAMES, name=str(gval)))
            records.append({
                "group": str(gval),
                "n": int(len(y_g)),
                "multiclass_log_loss": _multiclass_log_loss(y_g, p_g),
                "mean_brier_ovr": _mean_one_vs_rest_brier(y_g, p_g),
                "freq_l1_calibration": _freq_l1_calibration(y_g, p_g),
            })
        mdf = pd.DataFrame(records)
        if not mdf.empty:
            metrics_by_group[col] = (label, mdf.sort_values("group").reset_index(drop=True))
            cal_diff_heatmaps[col] = pd.DataFrame(heat_rows)
    return metrics_by_group, cal_diff_heatmaps


def _plot_subgroup_metrics(metrics_by_group, cal_diff_heatmaps):
    # Calibration heatmaps (pred - obs freq by outcome)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (col, (label, _)) in zip(axes.flat, metrics_by_group.items()):
        hm = cal_diff_heatmaps.get(col)
        if hm is None or hm.empty:
            ax.axis("off")
            continue
        mat = hm[OUTCOME_NAMES].to_numpy()
        vmax = float(np.max(np.abs(mat))) if mat.size else 0.01
        if vmax == 0:
            vmax = 0.01
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{label}: Pred - Obs freq by outcome")
        ax.set_xticks(range(len(OUTCOME_NAMES)))
        ax.set_xticklabels(OUTCOME_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(hm.index)))
        ax.set_yticklabels([str(x) for x in hm.index], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(fig, "subgroup_calibration_heatmaps.png")

    # Metric bars
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (col, (label, mdf)) in zip(axes.flat, metrics_by_group.items()):
        if mdf.empty:
            ax.axis("off")
            continue
        # Normalize and average three metrics to show "worse is higher"
        plot_df = mdf.copy()
        x = np.arange(len(plot_df))
        ax.bar(x - 0.25, plot_df["multiclass_log_loss"], width=0.25, label="Log loss")
        ax.bar(x, plot_df["mean_brier_ovr"], width=0.25, label="Mean Brier")
        ax.bar(x + 0.25, plot_df["freq_l1_calibration"], width=0.25, label="Freq L1 cal")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["group"], rotation=45, ha="right", fontsize=8)
        ax.set_title(label)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "subgroup_metrics_brier_logloss.png")


def _prepare_validation_with_pitcher_xfip(years=(2025,), subsample_frac=0.05):
    pa_df = vp.load_validation_data(years=list(years), subsample_frac=subsample_frac)
    pitcher_stats = pd.read_parquet(ROOT / "Data" / "pitcher_stats.parquet")
    ps = pitcher_stats[["xMLBAMID", "Season", "xFIP"]].copy()
    ps["xMLBAMID"] = ps["xMLBAMID"].astype("int32")
    ps["Season"] = ps["Season"].astype("int32")
    pa_df["pitcher"] = pa_df["pitcher"].astype("int32")
    pa_df["game_year"] = pa_df["game_year"].astype("int32")
    pa_df = pa_df.merge(ps.rename(columns={"xMLBAMID": "pitcher", "Season": "game_year"}), on=["pitcher", "game_year"], how="left")
    return pa_df


def run_all_extended():
    summary = {"notes": {}}

    idata = az.from_netcdf(THIS_DIR / "bayes_outcome_model.nc")
    with open(THIS_DIR / "scalers.json") as f:
        scalers = json.load(f)

    # Existing baseline plots
    try:
        from outcome_probs import prepare_model_data
        example_data = prepare_model_data(data_dir="Data", years=[2025], subsample_frac=0.05)
        base_diag.plot_all(idata, count_X_train=example_data["count_X"], example_data=example_data,
                           save_dir=str(PLOTS_DIR), show=False)
        plt.close("all")
    except Exception as e:
        summary["notes"]["base_plot_all_error"] = str(e)

    # Additional chain/sampler diagnostics
    ess_df = _plot_ess(idata)
    _plot_rank_and_autocorr(idata)
    summary["bfmi"] = _plot_bfmi_and_sampler_health(idata)
    div_info = _plot_divergence_localization(idata)
    summary["divergences"] = div_info
    _plot_posterior_corr_heatmap(idata)

    # PPC + approx LOO/WAIC from training subset
    ppc = _compute_loglik_and_ppc_subset(idata, n_obs=4000, draws_per_chain=100)
    _plot_ppc(ppc)
    loo, waic = _loo_waic_from_loglik(ppc["loglik"])
    _plot_loo_waic_pareto(loo, waic)
    summary["approx_loo_waic_subset"] = {
        "subset_n_obs": int(ppc["loglik"].shape[2]),
        "chains": int(ppc["loglik"].shape[0]),
        "draws_per_chain": int(ppc["loglik"].shape[1]),
        "elpd_loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "elpd_waic": float(waic.elpd_waic),
        "p_waic": float(waic.p_waic),
        "pareto_k_gt_0_7": int(np.sum(np.asarray(getattr(loo, "pareto_k", [])) > 0.7)),
    }

    # Validation + subgroup diagnostics
    pa_df = _prepare_validation_with_pitcher_xfip(years=(2025,), subsample_frac=0.05)
    pred_probs = vp.get_predictions_for_validation(pa_df, idata, scalers)
    valid_mask = ~np.isnan(pred_probs).any(axis=1)
    y_true = pa_df.loc[valid_mask, "y"].to_numpy(dtype=int)
    pp = pred_probs[valid_mask]
    summary["validation_subset"] = {
        "n_pa_total": int(len(pa_df)),
        "n_pa_valid_predictions": int(valid_mask.sum()),
        "multiclass_log_loss": _multiclass_log_loss(y_true, pp),
        "mean_brier_ovr": _mean_one_vs_rest_brier(y_true, pp),
        "freq_l1_calibration": _freq_l1_calibration(y_true, pp),
    }
    metrics_by_group, cal_heat = _compute_subgroup_metrics(pa_df, pred_probs)
    _plot_subgroup_metrics(metrics_by_group, cal_heat)
    subgroup_json = {}
    for col, (label, mdf) in metrics_by_group.items():
        subgroup_json[col] = {"label": label, "rows": mdf.to_dict(orient="records")}
    summary["subgroup_metrics"] = subgroup_json

    # Prior sensitivity / SBC not run automatically here (require multiple model fits/simulations)
    summary["notes"]["prior_sensitivity"] = (
        "Not run by this script; requires fitting multiple models with alternative priors."
    )
    summary["notes"]["sbc"] = (
        "Not run by this script; requires repeated simulation + refitting pipeline."
    )

    # Save summary + ESS table
    with open(PLOTS_DIR / "extended_diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    ess_df.to_csv(PLOTS_DIR / "ess_bulk_tail.csv", index=False)

    print(f"Wrote extended Bayes diagnostics to {PLOTS_DIR}")


if __name__ == "__main__":
    run_all_extended()
