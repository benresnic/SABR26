"""
Diagnostics for the joint PLS2 model used in `Bayes Outcomes/update_stats.py`.

This script mirrors the model fit in update_stats.py and writes diagnostic plots
and tables to a local output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
UPDATE_STATS_PATH = SCRIPT_DIR.parent / "update_stats.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "latest"


def _load_update_stats_module():
    spec = importlib.util.spec_from_file_location("update_stats_module", UPDATE_STATS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {UPDATE_STATS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _fit_mirrored_pls2(df: pd.DataFrame):
    """Mirror update_stats._fit_joint_model() and return artifacts for diagnostics."""
    X_raw = df[["median_bat_speed", "median_swing_length"]].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    zc_raw = df["Z-Contact_pct"].to_numpy(dtype=float)
    xiso_raw = df["xISO"].to_numpy(dtype=float)

    zc_clipped = np.clip(zc_raw, 1e-4, 1 - 1e-4)
    xiso_clipped = np.clip(xiso_raw, 1e-4, None)

    zc_logit = np.log(zc_clipped / (1 - zc_clipped))
    xiso_log = np.log(xiso_clipped)
    Y_trans = np.column_stack([zc_logit, xiso_log])

    max_valid_components = min(X_poly.shape[0] - 1, X_poly.shape[1], Y_trans.shape[1])
    n_components = max(1, max_valid_components)

    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_poly, Y_trans)

    Y_pred_trans = np.asarray(model.predict(X_poly), dtype=float)
    Y_pred_orig = np.column_stack(
        [
            1.0 / (1.0 + np.exp(-Y_pred_trans[:, 0])),
            np.exp(Y_pred_trans[:, 1]),
        ]
    )
    Y_true_orig = np.column_stack([zc_raw, xiso_raw])

    feature_names = list(poly.get_feature_names_out(["bat_speed_z", "swing_length_z"]))

    return {
        "scaler": scaler,
        "poly": poly,
        "model": model,
        "X_raw": X_raw,
        "X_poly": X_poly,
        "Y_trans": Y_trans,
        "Y_pred_trans": Y_pred_trans,
        "Y_true_orig": Y_true_orig,
        "Y_pred_orig": Y_pred_orig,
        "feature_names": feature_names,
        "n_components": int(n_components),
    }


def _build_predictions_frame(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    Y_trans = fit["Y_trans"]
    Y_pred_trans = fit["Y_pred_trans"]
    Y_true_orig = fit["Y_true_orig"]
    Y_pred_orig = fit["Y_pred_orig"]

    out = df.copy().reset_index(drop=True)
    out["zc_logit_true"] = Y_trans[:, 0]
    out["zc_logit_pred"] = Y_pred_trans[:, 0]
    out["zc_logit_resid"] = Y_trans[:, 0] - Y_pred_trans[:, 0]
    out["xiso_log_true"] = Y_trans[:, 1]
    out["xiso_log_pred"] = Y_pred_trans[:, 1]
    out["xiso_log_resid"] = Y_trans[:, 1] - Y_pred_trans[:, 1]

    out["z_contact_true"] = Y_true_orig[:, 0]
    out["z_contact_pred"] = Y_pred_orig[:, 0]
    out["z_contact_resid"] = Y_true_orig[:, 0] - Y_pred_orig[:, 0]
    out["xiso_true"] = Y_true_orig[:, 1]
    out["xiso_pred"] = Y_pred_orig[:, 1]
    out["xiso_resid"] = Y_true_orig[:, 1] - Y_pred_orig[:, 1]
    return out


def _compare_to_cached_predictor(us, df: pd.DataFrame):
    """Check parity between mirrored fit predictions and cached coefficient predictor path."""
    coeffs = us._load_or_fit_coefficients(force_refit=False)
    _, _, predict_both = us._build_joint_predictor(coeffs)

    rows = []
    for _, row in df.iterrows():
        zc_pred, xiso_pred = predict_both(float(row["median_bat_speed"]), float(row["median_swing_length"]))
        rows.append((zc_pred, xiso_pred))
    arr = np.asarray(rows, dtype=float)
    return coeffs, arr


def _save_observed_vs_predicted(pred_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    configs = [
        ("zc_logit_true", "zc_logit_pred", "Z-Contact (logit scale)"),
        ("xiso_log_true", "xiso_log_pred", "xISO (log scale)"),
        ("z_contact_true", "z_contact_pred", "Z-Contact (original scale)"),
        ("xiso_true", "xiso_pred", "xISO (original scale)"),
    ]

    for ax, (xcol, ycol, title) in zip(axes.flat, configs):
        x = pred_df[xcol].to_numpy()
        y = pred_df[ycol].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.7, edgecolors="none")
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_residual_diagnostics(pred_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    configs = [
        ("z_contact_pred", "z_contact_resid", "Z-Contact residuals vs fitted"),
        ("xiso_pred", "xiso_resid", "xISO residuals vs fitted"),
    ]
    for ax, (xcol, ycol, title) in zip(axes[0], configs):
        x = pred_df[xcol].to_numpy()
        y = pred_df[ycol].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.7, edgecolors="none")
        ax.axhline(0, color="black", lw=1, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.grid(alpha=0.2)

    axes[1, 0].hist(pred_df["z_contact_resid"], bins=25, color="#2E86AB", alpha=0.8)
    axes[1, 0].set_title("Z-Contact residual histogram", fontsize=10)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].hist(pred_df["xiso_resid"], bins=25, color="#E67E22", alpha=0.8)
    axes[1, 1].set_title("xISO residual histogram", fontsize=10)
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_latent_scores(fit: dict, pred_df: pd.DataFrame, output_path: Path):
    model = fit["model"]
    scores = np.asarray(getattr(model, "x_scores_", np.empty((len(pred_df), 0))), dtype=float)
    if scores.shape[1] == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x = scores[:, 0]
    if scores.shape[1] >= 2:
        y = scores[:, 1]
        c = pred_df["z_contact_true"].to_numpy()
        sc = ax.scatter(x, y, c=c, cmap="viridis", s=24, alpha=0.8)
        ax.set_ylabel("PLS score t2")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Observed Z-Contact")
    else:
        ax.scatter(x, np.zeros_like(x), s=24, alpha=0.8)
        ax.set_ylabel("(only one component)")
    ax.set_xlabel("PLS score t1")
    ax.set_title("PLS latent score space (X scores)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_coefficient_heatmap(fit: dict, output_path: Path):
    model = fit["model"]
    coef = np.asarray(model.coef_, dtype=float)

    # sklearn can return (n_features, n_targets) or transpose depending on version
    n_features = len(fit["feature_names"])
    if coef.shape[0] == n_features:
        coef_plot = coef
    elif coef.shape[1] == n_features:
        coef_plot = coef.T
    else:
        raise ValueError(f"Unexpected coefficient shape: {coef.shape}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vmax = float(np.max(np.abs(coef_plot))) if coef_plot.size else 1.0
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(coef_plot, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["z_contact (logit)", "xiso (log)"], rotation=0)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(fit["feature_names"])
    ax.set_title("PLS coefficient matrix")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Coefficient")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_t2_q_metrics(fit: dict):
    """Compute Hotelling's T² and Q (SPE) diagnostics on X-space."""
    model = fit["model"]
    X_poly = np.asarray(fit["X_poly"], dtype=float)
    scores = np.asarray(getattr(model, "x_scores_", np.empty((X_poly.shape[0], 0))), dtype=float)
    loadings = np.asarray(getattr(model, "x_loadings_", np.empty((X_poly.shape[1], 0))), dtype=float)

    if scores.size == 0 or loadings.size == 0:
        return {
            "scores": np.zeros((X_poly.shape[0], 0)),
            "t2": np.full(X_poly.shape[0], np.nan),
            "q": np.full(X_poly.shape[0], np.nan),
            "t2_threshold_95": np.nan,
            "q_threshold_95": np.nan,
        }

    # X reconstruction in latent space
    X_hat = scores @ loadings.T
    resid = X_poly - X_hat
    q = np.sum(resid ** 2, axis=1)

    comp_var = np.var(scores, axis=0, ddof=1)
    comp_var = np.where(comp_var <= 1e-12, 1e-12, comp_var)
    t2 = np.sum((scores ** 2) / comp_var[None, :], axis=1)

    return {
        "scores": scores,
        "t2": t2,
        "q": q,
        "t2_threshold_95": float(np.percentile(t2, 95)),
        "q_threshold_95": float(np.percentile(q, 95)),
    }


def _save_t2_q_plot(pred_df: pd.DataFrame, t2q: dict, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    t2 = np.asarray(t2q["t2"], dtype=float)
    q = np.asarray(t2q["q"], dtype=float)
    t2_thr = float(t2q["t2_threshold_95"])
    q_thr = float(t2q["q_threshold_95"])

    axes[0, 0].hist(t2, bins=30, color="#4C78A8", alpha=0.8)
    axes[0, 0].axvline(t2_thr, color="red", linestyle="--", lw=1)
    axes[0, 0].set_title("Hotelling's T²")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].hist(q, bins=30, color="#F58518", alpha=0.8)
    axes[0, 1].axvline(q_thr, color="red", linestyle="--", lw=1)
    axes[0, 1].set_title("Q residual (SPE)")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].scatter(t2, q, s=20, alpha=0.7, c=pred_df["z_contact_true"], cmap="viridis", edgecolors="none")
    axes[1, 0].axvline(t2_thr, color="red", linestyle="--", lw=1)
    axes[1, 0].axhline(q_thr, color="red", linestyle="--", lw=1)
    axes[1, 0].set_xlabel("T²")
    axes[1, 0].set_ylabel("Q (SPE)")
    axes[1, 0].set_title("Leverage vs residual distance")
    axes[1, 0].grid(alpha=0.2)

    # Label top combined outliers
    combo_rank = np.argsort((t2 / (t2_thr + 1e-9)) + (q / (q_thr + 1e-9)))[::-1][:5]
    for idx in combo_rank:
        axes[1, 0].annotate(str(idx), (t2[idx], q[idx]), fontsize=8, xytext=(4, 3), textcoords="offset points")

    scores = np.asarray(t2q["scores"])
    if scores.shape[1] >= 2:
        sc = axes[1, 1].scatter(scores[:, 0], scores[:, 1], c=t2, cmap="magma", s=20, alpha=0.8, edgecolors="none")
        plt.colorbar(sc, ax=axes[1, 1], label="T²")
        axes[1, 1].set_xlabel("t1")
        axes[1, 1].set_ylabel("t2")
        axes[1, 1].set_title("Latent scores colored by T²")
    else:
        axes[1, 1].scatter(np.arange(len(t2)), t2, s=18, alpha=0.8)
        axes[1, 1].set_title("T² by observation index")
        axes[1, 1].set_xlabel("Observation")
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compute_vip_scores(fit: dict) -> pd.DataFrame:
    """Compute approximate VIP scores from X scores/weights and Y variance explained."""
    model = fit["model"]
    X_poly = np.asarray(fit["X_poly"], dtype=float)
    Y = np.asarray(fit["Y_trans"], dtype=float)
    T = np.asarray(getattr(model, "x_scores_", np.empty((X_poly.shape[0], 0))), dtype=float)
    W = np.asarray(getattr(model, "x_weights_", np.empty((X_poly.shape[1], 0))), dtype=float)
    Q = np.asarray(getattr(model, "y_loadings_", np.empty((Y.shape[1], 0))), dtype=float)
    p = X_poly.shape[1]

    if T.size == 0 or W.size == 0 or Q.size == 0:
        return pd.DataFrame({"feature": fit["feature_names"], "vip": np.nan})

    # Sum of squares explained in Y per component (approx)
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    ss_total = np.sum(ss)
    if ss_total <= 0:
        vip = np.full(p, np.nan)
    else:
        w_norm = W / np.sqrt(np.sum(W ** 2, axis=0, keepdims=True))
        vip = np.sqrt(p * np.sum((w_norm ** 2) * ss[None, :], axis=1) / ss_total)

    return pd.DataFrame({"feature": fit["feature_names"], "vip": vip}).sort_values("vip", ascending=False)


def _save_vip_plot(vip_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = vip_df.sort_values("vip", ascending=True)
    ax.barh(plot_df["feature"], plot_df["vip"], color="#2E86AB", alpha=0.85)
    ax.axvline(1.0, color="red", linestyle="--", lw=1, label="VIP = 1")
    ax.set_xlabel("VIP score")
    ax.set_title("Variable Importance in Projection (VIP)")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_residual_region_heatmaps(pred_df: pd.DataFrame, output_path: Path):
    """Residual heatmaps over bat speed / swing length bins."""
    df = pred_df.copy()
    bs_bins = np.quantile(df["median_bat_speed"], np.linspace(0, 1, 9))
    sl_bins = np.quantile(df["median_swing_length"], np.linspace(0, 1, 9))
    # ensure uniqueness for cut
    bs_bins = np.unique(bs_bins)
    sl_bins = np.unique(sl_bins)
    if len(bs_bins) < 3 or len(sl_bins) < 3:
        return

    df["bs_bin"] = pd.cut(df["median_bat_speed"], bins=bs_bins, include_lowest=True, duplicates="drop")
    df["sl_bin"] = pd.cut(df["median_swing_length"], bins=sl_bins, include_lowest=True, duplicates="drop")

    mats = []
    titles = []
    for col, title in [("z_contact_resid", "Mean Z-Contact residual"), ("xiso_resid", "Mean xISO residual")]:
        pivot = df.pivot_table(index="bs_bin", columns="sl_bin", values=col, aggfunc="mean")
        mats.append(pivot)
        titles.append(title)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in zip(axes, mats, titles):
        arr = mat.to_numpy()
        vmax = np.nanmax(np.abs(arr))
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Swing length bin")
        ax.set_ylabel("Bat speed bin")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(range(1, mat.shape[1] + 1), fontsize=8)
        ax.set_yticklabels(range(1, mat.shape[0] + 1), fontsize=8)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cv_model_comparison(fit: dict, random_state: int = 42):
    X_poly = np.asarray(fit["X_poly"], dtype=float)
    Y_trans = np.asarray(fit["Y_trans"], dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    rows = []

    max_pls = min(5, X_poly.shape[1], Y_trans.shape[1], X_poly.shape[0] - 1)
    for n_comp in range(1, max_pls + 1):
        fold_rmse = []
        for tr, te in kf.split(X_poly):
            m = PLSRegression(n_components=n_comp, scale=False)
            m.fit(X_poly[tr], Y_trans[tr])
            pred = m.predict(X_poly[te])
            fold_rmse.append(np.sqrt(np.mean((Y_trans[te] - pred) ** 2)))
        rows.append({"model": f"PLS2_{n_comp}comp", "cv_rmse_trans": float(np.mean(fold_rmse)), "cv_rmse_trans_sd": float(np.std(fold_rmse))})

    for alpha in [0.0, 1.0, 10.0]:
        fold_rmse = []
        for tr, te in kf.split(X_poly):
            if alpha == 0.0:
                m = LinearRegression()
                name = "OLS_multioutput"
            else:
                m = Ridge(alpha=alpha)
                name = f"Ridge_alpha{alpha:g}"
            m.fit(X_poly[tr], Y_trans[tr])
            pred = m.predict(X_poly[te])
            fold_rmse.append(np.sqrt(np.mean((Y_trans[te] - pred) ** 2)))
        rows.append({"model": name, "cv_rmse_trans": float(np.mean(fold_rmse)), "cv_rmse_trans_sd": float(np.std(fold_rmse))})

    return pd.DataFrame(rows).sort_values("cv_rmse_trans")


def _save_cv_comparison_plot(cv_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = cv_df.sort_values("cv_rmse_trans", ascending=True).reset_index(drop=True)
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["cv_rmse_trans"], yerr=plot_df["cv_rmse_trans_sd"], capsize=4, color="#4C78A8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=30, ha="right")
    ax.set_ylabel("CV RMSE (transformed targets)")
    ax.set_title("Cross-validated model comparison (PLS2 vs baselines)")
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _leave_one_out_influence(fit: dict, pred_df: pd.DataFrame, top_n: int = 10):
    """Refit after removing top leverage points (by T²) and summarize impact."""
    X_poly = np.asarray(fit["X_poly"], dtype=float)
    Y_trans = np.asarray(fit["Y_trans"], dtype=float)
    base_model = fit["model"]
    base_coef = np.asarray(base_model.coef_, dtype=float)
    base_pred = np.asarray(fit["Y_pred_trans"], dtype=float)
    base_rmse = float(np.sqrt(np.mean((Y_trans - base_pred) ** 2)))

    t2 = pred_df["t2_hotelling"].to_numpy(dtype=float) if "t2_hotelling" in pred_df.columns else np.zeros(len(pred_df))
    top_idx = np.argsort(t2)[::-1][: min(top_n, len(t2))]

    rows = []
    n_comp = int(fit["n_components"])
    for idx in top_idx:
        mask = np.ones(len(X_poly), dtype=bool)
        mask[idx] = False
        m = PLSRegression(n_components=n_comp, scale=False)
        m.fit(X_poly[mask], Y_trans[mask])
        pred = m.predict(X_poly[mask])
        rmse = float(np.sqrt(np.mean((Y_trans[mask] - pred) ** 2)))
        coef = np.asarray(m.coef_, dtype=float)
        coef_diff = float(np.linalg.norm(coef.ravel() - base_coef.ravel()))
        rows.append({
            "obs_index": int(idx),
            "t2": float(t2[idx]),
            "q_spe": float(pred_df.loc[idx, "q_spe"]) if "q_spe" in pred_df.columns else np.nan,
            "base_rmse_trans": base_rmse,
            "loo_rmse_trans": rmse,
            "rmse_change": rmse - base_rmse,
            "coef_l2_diff": coef_diff,
        })

    return pd.DataFrame(rows).sort_values("t2", ascending=False)


def _save_influence_plot(infl_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_df = infl_df.copy().reset_index(drop=True)
    axes[0].bar(range(len(plot_df)), plot_df["coef_l2_diff"], color="#E45756", alpha=0.85)
    axes[0].set_xticks(range(len(plot_df)))
    axes[0].set_xticklabels(plot_df["obs_index"], rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Leave-one-out coefficient shift (top leverage points)")
    axes[0].set_ylabel("||coef_loo - coef_full||₂")
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(range(len(plot_df)), plot_df["rmse_change"], color="#54A24B", alpha=0.85)
    axes[1].axhline(0, color="black", lw=1, linestyle=":")
    axes[1].set_xticks(range(len(plot_df)))
    axes[1].set_xticklabels(plot_df["obs_index"], rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Leave-one-out RMSE change")
    axes[1].set_ylabel("Δ RMSE (transformed)")
    axes[1].grid(alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_coef_stability(fit: dict, n_boot: int = 50, random_state: int = 42):
    X_poly = np.asarray(fit["X_poly"], dtype=float)
    Y_trans = np.asarray(fit["Y_trans"], dtype=float)
    n = len(X_poly)
    rng = np.random.default_rng(random_state)
    n_comp = int(fit["n_components"])
    coefs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        m = PLSRegression(n_components=n_comp, scale=False)
        m.fit(X_poly[idx], Y_trans[idx])
        coefs.append(np.asarray(m.coef_, dtype=float).ravel())
    return np.asarray(coefs, dtype=float)


def _save_bootstrap_stability_plot(fit: dict, boot_coefs: np.ndarray, output_path: Path):
    coef = np.asarray(fit["model"].coef_, dtype=float)
    base = coef.ravel()
    lo = np.percentile(boot_coefs, 2.5, axis=0)
    hi = np.percentile(boot_coefs, 97.5, axis=0)
    med = np.percentile(boot_coefs, 50, axis=0)

    feat_names = fit["feature_names"]
    labels = [f"{feat}→zc" for feat in feat_names] + [f"{feat}→xiso" for feat in feat_names]
    # Align labels to coefficient flattening orientation
    if coef.shape[0] == len(feat_names):  # (features, targets)
        labels = [f"{feat}→{target}" for feat in feat_names for target in ["zc", "xiso"]]
    else:  # (targets, features)
        labels = [f"{target}:{feat}" for target in ["zc", "xiso"] for feat in feat_names]

    order = np.argsort(np.abs(base))[::-1][: min(12, len(base))]
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(order))
    ax.errorbar(
        med[order],
        y,
        xerr=np.vstack([med[order] - lo[order], hi[order] - med[order]]),
        fmt="o",
        color="#1f77b4",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Bootstrap coefficient estimate (95% interval)")
    ax.set_title("Bootstrap coefficient stability (top |coef| terms)")
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_prediction_intervals(fit: dict, boot_coefs: np.ndarray, output_path: Path):
    """Bootstrap prediction intervals along two one-dimensional mechanics paths.

    Top row: hold swing length constant (median), vary bat speed.
    Bottom row: hold bat speed constant (median), vary swing length.
    Columns: predicted Z-Contact and predicted xISO.
    """
    X_raw = np.asarray(fit["X_raw"], dtype=float)
    scaler = fit["scaler"]
    poly = fit["poly"]

    plot_blue = "#000E54"
    # Requested fixed constants for the one-variable paths
    bs_med = 75.0
    sl_med = 7.50
    pct_changes = np.arange(-12, 7, 3, dtype=float) / 100.0
    pct_axis = pct_changes * 100.0
    bs_vals = bs_med * (1.0 + pct_changes)
    sl_vals = sl_med * (1.0 + pct_changes)

    # Path A: vary bat speed, hold swing length constant
    pts_bs_path = np.column_stack([bs_vals, np.full_like(bs_vals, sl_med)])
    # Path B: vary swing length, hold bat speed constant
    pts_sl_path = np.column_stack([np.full_like(sl_vals, bs_med), sl_vals])

    X_poly_bs = poly.transform(scaler.transform(pts_bs_path))
    X_poly_sl = poly.transform(scaler.transform(pts_sl_path))

    coef_shape = np.asarray(fit["model"].coef_).shape
    intercept = np.asarray(fit["model"].intercept_, dtype=float).reshape(-1)

    def _predict_boot(X_poly_pts):
        preds = []
        for c in boot_coefs:
            coef = c.reshape(coef_shape)
            # Use full-model intercept (diagnostic approximation; bootstrap intercepts not tracked)
            if coef.shape[0] == X_poly_pts.shape[1]:
                yhat = intercept + X_poly_pts @ coef
            else:
                yhat = intercept + (coef @ X_poly_pts.T).T
            zc = 1.0 / (1.0 + np.exp(-yhat[:, 0]))
            xiso = np.exp(yhat[:, 1])
            preds.append(np.column_stack([zc, xiso]))
        return np.asarray(preds)  # (B, P, 2)

    preds_bs = _predict_boot(X_poly_bs)
    preds_sl = _predict_boot(X_poly_sl)

    def _bands(preds):
        return (
            np.percentile(preds, 50, axis=0),
            np.percentile(preds, 2.5, axis=0),
            np.percentile(preds, 97.5, axis=0),
        )

    med_bs, lo_bs, hi_bs = _bands(preds_bs)
    med_sl, lo_sl, hi_sl = _bands(preds_sl)

    # Two "current metric" baselines (latest available season in batter_stats at patch time):
    # Blue = Ian Happ (2025), Orange = Kyle Schwarber (2025).
    # We center on the model's 0%-mechanics prediction so x=0 maps to y=0, then scale by
    # the chosen current metric baseline so the curves differ by baseline level.
    baseline_targets_blue = np.array([0.8675, 0.2003], dtype=float)    # Ian Happ [Z-Contact, xISO]
    baseline_targets_orange = np.array([0.8002, 0.3327], dtype=float)  # Kyle Schwarber [Z-Contact, xISO]
    zero_idx = int(np.where(np.isclose(pct_axis, 0.0))[0][0])

    def _bands_centered_pct_change_from_zero(preds, metric_baseline):
        """Bootstrap bands for ((pred - pred_at_0pct) / current_metric_baseline) * 100."""
        denom = np.asarray(metric_baseline, dtype=float).reshape(1, 1, -1)
        denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        preds_zero = preds[:, zero_idx:zero_idx + 1, :]
        preds_pct = ((preds - preds_zero) / denom) * 100.0
        return _bands(preds_pct)

    med_bs_blue, lo_bs_blue, hi_bs_blue = _bands_centered_pct_change_from_zero(preds_bs, baseline_targets_blue)
    med_sl_blue, lo_sl_blue, hi_sl_blue = _bands_centered_pct_change_from_zero(preds_sl, baseline_targets_blue)
    med_bs_orange, lo_bs_orange, hi_bs_orange = _bands_centered_pct_change_from_zero(preds_bs, baseline_targets_orange)
    med_sl_orange, lo_sl_orange, hi_sl_orange = _bands_centered_pct_change_from_zero(preds_sl, baseline_targets_orange)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top row: vary bat speed (hold swing length constant)
    top_titles = [
        "Predicted Z-Contact vs Bat Speed Change",
        "Predicted xISO vs Bat Speed Change",
    ]
    for k, ax in enumerate(axes[0]):
        ax.errorbar(
            pct_axis,
            med_bs_blue[:, k],
            yerr=np.vstack([
                med_bs_blue[:, k] - lo_bs_blue[:, k],
                hi_bs_blue[:, k] - med_bs_blue[:, k],
            ]),
            fmt="o-",
            capsize=4,
            lw=1.5,
            color=plot_blue,
            ecolor=plot_blue,
            markerfacecolor=plot_blue,
            markeredgecolor=plot_blue,
            label=("Ian Happ (2025): ZCon .868" if k == 0 else "Ian Happ (2025): xISO .200"),
        )
        ax.plot(
            pct_axis,
            med_bs_orange[:, k],
            "o-",
            lw=1.5,
            color="#F76900",
            markerfacecolor="#F76900",
            markeredgecolor="#F76900",
            label=("Kyle Schwarber (2025): ZCon .800" if k == 0 else "Kyle Schwarber (2025): xISO .333"),
        )
        ax.set_title(top_titles[k], fontsize=10)
        ax.set_xlabel("Bat Speed Change (%)")
        ax.set_xlim(-12, 6)
        ax.set_xticks(np.arange(-12, 7, 3))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    # Bottom row: vary swing length (hold bat speed constant)
    bottom_titles = [
        "Predicted Z-Contact vs Swing Length Change",
        "Predicted xISO vs Swing Length Change",
    ]
    for k, ax in enumerate(axes[1]):
        ax.errorbar(
            pct_axis,
            med_sl_blue[:, k],
            yerr=np.vstack([
                med_sl_blue[:, k] - lo_sl_blue[:, k],
                hi_sl_blue[:, k] - med_sl_blue[:, k],
            ]),
            fmt="o-",
            capsize=4,
            lw=1.5,
            color=plot_blue,
            ecolor=plot_blue,
            markerfacecolor=plot_blue,
            markeredgecolor=plot_blue,
            label=("Ian Happ (2025): ZCon .868" if k == 0 else "Ian Happ (2025): xISO .200"),
        )
        ax.plot(
            pct_axis,
            med_sl_orange[:, k],
            "o-",
            lw=1.5,
            color="#F76900",
            markerfacecolor="#F76900",
            markeredgecolor="#F76900",
            label=("Kyle Schwarber (2025): ZCon .800" if k == 0 else "Kyle Schwarber (2025): xISO .333"),
        )
        ax.set_title(bottom_titles[k], fontsize=10)
        ax.set_xlabel("Swing Length Change (%)")
        ax.set_xlim(-12, 6)
        ax.set_xticks(np.arange(-12, 7, 3))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    axes[0, 0].set_ylabel("Z-Contact Change (%)")
    axes[1, 0].set_ylabel("Z-Contact Change (%)")
    axes[0, 1].set_ylabel("xISO Change (%)")
    axes[1, 1].set_ylabel("xISO Change (%)")

    fig.suptitle("Bootstrap prediction intervals along one-variable mechanics paths", fontsize=12, y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_training_data_with_keys(us):
    """Rebuild update_stats training data but keep batter/season keys for season holdout diagnostics."""
    data_dir = us.DATA_DIR
    pbp_files = [data_dir / f"pbp_{year}.parquet" for year in [2023, 2024, 2025]]
    pbp_list = [pd.read_parquet(f) for f in pbp_files if f.exists()]
    pbp = pd.concat(pbp_list, ignore_index=True)
    pbp = pbp[pbp["bat_speed"].notna()].copy()

    swing_stats = (
        pbp.groupby(["batter", "game_year"])
        .agg(
            median_bat_speed=("bat_speed", "median"),
            median_swing_length=("swing_length", "median"),
            swing_sample_n=("bat_speed", "size"),
        )
        .reset_index()
    )
    swing_stats = swing_stats[swing_stats["swing_sample_n"] >= us.MIN_SWINGS_PER_BATTER_SEASON].copy()

    batter_stats = pd.read_parquet(data_dir / "batter_stats.parquet")
    swing_stats["batter"] = swing_stats["batter"].astype("int32")
    swing_stats["game_year"] = swing_stats["game_year"].astype("int32")
    batter_stats["xMLBAMID"] = batter_stats["xMLBAMID"].astype("int32")
    batter_stats["Season"] = batter_stats["Season"].astype("int32")
    merged = swing_stats.merge(
        batter_stats[["xMLBAMID", "Season", "Z-Contact_pct", "xISO"]],
        left_on=["batter", "game_year"],
        right_on=["xMLBAMID", "Season"],
        how="inner",
    )
    merged = merged.dropna(subset=["median_bat_speed", "median_swing_length", "Z-Contact_pct", "xISO"])
    return merged.reset_index(drop=True)


def _season_holdout_evaluation(us, fit: dict):
    df = _build_training_data_with_keys(us)
    years = sorted(df["game_year"].dropna().astype(int).unique().tolist())
    rows = []

    for test_year in years:
        tr = df[df["game_year"] != test_year].copy()
        te = df[df["game_year"] == test_year].copy()
        if len(tr) < 10 or len(te) < 10:
            continue

        # Mirror fitting steps on train split only
        scaler = StandardScaler()
        Xtr_raw = tr[["median_bat_speed", "median_swing_length"]].to_numpy(float)
        Xte_raw = te[["median_bat_speed", "median_swing_length"]].to_numpy(float)
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xtrp = poly.fit_transform(Xtr)
        Xtep = poly.transform(Xte)

        ztr = np.clip(tr["Z-Contact_pct"].to_numpy(float), 1e-4, 1 - 1e-4)
        xtr = np.clip(tr["xISO"].to_numpy(float), 1e-4, None)
        ytr = np.column_stack([np.log(ztr / (1 - ztr)), np.log(xtr)])

        zte_true = te["Z-Contact_pct"].to_numpy(float)
        xte_true = te["xISO"].to_numpy(float)

        n_comp = min(int(fit["n_components"]), Xtrp.shape[0] - 1, Xtrp.shape[1], 2)
        n_comp = max(1, n_comp)
        m = PLSRegression(n_components=n_comp, scale=False)
        m.fit(Xtrp, ytr)
        yhat = np.asarray(m.predict(Xtep), dtype=float)
        zhat = 1.0 / (1.0 + np.exp(-yhat[:, 0]))
        xhat = np.exp(yhat[:, 1])

        rows.append({
            "test_year": int(test_year),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "z_contact_r2": _r2(zte_true, zhat),
            "xiso_r2": _r2(xte_true, xhat),
            "z_contact_rmse": _rmse(zte_true, zhat),
            "xiso_rmse": _rmse(xte_true, xhat),
        })

    return pd.DataFrame(rows)


def _save_season_holdout_plot(df: pd.DataFrame, output_path: Path):
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(df))
    axes[0].bar(x, df["z_contact_r2"], color="#4C78A8", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["test_year"].astype(str))
    axes[0].set_title("Season holdout R² (Z-Contact)")
    axes[0].axhline(0, color="black", linestyle=":", lw=1)
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(x, df["xiso_r2"], color="#F58518", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["test_year"].astype(str))
    axes[1].set_title("Season holdout R² (xISO)")
    axes[1].axhline(0, color="black", linestyle=":", lw=1)
    axes[1].grid(alpha=0.2, axis="y")
    for ax in axes:
        ax.set_xlabel("Held-out season")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(output_dir: Path, write_csv: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    us = _load_update_stats_module()

    df = us._build_training_data()
    fit = _fit_mirrored_pls2(df)
    pred_df = _build_predictions_frame(df, fit)
    t2q = _compute_t2_q_metrics(fit)
    scores = np.asarray(t2q["scores"], dtype=float)
    if scores.shape[1] >= 1:
        pred_df["t1"] = scores[:, 0]
    if scores.shape[1] >= 2:
        pred_df["t2"] = scores[:, 1]
    pred_df["t2_hotelling"] = t2q["t2"]
    pred_df["q_spe"] = t2q["q"]

    coeffs_cached, cached_preds = _compare_to_cached_predictor(us, df)
    pred_df["z_contact_pred_cached"] = cached_preds[:, 0]
    pred_df["xiso_pred_cached"] = cached_preds[:, 1]
    pred_df["z_contact_pred_diff_vs_cached"] = pred_df["z_contact_pred"] - pred_df["z_contact_pred_cached"]
    pred_df["xiso_pred_diff_vs_cached"] = pred_df["xiso_pred"] - pred_df["xiso_pred_cached"]

    summary = {
        "model": {
            "type": "joint_pls2_poly_transformed",
            "n_components": fit["n_components"],
            "poly_degree": 2,
            "poly_include_bias": False,
            "feature_names": fit["feature_names"],
        },
        "training_data": {
            "n_observations": int(len(df)),
            "bat_speed_mean": float(df["median_bat_speed"].mean()),
            "swing_length_mean": float(df["median_swing_length"].mean()),
            "z_contact_mean": float(df["Z-Contact_pct"].mean()),
            "xiso_mean": float(df["xISO"].mean()),
            "input_corr_bs_sl": float(np.corrcoef(df["median_bat_speed"], df["median_swing_length"])[0, 1]),
            "output_corr_zc_xiso": float(np.corrcoef(df["Z-Contact_pct"], df["xISO"])[0, 1]),
        },
        "fit_metrics_transformed": {
            "z_contact_logit_r2": _r2(pred_df["zc_logit_true"], pred_df["zc_logit_pred"]),
            "xiso_log_r2": _r2(pred_df["xiso_log_true"], pred_df["xiso_log_pred"]),
            "z_contact_logit_rmse": _rmse(pred_df["zc_logit_true"], pred_df["zc_logit_pred"]),
            "xiso_log_rmse": _rmse(pred_df["xiso_log_true"], pred_df["xiso_log_pred"]),
        },
        "fit_metrics_original": {
            "z_contact_r2": _r2(pred_df["z_contact_true"], pred_df["z_contact_pred"]),
            "xiso_r2": _r2(pred_df["xiso_true"], pred_df["xiso_pred"]),
            "z_contact_rmse": _rmse(pred_df["z_contact_true"], pred_df["z_contact_pred"]),
            "xiso_rmse": _rmse(pred_df["xiso_true"], pred_df["xiso_pred"]),
            "z_contact_mae": _mae(pred_df["z_contact_true"], pred_df["z_contact_pred"]),
            "xiso_mae": _mae(pred_df["xiso_true"], pred_df["xiso_pred"]),
        },
        "residuals": {
            "corr_z_contact_xiso_original": float(
                np.corrcoef(pred_df["z_contact_resid"], pred_df["xiso_resid"])[0, 1]
            ),
            "covariance_transformed": np.cov(
                np.column_stack([pred_df["zc_logit_resid"], pred_df["xiso_log_resid"]]).T
            ).tolist(),
        },
        "t2_q_diagnostics": {
            "t2_threshold_95": float(t2q["t2_threshold_95"]),
            "q_spe_threshold_95": float(t2q["q_threshold_95"]),
            "n_t2_gt_95pct": int(np.sum(pred_df["t2_hotelling"] > t2q["t2_threshold_95"])),
            "n_q_gt_95pct": int(np.sum(pred_df["q_spe"] > t2q["q_threshold_95"])),
        },
        "parity_with_cached_predictor": {
            "cached_model_type": coeffs_cached.get("model_type"),
            "cached_pls_n_components": coeffs_cached.get("pls_n_components"),
            "max_abs_z_contact_pred_diff": float(np.max(np.abs(pred_df["z_contact_pred_diff_vs_cached"]))),
            "max_abs_xiso_pred_diff": float(np.max(np.abs(pred_df["xiso_pred_diff_vs_cached"]))),
        },
    }

    vip_df = _compute_vip_scores(fit)
    summary["vip_scores"] = vip_df.to_dict(orient="records")

    cv_df = _cv_model_comparison(fit)
    summary["cv_model_comparison"] = cv_df.to_dict(orient="records")

    influence_df = _leave_one_out_influence(fit, pred_df, top_n=10)
    summary["influence_top_leverage"] = influence_df.to_dict(orient="records")

    boot_coefs = _bootstrap_coef_stability(fit, n_boot=50, random_state=42)
    summary["bootstrap_stability"] = {
        "n_boot": int(boot_coefs.shape[0]),
        "coef_vector_length": int(boot_coefs.shape[1]) if boot_coefs.ndim == 2 else 0,
    }

    season_holdout_df = _season_holdout_evaluation(us, fit)
    summary["season_holdout"] = season_holdout_df.to_dict(orient="records")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if write_csv:
        pred_df.to_csv(output_dir / "training_predictions.csv", index=False)
        vip_df.to_csv(output_dir / "vip_scores.csv", index=False)
        cv_df.to_csv(output_dir / "cv_model_comparison.csv", index=False)
        influence_df.to_csv(output_dir / "influence_top_leverage.csv", index=False)
        if not season_holdout_df.empty:
            season_holdout_df.to_csv(output_dir / "season_holdout_metrics.csv", index=False)

    _save_observed_vs_predicted(pred_df, output_dir / "observed_vs_predicted.png")
    _save_residual_diagnostics(pred_df, output_dir / "residual_diagnostics.png")
    _save_latent_scores(fit, pred_df, output_dir / "latent_scores.png")
    _save_coefficient_heatmap(fit, output_dir / "coefficient_heatmap.png")
    _save_t2_q_plot(pred_df, t2q, output_dir / "t2_q_diagnostics.png")
    _save_vip_plot(vip_df, output_dir / "vip_scores.png")
    _save_cv_comparison_plot(cv_df, output_dir / "cv_model_comparison.png")
    _save_residual_region_heatmaps(pred_df, output_dir / "residual_region_heatmaps.png")
    _save_influence_plot(influence_df, output_dir / "influence_top_leverage.png")
    _save_bootstrap_stability_plot(fit, boot_coefs, output_dir / "bootstrap_coefficient_stability.png")
    _bootstrap_prediction_intervals(fit, boot_coefs, output_dir / "bootstrap_prediction_intervals.png")
    _save_season_holdout_plot(season_holdout_df, output_dir / "season_holdout_r2.png")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostics for update_stats PLS2 model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write plots/tables (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing training_predictions.csv",
    )
    args = parser.parse_args()

    summary = run_diagnostics(args.output_dir, write_csv=not args.no_csv)
    print(f"Wrote diagnostics to: {args.output_dir}")
    print("Original-scale R^2:")
    print(f"  z_contact: {summary['fit_metrics_original']['z_contact_r2']:.4f}")
    print(f"  xiso:      {summary['fit_metrics_original']['xiso_r2']:.4f}")


if __name__ == "__main__":
    main()
