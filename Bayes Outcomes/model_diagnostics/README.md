# PLS2 Model Diagnostics (`update_stats.py`)

Diagnostics for the joint PLS2 model used in `Bayes Outcomes/update_stats.py` (mapping mechanics to `Z-Contact_pct` and `xISO`).

## What It Produces

Running the diagnostics script writes an output folder with:

- `summary.json`: fit metrics (transformed and original scale), training size, component count
- `training_predictions.csv`: observed vs predicted values and residuals
- `observed_vs_predicted.png`: parity plots (transformed + original scale)
- `residual_diagnostics.png`: residual histograms and fitted-vs-residual plots
- `latent_scores.png`: PLS score space (`t1` vs `t2` when available)
- `coefficient_heatmap.png`: learned coefficient matrix by feature/output

## Usage

From repo root:

```bash
python3 "Bayes Outcomes/model_diagnostics/pls2_update_stats_diagnostics.py"
```

Custom output directory:

```bash
python3 "Bayes Outcomes/model_diagnostics/pls2_update_stats_diagnostics.py" \
  --output-dir "Bayes Outcomes/model_diagnostics/outputs/run_01"
```

## Notes

- The script loads `update_stats.py` directly by path to ensure it reuses the same data-building logic and coefficient helpers.
- The model fit intentionally mirrors `update_stats._fit_joint_model()`:
  - input standardization
  - degree-2 polynomial features
  - PLS2 (`PLSRegression`, `scale=False`)
  - transformed targets (`logit(z_contact)`, `log(xiso)`)
