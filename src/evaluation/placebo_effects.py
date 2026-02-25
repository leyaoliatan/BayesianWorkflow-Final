"""
placebo_effects.py
==================
Utility helpers for placebo treatment-effect estimation and aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

def build_cohort_weights(
 df: pd.DataFrame,
 *,
 weight_mode: str,
 cohort_key: str = "cohort_idx_p5",
 cohort_label_col: str = "cohort",
 cohort_size_col: str = "N_cohort",
) -> pd.DataFrame:
 """
 Build cohort weights for ATT aggregation.
 """
 cohort_df = df[[cohort_key, cohort_label_col]].drop_duplicates.copy
 cohort_df = cohort_df.sort_values(cohort_key).reset_index(drop=True)

 if weight_mode == "equal":
  w = np.full(len(cohort_df), 1.0 / max(len(cohort_df), 1), dtype=float)
  cohort_df["N_cohort"] = np.nan
 elif weight_mode == "cohort_size":
  if cohort_size_col not in df.columns:
   raise ValueError(f"weight_mode='cohort_size' requires column '{cohort_size_col}'.")
  one_size = (
   df[[cohort_key, cohort_size_col]]
  .dropna
  .sort_values([cohort_key])
  .groupby(cohort_key, as_index=False)
  .first
  )
  cohort_df = cohort_df.merge(one_size, on=cohort_key, how="left")
  if cohort_df[cohort_size_col].isna.any:
   raise ValueError("Missing cohort sizes for some cohorts in cohort_size weighting.")
  n = cohort_df[cohort_size_col].to_numpy(dtype=float)
  if np.any(n <= 0):
   raise ValueError("All cohort sizes must be positive for cohort_size weighting.")
  w = n / np.sum(n)
 else:
  raise ValueError(f"Unknown weight_mode: {weight_mode!r}")

 cohort_df["weight"] = w
 cohort_df["weight_mode"] = weight_mode
 return cohort_df

def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
 err = y_true - y_pred
 return {
  "mae": float(np.mean(np.abs(err))),
  "rmse": float(np.sqrt(np.mean(err**2))),
  "resid_mean": float(np.mean(err)),
  "resid_std": float(np.std(err)),
 }

def attach_yhat_by_model(
 p5: pd.DataFrame,
 fit_pred: dict[str, np.ndarray],
 *,
 dv_log_col: str = "dv_log",
) -> pd.DataFrame:
 out = p5.copy
 out["y_obs"] = out[dv_log_col].to_numpy(dtype=float)
 out["y_fe"] = fit_pred["FE+AR"]
 out["y_cp"] = fit_pred["GP-CP"]
 out["y_ext"] = fit_pred["GP-CP-Extended"]
 return out

def build_support_mask(
 df: pd.DataFrame,
 *,
 n_cohorts: int,
 n_periods: int,
 cohort_key: str = "cohort_idx_p5",
 time_key: str = "time_idx_p5",
) -> np.ndarray:
 """
 Build observation-support mask M(c,t)=1 iff (c,t) is observed.
 """
 m = np.zeros((n_cohorts, n_periods), dtype=bool)
 cc = df[cohort_key].to_numpy(dtype=int)
 tt = df[time_key].to_numpy(dtype=int)
 valid = (cc >= 0) & (cc < n_cohorts) & (tt >= 0) & (tt < n_periods)
 m[cc[valid], tt[valid]] = True
 return m

def compute_beta_placebo_te_tables(
 *,
 category: str,
 train_fit: dict[str, dict[str, Any]],
 full_fit: dict[str, dict[str, Any]],
 n_t_tr: int,
 n_t: int,
 n_c: int,
 support_mask: np.ndarray | None = None,
 model_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
 """
 Compute canonical tau_t / tau_ct tables from y_ct gaps:
  tau_ct = yhat_ct(full-fit) - yhat_ct(train-fit)
 on valid cells:
  M(c,t)=1 OR t in placebo-post window.
 """
 if model_order is None:
  model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

 tau_t_rows = []
 tau_ct_rows = []
 summary_rows = []

 def _grid_yhat_and_std(fit_out: dict[str, Any], model_name: str) -> tuple[np.ndarray, np.ndarray]:
  mu = float(fit_out.get("mu", 0.0))
  alpha = np.asarray(fit_out.get("alpha", np.zeros(n_c)), dtype=float)
  beta = np.asarray(fit_out.get("beta", np.zeros(n_t)), dtype=float)
  std_alpha = np.asarray(fit_out.get("std_alpha", np.zeros_like(alpha)), dtype=float)
  std_beta = np.asarray(
   fit_out.get("beta_std", fit_out.get("std_beta", np.zeros_like(beta))), dtype=float
  )
  gamma = np.asarray(fit_out.get("gamma", np.zeros((len(alpha), len(beta)))), dtype=float)
  std_gamma = np.asarray(fit_out.get("std_gamma", np.zeros_like(gamma)), dtype=float)

  if gamma.ndim == 1 or gamma.shape != (len(alpha), len(beta)):
   gamma = np.zeros((len(alpha), len(beta)), dtype=float)
  if std_gamma.ndim == 1 or std_gamma.shape != gamma.shape:
   std_gamma = np.zeros_like(gamma, dtype=float)

  y_grid = np.full((n_c, n_t), np.nan, dtype=float)
  std_grid = np.full((n_c, n_t), np.nan, dtype=float)
  cc = min(n_c, len(alpha))
  tt = min(n_t, len(beta))
  for c in range(cc):
   for t in range(tt):
    y_grid[c, t] = mu + alpha[c] + beta[t] + gamma[c, t]
    v = (std_alpha[c] if c < len(std_alpha) else 0.0) ** 2 + (
     std_beta[t] if t < len(std_beta) else 0.0
    ) ** 2 + (std_gamma[c, t] if c < std_gamma.shape[0] and t < std_gamma.shape[1] else 0.0) ** 2
    std_grid[c, t] = float(np.sqrt(max(v, 0.0)))
  return y_grid, std_grid

 post_mask = np.zeros(n_t, dtype=bool)
 post_mask[n_t_tr:] = True
 for m in model_order:
  y_tr_grid, y_tr_std_grid = _grid_yhat_and_std(train_fit[m], m)
  y_full_grid, y_full_std_grid = _grid_yhat_and_std(full_fit[m], m)

  for t in range(n_t):
   if support_mask is None:
    valid_c = np.arange(n_c, dtype=int)
    observed_c = np.arange(n_c, dtype=int)
   else:
    observed_c = np.flatnonzero(support_mask[:, t]).astype(int)
    valid_c = np.flatnonzero(support_mask[:, t] | post_mask[t]).astype(int)
   observed_set = set(observed_c.tolist)
   for c_idx in valid_c:
    tau = y_full_grid[c_idx, t] - y_tr_grid[c_idx, t]
    tau_std = float(np.sqrt(y_tr_std_grid[c_idx, t] ** 2 + y_full_std_grid[c_idx, t] ** 2))
    tau_lo = float(tau - 1.96 * tau_std)
    tau_hi = float(tau + 1.96 * tau_std)
    tau_ct_rows.append(
     {
      "category": category,
      "model": m,
      "cohort_idx": int(c_idx),
      "time_idx": int(t),
      "tau_ct": float(tau),
      "tau_ct_std": float(tau_std),
      "tau_ct_lo95": tau_lo,
      "tau_ct_hi95": tau_hi,
      "is_post": bool(post_mask[t]),
      "is_observed_ct": bool(c_idx in observed_set),
      "truth": 0.0,
     }
    )

  # Build tau_t from tau_ct by time (average across active cohorts at time t)
  tau_m = pd.DataFrame([r for r in tau_ct_rows if r["model"] == m])
  for t, g in tau_m.groupby("time_idx", as_index=False):
   tau_vals = g["tau_ct"].to_numpy(dtype=float)
   tau_stds = g["tau_ct_std"].to_numpy(dtype=float)
   n_active = int(len(g))
   tau_mean = float(np.mean(tau_vals))
   tau_std = float(np.sqrt(np.sum(tau_stds**2)) / max(n_active, 1))
   tau_lo = tau_mean - 1.96 * tau_std
   tau_hi = tau_mean + 1.96 * tau_std
   tau_t_rows.append(
    {
     "category": category,
     "model": m,
     "time_idx": int(t),
     "tau_t": tau_mean,
     "tau_t_std": tau_std,
     "tau_t_lo95": tau_lo,
     "tau_t_hi95": tau_hi,
     "is_post": bool(int(t) >= int(n_t_tr)),
     "n_active_cohorts": n_active,
     "truth": 0.0,
    }
   )

  g_post = pd.DataFrame([r for r in tau_t_rows if r["model"] == m and r["is_post"]])
  if not g_post.empty:
   summary_rows.append(
    {
     "category": category,
     "model": m,
     "ATE_post_placebo": float(np.mean(g_post["tau_t"].to_numpy(dtype=float))),
     "ATE_95CI_low": float(np.mean(g_post["tau_t_lo95"].to_numpy(dtype=float))),
     "ATE_95CI_high": float(np.mean(g_post["tau_t_hi95"].to_numpy(dtype=float))),
     "MaxAbs_TE": float(np.max(np.abs(g_post["tau_t"].to_numpy(dtype=float)))),
     "bias_vs_zero": float(np.mean(g_post["tau_t"].to_numpy(dtype=float))),
     "rmse_vs_zero": float(np.sqrt(np.mean(g_post["tau_t"].to_numpy(dtype=float) ** 2))),
     "coverage_vs_zero": float(
      np.mean(
       (g_post["tau_t_lo95"].to_numpy(dtype=float) <= 0.0)
       & (0.0 <= g_post["tau_t_hi95"].to_numpy(dtype=float))
      )
     ),
     "ci_width_mean": float(
      np.mean(
       g_post["tau_t_hi95"].to_numpy(dtype=float)
       - g_post["tau_t_lo95"].to_numpy(dtype=float)
      )
     ),
     "ci_width_std": float(
      np.std(
       g_post["tau_t_hi95"].to_numpy(dtype=float)
       - g_post["tau_t_lo95"].to_numpy(dtype=float)
      )
     ),
    }
   )

 return pd.DataFrame(tau_t_rows), pd.DataFrame(tau_ct_rows), pd.DataFrame(summary_rows)

def aggregate_att_from_tau_ct(
 tau_ct_df: pd.DataFrame,
 weights_df: pd.DataFrame,
 *,
 category: str,
 cohort_key: str = "cohort_idx_p5",
) -> pd.DataFrame:
 """
 Aggregate ATT_t from tau_ct using supplied cohort weights.
 """
 if "weight" not in weights_df.columns:
  raise ValueError("weights_df must include 'weight' column.")

 w = weights_df.set_index(cohort_key)["weight"].to_dict
 att_rows = []
 for (m, t), g in tau_ct_df.groupby(["model", "time_idx"], as_index=False):
  rows = [row for row in g.itertuples(index=False)]
  w_raw = np.array([w.get(int(row.cohort_idx), 0.0) for row in rows], dtype=float)
  w_sum = float(np.sum(w_raw))
  if w_sum <= 0:
   continue
  # Dynamic ATT risk set: normalize weights over active cohorts at each t.
  w_dyn = w_raw / w_sum
  att = float(np.sum([row.tau_ct * w_dyn[i] for i, row in enumerate(rows)]))
  # conservative aggregation of uncertainty under independence
  att_std = float(np.sqrt(np.sum([(row.tau_ct_std * w_dyn[i]) ** 2 for i, row in enumerate(rows)])))
  att_rows.append(
   {
    "category": category,
    "model": m,
    "time_idx": int(t),
    "att_t": att,
    "att_t_std": att_std,
    "att_t_lo95": att - 1.96 * att_std,
    "att_t_hi95": att + 1.96 * att_std,
    "n_active_cohorts": int(len(rows)),
    "truth": 0.0,
   }
  )
 return pd.DataFrame(att_rows)

def build_placebo_split_metrics(
 *,
 category: str,
 model_order: list[str],
 fit_pred: dict[str, np.ndarray],
 y: np.ndarray,
 mask_tr: np.ndarray,
 mask_te: np.ndarray,
) -> pd.DataFrame:
 rows = []
 for m in model_order:
  y_hat = fit_pred[m]
  train_m = error_metrics(y[mask_tr], y_hat[mask_tr])
  test_m = error_metrics(y[mask_te], y_hat[mask_te])
  rows.append(
   {
    "category": category,
    "model": m,
    "Train_MAE": train_m["mae"],
    "Train_RMSE": train_m["rmse"],
    "PlaceboTest_MAE": test_m["mae"],
    "PlaceboTest_RMSE": test_m["rmse"],
    "Residual_mean": float(np.mean(y - y_hat)),
    "Residual_SD": float(np.std(y - y_hat)),
   }
  )
 return pd.DataFrame(rows)

def summarise_placebo_estimands(
 *,
 category: str,
 tau_ct_df: pd.DataFrame,
 tau_t_df: pd.DataFrame,
 model_order: list[str],
) -> pd.DataFrame:
 """
 Build placebo summary metrics aligned with estimands:
 - tau metrics: at each t, compare tau_ct across cohorts vs truth 0, then average over t
 - ATT metrics: ATT averaged over post periods, compared against truth 0
 """
 rows = []
 for m in model_order:
  tau_m = tau_ct_df[tau_ct_df["model"] == m].copy
  tau_t_m = tau_t_df[tau_t_df["model"] == m].copy
  if tau_m.empty or tau_t_m.empty:
   continue

  # tau(c,t): time-first aggregation (across cohorts at each t), then average over t
  per_t = []
  for t, g in tau_m.groupby("time_idx", as_index=False):
   tau = g["tau_ct"].to_numpy(dtype=float)
   lo = g["tau_ct_lo95"].to_numpy(dtype=float)
   hi = g["tau_ct_hi95"].to_numpy(dtype=float)
   width = hi - lo
   per_t.append(
    {
     "time_idx": int(t),
     "mae_over_cohorts": float(np.mean(np.abs(tau))),
     "rmse_over_cohorts": float(np.sqrt(np.mean(tau**2))),
     "coverage_over_cohorts": float(np.mean((lo <= 0.0) & (0.0 <= hi))),
     "ci_width_over_cohorts_mean": float(np.mean(width)),
     "n_active_cohorts": int(len(tau)),
    }
   )
  per_t_df = pd.DataFrame(per_t)

  # Overall ATT scalar from tau_t over post periods
  tau_t_post = tau_t_m[tau_t_m["is_post"].astype(bool)] if "is_post" in tau_t_m.columns else tau_t_m
  att_avg = float(np.mean(tau_t_post["tau_t"].to_numpy(dtype=float)))
  att_lo_avg = float(np.mean(tau_t_post["tau_t_lo95"].to_numpy(dtype=float)))
  att_hi_avg = float(np.mean(tau_t_post["tau_t_hi95"].to_numpy(dtype=float)))
  att_width_avg = float(att_hi_avg - att_lo_avg)

  rows.append(
   {
    "category": category,
    "model": m,
    "tau_mae_avg_over_cohorts": float(per_t_df["mae_over_cohorts"].mean),
    "tau_rmse_avg_over_cohorts": float(per_t_df["rmse_over_cohorts"].mean),
    "tau_coverage_avg_over_cohorts": float(per_t_df["coverage_over_cohorts"].mean),
    "tau_ci_width_avg_over_cohorts": float(per_t_df["ci_width_over_cohorts_mean"].mean),
    "tau_ci_width_std_over_cohorts": float(per_t_df["ci_width_over_cohorts_mean"].std),
    "att_avg_over_t": att_avg,
    "att_mae_vs_zero": float(abs(att_avg)),
    "att_rmse_vs_zero": float(abs(att_avg)),
    "att_coverage_vs_zero": float(att_lo_avg <= 0.0 <= att_hi_avg),
    "att_ci_width_avg_over_t": att_width_avg,
   }
  )
 return pd.DataFrame(rows)

def build_support_aware_model_time_summary(
 *,
 category: str,
 fit_results: dict[str, dict[str, Any]],
 support_mask: np.ndarray,
 model_order: list[str] | None = None,
 scope: str = "full_fit",
 extrapolation_start_idx: int | None = None,
) -> pd.DataFrame:
 """
 Support-aware per-time summaries for alpha/beta/y.
 Uses only cells with M(c,t)=1 for aggregation over cohorts.
 """
 if model_order is None:
  model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

 support_mask = np.asarray(support_mask, dtype=bool)
 n_c, n_t = support_mask.shape
 rows: list[dict[str, Any]] = []

 for m in model_order:
  if m not in fit_results:
   continue
  out = fit_results[m]
  mu = float(out.get("mu", 0.0))
  alpha = np.asarray(out.get("alpha", np.zeros(n_c)), dtype=float)
  beta = np.asarray(out.get("beta", np.zeros(n_t)), dtype=float)
  std_alpha = np.asarray(out.get("std_alpha", np.zeros_like(alpha)), dtype=float)
  std_beta = np.asarray(out.get("std_beta", np.zeros_like(beta)), dtype=float)
  gamma = np.asarray(out.get("gamma", np.zeros((len(alpha), len(beta)))), dtype=float)
  std_gamma = np.asarray(out.get("std_gamma", np.zeros_like(gamma)), dtype=float)

  if gamma.ndim == 1 or gamma.shape != (len(alpha), len(beta)):
   gamma = np.zeros((len(alpha), len(beta)), dtype=float)
  if std_gamma.ndim == 1 or std_gamma.shape != gamma.shape:
   std_gamma = np.zeros_like(gamma, dtype=float)

  tt_max = min(n_t, len(beta))
  cc_max = min(n_c, len(alpha))
  for t in range(tt_max):
   active = np.flatnonzero(support_mask[:cc_max, t]).astype(int)
   if active.size == 0:
    continue

   alpha_a = alpha[active]
   alpha_std_a = std_alpha[active] if std_alpha.size >= cc_max else np.zeros_like(alpha_a)
   beta_t = float(beta[t])
   beta_std_t = float(std_beta[t]) if t < len(std_beta) else 0.0
   gamma_at = gamma[active, t]
   gamma_std_at = std_gamma[active, t]

   y_cells = mu + alpha_a + beta_t + gamma_at
   var_cells = alpha_std_a**2 + beta_std_t**2 + gamma_std_at**2
   n_active = int(active.size)

   y_mean = float(np.mean(y_cells))
   y_se = float(np.sqrt(np.sum(var_cells)) / max(n_active, 1))
   y_lo = y_mean - 1.96 * y_se
   y_hi = y_mean + 1.96 * y_se

   alpha_mean = float(np.mean(alpha_a))
   alpha_se = float(np.sqrt(np.sum(alpha_std_a**2)) / max(n_active, 1))
   alpha_lo = alpha_mean - 1.96 * alpha_se
   alpha_hi = alpha_mean + 1.96 * alpha_se

   beta_lo = beta_t - 1.96 * beta_std_t
   beta_hi = beta_t + 1.96 * beta_std_t

   rows.append(
    {
     "category": category,
     "scope": scope,
     "model": m,
     "time_idx": int(t),
     "n_active_cohorts": n_active,
     "alpha_active_mean": alpha_mean,
     "alpha_active_se": alpha_se,
     "alpha_active_lo95": alpha_lo,
     "alpha_active_hi95": alpha_hi,
     "beta_t_mean": beta_t,
     "beta_t_se": beta_std_t,
     "beta_t_lo95": beta_lo,
     "beta_t_hi95": beta_hi,
     "y_active_mean": y_mean,
     "y_active_se": y_se,
     "y_active_lo95": y_lo,
     "y_active_hi95": y_hi,
     "is_extrapolation": bool(
      extrapolation_start_idx is not None and int(t) >= int(extrapolation_start_idx)
     ),
    }
   )
 return pd.DataFrame(rows)
