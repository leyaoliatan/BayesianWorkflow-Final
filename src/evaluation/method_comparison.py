"""
method_comparison.py
====================
Utilities for Scenario-A method comparison:

- Type-II MAP + Laplace (existing `CohortPeriodModel`)
- NUTS hyperparameters (`fit_hyperparam_nuts`)
- Full NUTS (`fit_full_nuts`)

This module runs one (N, seed) experiment and provides table builders plus
artifact saving helpers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data import simulate_cohort_data
from src.models import CohortPeriodModel
from src.models.gp_mcmc import fit_full_nuts, fit_hyperparam_nuts
from src.visualization import (
 plot_method_beta_posterior_comparison,
 plot_method_hyperparam_posteriors,
)

def _scenario_a_reps_from_n(
 n_total: int,
 n_cohorts: int,
 n_periods: int,
) -> int:
 base = int(n_cohorts) * int(n_periods)
 if n_total % base != 0:
  raise ValueError(
   f"n_total={n_total} is not divisible by n_cohorts*n_periods={base}. "
   "Use a compatible size such as 100/300/1000 with C=T=10."
  )
 reps = n_total // base
 if reps <= 0:
  raise ValueError("Computed n_reps must be positive.")
 return int(reps)

def _true_hparam_summary(sim_data: dict[str, Any]) -> dict[str, float]:
 hp = sim_data.get("hyperparams", {})
 ell_c = float(hp.get("ell_c", np.nan))
 ell_t = float(hp.get("ell_t", np.nan))
 sf_c = float(hp.get("sf_c", np.nan))
 sf_t = float(hp.get("sf_t", np.nan))
 sn = float(sim_data.get("sn_true", hp.get("sn", np.nan)))
 return {
  "ell": float(np.nanmean([ell_c, ell_t])),
  "sf": float(np.nanmean([sf_c, sf_t])),
  "sn": sn,
  "ell_c": ell_c,
  "ell_t": ell_t,
  "sf_c": sf_c,
  "sf_t": sf_t,
 }

def _method_summary_row(
 *,
 scenario: str,
 n_total: int,
 seed: int,
 method_label: str,
 result: dict[str, Any],
) -> dict[str, Any]:
 hp_post = result.get("hyperparams_posterior", {})
 hp_mean = hp_post.get("mean", {})
 hp_sd = hp_post.get("sd", {})

 # Fallback for Type-II MAP model that does not expose hyperparam posterior SD.
 hp_map = result.get("hyperparams_map", {})
 ell_map = float(np.nanmean([hp_map.get("ell_c", np.nan), hp_map.get("ell_t", np.nan)]))
 sf_map = float(np.nanmean([hp_map.get("sf_c", np.nan), hp_map.get("sf_t", np.nan)]))
 sn_map = float(hp_map.get("sn", np.nan))

 ell_mean = float(hp_mean.get("ell", ell_map))
 sf_mean = float(hp_mean.get("sf", sf_map))
 sn_mean = float(hp_mean.get("sn", sn_map))

 ell_sd = float(hp_sd.get("ell", np.nan))
 sf_sd = float(hp_sd.get("sf", np.nan))
 sn_sd = float(hp_sd.get("sn", np.nan))

 return {
  "Scenario": scenario,
  "N": int(n_total),
  "Seed": int(seed),
  "Method": method_label,
  "ell_mean": ell_mean,
  "sf_mean": sf_mean,
  "sn_mean": sn_mean,
  "ell_sd": ell_sd,
  "sf_sd": sf_sd,
  "sn_sd": sn_sd,
  "time_s": float(result.get("runtime_seconds", np.nan)),
 }

def run_scenario_a_single_n(
 *,
 n_total: int,
 seed: int = 0,
 n_cohorts: int = 10,
 n_periods: int = 10,
 dgp_params: Optional[dict[str, Any]] = None,
 type2map_cfg: Optional[dict[str, Any]] = None,
 nuts_hyper_cfg: Optional[dict[str, Any]] = None,
 full_nuts_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
 """
 Run one Scenario-A comparison at a single sample size and seed.
 """
 n_reps = _scenario_a_reps_from_n(n_total, n_cohorts, n_periods)

 dgp_defaults = {
  "ell_c": 2.0,
  "sf_c": 1.0,
  "ell_t": 3.0,
  "sf_t": 1.0,
  "sn": 0.3,
 }
 if dgp_params:
  dgp_defaults.update(dgp_params)

 sim_data = simulate_cohort_data(
  n_cohorts=int(n_cohorts),
  n_periods=int(n_periods),
  n_reps=int(n_reps),
  interaction=False,
  seed=int(seed),
  **dgp_defaults,
 )

 y = sim_data["y"]
 obs_c = sim_data["obs_c"]
 obs_t = sim_data["obs_t"]

 type2map_defaults = {"use_global_mean": False, "n_laplace_samples": 200, "n_map_steps": 50}
 if type2map_cfg:
  type2map_defaults.update(type2map_cfg)

 t0 = time.perf_counter
 m1 = CohortPeriodModel(
  use_global_mean=bool(type2map_defaults["use_global_mean"]),
  n_laplace_samples=int(type2map_defaults["n_laplace_samples"]),
  n_map_steps=int(type2map_defaults["n_map_steps"]),
 )
 m1.fit(
  y=y,
  obs_c=obs_c,
  obs_t=obs_t,
  n_cohorts=int(n_cohorts),
  n_periods=int(n_periods),
  seed=int(seed),
  true_hyperparams=sim_data.get("hyperparams"),
 )
 m1_res = m1.results_dict
 m1_res["runtime_seconds"] = float(time.perf_counter - t0)
 m1_res["method_label"] = "1) Type-II MAP + Laplace"
 m1_res["method_id"] = "type2map_laplace"

 nuts_hyper_defaults = {
  "seed": int(seed),
  "use_global_mean": False,
  "num_warmup": 500,
  "num_samples": 500,
  "num_chains": 1,
  "jitter": 1e-6,
  "return_samples": True,
 }
 if nuts_hyper_cfg:
  nuts_hyper_defaults.update(nuts_hyper_cfg)
 m2_res = fit_hyperparam_nuts(
  y=y,
  obs_c=obs_c,
  obs_t=obs_t,
  n_cohorts=int(n_cohorts),
  n_periods=int(n_periods),
  **nuts_hyper_defaults,
 )

 full_nuts_defaults = {
  "seed": int(seed),
  "use_global_mean": False,
  "num_warmup": 500,
  "num_samples": 500,
  "num_chains": 1,
  "jitter": 1e-6,
  "return_samples": True,
 }
 if full_nuts_cfg:
  full_nuts_defaults.update(full_nuts_cfg)
 m3_res = fit_full_nuts(
  y=y,
  obs_c=obs_c,
  obs_t=obs_t,
  n_cohorts=int(n_cohorts),
  n_periods=int(n_periods),
  **full_nuts_defaults,
 )

 method_results = {
  "method1": m1_res,
  "method2": m2_res,
  "method3": m3_res,
 }
 comparison_df = build_comparison_table(
  method_results=method_results,
  true_hparams=_true_hparam_summary(sim_data),
  scenario="A",
  n_total=int(n_total),
  seed=int(seed),
 )
 speed_df = build_speed_table(comparison_df)

 return {
  "scenario": "A",
  "seed": int(seed),
  "n_total": int(n_total),
  "n_cohorts": int(n_cohorts),
  "n_periods": int(n_periods),
  "n_reps": int(n_reps),
  "sim_data": sim_data,
  "true_hparams": _true_hparam_summary(sim_data),
  "method_results": method_results,
  "comparison_table": comparison_df,
  "speed_table": speed_df,
 }

def build_comparison_table(
 *,
 method_results: dict[str, dict[str, Any]],
 true_hparams: dict[str, float],
 scenario: str,
 n_total: int,
 seed: int,
) -> pd.DataFrame:
 """
 Build comparison table with speed + mean/SD hyperparameter summaries.
 """
 rows: list[dict[str, Any]] = [
  {
   "Scenario": scenario,
   "N": int(n_total),
   "Seed": int(seed),
   "Method": "Ground Truth",
   "ell_mean": float(true_hparams["ell"]),
   "sf_mean": float(true_hparams["sf"]),
   "sn_mean": float(true_hparams["sn"]),
   "ell_sd": np.nan,
   "sf_sd": np.nan,
   "sn_sd": np.nan,
   "time_s": np.nan,
  }
 ]

 ordered = ["method1", "method2", "method3"]
 for k in ordered:
  if k not in method_results:
   continue
  res = method_results[k]
  rows.append(
   _method_summary_row(
    scenario=scenario,
    n_total=int(n_total),
    seed=int(seed),
    method_label=str(res.get("method_label", k)),
    result=res,
   )
  )

 return pd.DataFrame(rows)

def build_speed_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
 """
 Build speed-only table from comparison table.
 """
 df = comparison_df.loc[comparison_df["Method"] != "Ground Truth", ["Scenario", "N", "Seed", "Method", "time_s"]].copy
 baseline = df.loc[df["Method"] == "1) Type-II MAP + Laplace", "time_s"]
 base_time = float(baseline.iloc[0]) if len(baseline) else np.nan
 if np.isfinite(base_time) and base_time > 0:
  df["relative_to_type2map"] = df["time_s"] / base_time
 else:
  df["relative_to_type2map"] = np.nan
 return df

def build_absolute_error_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
 """
 Build |estimate - truth| table for (ell, sf, sn) for one run.
 """
 truth = comparison_df.loc[comparison_df["Method"] == "Ground Truth"]
 if len(truth) == 0:
  raise ValueError("Ground Truth row missing from comparison table.")
 t_ell = float(truth["ell_mean"].iloc[0])
 t_sf = float(truth["sf_mean"].iloc[0])
 t_sn = float(truth["sn_mean"].iloc[0])

 methods = [
  "1) Type-II MAP + Laplace",
  "2) NUTS Hyperparams",
  "3) Full NUTS (Sample f + theta)",
 ]
 rows = []
 for param, col, tval, pname in [
  ("ell", "ell_mean", t_ell, "ℓ (lengthscale)"),
  ("sf", "sf_mean", t_sf, "σf (signal std)"),
  ("sn", "sn_mean", t_sn, "σn (noise std)"),
 ]:
  row = {"Parameter": pname}
  for m in methods:
   v = comparison_df.loc[comparison_df["Method"] == m, col]
   row[m] = float(abs(float(v.iloc[0]) - tval)) if len(v) else np.nan
  rows.append(row)
 return pd.DataFrame(rows)

def absolute_error_table_text(abs_df: pd.DataFrame) -> str:
 header = "=" * 100
 lines = [
  header,
  "Absolute Error from Ground Truth (|μ - true|)",
  header,
  f"{'Parameter':<15} {'M1b Error':<15} {'M2 Error':<15} {'M3 Error':<15}",
  "-" * 100,
 ]
 for _, r in abs_df.iterrows:
  lines.append(
   f"{str(r['Parameter']):<15} "
   f"{float(r['1) Type-II MAP + Laplace']):<15.5f} "
   f"{float(r['2) NUTS Hyperparams']):<15.5f} "
   f"{float(r['3) Full NUTS (Sample f + theta)']):<15.5f}"
  )
 lines.append(header)
 return "\n".join(lines) + "\n"

def save_method_artifacts(
 *,
 output_dir: str | Path,
 run_output: dict[str, Any],
) -> dict[str, str]:
 """
 Save per-run artifacts (tables + summaries + samples) to output_dir.
 """
 out = Path(output_dir)
 out.mkdir(parents=True, exist_ok=True)

 comparison_df = run_output["comparison_table"]
 speed_df = run_output["speed_table"]
 abs_df = build_absolute_error_table(comparison_df)
 method_results = run_output["method_results"]

 comparison_csv = out / "comparison_table_seed0.csv"
 speed_csv = out / "speed_table_seed0.csv"
 abs_csv = out / "absolute_error_table_seed0.csv"
 abs_txt = out / "absolute_error_table_seed0.txt"
 comparison_df.to_csv(comparison_csv, index=False)
 speed_df.to_csv(speed_csv, index=False)
 abs_df.to_csv(abs_csv, index=False)
 abs_txt.write_text(absolute_error_table_text(abs_df))

 metadata = {
  "scenario": run_output["scenario"],
  "seed": int(run_output["seed"]),
  "n_total": int(run_output["n_total"]),
  "n_cohorts": int(run_output["n_cohorts"]),
  "n_periods": int(run_output["n_periods"]),
  "n_reps": int(run_output["n_reps"]),
  "true_hparams": run_output["true_hparams"],
  "methods": {
   k: {
    "method_label": v.get("method_label"),
    "runtime_seconds": float(v.get("runtime_seconds", np.nan)),
    "hyperparams_map": v.get("hyperparams_map", {}),
    "hyperparams_posterior": v.get("hyperparams_posterior", {}),
    "mcmc_config": v.get("mcmc_config", {}),
   }
   for k, v in method_results.items
  },
 }
 metadata_json = out / "run_metadata.json"
 metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")

 sample_files: dict[str, str] = {}
 for key, fname in [
  ("method1", "method1_type2map_laplace_summary.npz"),
  ("method2", "method2_nuts_hyperparams_draws.npz"),
  ("method3", "method3_full_nuts_draws.npz"),
 ]:
  res = method_results.get(key, {})
  p = out / fname
  payload = {
   "alpha_mean": np.asarray(res.get("alpha", []), dtype=float),
   "beta_mean": np.asarray(res.get("beta", []), dtype=float),
   "std_alpha": np.asarray(res.get("std_alpha", []), dtype=float),
   "std_beta": np.asarray(res.get("std_beta", []), dtype=float),
   "y_hat": np.asarray(res.get("y_hat", []), dtype=float),
  }
  samples = res.get("posterior_samples")
  if isinstance(samples, dict):
   for sk, sv in samples.items:
    payload[f"sample_{sk}"] = np.asarray(sv, dtype=float)
  np.savez(p, **payload)
  sample_files[key] = str(p)

 beta_plot_png = out / "compare_posterior_beta_seed0.png"
 hp_plot_png = out / "compare_hyperparam_posteriors_seed0.png"
 plot_error = ""
 try:
  fig1 = plot_method_beta_posterior_comparison(run_output, save_path=str(beta_plot_png))
  fig2 = plot_method_hyperparam_posteriors(run_output, save_path=str(hp_plot_png))
  try:
   import matplotlib.pyplot as plt
   plt.close(fig1)
   plt.close(fig2)
  except Exception:
   pass
 except Exception as exc:
  plot_error = f"{type(exc).__name__}: {exc}"

 return {
  "comparison_csv": str(comparison_csv),
  "speed_csv": str(speed_csv),
  "absolute_error_csv": str(abs_csv),
  "absolute_error_txt": str(abs_txt),
  "metadata_json": str(metadata_json),
  "compare_posterior_beta_png": str(beta_plot_png),
  "compare_hyperparam_posteriors_png": str(hp_plot_png),
  "plot_error": plot_error,
  **{f"{k}_npz": v for k, v in sample_files.items},
 }
