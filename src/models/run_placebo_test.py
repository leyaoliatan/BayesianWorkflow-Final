"""
run_placebo_test.py
===================
Category-level placebo runner for real-data experiments (Part 5 style).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fixed_effects import FixedEffectsModel
from .cohort_period import CohortPeriodModel
from .cohort_period_extended import CohortPeriodExtendedModel
from src.evaluation.placebo_effects import (
    build_support_mask,
    attach_yhat_by_model,
    compute_beta_placebo_te_tables,
    build_placebo_split_metrics,
    summarise_placebo_estimands,
    build_support_aware_model_time_summary,
)
from src.visualization import (
    plot_placebo_tau_t_by_model_lines_ci,
    plot_placebo_tau_ct_extended_lines_ci,
    plot_placebo_beta_diagnostic_3panel,
    plot_placebo_cohort_trends_3x3,
    plot_residual_overlay,
)


def _reindex_placebo_sample(
    df: pd.DataFrame,
    *,
    covid_onset: pd.Timestamp,
    placebo_onset: pd.Timestamp,
    month_col: str,
    cohort_col: str,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    p5_raw = df[df[month_col] < covid_onset].copy()

    # Keep only cohorts that existed before placebo onset.
    # This mirrors the real-treatment setup where post-treatment entrants are excluded.
    cohort_dt = pd.to_datetime(p5_raw[cohort_col], errors="coerce")
    if cohort_dt.isna().any():
        raise ValueError(
            f"Could not parse '{cohort_col}' as datetime for placebo cohort filtering."
        )
    keep_mask = cohort_dt < placebo_onset
    p5 = p5_raw.loc[keep_mask].copy()
    if p5.empty:
        raise ValueError("No rows remain after dropping cohorts entering on/after placebo onset.")

    p5 = p5.sort_values([cohort_col, month_col]).reset_index(drop=True)
    p5["cohort_idx_p5"] = p5[cohort_col].astype("category").cat.codes
    p5["time_idx_p5"] = p5[month_col].astype("category").cat.codes
    start = p5.loc[p5[month_col] >= placebo_onset, "time_idx_p5"]
    if len(start) == 0:
        raise ValueError("Placebo onset has no matching month in pre-COVID sample.")
    placebo_start_idx = int(start.min())
    drop_stats = {
        "rows_pre_covid_raw": int(len(p5_raw)),
        "rows_after_cohort_placebo_filter": int(len(p5)),
        "rows_dropped_by_cohort_placebo_filter": int(len(p5_raw) - len(p5)),
        "n_cohorts_pre_covid_raw": int(p5_raw[cohort_col].nunique()),
        "n_cohorts_after_filter": int(p5[cohort_col].nunique()),
        "n_cohorts_dropped": int(p5_raw[cohort_col].nunique() - p5[cohort_col].nunique()),
    }
    return p5, placebo_start_idx, drop_stats


def run_placebo_test(
    cohort_df: pd.DataFrame,
    *,
    category: str,
    output_root: str | Path,
    covid_onset: str | pd.Timestamp = "2020-03-01",
    placebo_years_back: int = 1,
    month_col: str = "month",
    cohort_col: str = "cohort",
    dv_log_col: str = "dv_log",
    save_artifacts: bool = True,
    save_plots: bool = True,
) -> dict[str, Any]:
    """
    Run placebo split, fit three models once, and export core tables/metadata.
    """
    if dv_log_col not in cohort_df.columns:
        raise ValueError(f"Input cohort_df must include '{dv_log_col}'.")

    covid_onset = pd.to_datetime(covid_onset)
    placebo_onset = (covid_onset - pd.DateOffset(years=placebo_years_back)).normalize()

    p5, placebo_start_idx, drop_stats = _reindex_placebo_sample(
        cohort_df,
        covid_onset=covid_onset,
        placebo_onset=placebo_onset,
        month_col=month_col,
        cohort_col=cohort_col,
    )

    obs_c = p5["cohort_idx_p5"].to_numpy(dtype=int)
    obs_t = p5["time_idx_p5"].to_numpy(dtype=int)
    y = p5[dv_log_col].to_numpy(dtype=float)
    n = len(p5)
    n_c = int(p5["cohort_idx_p5"].nunique())
    n_t = int(p5["time_idx_p5"].nunique())
    support_mask = build_support_mask(
        p5,
        n_cohorts=n_c,
        n_periods=n_t,
        cohort_key="cohort_idx_p5",
        time_key="time_idx_p5",
    )
    n_t_tr = placebo_start_idx
    n_hold = n_t - n_t_tr
    if n_t_tr <= 2 or n_hold <= 0:
        raise ValueError("Invalid placebo split; check date range and placebo onset.")

    mask_tr = obs_t < n_t_tr
    mask_te = ~mask_tr
    obs_c_tr = obs_c[mask_tr]
    obs_t_tr = obs_t[mask_tr]
    y_tr = y[mask_tr]

    # ---- train-fit models (placebo protocol) ----
    # FE on train horizon then AR extrapolation
    fe_tr = FixedEffectsModel(use_global_mean=True).fit(
        y=y_tr, obs_c=obs_c_tr, obs_t=obs_t_tr, n_cohorts=n_c, n_periods=n_t_tr
    )
    beta_ex_mean, beta_ex_std = fe_tr.extrapolate(n_hold)
    beta_fe_all = np.concatenate([fe_tr.beta_, beta_ex_mean])
    beta_fe_std_all = np.concatenate([fe_tr.std_beta_, beta_ex_std])
    y_hat_fe = fe_tr.mu_ + fe_tr.alpha_[obs_c] + beta_fe_all[obs_t]

    # GP models trained on train rows, but full time domain
    cp_tr = CohortPeriodModel(use_global_mean=True).fit(
        y=y_tr, obs_c=obs_c_tr, obs_t=obs_t_tr, n_cohorts=n_c, n_periods=n_t
    )
    ext_tr = CohortPeriodExtendedModel(use_global_mean=True).fit(
        y=y_tr, obs_c=obs_c_tr, obs_t=obs_t_tr, n_cohorts=n_c, n_periods=n_t
    )
    y_hat_cp = cp_tr.predict(obs_c, obs_t)
    y_hat_ext = ext_tr.predict(obs_c, obs_t)

    fit_pred = {
        "FE+AR": y_hat_fe,
        "GP-CP": y_hat_cp,
        "GP-CP-Extended": y_hat_ext,
    }

    # ---- full fits for beta-based placebo comparison ----
    fe_full = FixedEffectsModel(use_global_mean=True).fit(
        y=y, obs_c=obs_c, obs_t=obs_t, n_cohorts=n_c, n_periods=n_t
    )
    cp_full = CohortPeriodModel(use_global_mean=True).fit(
        y=y, obs_c=obs_c, obs_t=obs_t, n_cohorts=n_c, n_periods=n_t
    )
    ext_full = CohortPeriodExtendedModel(use_global_mean=True).fit(
        y=y, obs_c=obs_c, obs_t=obs_t, n_cohorts=n_c, n_periods=n_t
    )

    train_fit = {
        "FE+AR": {
            "mu": fe_tr.mu_,
            "alpha": fe_tr.alpha_,
            "beta": beta_fe_all,
            "beta_std": beta_fe_std_all,
            "y_hat": y_hat_fe,
        },
        "GP-CP": {**cp_tr.results_dict(), "y_hat": y_hat_cp},
        "GP-CP-Extended": {**ext_tr.results_dict(), "y_hat": y_hat_ext},
    }
    full_fit = {
        "FE+AR": fe_full.results_dict(),
        "GP-CP": cp_full.results_dict(),
        "GP-CP-Extended": ext_full.results_dict(),
    }

    model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]
    tau_t_df, tau_ct_df, _legacy_te_summary_df = compute_beta_placebo_te_tables(
        category=category,
        train_fit=train_fit,
        full_fit=full_fit,
        n_t_tr=n_t_tr,
        n_t=n_t,
        n_c=n_c,
        support_mask=support_mask,
        model_order=model_order,
    )

    te_summary_df = summarise_placebo_estimands(
        category=category,
        tau_ct_df=tau_ct_df,
        tau_t_df=tau_t_df,
        model_order=model_order,
    )
    fit_support_summary_train_df = build_support_aware_model_time_summary(
        category=category,
        fit_results=train_fit,
        support_mask=support_mask,
        model_order=model_order,
        scope="train_fit",
        extrapolation_start_idx=n_t_tr,
    )
    fit_support_summary_full_df = build_support_aware_model_time_summary(
        category=category,
        fit_results=full_fit,
        support_mask=support_mask,
        model_order=model_order,
        scope="full_fit",
        extrapolation_start_idx=None,
    )

    # ---- forecast metrics on placebo split ----
    split_metrics_df = build_placebo_split_metrics(
        category=category,
        model_order=model_order,
        fit_pred=fit_pred,
        y=y,
        mask_tr=mask_tr,
        mask_te=mask_te,
    )

    # ---- optional saving ----
    output_root = Path(output_root)
    placebo_dir = output_root / category / "placebo"
    metadata_dir = output_root / category / "metadata"
    written: dict[str, str] = {}
    if save_artifacts:
        placebo_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        path_tau_t = placebo_dir / "tau_t_table.csv"
        path_tau_ct = placebo_dir / "tau_ct_table.csv"
        path_summary = placebo_dir / "te_post_summary_table.csv"
        path_support_train = placebo_dir / "fit_support_summary_train.csv"
        path_support_full = placebo_dir / "fit_support_summary_full.csv"
        path_split_metrics = placebo_dir / "placebo_split_metrics.csv"
        path_yhat = placebo_dir / "placebo_predictions_all_rows.csv"
        path_tau_t_plot = placebo_dir / "tau_t_by_model_lines_ci.png"
        path_tau_ct_ext_plot = placebo_dir / "tau_extended_cohort_lines_ci.png"
        path_beta_diag_plot = placebo_dir / "placebo_beta_diagnostic_3panel.png"
        path_trends_plot = placebo_dir / "placebo_cohort_trends_by_model.png"
        path_resid_plot = placebo_dir / "placebo_residual_overlay.png"
        path_meta = metadata_dir / "placebo_metadata.json"

        tau_t_df.to_csv(path_tau_t, index=False)
        tau_ct_df.to_csv(path_tau_ct, index=False)
        te_summary_df.to_csv(path_summary, index=False)
        fit_support_summary_train_df.to_csv(path_support_train, index=False)
        fit_support_summary_full_df.to_csv(path_support_full, index=False)
        split_metrics_df.to_csv(path_split_metrics, index=False)
        attach_yhat_by_model(p5, fit_pred, dv_log_col=dv_log_col).to_csv(path_yhat, index=False)
        if save_plots:
            fig_tau_t = plot_placebo_tau_t_by_model_lines_ci(
                tau_t_df=tau_t_df,
                model_order=model_order,
                treatment_time_idx=n_t_tr,
                treatment_label="Placebo onset",
                save_path=str(path_tau_t_plot),
            )
            fig_tau_ct_ext = plot_placebo_tau_ct_extended_lines_ci(
                tau_ct_df=tau_ct_df,
                model_order=model_order,
                treatment_time_idx=n_t_tr,
                treatment_label="Placebo onset",
                save_path=str(path_tau_ct_ext_plot),
            )
            fig_trends = plot_placebo_cohort_trends_3x3(
                train_fit=train_fit,
                full_fit=full_fit,
                model_order=model_order,
                support_mask=support_mask,
                treatment_time_idx=n_t_tr,
                treatment_label="Placebo onset",
                extrapolation_start_idx=n_t_tr,
                save_path=str(path_trends_plot),
            )
            fig_beta_diag = plot_placebo_beta_diagnostic_3panel(
                train_fit=train_fit,
                full_fit=full_fit,
                n_t_tr=n_t_tr,
                n_t=n_t,
                model_order=model_order,
                save_path=str(path_beta_diag_plot),
            )
            fig_resid = plot_residual_overlay(
                residuals_by_model={
                    m: y - np.asarray(fit_pred[m], dtype=float) for m in model_order if m in fit_pred
                },
                title="Task B placebo residual overlay by model",
                save_path=str(path_resid_plot),
            )

            plt.close(fig_tau_t)
            plt.close(fig_tau_ct_ext)
            plt.close(fig_trends)
            plt.close(fig_beta_diag)
            plt.close(fig_resid)

        meta = {
            "category": category,
            "covid_onset": str(covid_onset.date()),
            "placebo_onset": str(placebo_onset.date()),
            "n_rows_pre_covid": int(n),
            "n_cohorts": int(n_c),
            "n_periods_pre_covid": int(n_t),
            "train_periods": int(n_t_tr),
            "holdout_periods": int(n_hold),
            "true_effect": 0.0,
            "models": model_order,
            "cohort_placebo_filter": drop_stats,
            "support_mask_density": float(np.mean(support_mask)),
        }
        path_meta.write_text(json.dumps(meta, indent=2))

        written = {
            "tau_t_csv": str(path_tau_t),
            "tau_ct_csv": str(path_tau_ct),
            "te_summary_csv": str(path_summary),
            "fit_support_summary_train_csv": str(path_support_train),
            "fit_support_summary_full_csv": str(path_support_full),
            "split_metrics_csv": str(path_split_metrics),
            "predictions_csv": str(path_yhat),
            "metadata_json": str(path_meta),
        }
        if save_plots:
            written.update(
                {
                    "tau_t_plot_png": str(path_tau_t_plot),
                    "tau_ct_extended_plot_png": str(path_tau_ct_ext_plot),
                    "beta_diagnostic_plot_png": str(path_beta_diag_plot),
                    "cohort_trends_plot_png": str(path_trends_plot),
                    "residual_overlay_plot_png": str(path_resid_plot),
                }
            )

    return {
        "category": category,
        "placebo_onset": placebo_onset,
        "placebo_start_idx": int(placebo_start_idx),
        "train_fit": train_fit,
        "full_fit": full_fit,
        "split_metrics_df": split_metrics_df,
        "tau_t_df": tau_t_df,
        "tau_ct_df": tau_ct_df,
        "te_summary_df": te_summary_df,
        "fit_support_summary_train_df": fit_support_summary_train_df,
        "fit_support_summary_full_df": fit_support_summary_full_df,
        "written_artifacts": written,
    }
