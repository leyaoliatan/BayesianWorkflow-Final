"""
plots.py
========
Visualisation utilities for the BAPC simulation study.

Functions
---------
plot_simulated_data      : DGP diagnostic — show true effects and per-cohort trends.
plot_effect_recovery     : True vs estimated α and β for all models (per seed).
plot_extrapolation       : β trajectory with train/test split shading (per seed).
plot_interaction_heatmap : True vs estimated γ heatmap (GP-CP-Extended).
plot_metric_summary      : Grouped bar chart of Bias / RMSE / Coverage across conditions.
plot_coverage_heatmap    : Heatmap of empirical coverage across model × scenario × task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch


# ── Colour / style helpers ────────────────────────────────────────────────────

_MODEL_COLOURS = {
    "FE+AR": " #2E86AB".strip(),
    "GP-CP": "#A23B72",
    "GP-CP-Extended": "#F18F01",
    # Backward compatibility with old label
    "GP-Ext": "#F18F01",
}
_MODEL_MARKERS = {"FE+AR": "o", "GP-CP": "s", "GP-CP-Extended": "D", "GP-Ext": "D"}
_MODEL_ORDER = ["FE+AR", "GP-CP", "GP-CP-Extended"]
_EFFECT_LABELS = {"alpha": "Cohort effects", "beta": "Period effects", "gamma": "Interaction effects"}
_EST_LINEWIDTH = 1.8
_EST_MARKERSIZE = 4.8
_TRUE_LINEWIDTH = 1.6
_CI_ALPHA = 0.20

_COHORT_CMAP = "tab10"


def _cohort_palette(n: int) -> np.ndarray:
    """Return an (n, 4) RGBA array of cohort colours."""
    cmap = plt.get_cmap(_COHORT_CMAP)
    return cmap(np.linspace(0, 0.9, n))


def _canon_model_name(name: str) -> str:
    if name == "GP-Ext":
        return "GP-CP-Extended"
    return name


def _model_order_present(df: pd.DataFrame) -> list[str]:
    present = {_canon_model_name(m) for m in df["model"].astype(str).unique()}
    return [m for m in _MODEL_ORDER if m in present]


def _save(fig: matplotlib.figure.Figure, save_path: Optional[str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved -> {save_path}")


def _draw_treatment_vline(
    ax: matplotlib.axes.Axes,
    treatment_time_idx: int | None,
    treatment_label: str,
) -> None:
    if treatment_time_idx is None:
        return
    ax.axvline(
        int(treatment_time_idx),
        color="red",
        linestyle="--",
        linewidth=1.4,
        alpha=0.95,
        label=treatment_label,
    )


def plot_realdata_cohort_panel_raw(
    df: pd.DataFrame,
    dv_col: str = "spend_normalized",
    covid_onset: str | pd.Timestamp = "2020-03-01",
    title: str | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot raw cohort trajectories as a single figure (one line per cohort).
    """
    covid_onset = pd.to_datetime(covid_onset)
    cohort_keys = sorted(df["cohort"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(cohort_keys)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for color, cohort_val in zip(colors, cohort_keys):
        tmp = df.loc[df["cohort"] == cohort_val].sort_values("time_idx")
        ax.plot(tmp["time_idx"], tmp[dv_col], color=color, linewidth=1.5, alpha=0.7)

    post_rows = df.loc[df["month"] >= covid_onset, "time_idx"]
    if len(post_rows):
        covid_time_idx = int(post_rows.min())
        ax.axvline(covid_time_idx, color="red", linestyle=":", linewidth=2, label="COVID onset")
        ax.legend(loc="upper right")

    ax.set_title(title or f"Raw {dv_col} by cohort")
    ax.set_xlabel("time_idx (months since 2017-01)")
    ax.set_ylabel(dv_col)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_cohort_panel_log(
    df: pd.DataFrame,
    dv_col: str = "spend_normalized",
    covid_onset: str | pd.Timestamp = "2020-03-01",
    title: str | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot log-transformed cohort trajectories as a single figure (one line per cohort).
    """
    covid_onset = pd.to_datetime(covid_onset)
    cohort_keys = sorted(df["cohort"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(cohort_keys)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for color, cohort_val in zip(colors, cohort_keys):
        tmp = df.loc[df["cohort"] == cohort_val].sort_values("time_idx")
        ax.plot(tmp["time_idx"], tmp["dv_log"], color=color, linewidth=1.5, alpha=0.7)

    post_rows = df.loc[df["month"] >= covid_onset, "time_idx"]
    if len(post_rows):
        covid_time_idx = int(post_rows.min())
        ax.axvline(covid_time_idx, color="red", linestyle=":", linewidth=2, label="COVID onset")
        ax.legend(loc="upper right")

    ax.set_title(title or f"log({dv_col}) by cohort")
    ax.set_xlabel("time_idx (months since 2017-01)")
    ax.set_ylabel(f"log({dv_col})")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def save_realdata_cohort_panel_plots_separate(
    df: pd.DataFrame,
    output_dir: str | Path,
    category_name: str,
    dv_col: str = "spend_normalized",
    covid_onset: str | pd.Timestamp = "2020-03-01",
    include_pdf: bool = True,
) -> dict[str, str]:
    """
    Save cohort-level raw/log plots as two separate files.

    Output names:
    - cohort_panel_raw.png (+ optional .pdf)
    - cohort_panel_log.png (+ optional .pdf)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_png = out_dir / "cohort_panel_raw.png"
    log_png = out_dir / "cohort_panel_log.png"

    fig_raw = plot_realdata_cohort_panel_raw(
        df=df,
        dv_col=dv_col,
        covid_onset=covid_onset,
        title=f"Raw {dv_col} by cohort ({category_name})",
        save_path=str(raw_png),
    )
    plt.close(fig_raw)

    fig_log = plot_realdata_cohort_panel_log(
        df=df,
        dv_col=dv_col,
        covid_onset=covid_onset,
        title=f"log({dv_col}) by cohort ({category_name})",
        save_path=str(log_png),
    )
    plt.close(fig_log)

    outputs: dict[str, str] = {
        "raw_png": str(raw_png),
        "log_png": str(log_png),
    }

    if include_pdf:
        raw_pdf = out_dir / "cohort_panel_raw.pdf"
        log_pdf = out_dir / "cohort_panel_log.pdf"
        fig_raw_pdf = plot_realdata_cohort_panel_raw(
            df=df,
            dv_col=dv_col,
            covid_onset=covid_onset,
            title=f"Raw {dv_col} by cohort ({category_name})",
            save_path=str(raw_pdf),
        )
        plt.close(fig_raw_pdf)
        fig_log_pdf = plot_realdata_cohort_panel_log(
            df=df,
            dv_col=dv_col,
            covid_onset=covid_onset,
            title=f"log({dv_col}) by cohort ({category_name})",
            save_path=str(log_pdf),
        )
        plt.close(fig_log_pdf)
        outputs["raw_pdf"] = str(raw_pdf)
        outputs["log_pdf"] = str(log_pdf)

    return outputs


def plot_realdata_taska_observed_vs_fitted(
    df: pd.DataFrame,
    fit_results: dict[str, dict],
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Treatment onset",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot observed mean dv_log by time_idx and fitted means for each model.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    tmp = df.copy()
    tmp["y_obs"] = tmp["dv_log"].astype(float)
    agg = (
        tmp.groupby("time_idx", as_index=False)
        .agg(
            y_obs_mean=("y_obs", "mean"),
            y_obs_std=("y_obs", "std"),
            y_obs_n=("y_obs", "count"),
        )
        .sort_values("time_idx")
    )
    x = agg["time_idx"].to_numpy(dtype=int)
    y_obs = agg["y_obs_mean"].to_numpy(dtype=float)
    se_obs = agg["y_obs_std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(np.maximum(agg["y_obs_n"].to_numpy(dtype=float), 1.0))
    lo_obs = y_obs - 1.96 * se_obs
    hi_obs = y_obs + 1.96 * se_obs

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, y_obs, "ko-", linewidth=1.8, label="Observed mean (dv_log)")
    ax.fill_between(x, lo_obs, hi_obs, color="black", alpha=0.10)

    for model_name in model_order:
        if model_name not in fit_results:
            continue
        y_hat = np.asarray(fit_results[model_name]["y_hat"], dtype=float)
        tmp[f"y_hat_{model_name}"] = y_hat
        pred_agg = (
            tmp.groupby("time_idx", as_index=False)
            .agg(
                y_hat_mean=(f"y_hat_{model_name}", "mean"),
                y_hat_std=(f"y_hat_{model_name}", "std"),
                y_hat_n=(f"y_hat_{model_name}", "count"),
            )
            .sort_values("time_idx")
        )
        y_hat_mean = pred_agg["y_hat_mean"].to_numpy(dtype=float)
        se_hat = pred_agg["y_hat_std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(
            np.maximum(pred_agg["y_hat_n"].to_numpy(dtype=float), 1.0)
        )
        lo_hat = y_hat_mean - 1.96 * se_hat
        hi_hat = y_hat_mean + 1.96 * se_hat
        color = _MODEL_COLOURS.get(model_name, "tab:blue")
        marker = _MODEL_MARKERS.get(model_name, "o")
        ax.plot(
            pred_agg["time_idx"].to_numpy(dtype=int),
            y_hat_mean,
            marker=marker,
            linewidth=1.6,
            color=color,
            label=model_name,
            alpha=0.9,
        )
        ax.fill_between(
            pred_agg["time_idx"].to_numpy(dtype=int),
            lo_hat,
            hi_hat,
            color=color,
            alpha=0.12,
        )

    _draw_treatment_vline(ax, treatment_time_idx, treatment_label)

    ax.set_title("Task A: Observed vs fitted mean by time")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("dv_log")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_taska_beta_comparison(
    fit_results: dict[str, dict],
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Treatment onset",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot period effects beta with 95% CIs for all Task A models.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_name in model_order:
        if model_name not in fit_results:
            continue
        out = fit_results[model_name]
        beta = np.asarray(out["beta"], dtype=float)
        std_beta = np.asarray(out.get("std_beta", np.zeros_like(beta)), dtype=float)
        xt = np.arange(len(beta))
        lo = beta - 1.96 * std_beta
        hi = beta + 1.96 * std_beta
        color = _MODEL_COLOURS.get(model_name, "tab:blue")
        marker = _MODEL_MARKERS.get(model_name, "o")
        ax.plot(xt, beta, color=color, marker=marker, linewidth=1.7, label=model_name)
        if np.any(std_beta > 0):
            ax.fill_between(xt, lo, hi, color=color, alpha=0.17)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
    ax.set_title("Task A: Time effects (beta) with 95% CI")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("beta")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_taska_alpha_comparison(
    fit_results: dict[str, dict],
    model_order: list[str] | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot cohort effects alpha with 95% CIs for all Task A models.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model_name in model_order:
        if model_name not in fit_results:
            continue
        out = fit_results[model_name]
        alpha = np.asarray(out["alpha"], dtype=float)
        std_alpha = np.asarray(out.get("std_alpha", np.zeros_like(alpha)), dtype=float)
        xc = np.arange(len(alpha))
        lo = alpha - 1.96 * std_alpha
        hi = alpha + 1.96 * std_alpha
        color = _MODEL_COLOURS.get(model_name, "tab:blue")
        marker = _MODEL_MARKERS.get(model_name, "o")
        ax.plot(xc, alpha, color=color, marker=marker, linewidth=1.7, label=model_name)
        if np.any(std_alpha > 0):
            ax.fill_between(xc, lo, hi, color=color, alpha=0.17)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Task A: Cohort effects (alpha) with 95% CI")
    ax.set_xlabel("cohort_idx")
    ax.set_ylabel("alpha")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_taska_gamma_heatmap(
    gamma: np.ndarray,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Heatmap for extended-model interaction gamma estimates.
    """
    g = np.asarray(gamma, dtype=float)
    vmax = float(np.max(np.abs(g))) if g.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(g, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Task A: GP-CP-Extended interaction gamma")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("cohort_idx")
    plt.colorbar(im, ax=ax, shrink=0.9, label="gamma")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_taska_effects_small_multiples_fixed_ylim(
    fit_results: dict[str, dict],
    effect_key: str = "beta",
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Treatment onset",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Small-multiples effect plot with fixed y-axis across models.

    Produces 3 subplots (FE+AR, GP-CP, GP-CP-Extended), each showing mean ± 95% CI
    for the requested effect ('alpha' or 'beta').
    """
    if effect_key not in {"alpha", "beta"}:
        raise ValueError("effect_key must be 'alpha' or 'beta'.")

    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    present_models = [m for m in model_order if m in fit_results]
    if not present_models:
        raise ValueError("No requested models found in fit_results.")

    # Determine global y-limits across all models to keep panels comparable.
    all_lo = []
    all_hi = []
    for m in present_models:
        out = fit_results[m]
        mean = np.asarray(out[effect_key], dtype=float)
        std = np.asarray(out.get(f"std_{effect_key}", np.zeros_like(mean)), dtype=float)
        all_lo.append(mean - 1.96 * std)
        all_hi.append(mean + 1.96 * std)
    y_min = float(min(np.min(v) for v in all_lo))
    y_max = float(max(np.max(v) for v in all_hi))
    pad = 0.08 * (y_max - y_min + 1e-12)
    y_lim = (y_min - pad, y_max + pad)

    fig, axes = plt.subplots(1, len(present_models), figsize=(5.2 * len(present_models), 4.6), sharey=True)
    if len(present_models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, present_models):
        out = fit_results[model_name]
        mean = np.asarray(out[effect_key], dtype=float)
        std = np.asarray(out.get(f"std_{effect_key}", np.zeros_like(mean)), dtype=float)
        x = np.arange(len(mean))
        lo = mean - 1.96 * std
        hi = mean + 1.96 * std
        color = _MODEL_COLOURS.get(model_name, "tab:blue")
        marker = _MODEL_MARKERS.get(model_name, "o")

        ax.plot(x, mean, color=color, marker=marker, linewidth=1.7, markersize=4.5)
        if np.any(std > 0):
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        if effect_key == "beta":
            _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
        ax.set_ylim(*y_lim)
        ax.set_title(model_name)
        ax.set_xlabel("cohort_idx" if effect_key == "alpha" else "time_idx")
        ax.grid(alpha=0.28)

    axes[0].set_ylabel(effect_key)
    fig.suptitle(f"Task A: {effect_key} effects by model (fixed y-axis)", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_realdata_cohort_y_trends_by_model(
    fit_results: dict[str, dict],
    model_order: list[str] | None = None,
    max_cohorts: int = 20,
    support_mask: np.ndarray | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Treatment onset",
    extrapolation_start_idx: int | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot cohort-level fitted y-trends (mu + alpha + beta + gamma) in 3 model panels.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]
    present_models = [m for m in model_order if m in fit_results]
    if not present_models:
        raise ValueError("No requested models found in fit_results.")

    fig, axes = plt.subplots(1, len(present_models), figsize=(5.4 * len(present_models), 4.8), sharey=True)
    if len(present_models) == 1:
        axes = [axes]

    show_extrap_line = (
        extrapolation_start_idx is not None
        and (treatment_time_idx is None or int(extrapolation_start_idx) != int(treatment_time_idx))
    )

    for ax, model_name in zip(axes, present_models):
        out = fit_results[model_name]
        mu = float(out.get("mu", 0.0))
        alpha = np.asarray(out.get("alpha", []), dtype=float)
        beta = np.asarray(out.get("beta", []), dtype=float)
        gamma = np.asarray(out.get("gamma", np.zeros((len(alpha), len(beta)))), dtype=float)
        if gamma.ndim == 1:
            gamma = np.zeros((len(alpha), len(beta)), dtype=float)
        if gamma.shape != (len(alpha), len(beta)):
            gamma = np.zeros((len(alpha), len(beta)), dtype=float)

        n_c = len(alpha)
        n_t = len(beta)
        xt = np.arange(n_t, dtype=int)
        cohort_idxs = np.arange(n_c, dtype=int) if n_c <= max_cohorts else np.linspace(0, n_c - 1, max_cohorts).astype(int)
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(cohort_idxs), 1)))

        for color, c in zip(colors, cohort_idxs):
            trend = mu + alpha[c] + beta + gamma[c]
            if support_mask is not None and c < support_mask.shape[0] and n_t <= support_mask.shape[1]:
                mask_obs = support_mask[c, :n_t].astype(bool)
                if extrapolation_start_idx is not None:
                    mask_ex = xt >= int(extrapolation_start_idx)
                else:
                    mask_ex = np.zeros_like(mask_obs, dtype=bool)
                mask_pre = ~mask_ex

                in_support_pre = np.where(mask_obs & mask_pre, trend, np.nan)
                in_support_ex = np.where(mask_obs & mask_ex, trend, np.nan)
                out_support = np.where(~mask_obs, trend, np.nan)

                if np.any(mask_obs & mask_pre):
                    ax.plot(xt, in_support_pre, color=color, linewidth=1.2, alpha=0.92)
                if np.any(mask_obs & mask_ex):
                    ax.plot(xt, in_support_ex, color=color, linewidth=1.1, alpha=0.78, linestyle="--")
                if np.any(~mask_obs):
                    ax.plot(xt, out_support, color=color, linewidth=1.0, alpha=0.30, linestyle=":")
            else:
                if extrapolation_start_idx is not None:
                    pre = np.where(xt < int(extrapolation_start_idx), trend, np.nan)
                    ex = np.where(xt >= int(extrapolation_start_idx), trend, np.nan)
                    ax.plot(xt, pre, color=color, linewidth=1.2, alpha=0.90)
                    ax.plot(xt, ex, color=color, linewidth=1.1, alpha=0.78, linestyle="--")
                else:
                    ax.plot(xt, trend, color=color, linewidth=1.1, alpha=0.85)

        # Support-aware mean fitted y trend over active observed cohorts at each t.
        if support_mask is not None and n_t <= support_mask.shape[1]:
            y_mean = np.full(n_t, np.nan, dtype=float)
            for t in range(n_t):
                active = np.flatnonzero(support_mask[:n_c, t]).astype(int)
                if active.size == 0:
                    continue
                y_mean[t] = float(np.mean(mu + alpha[active] + beta[t] + gamma[active, t]))
            if extrapolation_start_idx is not None:
                pre = np.where(xt < int(extrapolation_start_idx), y_mean, np.nan)
                ex = np.where(xt >= int(extrapolation_start_idx), y_mean, np.nan)
                ax.plot(xt, pre, color="black", linestyle="-", linewidth=1.9, label="Support-aware mean fitted y")
                ax.plot(xt, ex, color="black", linestyle="--", linewidth=1.9)
            else:
                ax.plot(xt, y_mean, color="black", linestyle="-", linewidth=1.9, label="Support-aware mean fitted y")
        else:
            ax.plot(xt, mu + beta, color="black", linestyle="--", linewidth=1.8, label="Common additive trend")

        if extrapolation_start_idx is not None:
            ax.axvspan(int(extrapolation_start_idx) - 0.5, n_t - 0.5, color="grey", alpha=0.12)
            if show_extrap_line:
                ax.axvline(
                    int(extrapolation_start_idx),
                    color="grey",
                    linestyle="-.",
                    linewidth=1.1,
                    alpha=0.9,
                )
        _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
        ax.set_title(model_name)
        ax.set_xlabel("time_idx")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("fitted dv_log trend")
    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.9, label="Support-aware mean fitted y"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.4, label=treatment_label),
        Line2D([0], [0], color="#2ca02c", linestyle="-", linewidth=1.2, label="Observed support fit (M=1, pre)"),
    ]
    if extrapolation_start_idx is not None:
        legend_handles.append(
            Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.8, label="Observed support extrapolation (post)")
        )
    if support_mask is not None:
        legend_handles.append(
            Line2D([0], [0], color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.45, label="Out-of-support cells (M=0)")
        )
    if extrapolation_start_idx is not None:
        legend_handles.append(Patch(facecolor="grey", edgecolor="none", alpha=0.12, label="Extrapolation window"))
    if show_extrap_line:
        legend_handles.append(
            Line2D([0], [0], color="grey", linestyle="-.", linewidth=1.1, label="Start of extrapolation")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )
    fig.suptitle("Cohort-level fitted y-trends by model", y=1.03, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _save(fig, save_path)
    return fig


def plot_placebo_tau_t_by_model_lines_ci(
    tau_t_df: pd.DataFrame,
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Placebo treatment",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot tau_t by model with 95% CI bands.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model in model_order:
        sub = tau_t_df[tau_t_df["model"] == model].copy()
        if treatment_time_idx is not None:
            sub = sub[sub["time_idx"].astype(int) >= int(treatment_time_idx)]
        elif "is_post" in sub.columns:
            sub = sub[sub["is_post"].astype(bool)]
        sub = sub.sort_values("time_idx")
        if sub.empty:
            continue
        x = sub["time_idx"].to_numpy(dtype=int)
        y = sub["tau_t"].to_numpy(dtype=float)
        lo = sub["tau_t_lo95"].to_numpy(dtype=float)
        hi = sub["tau_t_hi95"].to_numpy(dtype=float)
        color = _MODEL_COLOURS.get(model, "tab:blue")
        marker = _MODEL_MARKERS.get(model, "o")
        ax.plot(x, y, color=color, marker=marker, linewidth=1.7, label=model)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
    ax.set_title("Placebo Task B: tau_t by model (95% CI)")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("tau_t")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_placebo_att_t_by_model_lines_ci(
    att_t_df: pd.DataFrame,
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Placebo treatment",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot ATT_t by model with 95% CI bands.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for model in model_order:
        sub = att_t_df[att_t_df["model"] == model].sort_values("time_idx")
        if sub.empty:
            continue
        x = sub["time_idx"].to_numpy(dtype=int)
        y = sub["att_t"].to_numpy(dtype=float)
        lo = sub["att_t_lo95"].to_numpy(dtype=float)
        hi = sub["att_t_hi95"].to_numpy(dtype=float)
        color = _MODEL_COLOURS.get(model, "tab:blue")
        marker = _MODEL_MARKERS.get(model, "o")
        ax.plot(x, y, color=color, marker=marker, linewidth=1.7, label=model)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
    ax.set_title("Placebo Task B: ATT_t by model (95% CI)")
    ax.set_xlabel("time_idx")
    ax.set_ylabel("ATT_t")
    ax.grid(alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_placebo_tau_ct_extended_lines_ci(
    tau_ct_df: pd.DataFrame,
    model_order: list[str] | None = None,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Placebo onset",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot cohort-level tau_c,t for all models in 3 subplots.
    Solid for observed cells, dashed for post extrapolation, dotted for other out-of-support.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]
    present_models = [m for m in model_order if m in set(tau_ct_df["model"].astype(str).unique())]
    if not present_models:
        raise ValueError("No requested models found in tau_ct_df.")

    fig, axes = plt.subplots(1, len(present_models), figsize=(5.4 * len(present_models), 4.9), sharey=True)
    if len(present_models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, present_models):
        sub = tau_ct_df[tau_ct_df["model"] == model_name].copy()
        cohorts = sorted(sub["cohort_idx"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(cohorts), 1)))

        for color, c in zip(colors, cohorts):
            g = sub[sub["cohort_idx"] == c].sort_values("time_idx")
            x = g["time_idx"].to_numpy(dtype=int)
            y = g["tau_ct"].to_numpy(dtype=float)
            lo = g["tau_ct_lo95"].to_numpy(dtype=float)
            hi = g["tau_ct_hi95"].to_numpy(dtype=float)
            is_obs = g.get("is_observed_ct", pd.Series([True] * len(g))).astype(bool).to_numpy()
            is_post = g.get("is_post", pd.Series([False] * len(g))).astype(bool).to_numpy()
            is_ex = (~is_obs) & is_post
            is_out = ~is_obs & ~is_post

            y_obs = np.where(is_obs, y, np.nan)
            y_ex = np.where(is_ex, y, np.nan)
            y_out = np.where(is_out, y, np.nan)
            lo_obs = np.where(is_obs, lo, np.nan)
            hi_obs = np.where(is_obs, hi, np.nan)
            lo_ex = np.where(is_ex, lo, np.nan)
            hi_ex = np.where(is_ex, hi, np.nan)

            if np.any(is_obs):
                ax.plot(x, y_obs, color=color, linewidth=1.2, alpha=0.90)
                ax.fill_between(x, lo_obs, hi_obs, color=color, alpha=0.10)
            if np.any(is_ex):
                ax.plot(x, y_ex, color=color, linewidth=1.1, alpha=0.78, linestyle="--")
                ax.fill_between(x, lo_ex, hi_ex, color=color, alpha=0.08)
            if np.any(is_out):
                ax.plot(x, y_out, color=color, linewidth=1.0, alpha=0.30, linestyle=":")

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
        ax.set_title(model_name)
        ax.set_xlabel("time_idx")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("tau_c,t")
    legend_handles = [
        Line2D([0], [0], color="#2ca02c", linestyle="-", linewidth=1.2, label="Observed support (M=1)"),
        Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.8, label="Post extrapolation"),
        Line2D([0], [0], color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.45, label="Other out-of-support"),
        Patch(facecolor="#2ca02c", alpha=0.10, edgecolor="none", label="95% CI"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.4, label=treatment_label),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(5, len(legend_handles)),
        frameon=False,
    )
    fig.suptitle("Placebo Task B: cohort-level tau_c,t by model", y=1.03, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _save(fig, save_path)
    return fig


def plot_placebo_beta_diagnostic_3panel(
    train_fit: dict[str, dict],
    full_fit: dict[str, dict],
    *,
    n_t_tr: int,
    n_t: int,
    model_order: list[str] | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Three-panel diagnostic: beta counterfactual vs observed fitted beta with 95% CI.
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]

    present = [m for m in model_order if (m in train_fit and m in full_fit)]
    if not present:
        raise ValueError("No model overlap between train_fit and full_fit.")

    fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.8), sharey=True)
    if len(present) == 1:
        axes = [axes]

    post_idx = np.arange(n_t_tr, n_t, dtype=int)
    pre_idx = np.arange(n_t_tr, dtype=int)

    for ax, model in zip(axes, present):
        beta_cf = np.asarray(train_fit[model]["beta"], dtype=float)
        beta_cf_std = np.asarray(
            train_fit[model].get("beta_std", train_fit[model].get("std_beta", np.zeros_like(beta_cf))),
            dtype=float,
        )
        beta_obs_raw = np.asarray(full_fit[model]["beta"], dtype=float)
        beta_obs_std = np.asarray(full_fit[model].get("std_beta", np.zeros_like(beta_obs_raw)), dtype=float)

        shift = float(np.mean(beta_cf[pre_idx] - beta_obs_raw[pre_idx])) if len(pre_idx) else 0.0
        beta_obs = beta_obs_raw + shift

        x = np.arange(n_t, dtype=int)
        color = _MODEL_COLOURS.get(model, "tab:blue")

        ax.axvspan(n_t_tr - 0.5, n_t - 0.5, color="grey", alpha=0.12, label="Placebo post window")
        ax.plot(x, beta_obs, color="black", linewidth=1.6, label="Observed fitted beta (aligned)")
        ax.fill_between(
            x,
            beta_obs - 1.96 * beta_obs_std,
            beta_obs + 1.96 * beta_obs_std,
            color="black",
            alpha=0.08,
        )
        ax.plot(x, beta_cf, color=color, linewidth=1.8, label="Counterfactual beta (train+extrap)")
        ax.fill_between(
            x,
            beta_cf - 1.96 * beta_cf_std,
            beta_cf + 1.96 * beta_cf_std,
            color=color,
            alpha=0.16,
        )
        ax.plot(post_idx, beta_obs[post_idx], color="black", linewidth=2.0)
        ax.plot(post_idx, beta_cf[post_idx], color=color, linewidth=2.1)
        ax.axvline(n_t_tr, color="red", linestyle="--", linewidth=1.2, label="Placebo onset")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(model)
        ax.set_xlabel("time_idx")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("beta")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.suptitle("Placebo Task B: beta diagnostic (counterfactual vs observed fitted)", y=1.03, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _save(fig, save_path)
    return fig


def plot_residual_overlay(
    residuals_by_model: dict[str, np.ndarray],
    title: str = "Residual overlay by model",
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Overlay residual distributions for multiple models with zero reference.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5))
    used = []
    for model in ["FE+AR", "GP-CP", "GP-CP-Extended"]:
        if model not in residuals_by_model:
            continue
        resid = np.asarray(residuals_by_model[model], dtype=float).ravel()
        resid = resid[np.isfinite(resid)]
        if resid.size == 0:
            continue
        ax.hist(
            resid,
            bins=35,
            density=True,
            alpha=0.30,
            label=model,
            color=_MODEL_COLOURS.get(model, "grey"),
            edgecolor="white",
            linewidth=0.4,
        )
        used.append(model)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.9, label="Zero residual")
    ax.set_title(title)
    ax.set_xlabel("residual")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    if used:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_placebo_cohort_trends_3x3(
    *,
    train_fit: dict[str, dict],
    full_fit: dict[str, dict],
    support_mask: np.ndarray,
    model_order: list[str] | None = None,
    max_cohorts: int = 20,
    treatment_time_idx: int | None = None,
    treatment_label: str = "Placebo onset",
    extrapolation_start_idx: int | None = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    3xK panel for placebo cohort trends (no CI):
      row 1: train-fit y_ct by cohort
      row 2: full-fit y_ct by cohort
      row 3: tau_ct = y_ct(full-fit) - y_ct(train-fit) by cohort
    """
    if model_order is None:
        model_order = ["FE+AR", "GP-CP", "GP-CP-Extended"]
    models = [m for m in model_order if m in train_fit and m in full_fit]
    if not models:
        raise ValueError("No model overlap between train_fit and full_fit.")

    msk = np.asarray(support_mask, dtype=bool)
    n_c, n_t = msk.shape
    fig, axes = plt.subplots(3, len(models), figsize=(5.4 * len(models), 11.2), sharex=True)
    if len(models) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    row_titles = ["Train-fit y_ct", "Full-fit y_ct", "tau_ct = full-fit - train-fit"]

    def _grid_from_fit(out: dict) -> np.ndarray:
        mu = float(out.get("mu", 0.0))
        alpha = np.asarray(out.get("alpha", np.zeros(n_c)), dtype=float)
        beta = np.asarray(out.get("beta", np.zeros(n_t)), dtype=float)
        gamma = np.asarray(out.get("gamma", np.zeros((len(alpha), len(beta)))), dtype=float)
        if gamma.ndim == 1 or gamma.shape != (len(alpha), len(beta)):
            gamma = np.zeros((len(alpha), len(beta)), dtype=float)
        cc = min(n_c, len(alpha))
        tt = min(n_t, len(beta))
        y = np.full((n_c, n_t), np.nan, dtype=float)
        for c in range(cc):
            for t in range(tt):
                y[c, t] = mu + alpha[c] + beta[t] + gamma[c, t]
        return y

    xt = np.arange(n_t, dtype=int)
    cohorts = np.arange(n_c, dtype=int) if n_c <= max_cohorts else np.linspace(0, n_c - 1, max_cohorts).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(cohorts), 1)))

    for j, model in enumerate(models):
        y_tr = _grid_from_fit(train_fit[model])
        y_full = _grid_from_fit(full_fit[model])
        y_tau = y_full - y_tr
        for i, panel in enumerate([y_tr, y_full, y_tau]):
            ax = axes[i, j]
            for color, c in zip(colors, cohorts):
                s = panel[c, :]
                obs = msk[c, :]
                post = (xt >= int(extrapolation_start_idx)) if extrapolation_start_idx is not None else np.zeros_like(obs, dtype=bool)
                obs_pre = obs & (~post)
                obs_post = obs & post
                out_support = ~obs

                if np.any(obs_pre):
                    ax.plot(xt, np.where(obs_pre, s, np.nan), color=color, linewidth=1.1, alpha=0.9)
                if np.any(obs_post):
                    ax.plot(xt, np.where(obs_post, s, np.nan), color=color, linewidth=1.1, alpha=0.75, linestyle="--")
                if np.any(out_support):
                    ax.plot(xt, np.where(out_support, s, np.nan), color=color, linewidth=1.0, alpha=0.3, linestyle=":")

            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
            _draw_treatment_vline(ax, treatment_time_idx, treatment_label)
            ax.grid(alpha=0.24)
            ax.set_title(f"{model}" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(row_titles[i])
            if i == 2:
                ax.set_xlabel("time_idx")

    legend_handles = [
        Line2D([0], [0], color="#2ca02c", linestyle="-", linewidth=1.1, label="Observed support (pre)"),
        Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=1.1, alpha=0.75, label="Observed post/extrapolation"),
        Line2D([0], [0], color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.35, label="Out-of-support"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.3, label=treatment_label),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(4, len(legend_handles)),
        frameon=False,
    )
    fig.suptitle("Placebo Cohort Trends: train-fit, full-fit, and tau_ct", y=1.01, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))
    _save(fig, save_path)
    return fig


def _effect_label(effect_key: str) -> str:
    return _EFFECT_LABELS.get(effect_key, effect_key)


def simulation_seed_output_dir(
    results_root: str | Path,
    scenario: str,
    seed: int | str,
    create: bool = True,
) -> Path:
    """
    Return output dir path following:
        results/simulations/{scenario}/{seed}
    """
    p = Path(results_root) / "simulations" / str(scenario) / str(seed)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _has_nonzero_gamma_truth(fit_info: dict) -> bool:
    # Pattern 1: top-level key
    if "gamma_true" in fit_info:
        g = np.asarray(fit_info["gamma_true"], dtype=float)
        if g.size > 0 and np.any(np.abs(g) > 0):
            return True
    # Pattern 2: per-model nested keys
    for model in _MODEL_ORDER:
        if model in fit_info and isinstance(fit_info[model], dict) and "gamma_true" in fit_info[model]:
            g = np.asarray(fit_info[model]["gamma_true"], dtype=float)
            if g.size > 0 and np.any(np.abs(g) > 0):
                return True
    return False


# ── Hyperparameter pretty-printer (shared by plot_simulated_data) ─────────────

def _print_dgp_header(data: dict) -> None:
    """
    Print a formatted summary of the DGP configuration to stdout.

    Shows the seed, scenario, grid dimensions, and all GP hyperparameters with
    a one-line description of what each controls.
    """
    hp = data["hyperparams"]
    scenario = "B  (PT-False, interaction=True)" if data["interaction"] else "A  (PT-True,  interaction=False)"
    seed_str  = str(data["seed"]) if data["seed"] is not None else "None (random)"

    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════╗",
        "║              DGP Configuration Summary                          ║",
        "╠══════════════════════════════════════════════════════════════════╣",
        f"║  Seed          : {seed_str:<48}║",
        f"║  Scenario      : {scenario:<48}║",
        f"║  Grid          : {data['n_cohorts']} cohorts × {data['n_periods']} periods × {data['n_reps']} reps/cell"
        f"  (N = {data['n_cohorts'] * data['n_periods'] * data['n_reps']}){'':<6}║",
        "╠══════════════════════════════════════════════════════════════════╣",
        "║  Hyperparameter       Value   Effect on generated data          ║",
        "╠══════════════════════════════════════════════════════════════════╣",
        f"║  ell_c  (cohort ℓ)  : {hp['ell_c']:>5.2f}   larger → smoother α across cohorts      ║",
        f"║  sf_c   (cohort σ)  : {hp['sf_c']:>5.2f}   larger → wider spread of cohort effects  ║",
        f"║  ell_t  (time ℓ)    : {hp['ell_t']:>5.2f}   larger → smoother β trend over time      ║",
        f"║  sf_t   (time σ)    : {hp['sf_t']:>5.2f}   larger → stronger common time trend       ║",
        f"║  sn     (noise σ)   : {hp['sn']:>5.2f}   larger → noisier obs, lower SNR           ║",
    ]

    if data["interaction"]:
        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  Interaction hyperparameters (Scenario B)                        ║",
            f"║  ell_gc (inter. cohort ℓ): {hp['ell_gc']:>5.2f}  larger → correlated PT violations    ║",
            f"║  sf_gc  (inter. cohort σ): {hp['sf_gc']:>5.2f}  larger → stronger PT violation scale  ║",
            f"║  ell_gt (inter. time  ℓ): {hp['ell_gt']:>5.2f}  larger → gradual PT-violation dynamics ║",
            f"║  sf_gt  (inter. time  σ): {hp['sf_gt']:>5.2f}  larger → stronger time-dim interaction  ║",
        ]
    else:
        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  Interaction hyperparameters: N/A  (interaction=False)           ║",
        ]

    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    lines.append("")
    print("\n".join(lines))


# ── DGP diagnostic plot ───────────────────────────────────────────────────────

def plot_simulated_data(
    data: dict,
    print_config: bool = True,
    show_observations: bool = True,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Diagnostic plot for a simulated dataset returned by ``simulate_cohort_data``.

    Panels produced
    ---------------
    Scenario A (interaction=False)  —  1 row × 3 columns:
        [0] Cohort effects α_c         (line + marker plot)
        [1] Time effects β_t           (line plot over periods)
        [2] Per-cohort time trends     (α_c + β_t, one coloured line per cohort)
            → lines should be exactly *parallel* (visual sanity check for PT-True)

    Scenario B (interaction=True)  —  2 rows × 3 columns:
        [0,0] Cohort effects α_c
        [0,1] Time effects β_t
        [0,2] Per-cohort time trends   (α_c + β_t + γ_{c,t})
              → lines are *non-parallel*; divergence reveals PT violation
        [1,0] Interaction γ heatmap    (true γ_{c,t}; diverging colormap)
        [1,1] Observed cell means      (ȳ_{c,t} averaged over reps)
        [1,2] Empty — reserved for notes / legend

    Parameters
    ----------
    data           : dict returned by ``simulate_cohort_data``
    print_config   : if True, print the formatted hyperparameter summary to stdout
    show_observations : if True, include observed cell-mean heatmap (Scenario B only)
    save_path      : if given, save figure to this path (e.g. ``"fig.pdf"``)

    Returns
    -------
    matplotlib.figure.Figure
    """
    if print_config:
        _print_dgp_header(data)

    # ── Unpack tensors to numpy ───────────────────────────────────────────────
    alpha = data["alpha_true"].numpy()          # [C]
    beta  = data["beta_true"].numpy()           # [T]
    gamma = data["gamma_true"].numpy()          # [C, T]
    y     = data["y"].numpy()                   # [N]
    obs_c = data["obs_c"].numpy()               # [N]
    obs_t = data["obs_t"].numpy()               # [N]

    C = data["n_cohorts"]
    T = data["n_periods"]
    has_interaction = data["interaction"]

    colours   = _cohort_palette(C)
    t_axis    = np.arange(1, T + 1)
    c_axis    = np.arange(1, C + 1)

    # ── Compute observed cell means (ȳ_{c,t}) ────────────────────────────────
    cell_mean = np.full((C, T), np.nan)
    for c in range(C):
        for t in range(T):
            mask = (obs_c == c) & (obs_t == t)
            if mask.any():
                cell_mean[c, t] = y[mask].mean()

    # ── Layout ────────────────────────────────────────────────────────────────
    if has_interaction:
        fig = plt.figure(figsize=(15, 9))
        gs  = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.45, wspace=0.35,
        )
        ax_alpha  = fig.add_subplot(gs[0, 0])
        ax_beta   = fig.add_subplot(gs[0, 1])
        ax_trends = fig.add_subplot(gs[0, 2])
        ax_gamma  = fig.add_subplot(gs[1, 0])
        ax_obs    = fig.add_subplot(gs[1, 1])
        ax_notes  = fig.add_subplot(gs[1, 2])
    else:
        fig, (ax_alpha, ax_beta, ax_trends) = plt.subplots(
            1, 3, figsize=(15, 4.5)
        )
        fig.subplots_adjust(wspace=0.35)

    scenario_tag = "B  (PT-False)" if has_interaction else "A  (PT-True)"
    seed_tag     = f"seed={data['seed']}" if data["seed"] is not None else "seed=None"
    fig.suptitle(
        f"Simulated DGP  —  Scenario {scenario_tag},  {seed_tag}",
        fontsize=13, fontweight="bold", y=1.01 if not has_interaction else 1.00,
    )

    # ── Panel 0: Cohort effects α_c ───────────────────────────────────────────
    # Keep style parallel to the beta panel (line + markers).
    ax_alpha.plot(
        c_axis,
        alpha,
        color="steelblue",
        lw=_EST_LINEWIDTH,
        marker="o",
        markersize=_EST_MARKERSIZE,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )
    ax_alpha.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_alpha.set_xlabel("Cohort index  c", fontsize=10)
    ax_alpha.set_ylabel("cohort effect", fontsize=10)
    ax_alpha.set_title("True cohort effects", fontsize=11)
    ax_alpha.set_xlim(0.5, C + 0.5)
    ax_alpha.grid(alpha=0.25)
    # Annotate range
    ax_alpha.text(
        0.97, 0.03,
        f"range: [{alpha.min():.2f}, {alpha.max():.2f}]",
        transform=ax_alpha.transAxes, ha="right", va="bottom",
        fontsize=8, color="grey",
    )

    # ── Panel 1: Time effects β_t ─────────────────────────────────────────────
    ax_beta.plot(
        t_axis,
        beta,
        color="darkorange",
        lw=_EST_LINEWIDTH,
        marker="o",
        markersize=_EST_MARKERSIZE,
        markerfacecolor="white",
        markeredgewidth=1.1,
    )
    ax_beta.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_beta.set_xlabel("Period  t", fontsize=10)
    ax_beta.set_ylabel("period effect", fontsize=10)
    ax_beta.set_title("True period effects", fontsize=11)
    ax_beta.grid(alpha=0.25)
    ax_beta.text(
        0.97, 0.03,
        f"ell_t={data['hyperparams']['ell_t']:.1f},  sf_t={data['hyperparams']['sf_t']:.1f}",
        transform=ax_beta.transAxes, ha="right", va="bottom",
        fontsize=8, color="grey",
    )

    # ── Panel 2: Per-cohort time trends ───────────────────────────────────────
    for c in range(C):
        trend = alpha[c] + beta + gamma[c]   # [T];  gamma[c] = 0 if PT-True
        ax_trends.plot(
            t_axis, trend,
            color=colours[c], lw=1.8,
            label=f"c={c + 1}" if C <= 12 else None,
            alpha=0.85,
        )
    # Common parallel trend (β only) shown as a thick black dashed reference
    ax_trends.plot(
        t_axis, beta, color="black", lw=2.5, linestyle="--",
        label="Common β\n(parallel trend)", zorder=5,
    )
    trend_title = (
        "Per-cohort trends  α_c + β_t + γ_{c,t}\n(non-parallel — PT violation)"
        if has_interaction
        else "Per-cohort trends  α_c + β_t\n(parallel — dashed = common β)"
    )
    ax_trends.set_title(trend_title, fontsize=11)
    ax_trends.set_xlabel("Period  t", fontsize=10)
    ax_trends.set_ylabel("α + β  (+γ)", fontsize=10)
    ax_trends.grid(alpha=0.25)
    if C <= 12:
        ax_trends.legend(fontsize=7, loc="upper left", ncol=2)
    else:
        # Just show a colourbar-style note
        ax_trends.text(
            0.97, 0.03,
            f"{C} cohorts (tab10 palette)",
            transform=ax_trends.transAxes, ha="right", va="bottom",
            fontsize=8, color="grey",
        )

    # ── Scenario B panels ─────────────────────────────────────────────────────
    if has_interaction:

        # Panel [1,0]: Interaction heatmap  γ_{c,t}
        vmax = np.abs(gamma).max()
        im = ax_gamma.imshow(
            gamma,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            origin="upper",
        )
        plt.colorbar(im, ax=ax_gamma, label="γ value", shrink=0.9)
        ax_gamma.set_title("True interaction  γ_{c,t}", fontsize=11)
        ax_gamma.set_xlabel("Period  t", fontsize=10)
        ax_gamma.set_ylabel("Cohort  c", fontsize=10)
        ax_gamma.set_xticks(np.arange(0, T, max(1, T // 5)))
        ax_gamma.set_xticklabels(np.arange(1, T + 1, max(1, T // 5)))
        ax_gamma.set_yticks(np.arange(C))
        ax_gamma.set_yticklabels(np.arange(1, C + 1), fontsize=7)
        # Annotate interaction scale
        ax_gamma.text(
            0.97, 0.02,
            f"sf_gc={data['hyperparams']['sf_gc']:.2f} × sf_gt={data['hyperparams']['sf_gt']:.2f}",
            transform=ax_gamma.transAxes, ha="right", va="bottom",
            fontsize=8, color="grey",
        )

        # Panel [1,1]: Observed cell means  ȳ_{c,t}
        if show_observations:
            im2 = ax_obs.imshow(
                cell_mean,
                cmap="viridis",
                aspect="auto",
                origin="upper",
            )
            plt.colorbar(im2, ax=ax_obs, label="ȳ_{c,t}", shrink=0.9)
            ax_obs.set_title(
                f"Observed cell means  ȳ_{{c,t}}\n(averaged over {data['n_reps']} reps)",
                fontsize=11,
            )
            ax_obs.set_xlabel("Period  t", fontsize=10)
            ax_obs.set_ylabel("Cohort  c", fontsize=10)
            ax_obs.set_xticks(np.arange(0, T, max(1, T // 5)))
            ax_obs.set_xticklabels(np.arange(1, T + 1, max(1, T // 5)))
            ax_obs.set_yticks(np.arange(C))
            ax_obs.set_yticklabels(np.arange(1, C + 1), fontsize=7)
        else:
            ax_obs.axis("off")

        # Panel [1,2]: Notes / DGP summary text
        ax_notes.axis("off")
        hp = data["hyperparams"]
        summary_lines = [
            "DGP Summary",
            "─" * 26,
            f"n_cohorts  = {data['n_cohorts']}",
            f"n_periods  = {data['n_periods']}",
            f"n_reps     = {data['n_reps']}",
            f"N (total)  = {len(data['y'])}",
            "",
            "Cohort GP",
            f"  ell_c  = {hp['ell_c']:.2f}",
            f"  sf_c   = {hp['sf_c']:.2f}",
            "",
            "Time GP",
            f"  ell_t  = {hp['ell_t']:.2f}",
            f"  sf_t   = {hp['sf_t']:.2f}",
            "",
            "Interaction GP",
            f"  ell_gc = {hp['ell_gc']:.2f}",
            f"  sf_gc  = {hp['sf_gc']:.2f}",
            f"  ell_gt = {hp['ell_gt']:.2f}",
            f"  sf_gt  = {hp['sf_gt']:.2f}",
            "",
            "Noise",
            f"  sn     = {hp['sn']:.2f}",
            "",
            f"Seed = {data['seed']}",
        ]
        ax_notes.text(
            0.05, 0.97, "\n".join(summary_lines),
            transform=ax_notes.transAxes,
            va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"),
        )

    # ── Save / return ─────────────────────────────────────────────────────────
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved → {save_path}")

    plt.tight_layout()
    return fig


# ── Remaining plot stubs (implemented in Step 4) ─────────────────────────────

def plot_effect_recovery(
    results_df,
    scenario: str,
    seed: int,
    fit_info: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """2-row × K-col figure of alpha/beta truth vs estimates; K=#models present."""
    plt.style.use("seaborn-v0_8-whitegrid")
    df = pd.DataFrame(results_df).copy()
    df["model"] = df["model"].astype(str).map(_canon_model_name)
    models = _model_order_present(df)
    if not models:
        raise ValueError("No model rows found in results_df.")

    fig, axes = plt.subplots(2, len(models), figsize=(5.0 * len(models), 8.0), sharex=False, sharey="row")
    if len(models) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, model in enumerate(models):
        for i, effect in enumerate(["alpha", "beta"]):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["effect_type"] == effect)].sort_values("index")
            if sub.empty:
                ax.set_title(f"{model} ({effect}) - no data")
                continue

            x = sub["index"].to_numpy(dtype=int)
            mean = sub["mean"].to_numpy(dtype=float)
            lo = sub["lo95"].to_numpy(dtype=float)
            hi = sub["hi95"].to_numpy(dtype=float)
            truth = sub["truth"].to_numpy(dtype=float)
            color = _MODEL_COLOURS.get(model, "steelblue")
            marker = _MODEL_MARKERS.get(model, "o")

            ax.fill_between(x, lo, hi, color=color, alpha=0.20, label="95% CI")
            ax.plot(
                x,
                mean,
                marker=marker,
                linestyle="-",
                color=color,
                linewidth=_EST_LINEWIDTH,
                markersize=_EST_MARKERSIZE,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label="Estimate",
            )
            ax.plot(x, truth, linestyle="--", linewidth=_TRUE_LINEWIDTH, color="black", alpha=0.85, label="Truth")
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.55)
            ax.set_title(f"{model} - {_effect_label(effect)}", fontsize=11)
            ax.set_xlabel("index (0-based)")
            if j == 0:
                ax.set_ylabel("effect value")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=True)

    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Effect Recovery (mean ± 95% CI)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)

    if fit_info is not None:
        if _has_nonzero_gamma_truth(fit_info):
            # Always generate residual/trend diagnostics when true interaction is non-zero.
            residual_path = None
            trends_path = None
            if save_path is not None:
                sp = Path(save_path)
                residual_path = str(sp.with_name(sp.stem + "_residuals" + sp.suffix))
                trends_path = str(sp.with_name(sp.stem + "_cohort_trends" + sp.suffix))
            plot_residual_comparison(fit_info, scenario=scenario, seed=seed, save_path=residual_path)
            plot_cohort_trends_comparison(fit_info, scenario=scenario, seed=seed, save_path=trends_path)

    return fig


def plot_extrapolation(
    fit_df,
    extrap_df,
    beta_true_full: np.ndarray,
    train_cutoff: int,
    scenario: str,
    seed: int,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """1-row × K-col beta trajectory with train/test split and 95% bands."""
    plt.style.use("seaborn-v0_8-whitegrid")

    def _normalise_model_col(df: pd.DataFrame) -> pd.DataFrame:
        if "model" in df.columns:
            return df
        for alt in ["model_name", "Model", "MODEL"]:
            if alt in df.columns:
                return df.rename(columns={alt: "model"})
        return df

    fdf = pd.DataFrame(fit_df).copy() if fit_df is not None else pd.DataFrame()
    edf = pd.DataFrame(extrap_df).copy()
    fdf = _normalise_model_col(fdf)
    edf = _normalise_model_col(edf)

    if "model" not in edf.columns:
        raise ValueError(
            "plot_extrapolation requires extrap_df with a 'model' column "
            f"(or alias model_name/Model/MODEL). Got columns: {list(edf.columns)}"
        )

    if not fdf.empty:
        if "model" not in fdf.columns:
            # If fit_df is passed but has no model column, ignore it and draw extrap-only.
            fdf = pd.DataFrame()
        else:
            fdf["model"] = fdf["model"].astype(str).map(_canon_model_name)
        # Keep only training-period effects when plotting extrapolation diagnostics.
        # This prevents post-cutoff full-data fit lines from overlapping extrapolated lines.
        if not fdf.empty and "index" in fdf.columns:
            fdf = fdf[fdf["index"].astype(int) <= int(train_cutoff)]
    edf["model"] = edf["model"].astype(str).map(_canon_model_name)

    model_series = [edf[["model"]]]
    if not fdf.empty and "model" in fdf.columns:
        model_series.append(fdf[["model"]])
    models = _model_order_present(pd.concat(model_series, axis=0))
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = np.array([axes])

    beta_true_full = np.asarray(beta_true_full, dtype=float).reshape(-1)
    xt = np.arange(len(beta_true_full), dtype=int)

    for j, model in enumerate(models):
        ax = axes[j]
        if {"model", "effect_type", "index"}.issubset(set(fdf.columns)):
            sub_fit = fdf[(fdf["model"] == model) & (fdf["effect_type"] == "beta")].sort_values("index")
        else:
            sub_fit = pd.DataFrame()
        sub_ex = edf[edf["model"] == model].sort_values("period")
        color = _MODEL_COLOURS.get(model, "steelblue")
        marker = _MODEL_MARKERS.get(model, "o")

        ax.plot(
            xt,
            beta_true_full,
            linestyle="--",
            linewidth=_TRUE_LINEWIDTH,
            color="black",
            alpha=0.90,
            label="True period effects",
        )

        if not sub_fit.empty:
            x_fit = sub_fit["index"].to_numpy(dtype=int)
            mu_fit = sub_fit["mean"].to_numpy(dtype=float)
            lo_fit = sub_fit["lo95"].to_numpy(dtype=float)
            hi_fit = sub_fit["hi95"].to_numpy(dtype=float)
            ax.fill_between(x_fit, lo_fit, hi_fit, color=color, alpha=_CI_ALPHA * 0.8)
            ax.plot(
                x_fit, mu_fit, marker=marker, linestyle="-", color=color,
                linewidth=_EST_LINEWIDTH, markersize=_EST_MARKERSIZE, markerfacecolor="white", markeredgewidth=1.1
            )

        if not sub_ex.empty:
            x_ex = sub_ex["period"].to_numpy(dtype=int)
            mu_ex = sub_ex["mean"].to_numpy(dtype=float)
            lo_ex = sub_ex["lo95"].to_numpy(dtype=float)
            hi_ex = sub_ex["hi95"].to_numpy(dtype=float)
            ax.fill_between(x_ex, lo_ex, hi_ex, color=color, alpha=_CI_ALPHA + 0.02)
            ax.plot(
                x_ex, mu_ex, marker=marker, linestyle="-", color=color,
                linewidth=_EST_LINEWIDTH, markersize=_EST_MARKERSIZE, markerfacecolor="white", markeredgewidth=1.2
            )

        ax.axvspan(train_cutoff + 0.5, len(beta_true_full) - 0.5, color="grey", alpha=0.13)
        ax.axvline(train_cutoff, linestyle="--", linewidth=1.4, color="red")
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("Period Index")
        if j == 0:
            ax.set_ylabel("Period Effects")
            legend_handles = [
                Line2D([0], [0], color="black", linestyle="--", linewidth=_TRUE_LINEWIDTH, label="True period effects"),
                Line2D([0], [0], color=color, linestyle="-", marker=marker, linewidth=_EST_LINEWIDTH, label="Estimated effects (train+extrap)"),
                Patch(facecolor=color, alpha=_CI_ALPHA + 0.02, edgecolor="none", label="95% interval"),
                Patch(facecolor="grey", alpha=0.13, edgecolor="none", label="Held-out periods"),
                Line2D([0], [0], color="red", linestyle="--", linewidth=1.4, label="Start of extrapolation"),
            ]
            ax.legend(handles=legend_handles, fontsize=8, frameon=True, loc="best")

    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Extrapolation (mean ± 95% CI)", fontsize=13, y=1.03)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_extrapolation_y(
    extrap_y_df,
    train_cutoff: int,
    scenario: str,
    seed: int,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Held-out y extrapolation diagnostics.

    Expected columns in extrap_y_df:
        model, period, y_true, y_hat
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    df = pd.DataFrame(extrap_y_df).copy()
    if "model" not in df.columns:
        raise ValueError("plot_extrapolation_y requires column 'model'.")
    for col in ["period", "y_true", "y_hat"]:
        if col not in df.columns:
            raise ValueError(f"plot_extrapolation_y requires column '{col}'.")

    df["model"] = df["model"].astype(str).map(_canon_model_name)
    models = _model_order_present(df[["model"]])
    if not models:
        raise ValueError("No model rows found in extrap_y_df.")

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = np.array([axes])

    legend_handles = None
    for j, model in enumerate(models):
        ax = axes[j]
        sub = df[df["model"] == model].copy()
        agg = (
            sub.groupby("period", as_index=False)
            .agg(
                y_true_mean=("y_true", "mean"),
                y_true_std=("y_true", "std"),
                y_true_n=("y_true", "count"),
                y_hat_mean=("y_hat", "mean"),
                y_hat_std=("y_hat", "std"),
                y_hat_n=("y_hat", "count"),
            )
            .sort_values("period")
        )
        x = agg["period"].to_numpy(dtype=int)
        yt = agg["y_true_mean"].to_numpy(dtype=float)
        yh = agg["y_hat_mean"].to_numpy(dtype=float)
        se_true = agg["y_true_std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(
            np.maximum(agg["y_true_n"].to_numpy(dtype=float), 1.0)
        )
        se_hat = agg["y_hat_std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(
            np.maximum(agg["y_hat_n"].to_numpy(dtype=float), 1.0)
        )
        yt_lo = yt - 1.96 * se_true
        yt_hi = yt + 1.96 * se_true
        yh_lo = yh - 1.96 * se_hat
        yh_hi = yh + 1.96 * se_hat

        color = _MODEL_COLOURS.get(model, "#3A7CA5")
        marker = _MODEL_MARKERS.get(model, "o")

        h_true = ax.plot(x, yt, linestyle="--", color="black", linewidth=1.7, label="True held-out y")[0]
        ax.fill_between(x, yt_lo, yt_hi, color="black", alpha=0.08)
        h_hat = ax.plot(
            x, yh, linestyle="-", color=color, marker=marker, markersize=4.2,
            markerfacecolor="white", markeredgewidth=1.0, linewidth=1.8, label="Predicted held-out y"
        )[0]
        h_hat_ci = ax.fill_between(x, yh_lo, yh_hi, color=color, alpha=0.18, label="Predicted y 95% CI")
        h_cut = ax.axvline(train_cutoff, linestyle="--", linewidth=1.3, color="red", label="Start of extrapolation")

        ax.set_title(model, fontsize=11)
        ax.set_xlabel("Period Index")
        if j == 0:
            ax.set_ylabel("Outcome y (period mean)")
        ax.grid(axis="y", alpha=0.28)
        ax.set_xlim(min(train_cutoff, x.min()) - 0.5, x.max() + 0.5)

        if legend_handles is None:
            legend_handles = [h_true, h_hat, h_hat_ci, h_cut]

    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Held-out y Extrapolation", fontsize=13, y=0.98)
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    _save(fig, save_path)
    return fig


def plot_interaction_heatmap(
    gamma_true: np.ndarray,
    gamma_hat: np.ndarray,
    scenario: str,
    seed: int,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """1-row × 2-col heatmap: true interaction vs GP-CP-Extended estimate."""
    plt.style.use("seaborn-v0_8-whitegrid")
    gt = np.asarray(gamma_true, dtype=float)
    gh = np.asarray(gamma_hat, dtype=float)
    vmax = float(max(np.max(np.abs(gt)), np.max(np.abs(gh)), 1e-10))

    # Use a dedicated colorbar axis so it never overlaps heatmap panels.
    fig = plt.figure(figsize=(12.0, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    cax = fig.add_subplot(gs[0, 2])

    im0 = ax0.imshow(gt, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    _ = im0
    ax0.set_title("True gamma")
    ax0.set_xlabel("period index (0-based)")
    ax0.set_ylabel("cohort index (0-based)")

    im1 = ax1.imshow(gh, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
    ax1.set_title("Estimated gamma (GP-CP-Extended)")
    ax1.set_xlabel("period index (0-based)")

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label("interaction effect")
    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Interaction Surface", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)
    return fig


def plot_metric_summary(
    summary_df,
    metrics: list[str] = ("Bias", "RMSE", "Coverage"),
    task: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Grouped bars by model with hue=scenario (for one task) or scenario×task."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sdf = pd.DataFrame(summary_df).copy()
    sdf["model"] = sdf["model"].astype(str).map(_canon_model_name)
    if task is not None:
        sdf = sdf[sdf["task"].astype(str) == str(task)].copy()
    if sdf.empty:
        raise ValueError(f"No rows available for plot_metric_summary(task={task!r}).")

    if task is None:
        sdf["condition"] = "Scen " + sdf["scenario"].astype(str) + " - " + sdf["task"].astype(str)
    else:
        sdf["condition"] = "Scen " + sdf["scenario"].astype(str)

    models = [m for m in _MODEL_ORDER if m in set(sdf["model"])]
    conditions = sorted(sdf["condition"].unique().tolist())
    color_list = ["#3A7CA5", "#6FB1A0", "#E07A5F", "#D1495B", "#7A5195", "#EDAE49"]
    cond_colours = {c: color_list[i % len(color_list)] for i, c in enumerate(conditions)}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.8 * len(metrics), 5.1), sharey=False)
    if len(metrics) == 1:
        axes = np.array([axes])

    for i, metric in enumerate(metrics):
        ax = axes[i]
        subm = sdf[sdf["metric"] == metric]
        width = 0.75 / max(len(conditions), 1)
        x = np.arange(len(models))
        for k, cond in enumerate(conditions):
            cond_df = subm[subm["condition"] == cond]
            means = []
            errs = []
            for m in models:
                row = cond_df[cond_df["model"] == m]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]))
            ax.bar(
                x + (k - (len(conditions) - 1) / 2) * width,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                label=cond,
                alpha=0.92,
                color=cond_colours[cond],
                edgecolor="white",
                linewidth=0.7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.set_title(metric)
        if metric == "Coverage":
            ax.axhline(0.95, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if title is None:
        title = "Summary Metrics Across Scenarios and Tasks" if task is None else f"Summary Metrics ({task})"
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=max(1, min(4, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    _save(fig, save_path)
    return fig


def plot_coverage_table(
    summary_df,
    task: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Coverage summary as a readable table (model×effect rows, scenario columns).
    """
    sdf = pd.DataFrame(summary_df).copy()
    sdf["model"] = sdf["model"].astype(str).map(_canon_model_name)
    cov = sdf[sdf["metric"] == "Coverage"].copy()
    if task is not None:
        cov = cov[cov["task"].astype(str) == str(task)].copy()
    if cov.empty:
        raise ValueError(f"No coverage rows available for plot_coverage_table(task={task!r}).")

    cov["row"] = cov["model"].astype(str) + " | " + cov["effect_type"].fillna("-").astype(str)
    cov["col"] = "Scen " + cov["scenario"].astype(str)
    if task is None:
        cov["col"] = cov["col"] + " | " + cov["task"].astype(str)

    table_df = cov.pivot_table(index="row", columns="col", values="mean", aggfunc="first")
    table_df = table_df.sort_index(axis=0).sort_index(axis=1)
    txt_df = table_df.copy()
    for col in txt_df.columns:
        txt_df[col] = txt_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")

    n_rows, n_cols = txt_df.shape
    fig_w = max(8.0, 1.8 * max(1, n_cols))
    fig_h = max(4.2, 0.45 * max(1, n_rows) + 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tb = ax.table(
        cellText=txt_df.values,
        rowLabels=txt_df.index.tolist(),
        colLabels=txt_df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.2)

    # Header styling
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(weight="bold")
        if c == -1:
            cell.set_facecolor("#F6F8FB")

    # Highlight under-/over-coverage cells.
    for i in range(n_rows):
        for j in range(n_cols):
            v = table_df.iloc[i, j]
            if pd.isna(v):
                continue
            cell = tb[(i + 1, j)]
            if v < 0.90:
                cell.set_facecolor("#FADBD8")
            elif v > 0.99:
                cell.set_facecolor("#FCF3CF")

    if title is None:
        title = "Coverage Table" if task is None else f"Coverage Table ({task})"
    ax.set_title(title, fontsize=13, pad=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_scenario_fit_summary(
    summary_fit_df,
    scenario: str,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot per-scenario fitting summary from summary_scenX_fit.csv.
    Expected columns: model, effect_type, metric, mean, std
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    sdf = pd.DataFrame(summary_fit_df).copy()
    sdf["model"] = sdf["model"].astype(str).map(_canon_model_name)
    sdf["effect_label"] = sdf["effect_type"].astype(str).map(_effect_label)

    metrics = ["Bias", "RMSE", "Coverage"]
    effects = [e for e in ["Cohort effects", "Period effects", "Interaction effects"] if e in set(sdf["effect_label"])]
    models = [m for m in _MODEL_ORDER if m in set(sdf["model"])]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.7 * len(metrics), 4.9), sharey=False)
    if len(metrics) == 1:
        axes = np.array([axes])

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = sdf[sdf["metric"] == metric]
        width = 0.75 / max(len(effects), 1)
        x = np.arange(len(models))
        for k, eff in enumerate(effects):
            eff_sub = sub[sub["effect_label"] == eff]
            means = []
            errs = []
            for m in models:
                row = eff_sub[eff_sub["model"] == m]
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]) if pd.notna(row["std"].iloc[0]) else 0.0)
            ax.bar(
                x + (k - (len(effects) - 1) / 2) * width,
                means,
                width=width,
                yerr=errs,
                capsize=3,
                alpha=0.88,
                label=eff,
            )
        if metric == "Coverage":
            ax.axhline(0.95, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(frameon=True, fontsize=8)

    fig.suptitle(f"Scenario {scenario}: Fitting Summary (mean ± std across seeds)", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_coverage_heatmap(
    summary_df,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Coverage heatmap with annotated values and under/over-coverage flags."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sdf = pd.DataFrame(summary_df).copy()
    sdf["model"] = sdf["model"].astype(str).map(_canon_model_name)
    cov = sdf[sdf["metric"] == "Coverage"].copy()
    cov["row"] = cov["model"].astype(str) + " | " + cov["effect_type"].fillna("-").astype(str)
    cov["col"] = "Scen " + cov["scenario"].astype(str) + " - " + cov["task"].astype(str)

    rows = sorted(cov["row"].unique().tolist())
    cols = sorted(cov["col"].unique().tolist())
    mat = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            sub = cov[(cov["row"] == r) & (cov["col"] == c)]
            if not sub.empty:
                mat[i, j] = float(sub["mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 * len(cols)), max(4.8, 0.5 * len(rows) + 1.8)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Coverage Heatmap")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}"
            weight = "bold" if (v < 0.90 or v > 0.99) else "normal"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9, fontweight=weight)
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("coverage")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def build_comparison_tables(
    fit_info: dict,
    include_interaction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build model comparison tables for one seed.

    Returns:
        metrics_df   : summary metrics by model
        residual_df  : residual mean/std by model
    """
    rows = []
    resid_rows = []
    for model in _MODEL_ORDER:
        if model not in fit_info:
            continue
        d = fit_info[model]
        y_true = np.asarray(d.get("y_true", np.nan))
        y_hat = np.asarray(d.get("y_hat", np.nan))
        resid = np.asarray(d.get("resid", np.nan))
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2))) if np.ndim(y_true) > 0 else np.nan
        mae = float(np.mean(np.abs(y_true - y_hat))) if np.ndim(y_true) > 0 else np.nan
        rows.append({"model": model, "RMSE_y": rmse, "MAE_y": mae})
        resid_rows.append({"model": model, "Residual_mean": float(np.nanmean(resid)), "Residual_SD": float(np.nanstd(resid))})

        if include_interaction and model == "GP-CP-Extended" and ("gamma" in d) and ("gamma_true" in d):
            g = np.asarray(d["gamma"], dtype=float)
            gt = np.asarray(d["gamma_true"], dtype=float)
            rows[-1]["RMSE_gamma"] = float(np.sqrt(np.mean((g - gt) ** 2)))
            rows[-1]["MAE_gamma"] = float(np.mean(np.abs(g - gt)))

    metrics_df = pd.DataFrame(rows)
    residual_df = pd.DataFrame(resid_rows)
    return metrics_df, residual_df


def plot_residual_comparison(
    fit_info: dict,
    scenario: str,
    seed: int,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Residual histogram comparison across FE+AR, GP-CP, GP-CP-Extended."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6))
    used = []
    for model in _MODEL_ORDER:
        if model not in fit_info:
            continue
        resid = np.asarray(fit_info[model].get("resid", []), dtype=float)
        if resid.size == 0:
            continue
        ax.hist(resid, bins=35, density=True, alpha=0.45, label=model, color=_MODEL_COLOURS.get(model, "grey"))
        used.append(model)
    ax.set_title("Residual Distribution Comparison")
    ax.set_xlabel("residual")
    ax.set_ylabel("density")
    if used:
        ax.legend(frameon=True)
    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Residual Diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_cohort_trends_comparison(
    fit_info: dict,
    scenario: str,
    seed: int,
    max_cohorts: int = 12,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Cohort-level trend comparison across models.
    Uses total effects mu + alpha_c + beta_t (+ gamma_{c,t} when available).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    models = [m for m in _MODEL_ORDER if m in fit_info]
    if not models:
        raise ValueError("No model entries found in fit_info for trend plot.")

    n_t = None
    for model in models:
        d = fit_info[model]
        beta = np.asarray(d.get("beta", []), dtype=float)
        n_t = len(beta) if n_t is None else n_t
    xt = np.arange(n_t, dtype=int)

    for ax_idx, model in enumerate(_MODEL_ORDER):
        ax = axes[ax_idx]
        if model not in fit_info:
            ax.axis("off")
            continue
        d = fit_info[model]
        mu = float(d.get("mu", 0.0))
        alpha = np.asarray(d.get("alpha"), dtype=float)
        beta = np.asarray(d.get("beta"), dtype=float)
        gamma = np.asarray(d.get("gamma", np.zeros((len(alpha), len(beta)))), dtype=float)

        c_total = len(alpha)
        if c_total <= max_cohorts:
            idxs = np.arange(c_total)
        else:
            idxs = np.linspace(0, c_total - 1, max_cohorts).astype(int)
        colors = plt.cm.tab20(np.linspace(0, 1, len(idxs)))
        for i, c in enumerate(idxs):
            trend = mu + alpha[c] + beta + gamma[c]
            ax.plot(xt, trend, color=colors[i], linewidth=1.2, alpha=0.80)
        ax.plot(xt, mu + beta, color="black", linestyle="--", linewidth=2.0, label="Common additive trend")
        ax.set_title(model)
        ax.set_xlabel("period index (0-based)")
        if ax_idx == 0:
            ax.set_ylabel("total effect")
            ax.legend(fontsize=8, frameon=True)

    fig.suptitle(f"Scenario {scenario} - Seed {seed}: Cohort-level Trends by Model", fontsize=13, y=1.03)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_method_beta_posterior_comparison(
    run_output: dict,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Compare beta posterior summaries across the three methods for one run.
    """
    sim_data = run_output["sim_data"]
    mr = run_output["method_results"]
    beta_true = np.asarray(sim_data["beta_true"], dtype=float)
    xt = np.arange(len(beta_true), dtype=int)

    method_specs = [
        ("method1", "1) Type-II MAP + Laplace", "#1f77b4"),
        ("method2", "2) NUTS Hyperparams", "#2ca02c"),
        ("method3", "3) Full NUTS (Sample f + theta)", "#d62728"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for ax, (key, title, color) in zip(axes, method_specs):
        if key not in mr:
            ax.axis("off")
            continue
        out = mr[key]
        beta = np.asarray(out["beta"], dtype=float)
        std = np.asarray(out.get("std_beta", np.zeros_like(beta)), dtype=float)
        lo = beta - 1.96 * std
        hi = beta + 1.96 * std
        ax.plot(xt, beta_true, color="black", linestyle="--", linewidth=1.6, label="Truth")
        ax.plot(xt, beta, color=color, linewidth=1.8, label="Estimate")
        if np.any(std > 0):
            ax.fill_between(xt, lo, hi, color=color, alpha=0.18, label="95% CI")
        ax.axhline(0.0, color="grey", linewidth=0.9, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("period index")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=True)

    axes[0].set_ylabel("beta")
    fig.suptitle(
        f"Scenario {run_output['scenario']} | N={run_output['n_total']} | seed={run_output['seed']}: Beta posterior comparison",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_method_hyperparam_posteriors(
    run_output: dict,
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Compare hyperparameter posteriors (ell, sf, sn) across methods for one run.

    Method 1 uses Laplace posterior samples (M1b-style) when available.
    """
    mr = run_output["method_results"]
    truth = run_output["true_hparams"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    params = [("ell", "Lengthscale"), ("sf", "Signal SD"), ("sn", "Noise SD")]
    method_colors = {"method2": "#2ca02c", "method3": "#d62728"}

    m1 = mr.get("method1", {})
    m1_map = m1.get("hyperparams_map", {})
    m1_samples = m1.get("posterior_samples", {})
    has_m1_samples = isinstance(m1_samples, dict) and "ell_c" in m1_samples

    for ax, (pkey, ptitle) in zip(axes, params):
        if has_m1_samples:
            if pkey == "ell":
                m1_draws = 0.5 * (np.asarray(m1_samples["ell_c"], dtype=float) + np.asarray(m1_samples["ell_t"], dtype=float))
            elif pkey == "sf":
                m1_draws = 0.5 * (np.asarray(m1_samples["sf_c"], dtype=float) + np.asarray(m1_samples["sf_t"], dtype=float))
            else:
                m1_draws = np.asarray(m1_samples["sn"], dtype=float)
            ax.hist(m1_draws, bins=30, density=True, alpha=0.35, color="#1f77b4", label="1) Type-II MAP + Laplace")
            ax.axvline(float(np.mean(m1_draws)), color="#1f77b4", linewidth=1.8, linestyle="-")
        else:
            m1_ell = float(np.nanmean([m1_map.get("ell_c", np.nan), m1_map.get("ell_t", np.nan)]))
            m1_sf = float(np.nanmean([m1_map.get("sf_c", np.nan), m1_map.get("sf_t", np.nan)]))
            m1_sn = float(m1_map.get("sn", np.nan))
            m1_vals = {"ell": m1_ell, "sf": m1_sf, "sn": m1_sn}
            if np.isfinite(m1_vals[pkey]):
                ax.axvline(m1_vals[pkey], color="#1f77b4", linewidth=2.0, label="Method 1 MAP")

        for mkey in ["method2", "method3"]:
            out = mr.get(mkey, {})
            ps = out.get("posterior_samples", {})
            if not isinstance(ps, dict) or "ell_c" not in ps:
                continue
            if pkey == "ell":
                samples = 0.5 * (np.asarray(ps["ell_c"], dtype=float) + np.asarray(ps["ell_t"], dtype=float))
            elif pkey == "sf":
                samples = 0.5 * (np.asarray(ps["sf_c"], dtype=float) + np.asarray(ps["sf_t"], dtype=float))
            else:
                samples = np.asarray(ps["sn"], dtype=float)
            ax.hist(samples, bins=30, density=True, alpha=0.35, color=method_colors[mkey], label=f"{mr[mkey].get('method_label', mkey)}")

        if pkey in truth:
            ax.axvline(float(truth[pkey]), color="black", linestyle="--", linewidth=1.5, label="Truth")
        ax.set_title(ptitle)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=True)

    fig.suptitle(
        f"Scenario {run_output['scenario']} | N={run_output['n_total']} | seed={run_output['seed']}: Hyperparameter posteriors",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_path)
    return fig
