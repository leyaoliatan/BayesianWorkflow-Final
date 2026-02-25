"""
Real-data loading and preprocessing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_category_path_spec(spec_path: Path) -> list[tuple[str, Path]]:
    """
    Parse an input spec file with lines in "{category}:{path}" format.

    Rules:
    - Empty lines are ignored.
    - Lines starting with '#' are treated as comments.
    - The split is performed on the first ':' only.
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Input spec file not found: {spec_path}")

    entries: list[tuple[str, Path]] = []
    for line_no, raw_line in enumerate(spec_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"Invalid spec line {line_no} in {spec_path}; expected "
                f"'{{category}}:{{path}}', got: {raw_line!r}"
            )
        category, path_str = line.split(":", 1)
        category = category.strip()
        path_str = path_str.strip()
        if not category or not path_str:
            raise ValueError(
                f"Invalid spec line {line_no} in {spec_path}; empty category/path."
            )
        entries.append((category, Path(path_str)))

    if not entries:
        raise ValueError(f"No valid dataset entries found in {spec_path}")
    return entries


def compute_Q_T_pre_event(T_pre: int, K3: int) -> np.ndarray:
    """Projection matrix used in downstream model code; shape (K3, K3)."""
    q = np.zeros((K3, K3), dtype=float)
    q[:T_pre, :T_pre] = np.eye(T_pre)
    if K3 > T_pre:
        q[T_pre:, T_pre:] = np.eye(K3 - T_pre)
    return q


def _drop_step(step_counts: dict[str, int], current_key: str, prev_key: str) -> int:
    return int(step_counts[prev_key] - step_counts[current_key])


def build_drop_summary(step_counts: dict[str, int]) -> dict[str, int]:
    """Return compact dropped-row counts by filter step."""
    return {
        "dropna_required": _drop_step(step_counts, "after_dropna_required", "raw_rows"),
        "outside_date_window": _drop_step(
            step_counts, "after_date_window", "after_dropna_required"
        ),
        "cohort_not_pre_covid": _drop_step(
            step_counts, "after_cohort_pre_covid", "after_date_window"
        ),
        "non_positive_cohort_age": _drop_step(
            step_counts, "after_positive_cohort_age", "after_cohort_pre_covid"
        ),
    }


def prepare_real_data_monthly(
    data_path: Path,
    dv: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    covid_onset: pd.Timestamp,
    *,
    date_col: str = "month",
    cohort_col: str = "cohort",
    cohort_time_col: str = "cohort_month",
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    """
    Prepare monthly cohort data using the same core rules as the reference notebook.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_df = pd.read_csv(data_path)

    required = [date_col, cohort_col, cohort_time_col, dv]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    # Drop duplicated header rows accidentally stored as data.
    raw_df = raw_df[raw_df[date_col] != date_col].copy()

    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
    raw_df[cohort_col] = pd.to_datetime(raw_df[cohort_col], errors="coerce")

    for col in [cohort_time_col, dv]:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    step_counts: dict[str, int] = {}
    step_counts["raw_rows"] = len(raw_df)

    df = raw_df.dropna(subset=required).copy()
    step_counts["after_dropna_required"] = len(df)

    mask_date = df[date_col].between(start_date, end_date)
    df = df.loc[mask_date].copy()
    step_counts["after_date_window"] = len(df)

    mask_cohort = df[cohort_col].between(start_date, covid_onset - pd.Timedelta(days=1))
    df = df.loc[mask_cohort].copy()
    step_counts["after_cohort_pre_covid"] = len(df)

    mask_age = df[cohort_time_col] > 0
    df = df.loc[mask_age].copy()
    step_counts["after_positive_cohort_age"] = len(df)

    if df.empty:
        raise ValueError(
            "No rows left after preprocessing filters. Check date range/cohort filters."
        )

    df = df.sort_values([cohort_col, date_col]).reset_index(drop=True)

    df["cohort_time_idx"] = df[cohort_time_col].astype("category").cat.codes
    df["cohort_idx"] = df[cohort_col].astype("category").cat.codes
    df["time_idx"] = df[date_col].astype("category").cat.codes
    df["is_post"] = (df[date_col] >= covid_onset).astype(int)

    K1 = int(df["cohort_time_idx"].nunique())
    K2 = int(df["cohort_idx"].nunique())
    K3 = int(df["time_idx"].nunique())

    pre_event_times = np.sort(df.loc[df["is_post"] == 0, date_col].unique())
    T_pre = int(len(pre_event_times))
    if T_pre <= 0:
        raise ValueError("Invalid pre/post split: no pre-COVID periods found (T_pre <= 0).")

    dv_raw = df[dv].to_numpy(dtype=float)
    if np.any(dv_raw <= 0):
        dv_log = np.log(dv_raw + 1.0)
    else:
        dv_log = np.log(dv_raw)
    df["dv_log"] = dv_log

    stan_like: dict[str, Any] = {
        "N": int(len(df)),
        "dv": dv_log,
        "K1": K1,
        "K2": K2,
        "K3": K3,
        "cohort_time_idx": df["cohort_time_idx"].to_numpy(),
        "cohort_idx": df["cohort_idx"].to_numpy(),
        "time_idx": df["time_idx"].to_numpy(),
        "is_post": df["is_post"].to_numpy(),
        "T_pre": T_pre,
        "Q_T": compute_Q_T_pre_event(T_pre, K3),
        "time_values": np.arange(K3),
        "cohort_values": np.arange(K2),
        "cohort_time_values": np.arange(K1),
    }

    return df, stan_like, step_counts


def make_real_data_qa(
    df: pd.DataFrame,
    stan_like: dict[str, Any],
    step_counts: dict[str, int],
) -> dict[str, Any]:
    """Create compact QA metadata for logs/reproducibility."""
    post = df.loc[df["is_post"] == 1, "time_idx"]
    return {
        "N": int(stan_like["N"]),
        "K1": int(stan_like["K1"]),
        "K2": int(stan_like["K2"]),
        "K3": int(stan_like["K3"]),
        "T_pre": int(stan_like["T_pre"]),
        "index_ranges": {
            "cohort_time_idx": [int(df["cohort_time_idx"].min()), int(df["cohort_time_idx"].max())],
            "cohort_idx": [int(df["cohort_idx"].min()), int(df["cohort_idx"].max())],
            "time_idx": [int(df["time_idx"].min()), int(df["time_idx"].max())],
        },
        "post_split": {
            "n_post_rows": int((df["is_post"] == 1).sum()),
            "post_start_time_idx": int(post.min()) if len(post) else None,
        },
        "step_counts": {k: int(v) for k, v in step_counts.items()},
        "dropped_rows": build_drop_summary(step_counts),
    }
