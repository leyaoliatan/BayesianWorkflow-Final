"""
Evaluation Metrics
===================
Utilities for fitting and extrapolation metrics used in simulation experiments.
"""

from __future__ import annotations

import numpy as np

def _as_1d(x) -> np.ndarray:
 return np.asarray(x, dtype=float).reshape(-1)

def _broadcast_truth(truth, shape: tuple[int,...]) -> np.ndarray:
 t = np.asarray(truth, dtype=float)
 if t.ndim == 0:
  return np.full(shape, float(t), dtype=float)
 return np.broadcast_to(t, shape).astype(float)

def bias(estimates: np.ndarray, truth) -> float:
 """Mean bias: mean(estimates - truth)."""
 e = np.asarray(estimates, dtype=float)
 t = _broadcast_truth(truth, e.shape)
 return float(np.mean(e - t))

def rmse(estimates: np.ndarray, truth) -> float:
 """Root mean squared error."""
 e = np.asarray(estimates, dtype=float)
 t = _broadcast_truth(truth, e.shape)
 return float(np.sqrt(np.mean((e - t) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
 """Mean Absolute Percentage Error (for forecast accuracy)."""
 y_true = _as_1d(y_true)
 y_pred = _as_1d(y_pred)
 mask = y_true != 0
 if not np.any(mask):
  return float("nan")
 return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
 """Mean Absolute Error."""
 y_true = _as_1d(y_true)
 y_pred = _as_1d(y_pred)
 return float(np.mean(np.abs(y_true - y_pred)))

def coverage(
 estimates: np.ndarray,
 lower: np.ndarray,
 upper: np.ndarray,
 truth,
) -> float:
 """
 Empirical coverage of nominal credible/confidence intervals.

 Parameters
 ----------
 estimates : not used directly, kept for API consistency.
 lower, upper : arrays of interval bounds (one per replicate).
 truth : scalar or array-like truth values.

 Returns
 -------
 Fraction of intervals that contain `truth`.
 """
 lower = np.asarray(lower, dtype=float)
 upper = np.asarray(upper, dtype=float)
 t = _broadcast_truth(truth, lower.shape)
 return float(np.mean((lower <= t) & (t <= upper)))

def summarise_effect(
 mean: np.ndarray,
 lo95: np.ndarray,
 hi95: np.ndarray,
 truth: np.ndarray,
) -> dict:
 """Return Bias/RMSE/Coverage for one estimated effect vector."""
 return {
  "Bias": bias(mean, truth),
  "RMSE": rmse(mean, truth),
  "Coverage": coverage(mean, lo95, hi95, truth),
 }

def summarise_extrapolation(pred_mean: np.ndarray, truth: np.ndarray) -> dict:
 """Return MAE/MAPE/RMSE for extrapolated period effects."""
 return {
  "MAE": mae(truth, pred_mean),
  "MAPE": mape(truth, pred_mean),
  "RMSE": rmse(pred_mean, truth),
 }

def summarise_simulation(results: list[dict], truth: float) -> dict:
 """
 Aggregate results over Monte Carlo replicates.

 Parameters
 ----------
 results : list of dicts, each with key 'att'.
 truth : true ATT.

 Returns
 -------
 dict with bias, rmse, and (optionally) coverage.
 """
 estimates = np.array([r["att"] for r in results])
 out = {
  "bias": bias(estimates, truth),
  "rmse": rmse(estimates, truth),
  "n_reps": len(estimates),
 }
 if "lower" in results[0] and "upper" in results[0]:
  lower = np.array([r["lower"] for r in results])
  upper = np.array([r["upper"] for r in results])
  out["coverage"] = coverage(estimates, lower, upper, truth)
 return out
