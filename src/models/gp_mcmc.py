"""
gp_mcmc.py
==========
MCMC-based estimators for the baseline constrained GP cohort-period model:

    y_i = [mu] + alpha_{c(i)} + beta_{t(i)} + eps_i
    eps_i ~ N(0, sn^2)

Method 2: NUTS over hyperparameters only (latent effects marginalized out).
Method 3: Full NUTS over latent reduced effects z and hyperparameters.

Both methods return dictionaries aligned with the `CohortPeriodModel.results_dict`
shape for downstream evaluation and plotting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_torch(x, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(np.asarray(x), dtype=dtype)


def _sum_to_zero_basis(k: int) -> Tuple[np.ndarray, np.ndarray]:
    i = np.eye(k)
    ones = np.ones((k, 1))
    p = i - (ones @ ones.T) / k
    q_drop = p[:, 1:]
    q_orth, r = np.linalg.qr(q_drop)
    q_pinv = np.linalg.inv(r) @ q_orth.T
    return q_drop, q_pinv


def _rbf_kernel_torch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    ell: torch.Tensor,
    sf: torch.Tensor,
) -> torch.Tensor:
    x1s = x1 / ell
    x2s = x2 / ell
    dist2 = (
        x1s.pow(2).sum(-1, keepdim=True)
        - 2.0 * (x1s @ x2s.T)
        + x2s.pow(2).sum(-1).unsqueeze(0)
    )
    return (sf ** 2) * torch.exp(-0.5 * dist2)


def _analytical_gp_posterior(
    y_centered: torch.Tensor,
    h: torch.Tensor,
    k_z: torch.Tensor,
    sn: torch.Tensor,
    jitter: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = h.shape[0]
    sigma_y = h @ k_z @ h.T + (sn ** 2 + jitter) * torch.eye(n, dtype=torch.float64)
    try:
        l = torch.linalg.cholesky(sigma_y)
    except RuntimeError:
        sigma_y = sigma_y + 1e-4 * torch.eye(n, dtype=torch.float64)
        l = torch.linalg.cholesky(sigma_y)

    alpha_v = torch.cholesky_solve(y_centered.unsqueeze(-1), l)
    mu_z = (k_z @ h.T @ alpha_v).squeeze(-1)
    x = torch.cholesky_solve(h @ k_z, l)
    sigma_z = k_z - (k_z @ h.T @ x)
    return mu_z, sigma_z


def _build_design(
    obs_c: np.ndarray,
    obs_t: np.ndarray,
    n_cohorts: int,
    n_periods: int,
) -> dict[str, Any]:
    c = int(n_cohorts)
    t = int(n_periods)
    n = len(obs_c)

    q_drop_c, q_pinv_c = _sum_to_zero_basis(c)
    q_drop_t, q_pinv_t = _sum_to_zero_basis(t)

    a = np.zeros((n, c), dtype=float)
    b = np.zeros((n, t), dtype=float)
    a[np.arange(n), obs_c] = 1.0
    b[np.arange(n), obs_t] = 1.0
    h_tilde = np.concatenate([a @ q_drop_c, b @ q_drop_t], axis=1)

    return {
        "q_drop_c": q_drop_c,
        "q_drop_t": q_drop_t,
        "q_pinv_c": q_pinv_c,
        "q_pinv_t": q_pinv_t,
        "h_tilde": h_tilde,
        "dc": c - 1,
        "dt": t - 1,
    }


def _summarise_theta_samples(
    ell_c: np.ndarray,
    sf_c: np.ndarray,
    ell_t: np.ndarray,
    sf_t: np.ndarray,
    sn: np.ndarray,
) -> dict[str, dict[str, float]]:
    ell = 0.5 * (ell_c + ell_t)
    sf = 0.5 * (sf_c + sf_t)
    return {
        "mean": {
            "ell": float(np.mean(ell)),
            "sf": float(np.mean(sf)),
            "sn": float(np.mean(sn)),
            "ell_c": float(np.mean(ell_c)),
            "sf_c": float(np.mean(sf_c)),
            "ell_t": float(np.mean(ell_t)),
            "sf_t": float(np.mean(sf_t)),
        },
        "sd": {
            "ell": float(np.std(ell, ddof=1)),
            "sf": float(np.std(sf, ddof=1)),
            "sn": float(np.std(sn, ddof=1)),
            "ell_c": float(np.std(ell_c, ddof=1)),
            "sf_c": float(np.std(sf_c, ddof=1)),
            "ell_t": float(np.std(ell_t, ddof=1)),
            "sf_t": float(np.std(sf_t, ddof=1)),
        },
    }


@dataclass
class _Bounds:
    ell_c_low: float
    ell_c_high: float
    sf_c_low: float
    sf_c_high: float
    ell_t_low: float
    ell_t_high: float
    sf_t_low: float
    sf_t_high: float
    sn_low: float
    sn_high: float


def _default_bounds(y: np.ndarray, n_cohorts: int, n_periods: int) -> _Bounds:
    y_std = float(np.std(y))
    y_mad = float(np.median(np.abs(y - np.median(y))) * 1.4826)
    y_scale = max(y_std, y_mad, 0.1)
    return _Bounds(
        ell_c_low=0.01,
        ell_c_high=max(8.0, 2.0 * float(n_cohorts)),
        sf_c_low=0.005,
        sf_c_high=max(8.0, 20.0 * y_scale),
        ell_t_low=0.01,
        ell_t_high=max(8.0, 2.0 * float(n_periods)),
        sf_t_low=0.005,
        sf_t_high=max(8.0, 20.0 * y_scale),
        sn_low=1e-4,
        sn_high=max(5.0, 12.0 * y_scale),
    )


def _require_numpyro() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC, NUTS
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "fit_hyperparam_nuts / fit_full_nuts require JAX + NumPyro. "
            "Install `jax`, `jaxlib`, and `numpyro`."
        ) from exc
    return jax, jnp, numpyro, MCMC, NUTS


def fit_hyperparam_nuts(
    y: "array-like",
    obs_c: "array-like",
    obs_t: "array-like",
    n_cohorts: int,
    n_periods: int,
    *,
    seed: int = 0,
    use_global_mean: bool = False,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    jitter: float = 1e-6,
    prior_log_sd: float = 2.0,
    prior_mu_sd: float = 5.0,
    return_samples: bool = True,
) -> dict[str, Any]:
    """
    Method 2: NUTS sampling over hyperparameters only.
    Latent effects are analytically marginalized in the likelihood.
    """
    t_total0 = time.perf_counter()
    y_np = _to_numpy(y).astype(float)
    obs_c_np = _to_numpy(obs_c).astype(int)
    obs_t_np = _to_numpy(obs_t).astype(int)
    n = len(y_np)
    if n == 0:
        raise ValueError("Empty observations provided.")

    c = int(n_cohorts)
    t = int(n_periods)
    if obs_c_np.min() < 0 or obs_c_np.max() >= c:
        raise ValueError("obs_c indices must be in [0, n_cohorts-1].")
    if obs_t_np.min() < 0 or obs_t_np.max() >= t:
        raise ValueError("obs_t indices must be in [0, n_periods-1].")

    design = _build_design(obs_c_np, obs_t_np, c, t)
    bounds = _default_bounds(y_np, c, t)

    q_pinv_c = design["q_pinv_c"]
    q_pinv_t = design["q_pinv_t"]
    h_tilde = design["h_tilde"]
    dc = design["dc"]
    dt = design["dt"]

    jax, jnp, numpyro, MCMC, NUTS = _require_numpyro()

    y_j = jnp.asarray(y_np)
    h_j = jnp.asarray(h_tilde)
    qpc_j = jnp.asarray(q_pinv_c)
    qpt_j = jnp.asarray(q_pinv_t)
    ci_j = jnp.arange(c, dtype=jnp.float64).reshape(-1, 1)
    ti_j = jnp.arange(t, dtype=jnp.float64).reshape(-1, 1)

    def _rbf_kernel_jnp(x1, x2, ell, sf):
        x1s = x1 / ell
        x2s = x2 / ell
        dist2 = (
            jnp.sum(x1s ** 2, axis=1, keepdims=True)
            - 2.0 * (x1s @ x2s.T)
            + jnp.sum(x2s ** 2, axis=1)[None, :]
        )
        return (sf ** 2) * jnp.exp(-0.5 * dist2)

    def _model(y_obs, h_mat):
        log_ell_c = numpyro.sample("log_ell_c", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sf_c = numpyro.sample("log_sf_c", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_ell_t = numpyro.sample("log_ell_t", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sf_t = numpyro.sample("log_sf_t", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sn = numpyro.sample("log_sn", numpyro.distributions.Normal(0.0, prior_log_sd))

        if use_global_mean:
            mu = numpyro.sample("mu", numpyro.distributions.Normal(0.0, prior_mu_sd))
        else:
            mu = 0.0

        ell_c = jnp.clip(jnp.exp(log_ell_c), bounds.ell_c_low, bounds.ell_c_high)
        sf_c = jnp.clip(jnp.exp(log_sf_c), bounds.sf_c_low, bounds.sf_c_high)
        ell_t = jnp.clip(jnp.exp(log_ell_t), bounds.ell_t_low, bounds.ell_t_high)
        sf_t = jnp.clip(jnp.exp(log_sf_t), bounds.sf_t_low, bounds.sf_t_high)
        sn = jnp.clip(jnp.exp(log_sn), bounds.sn_low, bounds.sn_high)

        k_c_full = _rbf_kernel_jnp(ci_j, ci_j, ell_c, sf_c)
        k_t_full = _rbf_kernel_jnp(ti_j, ti_j, ell_t, sf_t)
        k_c_red = qpc_j @ k_c_full @ qpc_j.T
        k_t_red = qpt_j @ k_t_full @ qpt_j.T
        k_z = jnp.block(
            [
                [k_c_red, jnp.zeros((dc, dt), dtype=jnp.float64)],
                [jnp.zeros((dt, dc), dtype=jnp.float64), k_t_red],
            ]
        )
        sigma = h_mat @ k_z @ h_mat.T + (sn ** 2 + jitter) * jnp.eye(n, dtype=jnp.float64)
        numpyro.sample("obs", numpyro.distributions.MultivariateNormal(loc=mu * jnp.ones(n), covariance_matrix=sigma), obs=y_obs)

    kernel = NUTS(_model, dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=int(num_chains), progress_bar=False)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)), y_obs=y_j, h_mat=h_j)
    mcmc_run_seconds = float(time.perf_counter() - t0)

    samps = mcmc.get_samples(group_by_chain=False)

    log_ell_c = np.asarray(samps["log_ell_c"])
    log_sf_c = np.asarray(samps["log_sf_c"])
    log_ell_t = np.asarray(samps["log_ell_t"])
    log_sf_t = np.asarray(samps["log_sf_t"])
    log_sn = np.asarray(samps["log_sn"])
    mu_samps = np.asarray(samps["mu"]) if use_global_mean else np.zeros_like(log_sn)

    ell_c_s = np.clip(np.exp(log_ell_c), bounds.ell_c_low, bounds.ell_c_high)
    sf_c_s = np.clip(np.exp(log_sf_c), bounds.sf_c_low, bounds.sf_c_high)
    ell_t_s = np.clip(np.exp(log_ell_t), bounds.ell_t_low, bounds.ell_t_high)
    sf_t_s = np.clip(np.exp(log_sf_t), bounds.sf_t_low, bounds.sf_t_high)
    sn_s = np.clip(np.exp(log_sn), bounds.sn_low, bounds.sn_high)

    q_drop_c_t = _to_torch(design["q_drop_c"])
    q_drop_t_t = _to_torch(design["q_drop_t"])
    q_pinv_c_t = _to_torch(design["q_pinv_c"])
    q_pinv_t_t = _to_torch(design["q_pinv_t"])
    h_tilde_t = _to_torch(design["h_tilde"])
    y_t = _to_torch(y_np)
    ci_t = torch.arange(c, dtype=torch.float64).unsqueeze(-1)
    ti_t = torch.arange(t, dtype=torch.float64).unsqueeze(-1)

    mu_red_samps = []
    sigma_red_samps = []
    mu_draws = []
    for i in range(len(sn_s)):
        ell_c_i = torch.tensor(float(ell_c_s[i]), dtype=torch.float64)
        sf_c_i = torch.tensor(float(sf_c_s[i]), dtype=torch.float64)
        ell_t_i = torch.tensor(float(ell_t_s[i]), dtype=torch.float64)
        sf_t_i = torch.tensor(float(sf_t_s[i]), dtype=torch.float64)
        sn_i = torch.tensor(float(sn_s[i]), dtype=torch.float64)
        mu_i = torch.tensor(float(mu_samps[i]), dtype=torch.float64)

        k_c_full = _rbf_kernel_torch(ci_t, ci_t, ell_c_i, sf_c_i)
        k_t_full = _rbf_kernel_torch(ti_t, ti_t, ell_t_i, sf_t_i)
        k_c_red = q_pinv_c_t @ k_c_full @ q_pinv_c_t.T
        k_t_red = q_pinv_t_t @ k_t_full @ q_pinv_t_t.T
        k_z = torch.block_diag(k_c_red, k_t_red)
        mu_red_i, sigma_red_i = _analytical_gp_posterior(y_t - mu_i, h_tilde_t, k_z, sn_i, jitter=jitter)
        mu_red_samps.append(mu_red_i)
        sigma_red_samps.append(sigma_red_i)
        mu_draws.append(mu_i)

    mu_red_stack = torch.stack(mu_red_samps, dim=0)
    sigma_red_stack = torch.stack(sigma_red_samps, dim=0)
    mu_draws_t = torch.stack(mu_draws, dim=0)

    mu_red = mu_red_stack.mean(dim=0)
    mu_outer = torch.einsum("si,sj->sij", mu_red_stack, mu_red_stack).mean(dim=0)
    e_cov = sigma_red_stack.mean(dim=0) + mu_outer
    cov_red = e_cov - torch.outer(mu_red, mu_red)
    cov_red = 0.5 * (cov_red + cov_red.T)

    mu_zc = mu_red[:dc]
    mu_zt = mu_red[dc:]
    cov_zc = cov_red[:dc, :dc]
    cov_zt = cov_red[dc:, dc:]

    alpha_t = q_drop_c_t @ mu_zc
    beta_t = q_drop_t_t @ mu_zt
    std_alpha_t = torch.diag(q_drop_c_t @ cov_zc @ q_drop_c_t.T).clamp_min(0.0).sqrt()
    std_beta_t = torch.diag(q_drop_t_t @ cov_zt @ q_drop_t_t.T).clamp_min(0.0).sqrt()

    mu_post = float(mu_draws_t.mean().detach()) if use_global_mean else 0.0
    y_hat_t = mu_post + (h_tilde_t @ mu_red)
    resid_t = y_t - y_hat_t

    theta_summary = _summarise_theta_samples(ell_c_s, sf_c_s, ell_t_s, sf_t_s, sn_s)
    out = {
        "model_name": "GP-CP",
        "method_label": "2) NUTS Hyperparams",
        "method_id": "nuts_hyperparams",
        "use_global_mean": bool(use_global_mean),
        "index_base": 0,
        "seed": int(seed),
        "mu": mu_post,
        "alpha": alpha_t.detach().cpu().numpy(),
        "beta": beta_t.detach().cpu().numpy(),
        "gamma": np.zeros((c, t), dtype=float),
        "std_alpha": std_alpha_t.detach().cpu().numpy(),
        "std_beta": std_beta_t.detach().cpu().numpy(),
        "lo95_alpha": (alpha_t - 1.96 * std_alpha_t).detach().cpu().numpy(),
        "hi95_alpha": (alpha_t + 1.96 * std_alpha_t).detach().cpu().numpy(),
        "lo95_beta": (beta_t - 1.96 * std_beta_t).detach().cpu().numpy(),
        "hi95_beta": (beta_t + 1.96 * std_beta_t).detach().cpu().numpy(),
        "y_hat": y_hat_t.detach().cpu().numpy(),
        "resid": resid_t.detach().cpu().numpy(),
        "hyperparams_map": {
            "ell_c": theta_summary["mean"]["ell_c"],
            "sf_c": theta_summary["mean"]["sf_c"],
            "ell_t": theta_summary["mean"]["ell_t"],
            "sf_t": theta_summary["mean"]["sf_t"],
            "sn": theta_summary["mean"]["sn"],
        },
        "hyperparams_posterior": theta_summary,
        "runtime_seconds": float(time.perf_counter() - t_total0),
        "mcmc_run_seconds": mcmc_run_seconds,
        "mcmc_config": {
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "jitter": float(jitter),
            "prior_log_sd": float(prior_log_sd),
            "prior_mu_sd": float(prior_mu_sd),
        },
    }

    if return_samples:
        out["posterior_samples"] = {
            "ell_c": ell_c_s,
            "sf_c": sf_c_s,
            "ell_t": ell_t_s,
            "sf_t": sf_t_s,
            "sn": sn_s,
            "mu": mu_samps,
        }
    return out


def fit_full_nuts(
    y: "array-like",
    obs_c: "array-like",
    obs_t: "array-like",
    n_cohorts: int,
    n_periods: int,
    *,
    seed: int = 0,
    use_global_mean: bool = False,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    jitter: float = 1e-6,
    prior_log_sd: float = 2.0,
    prior_mu_sd: float = 5.0,
    return_samples: bool = True,
) -> dict[str, Any]:
    """
    Method 3: Full NUTS sampling over latent reduced effects z and hyperparameters.
    """
    t_total0 = time.perf_counter()
    y_np = _to_numpy(y).astype(float)
    obs_c_np = _to_numpy(obs_c).astype(int)
    obs_t_np = _to_numpy(obs_t).astype(int)
    n = len(y_np)
    if n == 0:
        raise ValueError("Empty observations provided.")

    c = int(n_cohorts)
    t = int(n_periods)
    if obs_c_np.min() < 0 or obs_c_np.max() >= c:
        raise ValueError("obs_c indices must be in [0, n_cohorts-1].")
    if obs_t_np.min() < 0 or obs_t_np.max() >= t:
        raise ValueError("obs_t indices must be in [0, n_periods-1].")

    design = _build_design(obs_c_np, obs_t_np, c, t)
    bounds = _default_bounds(y_np, c, t)

    q_pinv_c = design["q_pinv_c"]
    q_pinv_t = design["q_pinv_t"]
    h_tilde = design["h_tilde"]
    dc = design["dc"]
    dt = design["dt"]
    dz = dc + dt

    jax, jnp, numpyro, MCMC, NUTS = _require_numpyro()

    y_j = jnp.asarray(y_np)
    h_j = jnp.asarray(h_tilde)
    qpc_j = jnp.asarray(q_pinv_c)
    qpt_j = jnp.asarray(q_pinv_t)
    ci_j = jnp.arange(c, dtype=jnp.float64).reshape(-1, 1)
    ti_j = jnp.arange(t, dtype=jnp.float64).reshape(-1, 1)

    def _rbf_kernel_jnp(x1, x2, ell, sf):
        x1s = x1 / ell
        x2s = x2 / ell
        dist2 = (
            jnp.sum(x1s ** 2, axis=1, keepdims=True)
            - 2.0 * (x1s @ x2s.T)
            + jnp.sum(x2s ** 2, axis=1)[None, :]
        )
        return (sf ** 2) * jnp.exp(-0.5 * dist2)

    def _model(y_obs, h_mat):
        log_ell_c = numpyro.sample("log_ell_c", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sf_c = numpyro.sample("log_sf_c", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_ell_t = numpyro.sample("log_ell_t", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sf_t = numpyro.sample("log_sf_t", numpyro.distributions.Normal(0.0, prior_log_sd))
        log_sn = numpyro.sample("log_sn", numpyro.distributions.Normal(0.0, prior_log_sd))

        if use_global_mean:
            mu = numpyro.sample("mu", numpyro.distributions.Normal(0.0, prior_mu_sd))
        else:
            mu = 0.0

        ell_c = jnp.clip(jnp.exp(log_ell_c), bounds.ell_c_low, bounds.ell_c_high)
        sf_c = jnp.clip(jnp.exp(log_sf_c), bounds.sf_c_low, bounds.sf_c_high)
        ell_t = jnp.clip(jnp.exp(log_ell_t), bounds.ell_t_low, bounds.ell_t_high)
        sf_t = jnp.clip(jnp.exp(log_sf_t), bounds.sf_t_low, bounds.sf_t_high)
        sn = jnp.clip(jnp.exp(log_sn), bounds.sn_low, bounds.sn_high)

        k_c_full = _rbf_kernel_jnp(ci_j, ci_j, ell_c, sf_c)
        k_t_full = _rbf_kernel_jnp(ti_j, ti_j, ell_t, sf_t)
        k_c_red = qpc_j @ k_c_full @ qpc_j.T
        k_t_red = qpt_j @ k_t_full @ qpt_j.T
        k_z = jnp.block(
            [
                [k_c_red, jnp.zeros((dc, dt), dtype=jnp.float64)],
                [jnp.zeros((dt, dc), dtype=jnp.float64), k_t_red],
            ]
        )
        sigma_f = h_mat @ k_z @ h_mat.T + jitter * jnp.eye(n, dtype=jnp.float64)
        f_latent = numpyro.sample(
            "f",
            numpyro.distributions.MultivariateNormal(
                loc=jnp.zeros(n, dtype=jnp.float64),
                covariance_matrix=sigma_f,
            ),
        )
        mean_y = mu + f_latent
        numpyro.sample("obs", numpyro.distributions.Normal(mean_y, sn).to_event(1), obs=y_obs)

    kernel = NUTS(_model, dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=int(num_chains), progress_bar=False)

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(int(seed)), y_obs=y_j, h_mat=h_j)
    mcmc_run_seconds = float(time.perf_counter() - t0)

    samps = mcmc.get_samples(group_by_chain=False)

    log_ell_c = np.asarray(samps["log_ell_c"])
    log_sf_c = np.asarray(samps["log_sf_c"])
    log_ell_t = np.asarray(samps["log_ell_t"])
    log_sf_t = np.asarray(samps["log_sf_t"])
    log_sn = np.asarray(samps["log_sn"])
    f_samps = np.asarray(samps["f"])
    mu_samps = np.asarray(samps["mu"]) if use_global_mean else np.zeros(len(log_sn), dtype=float)

    ell_c_s = np.clip(np.exp(log_ell_c), bounds.ell_c_low, bounds.ell_c_high)
    sf_c_s = np.clip(np.exp(log_sf_c), bounds.sf_c_low, bounds.sf_c_high)
    ell_t_s = np.clip(np.exp(log_ell_t), bounds.ell_t_low, bounds.ell_t_high)
    sf_t_s = np.clip(np.exp(log_sf_t), bounds.sf_t_low, bounds.sf_t_high)
    sn_s = np.clip(np.exp(log_sn), bounds.sn_low, bounds.sn_high)

    h_pinv = np.linalg.pinv(h_tilde)  # [dz, n]
    z_samps = (h_pinv @ f_samps.T).T  # [S, dz]

    alpha_draws = z_samps[:, :dc] @ design["q_drop_c"].T
    beta_draws = z_samps[:, dc:] @ design["q_drop_t"].T
    alpha_mean = alpha_draws.mean(axis=0)
    beta_mean = beta_draws.mean(axis=0)
    alpha_std = alpha_draws.std(axis=0, ddof=1)
    beta_std = beta_draws.std(axis=0, ddof=1)

    y_hat_draws = mu_samps[:, None] + f_samps
    y_hat = y_hat_draws.mean(axis=0)
    resid = y_np - y_hat
    mu_post = float(np.mean(mu_samps)) if use_global_mean else 0.0

    theta_summary = _summarise_theta_samples(ell_c_s, sf_c_s, ell_t_s, sf_t_s, sn_s)
    out = {
        "model_name": "GP-CP",
        "method_label": "3) Full NUTS (Sample f + theta)",
        "method_id": "full_nuts",
        "use_global_mean": bool(use_global_mean),
        "index_base": 0,
        "seed": int(seed),
        "mu": mu_post,
        "alpha": alpha_mean,
        "beta": beta_mean,
        "gamma": np.zeros((c, t), dtype=float),
        "std_alpha": alpha_std,
        "std_beta": beta_std,
        "lo95_alpha": alpha_mean - 1.96 * alpha_std,
        "hi95_alpha": alpha_mean + 1.96 * alpha_std,
        "lo95_beta": beta_mean - 1.96 * beta_std,
        "hi95_beta": beta_mean + 1.96 * beta_std,
        "y_hat": y_hat,
        "resid": resid,
        "hyperparams_map": {
            "ell_c": theta_summary["mean"]["ell_c"],
            "sf_c": theta_summary["mean"]["sf_c"],
            "ell_t": theta_summary["mean"]["ell_t"],
            "sf_t": theta_summary["mean"]["sf_t"],
            "sn": theta_summary["mean"]["sn"],
        },
        "hyperparams_posterior": theta_summary,
        "runtime_seconds": float(time.perf_counter() - t_total0),
        "mcmc_run_seconds": mcmc_run_seconds,
        "mcmc_config": {
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "jitter": float(jitter),
            "prior_log_sd": float(prior_log_sd),
            "prior_mu_sd": float(prior_mu_sd),
        },
    }

    if return_samples:
        out["posterior_samples"] = {
            "ell_c": ell_c_s,
            "sf_c": sf_c_s,
            "ell_t": ell_t_s,
            "sf_t": sf_t_s,
            "sn": sn_s,
            "mu": mu_samps,
            "f": f_samps,
        }
    return out
