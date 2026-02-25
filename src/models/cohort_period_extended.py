"""
cohort_period_extended.py
=========================
Constrained extended GP model for cohort + period + interaction:

    y_i = [mu] + alpha_{c(i)} + beta_{t(i)} + gamma_{c(i), t(i)} + eps_i
    eps_i ~ N(0, sn^2)

alpha and beta use sum-to-zero bases; gamma uses doubly-centered reduced basis.
Inference follows MAP + Laplace.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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


def _rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, ell: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    x1s = x1 / ell
    x2s = x2 / ell
    dist2 = (
        x1s.pow(2).sum(-1, keepdim=True)
        - 2.0 * (x1s @ x2s.T)
        + x2s.pow(2).sum(-1).unsqueeze(0)
    )
    return (sf ** 2) * torch.exp(-0.5 * dist2)


def _hessian_of_scalar(fn: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    grads = torch.autograd.grad(fn, params, create_graph=True)
    rows = []
    for g in grads:
        row = [torch.autograd.grad(g, p, retain_graph=True)[0] for p in params]
        rows.append(torch.stack(row))
    return torch.stack(rows)


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


class CohortPeriodExtendedModel:
    """
    Constrained extended GP model with additive + interaction components.
    """

    def __init__(
        self,
        use_global_mean: bool = False,
        n_laplace_samples: int = 200,
        n_map_steps: int = 50,
    ) -> None:
        self.use_global_mean = use_global_mean
        self.n_laplace_samples = int(n_laplace_samples)
        self.n_map_steps = int(n_map_steps)

        self.mu_: float = 0.0
        self.alpha_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.std_alpha_: Optional[np.ndarray] = None
        self.std_beta_: Optional[np.ndarray] = None
        self.std_gamma_: Optional[np.ndarray] = None
        self.y_hat_: Optional[np.ndarray] = None
        self.resid_: Optional[np.ndarray] = None
        self.hyperparams_map_: Optional[dict] = None
        self.bounds_used_: Optional[dict] = None

        self.seed_: Optional[int] = None
        self.true_hyperparams_: Optional[dict] = None

        self._n_cohorts: Optional[int] = None
        self._n_periods: Optional[int] = None
        self._is_fitted: bool = False

    def fit(
        self,
        y: "array-like",
        obs_c: "array-like",
        obs_t: "array-like",
        n_cohorts: int,
        n_periods: int,
        seed: Optional[int] = None,
        true_hyperparams: Optional[dict] = None,
    ) -> "CohortPeriodExtendedModel":
        y_np = _to_numpy(y).astype(float)
        obs_c_np = _to_numpy(obs_c).astype(int)
        obs_t_np = _to_numpy(obs_t).astype(int)

        c = int(n_cohorts)
        t = int(n_periods)
        n = len(y_np)
        if n == 0:
            raise ValueError("Empty observations provided to CohortPeriodExtendedModel.fit.")
        if obs_c_np.min() < 0 or obs_c_np.max() >= c:
            raise ValueError("obs_c indices must be in [0, n_cohorts-1].")
        if obs_t_np.min() < 0 or obs_t_np.max() >= t:
            raise ValueError("obs_t indices must be in [0, n_periods-1].")

        q_drop_c, q_pinv_c = _sum_to_zero_basis(c)
        q_drop_t, q_pinv_t = _sum_to_zero_basis(t)
        q_drop_c_t = _to_torch(q_drop_c)
        q_drop_t_t = _to_torch(q_drop_t)
        q_pinv_c_t = _to_torch(q_pinv_c)
        q_pinv_t_t = _to_torch(q_pinv_t)

        dc = c - 1
        dt = t - 1
        dg = dc * dt

        a = np.zeros((n, c), dtype=float)
        b = np.zeros((n, t), dtype=float)
        a[np.arange(n), obs_c_np] = 1.0
        b[np.arange(n), obs_t_np] = 1.0
        aq = a @ q_drop_c
        bq = b @ q_drop_t

        # Interaction design in reduced basis:
        # row i = kron(Qc[c_i,:], Qt[t_i,:])
        gq = np.einsum("ni,nj->nij", aq, bq).reshape(n, dg)
        h_ext = np.concatenate([aq, bq, gq], axis=1)
        h_ext_t = _to_torch(h_ext)
        y_t = _to_torch(y_np)

        ci = torch.arange(c, dtype=torch.float64).unsqueeze(-1)
        ti = torch.arange(t, dtype=torch.float64).unsqueeze(-1)

        def marginal_loglik_extended(
            y_centered: torch.Tensor,
            ell_c: torch.Tensor,
            sf_c: torch.Tensor,
            ell_t: torch.Tensor,
            sf_t: torch.Tensor,
            ell_gc: torch.Tensor,
            sf_gc: torch.Tensor,
            ell_gt: torch.Tensor,
            sf_gt: torch.Tensor,
            sn: torch.Tensor,
            jitter: float = 1e-6,
        ) -> torch.Tensor:
            k_c_full = _rbf_kernel(ci, ci, ell_c, sf_c)
            k_t_full = _rbf_kernel(ti, ti, ell_t, sf_t)
            k_gc_full = _rbf_kernel(ci, ci, ell_gc, sf_gc)
            k_gt_full = _rbf_kernel(ti, ti, ell_gt, sf_gt)

            k_c_red = q_pinv_c_t @ k_c_full @ q_pinv_c_t.T
            k_t_red = q_pinv_t_t @ k_t_full @ q_pinv_t_t.T
            k_gc_red = q_pinv_c_t @ k_gc_full @ q_pinv_c_t.T
            k_gt_red = q_pinv_t_t @ k_gt_full @ q_pinv_t_t.T
            k_g_red = torch.kron(k_gc_red, k_gt_red)

            k_z = torch.block_diag(k_c_red, k_t_red, k_g_red)
            sigma_y = h_ext_t @ k_z @ h_ext_t.T + (sn ** 2 + jitter) * torch.eye(n, dtype=torch.float64)
            try:
                l = torch.linalg.cholesky(sigma_y)
            except RuntimeError:
                sigma_y = sigma_y + 1e-4 * torch.eye(n, dtype=torch.float64)
                l = torch.linalg.cholesky(sigma_y)
            alpha_v = torch.cholesky_solve(y_centered.unsqueeze(-1), l).squeeze(-1)
            logdet = 2.0 * torch.sum(torch.log(torch.diag(l)))
            return -0.5 * (y_centered @ alpha_v) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)

        if self.use_global_mean:
            mu_raw = torch.tensor(float(np.mean(y_np)), requires_grad=True, dtype=torch.float64)
        else:
            mu_raw = torch.tensor(0.0, requires_grad=False, dtype=torch.float64)

        # Data-aware scale (robust + standard) for conservative real-data bounds.
        y_std = float(np.std(y_np))
        y_mad = float(np.median(np.abs(y_np - np.median(y_np))) * 1.4826)
        y_scale = max(y_std, y_mad, 0.1)

        sf_init = max(1.0, 0.5 * y_scale)
        sf_g_init = max(0.5, 0.25 * y_scale)
        sn_init = max(0.3, 0.10 * y_scale)

        logs = [
            torch.tensor(math.log(1.0), requires_grad=True, dtype=torch.float64),      # ell_c
            torch.tensor(math.log(sf_init), requires_grad=True, dtype=torch.float64),  # sf_c
            torch.tensor(math.log(1.0), requires_grad=True, dtype=torch.float64),      # ell_t
            torch.tensor(math.log(sf_init), requires_grad=True, dtype=torch.float64),  # sf_t
            torch.tensor(math.log(1.0), requires_grad=True, dtype=torch.float64),      # ell_gc
            torch.tensor(math.log(sf_g_init), requires_grad=True, dtype=torch.float64),# sf_gc
            torch.tensor(math.log(1.0), requires_grad=True, dtype=torch.float64),      # ell_gt
            torch.tensor(math.log(sf_g_init), requires_grad=True, dtype=torch.float64),# sf_gt
            torch.tensor(math.log(sn_init), requires_grad=True, dtype=torch.float64),  # sn
        ]

        if self.use_global_mean:
            params = [mu_raw] + logs
            log_params = logs
        else:
            params = logs
            log_params = logs

        # Data-aware conservative bounds: wide enough to avoid clipping plausible real-data values.
        ell_upper_c = max(8.0, 2.0 * float(c))
        ell_upper_t = max(8.0, 2.0 * float(t))
        sf_upper = max(8.0, 20.0 * y_scale)
        sf_g_upper = max(8.0, 20.0 * y_scale)
        sn_upper = max(5.0, 12.0 * y_scale)
        bounds = [
            (0.01, ell_upper_c),  # ell_c
            (0.005, sf_upper),    # sf_c
            (0.01, ell_upper_t),  # ell_t
            (0.005, sf_upper),    # sf_t
            (0.01, ell_upper_c),  # ell_gc
            (0.005, sf_g_upper),  # sf_gc
            (0.01, ell_upper_t),  # ell_gt
            (0.005, sf_g_upper),  # sf_gt
            (1e-4, sn_upper),     # sn
        ]
        self.bounds_used_ = {
            "ell_c": [float(bounds[0][0]), float(bounds[0][1])],
            "sf_c": [float(bounds[1][0]), float(bounds[1][1])],
            "ell_t": [float(bounds[2][0]), float(bounds[2][1])],
            "sf_t": [float(bounds[3][0]), float(bounds[3][1])],
            "ell_gc": [float(bounds[4][0]), float(bounds[4][1])],
            "sf_gc": [float(bounds[5][0]), float(bounds[5][1])],
            "ell_gt": [float(bounds[6][0]), float(bounds[6][1])],
            "sf_gt": [float(bounds[7][0]), float(bounds[7][1])],
            "sn": [float(bounds[8][0]), float(bounds[8][1])],
            "y_scale": float(y_scale),
        }

        def log_prior_hyper(lps: list[torch.Tensor]) -> torch.Tensor:
            return sum(-0.5 * (lp / 2.0) ** 2 for lp in lps)

        def log_prior_mu(mu: torch.Tensor) -> torch.Tensor:
            return -0.5 * (mu / 5.0) ** 2

        opt = torch.optim.LBFGS(params, lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            vals = [torch.exp(lp).clamp(lo, hi) for lp, (lo, hi) in zip(log_params, bounds)]
            if self.use_global_mean:
                y_centered = y_t - mu_raw
                lp = log_prior_hyper(log_params) + log_prior_mu(mu_raw)
            else:
                y_centered = y_t
                lp = log_prior_hyper(log_params)
            ll = marginal_loglik_extended(
                y_centered,
                vals[0], vals[1], vals[2], vals[3],
                vals[4], vals[5], vals[6], vals[7], vals[8],
            )
            loss = -(ll + lp)
            loss.backward()
            return loss

        for _ in range(self.n_map_steps):
            opt.step(closure)

        vals_map = [torch.exp(lp).clamp(lo, hi).detach() for lp, (lo, hi) in zip(log_params, bounds)]

        for p in params:
            p.requires_grad_(True)

        vals_now = [torch.exp(lp).clamp(lo, hi) for lp, (lo, hi) in zip(log_params, bounds)]
        if self.use_global_mean:
            y_centered_now = y_t - mu_raw
            nll = -(marginal_loglik_extended(
                y_centered_now,
                vals_now[0], vals_now[1], vals_now[2], vals_now[3],
                vals_now[4], vals_now[5], vals_now[6], vals_now[7], vals_now[8],
            ) + log_prior_hyper(log_params) + log_prior_mu(mu_raw))
        else:
            nll = -(marginal_loglik_extended(
                y_t,
                vals_now[0], vals_now[1], vals_now[2], vals_now[3],
                vals_now[4], vals_now[5], vals_now[6], vals_now[7], vals_now[8],
            ) + log_prior_hyper(log_params))

        hess = _hessian_of_scalar(nll, params).detach().cpu().numpy()
        eigvals = np.linalg.eigvalsh(hess)
        reg = max(1e-6, -float(eigvals.min()) + 1e-4) if float(eigvals.min()) < 1e-6 else 1e-6
        cov_params = np.linalg.inv(hess + reg * np.eye(hess.shape[0]))

        param_mean = np.array([float(p.detach()) for p in params], dtype=float)
        mvn = torch.distributions.MultivariateNormal(
            torch.tensor(param_mean, dtype=torch.float64),
            torch.tensor(cov_params, dtype=torch.float64),
        )
        draws = mvn.sample((self.n_laplace_samples,))

        mu_red_samps = []
        sigma_red_samps = []
        mu_draws = []

        for s in range(self.n_laplace_samples):
            row = draws[s]
            if self.use_global_mean:
                mu_s = row[0]
                lp_s = row[1:]
            else:
                mu_s = torch.tensor(0.0, dtype=torch.float64)
                lp_s = row
            vals_s = [torch.exp(lp).clamp(lo, hi) for lp, (lo, hi) in zip(lp_s, bounds)]

            k_c_full = _rbf_kernel(ci, ci, vals_s[0], vals_s[1])
            k_t_full = _rbf_kernel(ti, ti, vals_s[2], vals_s[3])
            k_gc_full = _rbf_kernel(ci, ci, vals_s[4], vals_s[5])
            k_gt_full = _rbf_kernel(ti, ti, vals_s[6], vals_s[7])

            k_c_red = q_pinv_c_t @ k_c_full @ q_pinv_c_t.T
            k_t_red = q_pinv_t_t @ k_t_full @ q_pinv_t_t.T
            k_gc_red = q_pinv_c_t @ k_gc_full @ q_pinv_c_t.T
            k_gt_red = q_pinv_t_t @ k_gt_full @ q_pinv_t_t.T
            k_g_red = torch.kron(k_gc_red, k_gt_red)
            k_z = torch.block_diag(k_c_red, k_t_red, k_g_red)

            mu_red_s, sigma_red_s = _analytical_gp_posterior(y_t - mu_s, h_ext_t, k_z, vals_s[8])
            mu_red_samps.append(mu_red_s)
            sigma_red_samps.append(sigma_red_s)
            mu_draws.append(mu_s)

        mu_red_stack = torch.stack(mu_red_samps, dim=0)
        sigma_red_stack = torch.stack(sigma_red_samps, dim=0)
        mu_draws_t = torch.stack(mu_draws, dim=0)

        mu_red = mu_red_stack.mean(dim=0)
        mu_outer = torch.einsum("si,sj->sij", mu_red_stack, mu_red_stack).mean(dim=0)
        e_cov = sigma_red_stack.mean(dim=0) + mu_outer
        cov_red = e_cov - torch.outer(mu_red, mu_red)
        cov_red = 0.5 * (cov_red + cov_red.T)

        idx_a = dc
        idx_b = dc + dt

        mu_zc = mu_red[:idx_a]
        mu_zt = mu_red[idx_a:idx_b]
        mu_zg = mu_red[idx_b:].reshape(dc, dt)

        cov_zc = cov_red[:idx_a, :idx_a]
        cov_zt = cov_red[idx_a:idx_b, idx_a:idx_b]
        cov_zg = cov_red[idx_b:, idx_b:]

        alpha_t = q_drop_c_t @ mu_zc
        beta_t = q_drop_t_t @ mu_zt
        gamma_t = q_drop_c_t @ mu_zg @ q_drop_t_t.T

        cov_alpha = q_drop_c_t @ cov_zc @ q_drop_c_t.T
        cov_beta = q_drop_t_t @ cov_zt @ q_drop_t_t.T
        std_alpha_t = torch.diag(cov_alpha).clamp_min(0.0).sqrt()
        std_beta_t = torch.diag(cov_beta).clamp_min(0.0).sqrt()

        # Interaction std from reduced covariance: diag((Qt ⊗ Qc) Cov(zg) (Qt ⊗ Qc)^T)
        qg = torch.kron(q_drop_c_t, q_drop_t_t)  # [C*T, dc*dt]
        var_gamma_vec = torch.diag(qg @ cov_zg @ qg.T).clamp_min(0.0)
        std_gamma_t = var_gamma_vec.sqrt().reshape(c, t)

        mu_post = float(mu_draws_t.mean().detach()) if self.use_global_mean else 0.0
        y_hat_t = mu_post + alpha_t[obs_c_np] + beta_t[obs_t_np] + gamma_t[obs_c_np, obs_t_np]
        resid_t = y_t - y_hat_t

        self.mu_ = mu_post
        self.alpha_ = alpha_t.detach().cpu().numpy()
        self.beta_ = beta_t.detach().cpu().numpy()
        self.gamma_ = gamma_t.detach().cpu().numpy()
        self.std_alpha_ = std_alpha_t.detach().cpu().numpy()
        self.std_beta_ = std_beta_t.detach().cpu().numpy()
        self.std_gamma_ = std_gamma_t.detach().cpu().numpy()
        self.y_hat_ = y_hat_t.detach().cpu().numpy()
        self.resid_ = resid_t.detach().cpu().numpy()
        self.hyperparams_map_ = {
            "ell_c": float(vals_map[0]),
            "sf_c": float(vals_map[1]),
            "ell_t": float(vals_map[2]),
            "sf_t": float(vals_map[3]),
            "ell_gc": float(vals_map[4]),
            "sf_gc": float(vals_map[5]),
            "ell_gt": float(vals_map[6]),
            "sf_gt": float(vals_map[7]),
            "sn": float(vals_map[8]),
        }
        self.seed_ = seed
        self.true_hyperparams_ = dict(true_hyperparams) if true_hyperparams is not None else None
        self._n_cohorts = c
        self._n_periods = t
        self._is_fitted = True
        return self

    def predict(self, obs_c: "array-like", obs_t: "array-like") -> np.ndarray:
        self._check_fitted()
        c_idx = _to_numpy(obs_c).astype(int)
        t_idx = _to_numpy(obs_t).astype(int)
        return self.mu_ + self.alpha_[c_idx] + self.beta_[t_idx] + self.gamma_[c_idx, t_idx]

    def predict_beta(self, periods: Optional["array-like"] = None) -> dict:
        self._check_fitted()
        if periods is None:
            p_idx = np.arange(self._n_periods, dtype=int)
        else:
            p_idx = _to_numpy(periods).astype(int)
        beta = self.beta_[p_idx]
        std = self.std_beta_[p_idx]
        return {
            "periods": p_idx,
            "mean": beta,
            "std": std,
            "lo95": beta - 1.96 * std,
            "hi95": beta + 1.96 * std,
        }

    def results_dict(self) -> dict:
        self._check_fitted()
        return {
            "model_name": "GP-CP-Extended",
            "use_global_mean": self.use_global_mean,
            "index_base": 0,
            "seed": self.seed_,
            "true_hyperparams": self.true_hyperparams_,
            "mu": self.mu_,
            "alpha": self.alpha_,
            "beta": self.beta_,
            "gamma": self.gamma_,
            "std_alpha": self.std_alpha_,
            "std_beta": self.std_beta_,
            "std_gamma": self.std_gamma_,
            "lo95_alpha": self.alpha_ - 1.96 * self.std_alpha_,
            "hi95_alpha": self.alpha_ + 1.96 * self.std_alpha_,
            "lo95_beta": self.beta_ - 1.96 * self.std_beta_,
            "hi95_beta": self.beta_ + 1.96 * self.std_beta_,
            "lo95_gamma": self.gamma_ - 1.96 * self.std_gamma_,
            "hi95_gamma": self.gamma_ + 1.96 * self.std_gamma_,
            "y_hat": self.y_hat_,
            "resid": self.resid_,
            "hyperparams_map": self.hyperparams_map_,
            "bounds_used": self.bounds_used_,
        }

    def summary(self) -> None:
        self._check_fitted()
        print(
            f"\n{'-' * 66}\n"
            f"  CohortPeriodExtendedModel ({'with' if self.use_global_mean else 'without'} global mean)\n"
            f"{'-' * 66}\n"
            f"  Grid         : C={self._n_cohorts}, T={self._n_periods}\n"
            f"  mu           : {self.mu_:.5f}\n"
            f"  alpha range  : [{self.alpha_.min():.3f}, {self.alpha_.max():.3f}]  sum={self.alpha_.sum():.2e}\n"
            f"  beta range   : [{self.beta_.min():.3f}, {self.beta_.max():.3f}]  sum={self.beta_.sum():.2e}\n"
            f"  gamma range  : [{self.gamma_.min():.3f}, {self.gamma_.max():.3f}]\n"
            f"  MAP theta    : {self.hyperparams_map_}\n"
            f"{'-' * 66}\n"
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
