"""
fixed_effects.py
================
Two-way fixed effects (FE) baseline model with sum-to-zero constraints.

Model
-----
Fitting task (in-sample):

    y_i = [μ]  +  α_{c(i)}  +  β_{t(i)}  +  ε_i,    ε_i ~ N(0, σ²)

    where μ is an optional global intercept (use_global_mean=True).

    Sum-to-zero constraints  Σ_c α_c = 0,  Σ_t β_t = 0  are enforced
    via the reparameterisation  α = Q_drop_c · z_c,  β = Q_drop_t · z_t,
    giving the constrained design matrix

        H̃ = [ A Q_drop_c  |  B Q_drop_t ]      (shape [N, C-1+T-1])

    optionally prefixed with a ones column for μ.  OLS gives the MLE.
    Uncertainty is propagated to the full α, β space via the delta method.

Extrapolation task (out-of-sample):

    1.  fit() is called on TRAINING observations only (periods 0 .. T_tr-1).
    2.  extrapolate(n_extrap) fits an AR(1) process on the estimated β̂_{1:T_tr}:

            β_t  =  c_ar  +  φ · β_{t-1}  +  ε_ar,    ε_ar ~ N(0, σ²_ar)

        and steps it forward n_extrap periods, accumulating forecast variance:

            Var(β̂_{T_tr+h}) = σ²_ar · Σ_{j=0}^{h-1} φ^{2j}

Reference implementations
--------------------------
- OLS + delta method:  notebook 00_test_models.ipynb, Cell 35
- With global mean μ:  notebook 01_realdata_part45_model_comparison_draft.ipynb, Cell 31
- AR(1) extrapolation: notebook 00_test_models.ipynb, Cell 41
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── Internal helper ───────────────────────────────────────────────────────────

def _sum_to_zero_basis(K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the sum-to-zero reparameterisation matrices for a K-vector.

    The constraint  Σ_k θ_k = 0  is equivalent to  θ ∈ col(Q_drop).
    Any vector in the sum-to-zero subspace can be written as  θ = Q_drop · z,
    and the reduced coefficients recovered as  z = Q_pinv · θ.

    Parameters
    ----------
    K : int
        Number of levels (cohorts or periods).

    Returns
    -------
    Q_drop : ndarray [K, K-1]
        Columns span the sum-to-zero subspace.
        Defined as  P[:, 1:]  where  P = I - (1/K) 11ᵀ.
    Q_pinv : ndarray [K-1, K]
        Left pseudo-inverse of Q_drop  (Q_pinv @ Q_drop = I_{K-1}).
        Computed from the thin QR decomposition of Q_drop.
    """
    I    = np.eye(K)
    ones = np.ones((K, 1))
    P      = I - (ones @ ones.T) / K          # [K, K]  projection matrix
    Q_drop = P[:, 1:]                          # [K, K-1]
    Q_orth, R = np.linalg.qr(Q_drop)          # thin QR: Q_orth [K,K-1], R [K-1,K-1]
    Q_pinv = np.linalg.inv(R) @ Q_orth.T      # [K-1, K]
    return Q_drop, Q_pinv


def _to_numpy(x) -> np.ndarray:
    """Convert a torch.Tensor or any array-like to a numpy array."""
    if hasattr(x, "numpy"):          # torch.Tensor
        return x.detach().numpy()
    return np.asarray(x)


# ── Model class ───────────────────────────────────────────────────────────────

class FixedEffectsModel:
    """
    Two-way fixed effects model with optional global intercept.

    Parameters
    ----------
    use_global_mean : bool, default False
        If False  (default, matches 00_test_models.ipynb):
            y = α_c + β_t + ε    (sum-to-zero on both effects)
        If True  (matches 01_realdata_part45_model_comparison_draft.ipynb Section 4b):
            y = μ + α_c + β_t + ε    (intercept absorbs the grand mean)

    Fitted Attributes (available after fit())
    ------------------------------------------
    mu_         : float        Global mean (0.0 when use_global_mean=False)
    alpha_      : ndarray [C]  Cohort effects  (sum-to-zero)
    beta_       : ndarray [T]  Time effects    (sum-to-zero)
    std_alpha_  : ndarray [C]  Delta-method standard errors for α
    std_beta_   : ndarray [T]  Delta-method standard errors for β
    sigma2_     : float        Residual variance  σ̂²
    y_hat_      : ndarray [N]  Fitted values
    resid_      : ndarray [N]  Residuals  y − ŷ

    After extrapolate()
    -------------------
    beta_extrap_mean_ : ndarray [n_extrap]
    beta_extrap_std_  : ndarray [n_extrap]
    ar_phi_           : float   AR(1) coefficient φ
    ar_c_             : float   AR(1) intercept c_ar
    ar_sigma2_        : float   AR(1) residual variance
    """

    def __init__(self, use_global_mean: bool = False) -> None:
        self.use_global_mean = use_global_mean

        # Fitted state — set by fit()
        self.mu_        : float               = 0.0
        self.alpha_     : Optional[np.ndarray] = None
        self.beta_      : Optional[np.ndarray] = None
        self.std_alpha_ : Optional[np.ndarray] = None
        self.std_beta_  : Optional[np.ndarray] = None
        self.sigma2_    : Optional[float]      = None
        self.y_hat_     : Optional[np.ndarray] = None
        self.resid_     : Optional[np.ndarray] = None

        # Extrapolation state — set by extrapolate()
        self.beta_extrap_mean_ : Optional[np.ndarray] = None
        self.beta_extrap_std_  : Optional[np.ndarray] = None
        self.ar_phi_           : Optional[float]      = None
        self.ar_c_             : Optional[float]      = None
        self.ar_sigma2_        : Optional[float]      = None

        # Metadata (set by fit; used by results_dict for experiment tracking)
        self.seed_             : Optional[int]        = None
        self.true_hyperparams_ : Optional[dict]       = None

        # Internals for downstream use (e.g. extrapolate, predict)
        self._Q_drop_c : Optional[np.ndarray] = None
        self._Q_drop_t : Optional[np.ndarray] = None
        self._n_cohorts: Optional[int]         = None
        self._n_periods: Optional[int]         = None
        self._is_fitted: bool                  = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        y: "array-like",
        obs_c: "array-like",
        obs_t: "array-like",
        n_cohorts: int,
        n_periods: int,
        seed: Optional[int] = None,
        true_hyperparams: Optional[dict] = None,
    ) -> "FixedEffectsModel":
        """
        Fit two-way FE via OLS on the provided observations.

        For the in-sample fitting task  pass all T periods.
        For the extrapolation task      pass training observations only
        (with n_periods = T_train), then call extrapolate().

        Parameters
        ----------
        y        : array-like [N]   Observations (torch.Tensor or ndarray).
        obs_c    : array-like [N]   Cohort indices, 0-based.
        obs_t    : array-like [N]   Period indices, 0-based.
                                    Must satisfy  0 ≤ obs_t[i] < n_periods.
        n_cohorts: int              Number of distinct cohorts C.
        n_periods: int              Number of distinct periods T.
                                    For extrapolation training, T = T_train.
        seed     : Optional[int]    Simulation seed metadata to store.
        true_hyperparams : Optional[dict]
                                    DGP hyperparameters metadata to store.

        Returns
        -------
        self  (for method chaining)
        """
        y_np    = _to_numpy(y).astype(float)
        obs_c_np = _to_numpy(obs_c).astype(int)
        obs_t_np = _to_numpy(obs_t).astype(int)

        C, T, N = n_cohorts, n_periods, len(y_np)

        # ── Sum-to-zero basis ─────────────────────────────────────────────────
        Q_drop_c, _ = _sum_to_zero_basis(C)   # [C, C-1]
        Q_drop_t, _ = _sum_to_zero_basis(T)   # [T, T-1]

        # ── Incidence matrices ────────────────────────────────────────────────
        A = np.zeros((N, C))
        B = np.zeros((N, T))
        A[np.arange(N), obs_c_np] = 1.0
        B[np.arange(N), obs_t_np] = 1.0

        # ── Constrained design matrix  H̃ = [A Q_drop_c | B Q_drop_t] ─────────
        AQ = A @ Q_drop_c   # [N, C-1]
        BQ = B @ Q_drop_t   # [N, T-1]
        H_tilde = np.concatenate([AQ, BQ], axis=1)   # [N, C-1+T-1]

        # Optionally prepend intercept column
        if self.use_global_mean:
            X = np.column_stack([np.ones(N), H_tilde])   # [N, 1+C-1+T-1]
        else:
            X = H_tilde                                   # [N, C-1+T-1]

        # ── OLS ───────────────────────────────────────────────────────────────
        theta, _, _, _ = np.linalg.lstsq(X, y_np, rcond=None)
        y_hat  = X @ theta
        resid  = y_np - y_hat
        n_params = X.shape[1]
        sigma2 = float(np.sum(resid ** 2) / max(N - n_params, 1))

        # OLS covariance  Cov(θ̂) = σ² (XᵀX)⁻¹.
        # In placebo/train splits, the design can be rank-deficient (e.g., some
        # cohort levels absent before cutoff). Fall back to pseudo-inverse so
        # uncertainty remains computable instead of failing with singular XtX.
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            ridge = 1e-8 * np.eye(XtX.shape[0], dtype=float)
            XtX_inv = np.linalg.pinv(XtX + ridge, rcond=1e-10)
        theta_cov = sigma2 * XtX_inv

        # ── Extract effects and their covariance blocks ───────────────────────
        dc = C - 1   # dimension of reduced cohort space
        dt = T - 1   # dimension of reduced time space

        if self.use_global_mean:
            # theta = [μ  |  z_c (C-1)  |  z_t (T-1)]
            mu    = float(theta[0])
            z_c   = theta[1 : 1 + dc]
            z_t   = theta[1 + dc : 1 + dc + dt]
            cov_zc = theta_cov[1 : 1+dc,     1 : 1+dc]
            cov_zt = theta_cov[1+dc : 1+dc+dt, 1+dc : 1+dc+dt]
        else:
            # theta = [z_c (C-1)  |  z_t (T-1)]
            mu   = 0.0
            z_c  = theta[:dc]
            z_t  = theta[dc:]
            cov_zc = theta_cov[:dc, :dc]
            cov_zt = theta_cov[dc:, dc:]

        # ── Back-transform to full effect space ───────────────────────────────
        alpha = Q_drop_c @ z_c   # [C],  Σ α_c = 0  by construction
        beta  = Q_drop_t @ z_t   # [T],  Σ β_t = 0  by construction

        # Delta method:  Cov(α̂) = Q_drop_c · Cov(ẑ_c) · Q_drop_cᵀ
        cov_alpha = Q_drop_c @ cov_zc @ Q_drop_c.T    # [C, C]
        cov_beta  = Q_drop_t @ cov_zt @ Q_drop_t.T    # [T, T]
        std_alpha = np.sqrt(np.maximum(np.diag(cov_alpha), 0.0))
        std_beta  = np.sqrt(np.maximum(np.diag(cov_beta),  0.0))

        # ── Store ─────────────────────────────────────────────────────────────
        self.mu_        = mu
        self.alpha_     = alpha
        self.beta_      = beta
        self.std_alpha_ = std_alpha
        self.std_beta_  = std_beta
        self.sigma2_    = sigma2
        self.y_hat_     = y_hat
        self.resid_     = resid
        self._Q_drop_c  = Q_drop_c
        self._Q_drop_t  = Q_drop_t
        self._n_cohorts = C
        self._n_periods = T
        self.seed_      = seed
        self.true_hyperparams_ = (
            dict(true_hyperparams) if true_hyperparams is not None else None
        )
        self._is_fitted = True

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, obs_c: "array-like", obs_t: "array-like") -> np.ndarray:
        """
        Compute fitted values  μ + α_{c(i)} + β_{t(i)}  for arbitrary index pairs.

        obs_t indices must lie within the range used during fit() (0-based,
        0 ≤ t < n_periods_train).  For held-out periods use beta_extrap_mean_.

        Parameters
        ----------
        obs_c, obs_t : array-like [N]  Index arrays (0-based).

        Returns
        -------
        y_hat : ndarray [N]
        """
        self._check_fitted()
        obs_c_np = _to_numpy(obs_c).astype(int)
        obs_t_np = _to_numpy(obs_t).astype(int)
        return self.mu_ + self.alpha_[obs_c_np] + self.beta_[obs_t_np]

    # ── extrapolate ───────────────────────────────────────────────────────────

    def extrapolate(self, n_extrap: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        AR(1) forecast of β for n_extrap periods beyond the training range.

        Must be called after fit() on TRAINING data only.

        AR(1) model (fit on estimated β̂_{0:T_tr-1}):
            β_t = c_ar + φ · β_{t-1} + ε_ar,    ε_ar ~ N(0, σ²_ar)

        Forecast variance accumulates step-by-step:
            Var(β̂_{T_tr+h}) = σ²_ar + φ² · Var(β̂_{T_tr+h-1})
                             = σ²_ar · Σ_{j=0}^{h-1} φ^{2j}

        Parameters
        ----------
        n_extrap : int
            Number of periods to forecast beyond the last training period.
            Corresponds to T - T_train  (e.g. 5 when T=20, T_train=15).

        Returns
        -------
        beta_extrap_mean : ndarray [n_extrap]   Point forecasts.
        beta_extrap_std  : ndarray [n_extrap]   Forecast standard deviations.
        """
        self._check_fitted()
        beta_tr = self.beta_                # [T_train]
        T_tr    = len(beta_tr)

        if T_tr < 3:
            raise ValueError(
                f"Need at least 3 training periods to fit AR(1); got {T_tr}."
            )

        # ── Fit AR(1) on training β̂ ──────────────────────────────────────────
        beta_lag  = beta_tr[:-1]                              # [T_tr-1]
        beta_lead = beta_tr[1:]                               # [T_tr-1]
        X_ar      = np.column_stack([np.ones(T_tr - 1), beta_lag])
        ar_coef, _, _, _ = np.linalg.lstsq(X_ar, beta_lead, rcond=None)
        c_ar, phi_ar = float(ar_coef[0]), float(ar_coef[1])
        ar_resid   = beta_lead - X_ar @ ar_coef
        sigma2_ar  = float(np.sum(ar_resid ** 2) / max(T_tr - 3, 1))
        # denominator: T_tr-1 obs, 2 AR params fitted → df = T_tr-3

        # ── Step-ahead forecast ───────────────────────────────────────────────
        beta_extrap_mean = np.zeros(n_extrap)
        beta_extrap_var  = np.zeros(n_extrap)
        last_b   = beta_tr[-1]
        cum_var  = 0.0
        for h in range(n_extrap):
            beta_extrap_mean[h] = c_ar + phi_ar * last_b
            cum_var = sigma2_ar + (phi_ar ** 2) * cum_var
            beta_extrap_var[h]  = cum_var
            last_b = beta_extrap_mean[h]

        beta_extrap_std = np.sqrt(beta_extrap_var)

        # ── Store ─────────────────────────────────────────────────────────────
        self.beta_extrap_mean_ = beta_extrap_mean
        self.beta_extrap_std_  = beta_extrap_std
        self.ar_phi_           = phi_ar
        self.ar_c_             = c_ar
        self.ar_sigma2_        = sigma2_ar

        return beta_extrap_mean, beta_extrap_std

    # ── results_dict ──────────────────────────────────────────────────────────

    def results_dict(self) -> dict:
        """
        Return fitted parameters as a standardised dict.

        The schema is shared across all three models (FE+AR, GP-CP, GP-CP-Extended) so
        that experiment loops and plotting functions can treat them uniformly.

        Keys
        ----
        Always present (after fit()):
            model_name  : str          Model label ("FE+AR")
            use_global_mean : bool     Whether intercept μ is enabled
            index_base  : int          Index convention (0-based)
            seed        : Optional[int]
            true_hyperparams : Optional[dict]
            mu          : float        Global mean (0.0 if use_global_mean=False)
            alpha       : ndarray [C]  Cohort effects
            beta        : ndarray [T]  Time effects  (training range)
            std_alpha   : ndarray [C]  Standard errors for α
            std_beta    : ndarray [T]  Standard errors for β
            gamma       : ndarray [C, T]  Zeros — FE has no interaction term
            sigma2      : float        Residual variance
            y_hat       : ndarray [N]  Fitted values
            resid       : ndarray [N]  Residuals
            lo95_alpha  : ndarray [C]  Lower 95% CI bound  (mean − 1.96·std)
            hi95_alpha  : ndarray [C]  Upper 95% CI bound
            lo95_beta   : ndarray [T]
            hi95_beta   : ndarray [T]

        Present after extrapolate():
            beta_extrap_mean : ndarray [n_extrap]
            beta_extrap_std  : ndarray [n_extrap]
            ar_phi           : float
            ar_c             : float
            ar_sigma2        : float
        """
        self._check_fitted()
        C = self._n_cohorts
        T = self._n_periods

        d: dict = dict(
            model_name="FE+AR",
            use_global_mean=self.use_global_mean,
            index_base=0,
            seed=self.seed_,
            true_hyperparams=self.true_hyperparams_,
            mu=self.mu_,
            alpha=self.alpha_,
            beta=self.beta_,
            std_alpha=self.std_alpha_,
            std_beta=self.std_beta_,
            gamma=np.zeros((C, T)),          # FE has no interaction term
            sigma2=self.sigma2_,
            y_hat=self.y_hat_,
            resid=self.resid_,
            lo95_alpha=self.alpha_ - 1.96 * self.std_alpha_,
            hi95_alpha=self.alpha_ + 1.96 * self.std_alpha_,
            lo95_beta=self.beta_ - 1.96 * self.std_beta_,
            hi95_beta=self.beta_ + 1.96 * self.std_beta_,
        )

        if self.beta_extrap_mean_ is not None:
            d.update(
                beta_extrap_mean=self.beta_extrap_mean_,
                beta_extrap_std=self.beta_extrap_std_,
                ar_phi=self.ar_phi_,
                ar_c=self.ar_c_,
                ar_sigma2=self.ar_sigma2_,
            )
        return d

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """
        Print a concise summary of the fitted model to stdout.

        Shows residual variance, effect ranges, and (if available) AR(1)
        parameters and n_extrap forecast periods.
        """
        self._check_fitted()
        mu_str = f"  μ (global mean) : {self.mu_:.4f}\n" if self.use_global_mean else ""
        print(
            f"\n{'─'*55}\n"
            f"  FixedEffectsModel  "
            f"({'with' if self.use_global_mean else 'without'} global mean)\n"
            f"{'─'*55}\n"
            f"  Grid            : C={self._n_cohorts}, T={self._n_periods}\n"
            f"{mu_str}"
            f"  σ̂² (residual)   : {self.sigma2_:.5f}\n"
            f"  α range         : [{self.alpha_.min():.3f}, {self.alpha_.max():.3f}]"
            f"   Σα = {self.alpha_.sum():.2e}\n"
            f"  β range         : [{self.beta_.min():.3f},  {self.beta_.max():.3f}]"
            f"   Σβ = {self.beta_.sum():.2e}\n"
        )
        if self.beta_extrap_mean_ is not None:
            n_e = len(self.beta_extrap_mean_)
            print(
                f"  AR(1) φ         : {self.ar_phi_:.4f}\n"
                f"  AR(1) c         : {self.ar_c_:.4f}\n"
                f"  AR(1) σ²_ar     : {self.ar_sigma2_:.5f}\n"
                f"  Extrap periods  : {n_e}"
                f"  β̂ ∈ [{self.beta_extrap_mean_.min():.3f},"
                f" {self.beta_extrap_mean_.max():.3f}]\n"
            )
        print(f"{'─'*55}\n")

    # ── private ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not fitted yet.  Call fit() before using this method."
            )
