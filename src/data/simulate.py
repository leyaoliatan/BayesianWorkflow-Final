"""
simulate.py
===========
Synthetic cohort-period data generator for BAPC simulation experiments.

Data-Generating Process
-----------------------
Scenario A  (interaction=False):
    y_{c,t,r} = α_c + β_t + ε_{c,t,r},    ε ~ N(0, sn²)

Scenario B  (interaction=True):
    y_{c,t,r} = α_c + β_t + γ_{c,t} + ε_{c,t,r},    ε ~ N(0, sn²)

True effects are drawn from zero-mean Gaussian processes with RBF kernels and
projected onto the sum-to-zero subspace for identifiability:

    Σ_c α_c = 0
    Σ_t β_t = 0
    Σ_c γ_{c,t} = 0  ∀t  and  Σ_t γ_{c,t} = 0  ∀c   (Scenario B only)

Observation design: full balanced grid — every (cohort, period) cell has
exactly n_reps independent draws.  Total N = n_cohorts × n_periods × n_reps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Use float64 throughout (matches notebook precision).
torch.set_default_dtype(torch.float64)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rbf_kernel(
    X1: torch.Tensor,
    X2: torch.Tensor,
    ell: float,
    sf: float,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Squared-exponential (RBF) kernel.

        k(i, j) = sf² · exp( −(i−j)² / (2·ell²) )

    Parameters
    ----------
    X1, X2  : index tensors of shape [n, 1]
    ell     : lengthscale
    sf      : signal standard deviation
    jitter  : small constant added to the diagonal for numerical stability
    """
    X1s = X1 / ell
    X2s = X2 / ell
    dist2 = (
        X1s.pow(2).sum(-1, keepdim=True)
        - 2.0 * (X1s @ X2s.T)
        + X2s.pow(2).sum(-1).unsqueeze(0)
    )
    K = sf ** 2 * torch.exp(-0.5 * dist2)
    if jitter > 0 and X1.shape[0] == X2.shape[0]:
        K = K + jitter * torch.eye(X1.shape[0], dtype=X1.dtype)
    return K


def _sum_to_zero(v: torch.Tensor) -> torch.Tensor:
    """Project a 1-D tensor onto the sum-to-zero subspace: v̄ ← v − mean(v)."""
    return v - v.mean()


def _double_centre(M: torch.Tensor) -> torch.Tensor:
    """
    Double-centre a 2-D tensor so that every row sum and every column sum is zero.

    Uses Tukey's formula:
        M̃_{c,t} = M_{c,t} − row_mean_c − col_mean_t + grand_mean

    This guarantees:
        Σ_c γ_{c,t} = 0  for all t   (column margins zero)
        Σ_t γ_{c,t} = 0  for all c   (row margins zero)
    """
    return (
        M
        - M.mean(dim=0, keepdim=True)   # subtract period (column) means
        - M.mean(dim=1, keepdim=True)   # subtract cohort (row) means
        + M.mean()                       # add grand mean back (avoids double subtraction)
    )


# ── Main simulation function ──────────────────────────────────────────────────

def simulate_cohort_data(
    # ── Grid dimensions ──────────────────────────────────────────────────────
    n_cohorts: int = 20,
    # Number of cohorts C.
    # ↑ more cohorts  → denser α recovery, larger dataset, more diverse cohort trajectories.
    # ↓ fewer cohorts → sparser α estimation; interaction surface less identifiable.

    n_periods: int = 20,
    # Number of time periods T.
    # ↑ more periods  → longer β trend, larger held-out window for extrapolation tasks.
    # ↓ fewer periods → shorter trend; extrapolation is a larger fraction of total span.

    n_reps: int = 3,
    # Within-cell replicates R — independent observations per (cohort, period) cell.
    # ↑ more reps  → lower per-cell noise, SNR improves, dataset scales as C×T×R.
    # ↓ fewer reps → noisier per-cell mean; minimum is 1 (no within-cell replication).

    # ── Cohort GP hyperparameters ─────────────────────────────────────────────
    ell_c: float = 2.0,
    # Cohort GP lengthscale.
    # Controls how rapidly cohort effects α_c vary across the cohort index.
    # ↑ larger ell_c → smooth, slowly-varying α; neighbouring cohorts are highly similar.
    # ↓ smaller ell_c → idiosyncratic α; cohorts vary independently (rough surface).

    sf_c: float = 1.0,
    # Cohort GP signal standard deviation (amplitude).
    # Controls the overall scale / spread of cohort-level intercepts.
    # ↑ larger sf_c → large cross-cohort heterogeneity (wide range of α values).
    # ↓ smaller sf_c → all cohorts cluster tightly around zero (low heterogeneity).

    # ── Time GP hyperparameters ───────────────────────────────────────────────
    ell_t: float = 3.0,
    # Time GP lengthscale.
    # Controls how rapidly the common trend β_t varies across periods.
    # ↑ larger ell_t → smooth, slowly-drifting trend (long business cycle).
    # ↓ smaller ell_t → rapidly oscillating β (seasonal or high-frequency pattern).

    sf_t: float = 1.0,
    # Time GP signal standard deviation (amplitude).
    # Controls the overall scale of the common time effect.
    # ↑ larger sf_t → strong common trend dominates cross-cohort differences.
    # ↓ smaller sf_t → weak time effect; cohort effects dominate the outcome.

    # ── Interaction GP hyperparameters  [Scenario B / interaction=True only] ──
    ell_gc: float = 1.5,
    # Interaction cohort lengthscale.  [ignored when interaction=False]
    # Controls smoothness of γ_{c,t} along the cohort axis.
    # ↑ larger ell_gc → all cohorts deviate from parallel trends in a similar, correlated way.
    # ↓ smaller ell_gc → each cohort has a distinct, independent PT-violation pattern.

    sf_gc: float = 0.5,
    # Interaction cohort GP signal SD.  [ignored when interaction=False]
    # Together with sf_gt, sets the scale of the interaction surface.
    # ↑ larger sf_gc → more severe PT violations, harder for FE/GP-CP to recover α and β.
    # ↓ smaller sf_gc → mild interaction; misspecification bias is smaller.

    ell_gt: float = 1.5,
    # Interaction time lengthscale.  [ignored when interaction=False]
    # Controls smoothness of γ_{c,t} along the time axis.
    # ↑ larger ell_gt → PT violations evolve gradually (slow cohort divergence).
    # ↓ smaller ell_gt → PT violations appear/disappear rapidly (spiky interaction).

    sf_gt: float = 0.5,
    # Interaction time GP signal SD.  [ignored when interaction=False]
    # See sf_gc for combined amplitude interpretation.
    # ↑ larger sf_gt → stronger time-dimension variation in the interaction surface.

    # ── Noise ─────────────────────────────────────────────────────────────────
    sn: float = 0.3,
    # Observation noise standard deviation.
    # SNR ≈ (sf_c² + sf_t²) / sn²
    # ↑ larger sn → noisier observations, lower SNR, harder estimation, lower coverage.
    # ↓ smaller sn → near-deterministic observations; all models recover truth easily.

    # ── Experiment controls ───────────────────────────────────────────────────
    interaction: bool = False,
    # Whether to include the cohort×period interaction γ in the DGP.
    # False → Scenario A (PT-True):  y = α + β + ε
    # True  → Scenario B (PT-False): y = α + β + γ + ε
    # Interaction hyperparameters (ell_gc, sf_gc, ell_gt, sf_gt) are ignored when False.

    seed: Optional[int] = None,
    # Random seed for full reproducibility (controls GP draws AND observation noise).
    # None → fresh random draw on every call.
    # Set to an integer (e.g. seed=0) to get identical data across runs.

) -> dict:
    """
    Generate synthetic cohort-period panel data from a GP-based DGP.

    Returns a dict — see 'Returns' section below for the complete key listing.

    Parameters  (see inline comments on the function signature above for full detail)
    ----------
    n_cohorts, n_periods, n_reps : grid dimensions
    ell_c, sf_c   : cohort GP hyperparameters
    ell_t, sf_t   : time GP hyperparameters
    ell_gc, sf_gc : interaction cohort GP hyperparameters  (Scenario B only)
    ell_gt, sf_gt : interaction time  GP hyperparameters  (Scenario B only)
    sn            : observation noise SD
    interaction   : False = PT-True (Scenario A), True = PT-False (Scenario B)
    seed          : integer seed or None

    Returns
    -------
    dict with keys:

    Observations
        y          : Tensor [N]        noisy observations
        obs_c      : Tensor [N] int    cohort index per observation  (0-based)
        obs_t      : Tensor [N] int    period index per observation  (0-based)

    True effects
        alpha_true : Tensor [C]        true cohort effects      (sum-to-zero)
        beta_true  : Tensor [T]        true time effects        (sum-to-zero)
        gamma_true : Tensor [C, T]     true interaction matrix  (double-centred;
                                       all-zero when interaction=False)

    Kernel matrices
        K_c_true   : Tensor [C, C]     cohort RBF kernel at true hyperparameters
        K_t_true   : Tensor [T, T]     time   RBF kernel at true hyperparameters

    Design matrices
        A          : Tensor [N, C]     cohort incidence matrix  (one-hot rows)
        B          : Tensor [N, T]     time   incidence matrix  (one-hot rows)

    Metadata
        n_cohorts, n_periods, n_reps : int  grid dimensions
        sn_true    : float             noise SD used
        interaction: bool              DGP scenario flag
        seed       : int or None       seed used
        hyperparams: dict              all GP hyperparameters (for logging / plotting)
    """
    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    C, T = n_cohorts, n_periods

    # ── Index tensors (1-based, shape [K, 1] for kernel computation) ──────────
    cohort_idx = torch.arange(1, C + 1, dtype=torch.float64).unsqueeze(-1)  # [C, 1]
    time_idx   = torch.arange(1, T + 1, dtype=torch.float64).unsqueeze(-1)  # [T, 1]

    # ── RBF kernel matrices ────────────────────────────────────────────────────
    K_c = _rbf_kernel(cohort_idx, cohort_idx, ell_c, sf_c)   # [C, C]
    K_t = _rbf_kernel(time_idx,   time_idx,   ell_t, sf_t)   # [T, T]

    # ── Draw and centre cohort effects α ~ GP(0, K_c) ─────────────────────────
    alpha_raw  = torch.distributions.MultivariateNormal(
        torch.zeros(C, dtype=torch.float64), K_c
    ).sample()
    alpha_true = _sum_to_zero(alpha_raw)   # Σ_c α_c = 0  [C]

    # ── Draw and centre time effects β ~ GP(0, K_t) ───────────────────────────
    beta_raw  = torch.distributions.MultivariateNormal(
        torch.zeros(T, dtype=torch.float64), K_t
    ).sample()
    beta_true = _sum_to_zero(beta_raw)     # Σ_t β_t = 0  [T]

    # ── Draw and double-centre interaction γ (Scenario B only) ────────────────
    if interaction:
        K_gc = _rbf_kernel(cohort_idx, cohort_idx, ell_gc, sf_gc)   # [C, C]
        K_gt = _rbf_kernel(time_idx,   time_idx,   ell_gt, sf_gt)   # [T, T]
        # Kronecker product: covariance across all (cohort, period) pairs
        K_g  = torch.kron(K_gc, K_gt)                                # [C*T, C*T]
        gamma_vec_raw = torch.distributions.MultivariateNormal(
            torch.zeros(C * T, dtype=torch.float64), K_g
        ).sample()
        gamma_mat_raw = gamma_vec_raw.reshape(C, T)
        gamma_true    = _double_centre(gamma_mat_raw)                 # [C, T]
    else:
        # Scenario A: interaction is identically zero
        gamma_true = torch.zeros(C, T, dtype=torch.float64)          # [C, T]

    # ── Full-grid balanced observation design ──────────────────────────────────
    # Each cell (c, t) appears exactly n_reps times.
    # Ordering: for each rep, iterate over all cohorts and all periods.
    c_grid   = np.repeat(np.arange(C), T)        # [C*T]: 0,..,0, 1,..,1, ..., C-1,..,C-1
    t_grid   = np.tile(np.arange(T), C)          # [C*T]: 0,1,..,T-1, 0,1,..,T-1, ...
    obs_c_np = np.tile(c_grid, n_reps)            # [N]: n_reps full copies
    obs_t_np = np.tile(t_grid, n_reps)            # [N]

    N     = len(obs_c_np)                         # = C * T * n_reps
    obs_c = torch.tensor(obs_c_np, dtype=torch.long)
    obs_t = torch.tensor(obs_t_np, dtype=torch.long)

    # ── Incidence matrices ─────────────────────────────────────────────────────
    A = torch.zeros(N, C, dtype=torch.float64)    # A[i, c] = 1 iff obs i belongs to cohort c
    B = torch.zeros(N, T, dtype=torch.float64)    # B[i, t] = 1 iff obs i belongs to period t
    A[torch.arange(N), obs_c] = 1.0
    B[torch.arange(N), obs_t] = 1.0

    # ── Generate noisy observations ────────────────────────────────────────────
    f_true = (
        alpha_true[obs_c]             # cohort effect for each observation
        + beta_true[obs_t]            # time effect for each observation
        + gamma_true[obs_c, obs_t]    # interaction (zero matrix if Scenario A)
    )
    eps = sn * torch.randn(N, dtype=torch.float64)
    y   = f_true + eps                # [N]

    # ── Package and return ─────────────────────────────────────────────────────
    hyperparams = dict(
        ell_c=ell_c,   sf_c=sf_c,
        ell_t=ell_t,   sf_t=sf_t,
        ell_gc=ell_gc, sf_gc=sf_gc,
        ell_gt=ell_gt, sf_gt=sf_gt,
        sn=sn,
    )

    return dict(
        # observations
        y=y,
        obs_c=obs_c,
        obs_t=obs_t,
        # true effects
        alpha_true=alpha_true,
        beta_true=beta_true,
        gamma_true=gamma_true,
        # kernel matrices
        K_c_true=K_c,
        K_t_true=K_t,
        # design matrices
        A=A,
        B=B,
        # metadata
        n_cohorts=n_cohorts,
        n_periods=n_periods,
        n_reps=n_reps,
        sn_true=sn,
        interaction=interaction,
        seed=seed,
        hyperparams=hyperparams,
    )


def simulate_cohort_data2(
    n_cohorts: int = 20,
    n_periods: int = 20,
    n_reps: int = 3,
    # Simple smooth sinusoid controls
    sf_c: float = 1.0,
    sf_t: float = 1.0,
    freq_t: float = 1.2,
    # Interaction controls (for Scenario D)
    sf_g: float = 1.5,
    slope_g: float = 1.2,
    sn: float = 0.3,
    interaction: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Synthetic cohort-period data with simple sinusoidal alpha/beta effects.

    Scenario C (interaction=False):
        y_{c,t,r} = alpha_c + beta_t + eps

    Scenario D (interaction=True):
        y_{c,t,r} = alpha_c + beta_t + gamma_{c,t} + eps

    where alpha/beta/gamma are deterministic smooth functions (no GP draws),
    and gamma introduces substantial non-parallel trends via cohort-specific slopes.

    Return schema matches ``simulate_cohort_data``.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    C, T = int(n_cohorts), int(n_periods)
    if C < 2 or T < 2:
        raise ValueError("n_cohorts and n_periods must both be >= 2.")

    # Normalized indices in [0, 1]
    u = torch.linspace(0.0, 1.0, C, dtype=torch.float64)
    v = torch.linspace(0.0, 1.0, T, dtype=torch.float64)

    # Minimal seed-reproducible randomization.
    # Same seed => same DGP; different seeds => different smooth curves.
    phase_c = float(np.random.uniform(0.0, 2.0 * np.pi))
    phase_t = float(np.random.uniform(0.0, 2.0 * np.pi))
    h2_c = float(np.random.uniform(0.20, 0.40))
    h2_t = float(np.random.uniform(0.20, 0.40))
    freq_t_eff = max(0.4, float(freq_t) * float(np.random.uniform(0.85, 1.20)))
    slope_jitter = float(np.random.uniform(0.85, 1.15))

    # Cohort effect: simple sinusoid + harmonic; then sum-to-zero.
    alpha_raw = sf_c * (
        torch.sin(2.0 * np.pi * u + phase_c)
        + h2_c * torch.cos(4.0 * np.pi * u + phase_c)
    )
    alpha_true = _sum_to_zero(alpha_raw)

    # Period effect: same functional style as alpha (parallel design philosophy).
    beta_raw = sf_t * (
        torch.sin(2.0 * np.pi * freq_t_eff * v + phase_t)
        + h2_t * torch.cos(4.0 * np.pi * freq_t_eff * v + phase_t)
    )
    beta_true = _sum_to_zero(beta_raw)

    # Interaction for non-parallel trends:
    # pure cohort-specific linear slopes over time (mixed signs).
    if interaction:
        cohort_loading = torch.linspace(-1.0, 1.0, C, dtype=torch.float64)
        t_center = v - 0.5
        gamma_raw = sf_g * slope_g * slope_jitter * torch.outer(cohort_loading, t_center)
        gamma_true = _double_centre(gamma_raw)
    else:
        gamma_true = torch.zeros(C, T, dtype=torch.float64)

    # Full balanced grid design
    c_grid = np.repeat(np.arange(C), T)
    t_grid = np.tile(np.arange(T), C)
    obs_c_np = np.tile(c_grid, n_reps)
    obs_t_np = np.tile(t_grid, n_reps)
    N = len(obs_c_np)

    obs_c = torch.tensor(obs_c_np, dtype=torch.long)
    obs_t = torch.tensor(obs_t_np, dtype=torch.long)

    A = torch.zeros(N, C, dtype=torch.float64)
    B = torch.zeros(N, T, dtype=torch.float64)
    A[torch.arange(N), obs_c] = 1.0
    B[torch.arange(N), obs_t] = 1.0

    f_true = alpha_true[obs_c] + beta_true[obs_t] + gamma_true[obs_c, obs_t]
    y = f_true + sn * torch.randn(N, dtype=torch.float64)

    # Keep compatibility keys used by notebooks/plots.
    hyperparams = dict(
        ell_c=np.nan,
        sf_c=sf_c,
        ell_t=np.nan,
        sf_t=sf_t,
        freq_t=freq_t,
        freq_t_eff=freq_t_eff,
        phase_c=phase_c,
        phase_t=phase_t,
        h2_c=h2_c,
        h2_t=h2_t,
        sf_g=sf_g,
        slope_g=slope_g,
        slope_jitter=slope_jitter,
        ell_gc=np.nan,
        sf_gc=np.nan,
        ell_gt=np.nan,
        sf_gt=np.nan,
        sn=sn,
        base_dgp="simple_sinusoid",
    )

    return dict(
        y=y,
        obs_c=obs_c,
        obs_t=obs_t,
        alpha_true=alpha_true,
        beta_true=beta_true,
        gamma_true=gamma_true,
        K_c_true=None,
        K_t_true=None,
        A=A,
        B=B,
        n_cohorts=n_cohorts,
        n_periods=n_periods,
        n_reps=n_reps,
        sn_true=sn,
        interaction=interaction,
        seed=seed,
        hyperparams=hyperparams,
    )
