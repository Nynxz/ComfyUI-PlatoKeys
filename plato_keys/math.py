"""Closed-form cross-encoder alignment math (Plato's Key & regularized linear maps).

Every solver here operates directly on exact float64 **sufficient statistics**:
  - `weight_sum` : sum of token weights W = sum_i w_i
  - `sum_x`      : [d_x] = sum_i w_i x_i
  - `sum_y`      : [d_y] = sum_i w_i y_i
  - `sum_xx`     : [d_x, d_x] = sum_i w_i x_i^T x_i
  - `sum_xy`     : [d_x, d_y] = sum_i w_i x_i^T y_i
  - `sum_yy`     : [d_y, d_y] = sum_i w_i y_i^T y_i

Why sufficient statistics instead of stacking an `[N, D]` matrix?
  1. Constant memory (~224 MB for 2048 <-> 4096 in float64) whether you calibrate on 5,000 tokens
     or 5,000,000 tokens — fits easily on free Colab GPUs.
  2. Exact float64 precision (zero catastrophic cancellation when centering).
  3. One single calibration pass yields BOTH directions (`X -> Y` and `Y -> X`) and lets you fit
     and compare Procrustes, variance-matched Procrustes, PCA-Procrustes, and Ridge across an
     entire regularization grid in seconds on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .format import PlatoKey, PlatoKeyMeta

_EPS = 1e-12


@dataclass
class SufficientStats:
    """Streaming float64 second-order statistics for paired embeddings `(X, Y)`."""

    d_x: int
    d_y: int
    weight_sum: float = 0.0
    token_count: int = 0
    prompt_count: int = 0
    sum_x: torch.Tensor = field(init=False)
    sum_y: torch.Tensor = field(init=False)
    sum_xx: torch.Tensor = field(init=False)
    sum_xy: torch.Tensor = field(init=False)
    sum_yy: torch.Tensor = field(init=False)
    sum_x_norm_sq: float = 0.0
    sum_y_norm_sq: float = 0.0

    def __post_init__(self) -> None:
        self.sum_x = torch.zeros(self.d_x, dtype=torch.float64)
        self.sum_y = torch.zeros(self.d_y, dtype=torch.float64)
        self.sum_xx = torch.zeros((self.d_x, self.d_x), dtype=torch.float64)
        self.sum_xy = torch.zeros((self.d_x, self.d_y), dtype=torch.float64)
        self.sum_yy = torch.zeros((self.d_y, self.d_y), dtype=torch.float64)

    @torch.no_grad()
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor | None = None,
        num_prompts: int = 0,
    ) -> None:
        """Accumulate a batch of paired token vectors `x: [N, d_x]`, `y: [N, d_y]`."""
        x64 = x.detach().to(device="cpu", dtype=torch.float64)
        y64 = y.detach().to(device="cpu", dtype=torch.float64)
        if x64.ndim != 2 or y64.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got {tuple(x64.shape)} and {tuple(y64.shape)}")
        if x64.shape[0] != y64.shape[0]:
            raise ValueError(f"Row mismatch: x has {x64.shape[0]}, y has {y64.shape[0]}")
        if x64.shape[1] != self.d_x or y64.shape[1] != self.d_y:
            raise ValueError(
                f"Dim mismatch: expected ({self.d_x}, {self.d_y}), got ({x64.shape[1]}, {y64.shape[1]})"
            )
        if x64.shape[0] == 0:
            return

        if weights is None:
            self.weight_sum += float(x64.shape[0])
            self.sum_x.add_(x64.sum(dim=0))
            self.sum_y.add_(y64.sum(dim=0))
            self.sum_xx.addmm_(x64.T, x64)
            self.sum_xy.addmm_(x64.T, y64)
            self.sum_yy.addmm_(y64.T, y64)
            self.sum_x_norm_sq += float((x64 * x64).sum().item())
            self.sum_y_norm_sq += float((y64 * y64).sum().item())
        else:
            w = weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if w.shape[0] != x64.shape[0]:
                raise ValueError(f"Weight length {w.shape[0]} != batch rows {x64.shape[0]}")
            wx = x64 * w.unsqueeze(1)
            wy = y64 * w.unsqueeze(1)
            self.weight_sum += float(w.sum().item())
            self.sum_x.add_(wx.sum(dim=0))
            self.sum_y.add_(wy.sum(dim=0))
            self.sum_xx.addmm_(wx.T, x64)
            self.sum_xy.addmm_(wx.T, y64)
            self.sum_yy.addmm_(wy.T, y64)
            self.sum_x_norm_sq += float((wx * x64).sum().item())
            self.sum_y_norm_sq += float((wy * y64).sum().item())

        self.token_count += int(x64.shape[0])
        self.prompt_count += int(num_prompts)

    def swap(self) -> SufficientStats:
        """Return a view with X and Y swapped, enabling reverse-direction fitting for free."""
        rev = SufficientStats(self.d_y, self.d_x)
        rev.weight_sum = self.weight_sum
        rev.token_count = self.token_count
        rev.prompt_count = self.prompt_count
        rev.sum_x = self.sum_y.clone()
        rev.sum_y = self.sum_x.clone()
        rev.sum_xx = self.sum_yy.clone()
        rev.sum_xy = self.sum_xy.T.contiguous().clone()
        rev.sum_yy = self.sum_xx.clone()
        rev.sum_x_norm_sq = self.sum_y_norm_sq
        rev.sum_y_norm_sq = self.sum_x_norm_sq
        return rev

    def centered(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(mu_x, mu_y, M_xx, M_xy, M_yy)` where `M_xy = X_c^T W Y_c`."""
        if self.weight_sum <= 0.0:
            raise ValueError("Cannot compute centered statistics from empty SufficientStats.")
        W = self.weight_sum
        mu_x = self.sum_x / W
        mu_y = self.sum_y / W
        m_xx = self.sum_xx - W * torch.outer(mu_x, mu_x)
        m_xy = self.sum_xy - W * torch.outer(mu_x, mu_y)
        m_yy = self.sum_yy - W * torch.outer(mu_y, mu_y)
        # Enforce exact symmetry against tiny fp64 rounding asymmetry.
        m_xx = 0.5 * (m_xx + m_xx.T)
        m_yy = 0.5 * (m_yy + m_yy.T)
        return mu_x, mu_y, m_xx, m_xy, m_yy

    @property
    def source_token_rms(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0
        return math.sqrt(max(self.sum_x_norm_sq / (self.weight_sum * self.d_x), 0.0))

    @property
    def target_token_rms(self) -> float:
        if self.weight_sum <= 0.0:
            return 0.0
        return math.sqrt(max(self.sum_y_norm_sq / (self.weight_sum * self.d_y), 0.0))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "d_x": self.d_x,
                "d_y": self.d_y,
                "weight_sum": self.weight_sum,
                "token_count": self.token_count,
                "prompt_count": self.prompt_count,
                "sum_x": self.sum_x,
                "sum_y": self.sum_y,
                "sum_xx": self.sum_xx,
                "sum_xy": self.sum_xy,
                "sum_yy": self.sum_yy,
                "sum_x_norm_sq": self.sum_x_norm_sq,
                "sum_y_norm_sq": self.sum_y_norm_sq,
            },
            str(p),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> SufficientStats:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        stats = cls(int(data["d_x"]), int(data["d_y"]))
        stats.weight_sum = float(data["weight_sum"])
        stats.token_count = int(data["token_count"])
        stats.prompt_count = int(data["prompt_count"])
        stats.sum_x = data["sum_x"].to(torch.float64)
        stats.sum_y = data["sum_y"].to(torch.float64)
        stats.sum_xx = data["sum_xx"].to(torch.float64)
        stats.sum_xy = data["sum_xy"].to(torch.float64)
        stats.sum_yy = data["sum_yy"].to(torch.float64)
        stats.sum_x_norm_sq = float(data["sum_x_norm_sq"])
        stats.sum_y_norm_sq = float(data["sum_y_norm_sq"])
        return stats


@dataclass
class EvalMetrics:
    """Validation metrics on held-out paired token embeddings."""

    token_cosine: float  # mean cosine similarity across tokens
    centered_cosine: float  # cosine similarity after subtracting target mean mu_y (measures semantic direction!)
    rel_l2: float  # ||Y_hat - Y||_F / ||Y||_F
    r2: float  # 1 - ||Y_hat - Y||_F^2 / ||Y - mu_y||_F^2
    norm_ratio: float  # mean(||y_hat|| / ||y||)
    channel_std_corr: float  # Pearson correlation between per-channel std of Y_hat and Y

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}token_cosine": self.token_cosine,
            f"{p}centered_cosine": self.centered_cosine,
            f"{p}rel_l2": self.rel_l2,
            f"{p}r2": self.r2,
            f"{p}norm_ratio": self.norm_ratio,
            f"{p}channel_std_corr": self.channel_std_corr,
        }


@dataclass
class FitResult:
    """Fitted PlatoKey paired with its validation metrics (if evaluated)."""

    name: str
    key: PlatoKey
    metrics: EvalMetrics | None = None


def _base_meta(
    stats: SufficientStats,
    meta: PlatoKeyMeta | None,
    *,
    method: str,
    scale: float,
    ridge_alpha: float | None = None,
    pca_rank: int | None = None,
) -> PlatoKeyMeta:
    base = PlatoKeyMeta(**vars(meta)) if meta is not None else PlatoKeyMeta()
    base.method = method
    base.d_source = stats.d_x
    base.d_target = stats.d_y
    base.scale = float(scale)
    base.ridge_alpha = ridge_alpha
    base.pca_rank = pca_rank
    base.num_prompts = stats.prompt_count or base.num_prompts
    base.num_tokens = stats.token_count or base.num_tokens
    base.metrics = dict(base.metrics)
    return base


def fit_procrustes(
    stats: SufficientStats,
    *,
    scale_mode: str = "ls",
    meta: PlatoKeyMeta | None = None,
) -> PlatoKey:
    """Closed-form Semi-Orthogonal Procrustes alignment (Plato's Key).

    Computes the thin SVD of the centered cross-covariance `M_xy = U @ diag(S) @ V^T` and forms
    the semi-orthogonal rotation `R = U @ V^T` of shape `[d_x, d_y]`:
      - When `d_x <= d_y` (e.g. 2048 -> 4096): `R @ R^T = I_{d_x}` (exact isometric embedding into
        the optimal `d_x`-dimensional subspace of target space, with zero wasted zero-padding).
      - When `d_x > d_y` (e.g. 4096 -> 2048): `R^T @ R = I_{d_y}` (optimal semi-orthogonal projection).

    `scale_mode`:
      - `"ls"` (default, exact Plato's Key): least-squares optimal global scalar
        `s = sum(S) / tr(R^T M_xx R)`.
      - `"variance"`: global scalar matching total target variance
        `s = sqrt(tr(M_yy) / tr(R^T M_xx R))`. Prevents shrinkage toward `mu_y` when mapping
        from a smaller space into a larger DiT conditioning space.
      - `"unit"`: pure rotation `s = 1.0`.
    """
    mu_x, mu_y, m_xx, m_xy, m_yy = stats.centered()
    u, s_vals, vt = torch.linalg.svd(m_xy, full_matrices=False)
    # R: [d_x, d_y]
    r_mat = u @ vt

    # Projected source variance: tr(R^T M_xx R) = tr(U^T M_xx U)
    # (Equals tr(M_xx) when d_x <= d_y; strictly smaller when d_x > d_y.)
    proj_energy = float(torch.sum(u * (m_xx @ u)).clamp_min(_EPS).item())
    target_energy = float(torch.trace(m_yy).clamp_min(_EPS).item())

    if scale_mode == "ls":
        scale = float(s_vals.sum().item()) / proj_energy
        method = "procrustes"
    elif scale_mode == "variance":
        scale = math.sqrt(target_energy / proj_energy)
        method = "procrustes_norm"
    elif scale_mode == "unit":
        scale = 1.0
        method = "procrustes_unit"
    else:
        raise ValueError(f"Unknown scale_mode {scale_mode!r}; expected 'ls', 'variance', or 'unit'")

    # PyTorch F.linear expects weight of shape [d_target, d_source]
    weight = (scale * r_mat).T.contiguous().to(torch.float32)
    key_meta = _base_meta(stats, meta, method=method, scale=scale)
    return PlatoKey(
        weight=weight,
        mu_x=mu_x.to(torch.float32),
        mu_y=mu_y.to(torch.float32),
        meta=key_meta,
        scale=scale,
        target_token_rms=stats.target_token_rms,
        source_token_rms=stats.source_token_rms,
    )


def fit_procrustes_pca(
    stats: SufficientStats,
    rank: int = 256,
    *,
    scale_mode: str = "ls",
    meta: PlatoKeyMeta | None = None,
) -> PlatoKey:
    """Rank-`r` PCA-projected Procrustes (aligns the top-`r` principal subspaces of X and Y).

    Directly implements the low-dimensional compression finding from Plato's Key (where ~256 PCs
    captured most of the transferable geometric signal). Useful for ablation or ultra-regularized
    cross-family alignment.
    """
    max_rank = min(stats.d_x, stats.d_y)
    r = max(1, min(int(rank), max_rank))
    mu_x, mu_y, m_xx, m_xy, m_yy = stats.centered()

    # Top-r eigenvectors of M_xx and M_yy (eigh returns ascending order)
    _, q_x_all = torch.linalg.eigh(m_xx)
    _, q_y_all = torch.linalg.eigh(m_yy)
    q_x = q_x_all[:, -r:]  # [d_x, r]
    q_y = q_y_all[:, -r:]  # [d_y, r]

    # Cross-covariance inside the r-dimensional PC subspaces
    core = q_x.T @ m_xy @ q_y  # [r, r]
    u, s_vals, vt = torch.linalg.svd(core, full_matrices=False)
    r_mat = (q_x @ (u @ vt)) @ q_y.T  # [d_x, d_y], rank r

    proj_energy = float(torch.sum(q_x * (m_xx @ q_x)).clamp_min(_EPS).item())
    target_energy = float(torch.trace(m_yy).clamp_min(_EPS).item())

    if scale_mode == "ls":
        scale = float(s_vals.sum().item()) / proj_energy
    elif scale_mode == "variance":
        scale = math.sqrt(target_energy / proj_energy)
    elif scale_mode == "unit":
        scale = 1.0
    else:
        raise ValueError(f"Unknown scale_mode {scale_mode!r}")

    weight = (scale * r_mat).T.contiguous().to(torch.float32)
    key_meta = _base_meta(stats, meta, method="procrustes_pca", scale=scale, pca_rank=r)
    return PlatoKey(
        weight=weight,
        mu_x=mu_x.to(torch.float32),
        mu_y=mu_y.to(torch.float32),
        meta=key_meta,
        scale=scale,
        target_token_rms=stats.target_token_rms,
        source_token_rms=stats.source_token_rms,
    )


def fit_ridge(
    stats: SufficientStats,
    *,
    alpha: float | None = None,
    alphas: Iterable[float] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0),
    val_x: torch.Tensor | None = None,
    val_y: torch.Tensor | None = None,
    match_variance: bool = False,
    meta: PlatoKeyMeta | None = None,
) -> PlatoKey:
    """Closed-form Ridge regression `A = (M_xx + lambda * I)^-1 M_xy`.

    `alpha` is scale-invariant: `lambda = alpha * mean_eigenvalue(M_xx)`, so the same alpha grid
    works regardless of dataset size or model hidden norm.

    When `alpha is None` and `(val_x, val_y)` are provided, eigendecomposes `M_xx` ONCE and sweeps
    `alphas` in milliseconds to pick the regularization maximizing held-out centered cosine
    similarity (tie-broken by lower relative L2 error).

    When `match_variance=True`, rescales the Ridge matrix by a single global scalar so total
    mapped variance matches `tr(M_yy)` (preventing shrinkage toward `mu_y`).
    """
    mu_x, mu_y, m_xx, m_xy, m_yy = stats.centered()

    evals, evecs = torch.linalg.eigh(m_xx)
    evals = evals.clamp_min(0.0)
    mean_eig = float(evals.mean().clamp_min(_EPS).item())
    # Precompute Q^T M_xy once so solving for any alpha is O(d_x^2 * d_y) via one matmul.
    qt_mxy = evecs.T @ m_xy  # [d_x, d_y]

    def _solve_for(a_val: float) -> torch.Tensor:
        lam = a_val * mean_eig
        inv = 1.0 / (evals + lam)
        return evecs @ (inv.unsqueeze(1) * qt_mxy)  # [d_x, d_y]

    if alpha is not None:
        best_alpha = float(alpha)
        a_mat = _solve_for(best_alpha)
    elif val_x is not None and val_y is not None and val_x.shape[0] > 0:
        vx_c = val_x.detach().to(device="cpu", dtype=torch.float64) - mu_x
        vy = val_y.detach().to(device="cpu", dtype=torch.float64)
        vy_c = vy - mu_y
        vy_c_norm = vy_c.norm(dim=-1).clamp_min(_EPS)

        best_score = -1e9
        best_alpha = 1e-2
        best_a = None
        for cand in alphas:
            cand_a = _solve_for(float(cand))
            pred_c = vx_c @ cand_a
            pred_c_norm = pred_c.norm(dim=-1).clamp_min(_EPS)
            centered_cos = float(((pred_c * vy_c).sum(dim=-1) / (pred_c_norm * vy_c_norm)).mean().item())
            # Mild penalty for large L2 mismatch so we don't pick wildly ill-conditioned alpha.
            mse_ratio = float(((pred_c - vy_c).pow(2).sum() / vy_c.pow(2).sum().clamp_min(_EPS)).item())
            score = centered_cos - 0.1 * mse_ratio
            if score > best_score:
                best_score = score
                best_alpha = float(cand)
                best_a = cand_a
        assert best_a is not None
        a_mat = best_a
    else:
        best_alpha = 1e-2
        a_mat = _solve_for(best_alpha)

    scale = 1.0
    if match_variance:
        # Mapped variance: tr(A^T M_xx A) = sum((M_xx @ A) * A)
        pred_energy = float(torch.sum((m_xx @ a_mat) * a_mat).clamp_min(_EPS).item())
        target_energy = float(torch.trace(m_yy).clamp_min(_EPS).item())
        scale = math.sqrt(target_energy / pred_energy)
        a_mat = a_mat * scale

    method = "ridge_norm" if match_variance else "ridge"
    weight = a_mat.T.contiguous().to(torch.float32)
    key_meta = _base_meta(stats, meta, method=method, scale=scale, ridge_alpha=best_alpha)
    return PlatoKey(
        weight=weight,
        mu_x=mu_x.to(torch.float32),
        mu_y=mu_y.to(torch.float32),
        meta=key_meta,
        scale=scale,
        target_token_rms=stats.target_token_rms,
        source_token_rms=stats.source_token_rms,
    )


@torch.no_grad()
def evaluate_key(key: PlatoKey, val_x: torch.Tensor, val_y: torch.Tensor) -> EvalMetrics:
    """Evaluate a fitted PlatoKey on held-out paired vectors `(val_x, val_y)` of shape `[N, D]`."""
    x = val_x.detach().to(device="cpu", dtype=torch.float32)
    y = val_y.detach().to(device="cpu", dtype=torch.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("evaluate_key expects 2D [N, D] tensors")

    mu_x = key.mu_x.to(torch.float32)
    mu_y = key.mu_y.to(torch.float32)
    w = key.weight.to(torch.float32)

    y_hat = torch.nn.functional.linear(x - mu_x, w) + mu_y

    # Raw token cosine similarity
    x_norm_hat = y_hat.norm(dim=-1).clamp_min(_EPS)
    y_norm = y.norm(dim=-1).clamp_min(_EPS)
    token_cos = float(((y_hat * y).sum(dim=-1) / (x_norm_hat * y_norm)).mean().item())

    # Mean-centered cosine similarity (critical for LLMs whose tokens share a large constant mean!)
    yh_c = y_hat - mu_y
    y_c = y - mu_y
    centered_cos = float(
        ((yh_c * y_c).sum(dim=-1) / (yh_c.norm(dim=-1).clamp_min(_EPS) * y_c.norm(dim=-1).clamp_min(_EPS)))
        .mean()
        .item()
    )

    diff_sq = float((y_hat - y).pow(2).sum().item())
    y_sq = float(y.pow(2).sum().clamp_min(_EPS).item())
    yc_sq = float(y_c.pow(2).sum().clamp_min(_EPS).item())
    rel_l2 = math.sqrt(diff_sq / y_sq)
    r2 = 1.0 - (diff_sq / yc_sq)

    norm_ratio = float((x_norm_hat / y_norm).mean().item())

    # Per-channel std correlation (checks whether outlier channels in Qwen3.5 are faithfully reproduced)
    std_hat = y_hat.std(dim=0, unbiased=False)
    std_ref = y.std(dim=0, unbiased=False)
    sh_c = std_hat - std_hat.mean()
    sr_c = std_ref - std_ref.mean()
    denom = float((sh_c.norm() * sr_c.norm()).clamp_min(_EPS).item())
    channel_std_corr = float(((sh_c * sr_c).sum() / denom).item())

    return EvalMetrics(
        token_cosine=token_cos,
        centered_cosine=centered_cos,
        rel_l2=rel_l2,
        r2=r2,
        norm_ratio=norm_ratio,
        channel_std_corr=channel_std_corr,
    )


def fit_all_methods(
    stats: SufficientStats,
    *,
    val_x: torch.Tensor | None = None,
    val_y: torch.Tensor | None = None,
    meta: PlatoKeyMeta | None = None,
    pca_rank: int | None = None,
) -> list[FitResult]:
    """Fit Procrustes, variance-matched Procrustes, Ridge, variance-matched Ridge (and optional PCA-Procrustes).

    Evaluates each on `(val_x, val_y)` when provided and stores the metrics right in `key.meta.metrics`.
    """
    candidates: list[tuple[str, PlatoKey]] = [
        ("procrustes", fit_procrustes(stats, scale_mode="ls", meta=meta)),
        ("procrustes_norm", fit_procrustes(stats, scale_mode="variance", meta=meta)),
        ("ridge", fit_ridge(stats, val_x=val_x, val_y=val_y, match_variance=False, meta=meta)),
        ("ridge_norm", fit_ridge(stats, val_x=val_x, val_y=val_y, match_variance=True, meta=meta)),
    ]
    if pca_rank is not None and pca_rank > 0:
        candidates.append(
            (f"procrustes_pca{pca_rank}", fit_procrustes_pca(stats, rank=pca_rank, scale_mode="ls", meta=meta))
        )

    results: list[FitResult] = []
    for name, key in candidates:
        metrics = None
        if val_x is not None and val_y is not None and val_x.shape[0] > 0:
            metrics = evaluate_key(key, val_x, val_y)
            key.meta.metrics.update(metrics.to_dict())
        results.append(FitResult(name=name, key=key, metrics=metrics))
    return results


@torch.no_grad()
def compose_keys(first: PlatoKey, second: PlatoKey) -> PlatoKey:
    """Analytically compose two PlatoKeys `first: A -> Hub` and `second: Hub -> B` into `A -> B`.

    Given:
      h = mu_h1 + W1 (x - mu_x)
      y = mu_y  + W2 (h - mu_h2)
    their exact composition is a single affine map with zero runtime overhead:
      y = (mu_y + W2 (mu_h1 - mu_h2)) + (W2 W1) (x - mu_x)

    Supports:
      - 2D o 2D: `[d_B, d_H] @ [d_H, d_A] -> [d_B, d_A]`
      - 3D o 3D: `[N, d_B, d_H] @ [N, d_H, d_A] -> [N, d_B, d_A]` (per-layer hub)
      - 2D o 3D (flattened): `[d_B, N*d_H]` composed with `[N, d_H, d_A]` -> `[d_B, N*d_A]`
    """
    w1 = first.weight.detach().to(device="cpu", dtype=torch.float64)
    mx1 = first.mu_x.detach().to(device="cpu", dtype=torch.float64)
    my1 = first.mu_y.detach().to(device="cpu", dtype=torch.float64)

    w2 = second.weight.detach().to(device="cpu", dtype=torch.float64)
    mx2 = second.mu_x.detach().to(device="cpu", dtype=torch.float64)
    my2 = second.mu_y.detach().to(device="cpu", dtype=torch.float64)

    layer_scales: list[float] | None = None
    if w1.ndim == 2 and w2.ndim == 2:
        if w1.shape[0] != w2.shape[1]:
            raise ValueError(
                f"Cannot compose 2D PlatoKeys: first target dim {w1.shape[0]} != second source dim {w2.shape[1]}"
            )
        w_comp = w2 @ w1
        mx_comp = mx1
        my_comp = my2 + w2 @ (my1 - mx2)
    elif w1.ndim == 3 and w2.ndim == 3:
        if w1.shape[0] != w2.shape[0] or w1.shape[1] != w2.shape[2]:
            raise ValueError(
                f"Cannot compose 3D PlatoKeys: first target shape ({w1.shape[0]}, {w1.shape[1]}) "
                f"!= second source shape ({w2.shape[0]}, {w2.shape[2]})"
            )
        w_comp = torch.bmm(w2, w1)  # [N, d_B, d_A]
        mx_comp = mx1
        my_comp = my2 + torch.bmm(w2, (my1 - mx2).unsqueeze(-1)).squeeze(-1)
        s1 = first.layer_scales or [first.scale] * w1.shape[0]
        s2 = second.layer_scales or [second.scale] * w2.shape[0]
        layer_scales = [float(a * b) for a, b in zip(s1, s2, strict=True)]
    elif w1.ndim == 3 and w2.ndim == 2:
        n_layers, d_h, d_a = w1.shape
        if n_layers * d_h != w2.shape[1]:
            raise ValueError(
                f"Cannot compose 3D->2D PlatoKeys: flattened first target {n_layers}*{d_h}={n_layers * d_h} "
                f"!= second source dim {w2.shape[1]}"
            )
        # Apply each layer block of W2 ([d_B, N, d_H]) against W1 ([N, d_H, d_A]) -> [d_B, N*d_A]
        w2_blocks = w2.view(w2.shape[0], n_layers, d_h)
        w_comp = torch.einsum("bnh,nha->bna", w2_blocks, w1).reshape(w2.shape[0], n_layers * d_a)
        mx_comp = mx1.reshape(n_layers * d_a)
        my_comp = my2 + w2 @ (my1.reshape(n_layers * d_h) - mx2)
    else:
        raise ValueError(
            f"Unsupported PlatoKey composition shapes: first={tuple(w1.shape)}, second={tuple(w2.shape)}"
        )

    pooled_w = pooled_mx = pooled_my = None
    if (
        first.pooled_weight is not None
        and second.pooled_weight is not None
        and first.pooled_mu_x is not None
        and first.pooled_mu_y is not None
        and second.pooled_mu_x is not None
        and second.pooled_mu_y is not None
        and first.pooled_weight.shape[0] == second.pooled_weight.shape[1]
    ):
        pw1 = first.pooled_weight.detach().to(device="cpu", dtype=torch.float64)
        pmx1 = first.pooled_mu_x.detach().to(device="cpu", dtype=torch.float64)
        pmy1 = first.pooled_mu_y.detach().to(device="cpu", dtype=torch.float64)
        pw2 = second.pooled_weight.detach().to(device="cpu", dtype=torch.float64)
        pmx2 = second.pooled_mu_x.detach().to(device="cpu", dtype=torch.float64)
        pmy2 = second.pooled_mu_y.detach().to(device="cpu", dtype=torch.float64)
        pooled_w = (pw2 @ pw1).to(torch.float32)
        pooled_mx = pmx1.to(torch.float32)
        pooled_my = (pmy2 + pw2 @ (pmy1 - pmx2)).to(torch.float32)

    hub_name = first.meta.target_model or second.meta.source_model or "hub"
    comp_meta = PlatoKeyMeta(
        source_model=first.meta.source_model,
        target_model=second.meta.target_model,
        target_preset=second.meta.target_preset or first.meta.target_preset,
        method=f"composed({first.meta.method}->{second.meta.method})",
        source_layer=first.meta.source_layer,
        target_layer=second.meta.target_layer,
        source_layer_norm=first.meta.source_layer_norm,
        target_layer_norm=second.meta.target_layer_norm,
        source_template=second.meta.source_template or first.meta.source_template,
        strip_prompt=second.meta.strip_prompt if second.meta.source_template else first.meta.strip_prompt,
        strip_prefix_mode=second.meta.strip_prefix_mode or first.meta.strip_prefix_mode,
        flatten_layers=second.meta.flatten_layers,
        scale=float(first.scale * second.scale),
        notes=f"Platonic Hub composition via {hub_name}",
    )

    return PlatoKey(
        weight=w_comp.to(torch.float32).contiguous(),
        mu_x=mx_comp.to(torch.float32).contiguous(),
        mu_y=my_comp.to(torch.float32).contiguous(),
        meta=comp_meta,
        scale=float(first.scale * second.scale),
        layer_scales=layer_scales,
        source_token_rms=first.source_token_rms,
        target_token_rms=second.target_token_rms,
        pooled_weight=pooled_w,
        pooled_mu_x=pooled_mx,
        pooled_mu_y=pooled_my,
    )
