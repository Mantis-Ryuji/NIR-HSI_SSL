from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Sphere geometry (S^{D-1}) helpers
# =============================================================================
def _safe_acos(x: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    r"""
    数値安定化付き arccos。

    Notes
    -----
    これは「外れ値を潰す」ためのスコアクリップではなく、acos の定義域を守るための
    数値安定化です（NaN 回避）。
    """
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def sphere_log(mu: torch.Tensor, z: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Log 写像：S^{D-1} → T_μ S^{D-1}。

    Parameters
    ----------
    mu : torch.Tensor, shape (D,)
        基準点 μ（内部で正規化）。
    z : torch.Tensor, shape (..., D)
        球面上の点（内部で正規化）。
    eps : float, default=1e-8
        ゼロ割回避。

    Returns
    -------
    v : torch.Tensor, shape (..., D)
        接空間ベクトル。理想的には v ⟂ μ。
    """
    mu = F.normalize(mu, dim=0)
    z = F.normalize(z, dim=-1)

    cos = (z * mu).sum(dim=-1)  # (...)
    theta = _safe_acos(cos, eps=1e-7)  # (...)
    # u = (z - cos*mu) / sin(theta)
    sin = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0))
    u = z - cos[..., None] * mu
    u = u / torch.clamp(sin[..., None], min=eps)
    return theta[..., None] * u


def sphere_exp(mu: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Exp 写像：T_μ S^{D-1} → S^{D-1}。

    Parameters
    ----------
    mu : torch.Tensor, shape (D,)
        基準点 μ（内部で正規化）。
    v : torch.Tensor, shape (..., D)
        接空間ベクトル。
    eps : float, default=1e-8
        ゼロ割回避。

    Returns
    -------
    z : torch.Tensor, shape (..., D)
        球面上の点（正規化済み）。
    """
    mu = F.normalize(mu, dim=0)
    nv = torch.linalg.norm(v, dim=-1)  # (...)
    nv_safe = torch.clamp(nv, min=eps)

    a = torch.cos(nv)[..., None] * mu
    b = torch.sin(nv)[..., None] * (v / nv_safe[..., None])
    z = a + b
    return F.normalize(z, dim=-1)


def frechet_mean_sphere(
    z: torch.Tensor,
    *,
    iters: int = 10,
    step: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    球面 S^{D-1} 上の Fréchet 平均（近似）を計算。

    Parameters
    ----------
    z : torch.Tensor, shape (N, D)
        点群（内部で正規化）。
    iters : int, default=10
        反復回数。0 の場合は `normalize(mean(z))` を返す。
    step : float, default=1.0
        更新係数（大きいほど速いが不安定化しうる）。
    eps : float, default=1e-8
        Log/Exp の安定化定数。

    Returns
    -------
    mu : torch.Tensor, shape (D,)
        Fréchet mean（近似）。
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (N,D), got {tuple(z.shape)}")
    z = F.normalize(z, dim=1)
    mu = F.normalize(z.mean(dim=0), dim=0)
    if iters <= 0:
        return mu
    for _ in range(iters):
        v = sphere_log(mu, z, eps=eps)
        mu = sphere_exp(mu, step * v.mean(dim=0), eps=eps)
    return mu


# =============================================================================
# Tangent PCA (1D)
# =============================================================================
@dataclass
class TangentPCA1D:
    r"""
    球面上の Tangent PCA により 1D スコアを得る。

    目的
    ----
    L2 正規化されたデータ z_i ∈ S^{D-1} に対し、
    Fréchet 平均 μ を基準点として接空間 T_μ S^{D-1} に Log 写像し、
    その接空間内で 1D PCA（主成分）を求めてスカラー t を返す。

    手順（学習）
    -----------
    1. Fréchet mean μ を推定
    2. v_i = Log_μ(z_i) を計算（v_i ∈ T_μ S^{D-1}）
    3. v を（数値誤差を考慮して）平均中心化して共分散を作り、最大固有ベクトル u を得る
    4. u を接空間に再射影し L2 正規化して保持

    スコア（推論）
    -------------
    t(z) = < Log_μ(z), u >

    既定の原点は μ です（Log_μ(μ)=0 なので t(μ)=0）。

    Parameters
    ----------
    eps_acos : float, default=1e-7
        acos 安定化。
    mean_iters : int, default=10
        Fréchet mean 反復回数。
    mean_step : float, default=1.0
        Fréchet mean 更新係数。
    init_eps : float, default=1e-8
        Log/Exp 安定化。

    Attributes
    ----------
    `mu_` : torch.Tensor, shape (D,)
        学習データの Fréchet mean（基準点）。
    `u_` : torch.Tensor, shape (D,)
        接空間の主方向（単位ベクトル、理想的には u_ ⟂ mu_）。
    `v_mean_` : torch.Tensor, shape (D,)
        学習データの接空間ベクトル v の平均（デバッグ用）。
    `is_fitted_` : bool
        学習済みフラグ。

    Notes
    -----
    - **符号の不定性**：
      主方向 u は u ↦ -u の自由度を持つため、スコアも t ↦ -t が同値です。
      進行度（温度・時間など）に対して単調増加の向きを揃えたい場合は、
      `corr(t, y)` が正になるように **必要なら u_ を反転**してください。
    - inverse_transform は「接空間の 1D 直線」を Exp で球面に戻す近似であり、
      大きい |t| では球面上で周回（折り返し）得ます（一般に |t|≳π で解釈が難しくなります）。

    Examples
    --------
    >>> model = TangentPCA1D(mean_iters=20)
    >>> model.fit(z_train)
    >>> t = model.transform(z_test)
    >>> z_recon = model.inverse_transform(t)
    """

    eps_acos: float = 1e-7
    mean_iters: int = 10
    mean_step: float = 1.0
    init_eps: float = 1e-8

    mu_: Optional[torch.Tensor] = None
    u_: Optional[torch.Tensor] = None
    v_mean_: Optional[torch.Tensor] = None
    is_fitted_: bool = False

    def _check_z(self, z: torch.Tensor) -> torch.Tensor:
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor")
        if z.ndim != 2:
            raise ValueError(f"z must be 2D (N,D), got {tuple(z.shape)}")
        return F.normalize(z, dim=1)

    def _check_fitted(self) -> None:
        if (not self.is_fitted_) or (self.mu_ is None) or (self.u_ is None):
            raise RuntimeError("Call fit() first.")

    @torch.no_grad()
    def fit(self, z: torch.Tensor) -> "TangentPCA1D":
        z = self._check_z(z)
        mu = frechet_mean_sphere(z, iters=self.mean_iters, step=self.mean_step, eps=self.init_eps)
        v = sphere_log(mu, z, eps=self.init_eps)

        v_mean = v.mean(dim=0)
        v_center = v - v_mean
        # Covariance in tangent space
        C = (v_center.T @ v_center) / max(1, v_center.shape[0])

        # Largest eigenvector
        evals, evecs = torch.linalg.eigh(C)
        u = evecs[:, -1]

        # Re-project to tangent space at mu (for numerical safety)
        u = u - (u @ mu) * mu
        u = u / torch.clamp(torch.linalg.norm(u), min=1e-12)

        self.mu_ = mu.detach()
        self.u_ = u.detach()
        self.v_mean_ = v_mean.detach()
        self.is_fitted_ = True
        return self

    def transform(self, z: torch.Tensor, *, return_v: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        self._check_fitted()
        z = self._check_z(z)

        mu = self.mu_.to(device=z.device, dtype=z.dtype)
        u = self.u_.to(device=z.device, dtype=z.dtype)

        v = sphere_log(mu, z, eps=self.init_eps)
        t = v @ u  # (N,)

        if return_v:
            return t, v
        return t

    def inverse_transform(self, t: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor")
        if t.ndim != 1:
            raise ValueError(f"t must be 1D (N,), got {tuple(t.shape)}")

        mu = self.mu_.to(device=t.device, dtype=t.dtype)
        u = self.u_.to(device=t.device, dtype=t.dtype)

        v = t[:, None] * u[None, :]
        z = sphere_exp(mu, v, eps=self.init_eps)
        return z

    @torch.no_grad()
    def flip_sign_(self) -> "TangentPCA1D":
        r"""
        主方向の符号を反転する（u_ ← -u_）。

        Notes
        -----
        u_ の符号は任意なので、外部の指標（温度・時間など）と単調性を揃える用途で使います。
        """
        self._check_fitted()
        self.u_.mul_(-1)
        return self