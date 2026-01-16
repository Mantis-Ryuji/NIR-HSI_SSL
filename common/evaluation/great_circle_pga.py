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

    Parameters
    ----------
    x : torch.Tensor
        arccos の入力。通常は内積やノルム（本実装では r=||U^T x||）。
    eps : float, default=1e-7
        浮動小数誤差で [-1,1] を僅かに外れるのを防ぐための clamp 余裕。

    Returns
    -------
    theta : torch.Tensor
        `acos(clamp(x, -1+eps, 1-eps))`。

    Notes
    -----
    これは「外れ値を潰す」ためのスコアクリップではなく、acos の定義域を守るための
    数値安定化です（NaN 回避）。
    """
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))

def _wrap_pi(t: torch.Tensor) -> torch.Tensor:
    r"""Wrap angles to (-π, π].

    Parameters
    ----------
    t : torch.Tensor
        角度（ラジアン）。

    Returns
    -------
    t_wrapped : torch.Tensor
        (-π, π] に折り返した角度。
    """
    two_pi = 2.0 * torch.pi
    return torch.remainder(t + torch.pi, two_pi) - torch.pi



def sphere_log(mu: torch.Tensor, z: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    r"""
    Log 写像：S^{D-1} → T_μ S^{D-1}。

    Parameters
    ----------
    mu : torch.Tensor, shape (D,)
        基準点 μ（単位ベクトルが望ましいが内部で正規化）。
    z : torch.Tensor, shape (..., D)
        球面上の点（内部で最後次元を正規化）。
    eps : float, default=1e-8
        ゼロ割回避。

    Returns
    -------
    v : torch.Tensor, shape (..., D)
        μ における接空間ベクトル（Log_μ(z)）。

    Notes
    -----
    c = <μ,z>, θ = arccos(c), u = z - cμ, v = (θ/||u||)u。
    """
    mu = F.normalize(mu, dim=0)
    z = F.normalize(z, dim=-1)

    c = (z * mu).sum(dim=-1, keepdim=True)  # (..., 1)
    theta = _safe_acos(c)                   # (..., 1)

    u = z - c * mu                          # (..., D)
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    v = (theta / u_norm) * u

    # θ→0 では v→0
    v = torch.where(theta < 1e-6, torch.zeros_like(v), v)
    return v


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
        Exp_μ(v)（最後次元を正規化して返す）。

    Notes
    -----
    θ = ||v||, z = cos(θ)μ + sin(θ) v/θ。
    """
    mu = F.normalize(mu, dim=0)
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    v_dir = v / theta.clamp_min(eps)

    z = torch.cos(theta) * mu + torch.sin(theta) * v_dir
    z = torch.where(theta < 1e-6, mu.expand_as(z), z)
    return F.normalize(z, dim=-1)


@torch.no_grad()
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
        更新係数。
    eps : float, default=1e-8
        Log/Exp の安定化。

    Returns
    -------
    mu : torch.Tensor, shape (D,)
        Fréchet 平均の近似（単位ベクトル）。

    Notes
    -----
    Riemannian 勾配法（簡易）：
      μ ← Exp_μ(step * mean(Log_μ(z_i)) )
    """
    z = F.normalize(z, dim=1)
    mu = F.normalize(z.mean(dim=0), dim=0)
    if iters <= 0:
        return mu
    for _ in range(iters):
        v = sphere_log(mu, z, eps=eps)
        mu = sphere_exp(mu, step * v.mean(dim=0), eps=eps)
    return mu


# =============================================================================
# Great-circle PGA (strict): optimize a great circle directly on Stiefel(D,2)
# =============================================================================
def _orthonormalize_2col(U: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Gram-Schmidt: U (D,2) -> columns orthonormal.
    """
    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError(f"U must be (D,2), got {tuple(U.shape)}")
    a = U[:, 0]
    b = U[:, 1]
    a = a / (a.norm() + eps)
    b = b - a * (a @ b)
    b = b / (b.norm() + eps)
    return torch.stack([a, b], dim=1)


@torch.no_grad()
def _tangent_pca_init(
    z: torch.Tensor,
    *,
    center_in_tangent: bool = True,
    mean_iters: int = 20,
    mean_step: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    厳密最適化の初期値として、(μ, p) を tangent PCA から作る（インメモリ版）。

    Parameters
    ----------
    z : torch.Tensor, shape (N, D)
        単位ベクトル群（内部で正規化）。
    center_in_tangent : bool, default=True
        接空間で平均 0 に中心化してから共分散を推定する。
    mean_iters : int, default=20
        Fréchet mean 反復回数。
    mean_step : float, default=1.0
        Fréchet mean 更新係数。
    eps : float, default=1e-8
        Log/Exp 安定化。

    Returns
    -------
    mu : torch.Tensor, shape (D,)
        球面平均。
    p : torch.Tensor, shape (D,)
        μ に直交する接空間主方向（単位ベクトル）。

    Notes
    -----
    以前の実装にあったチャンク処理（streaming）は削除した。
    N が極端に大きい場合はメモリ使用量が増える点に注意。
    """
    z = F.normalize(z, dim=1)
    mu = frechet_mean_sphere(z, iters=mean_iters, step=mean_step, eps=eps)

    v = sphere_log(mu, z, eps=eps)  # (N, D)
    if center_in_tangent:
        v = v - v.mean(dim=0, keepdim=True)

    # covariance in tangent space
    C = (v.transpose(0, 1) @ v) / max(v.shape[0], 1)  # (D, D)

    evals, evecs = torch.linalg.eigh(C)
    p = evecs[:, -1]

    # ensure orthogonal to mu (numerical)
    p = p - mu * (mu @ p)
    p = F.normalize(p, dim=0)
    return mu, p


@dataclass
class GreatCirclePGA1D:
    r"""
    球面上の「主測地線（大円）」を **直接** 最適化して 1D 角度座標を返す（厳密版）。

    目的
    ----
    L2 正規化されたデータ x_i ∈ S^{D-1} に対し、
    大円（great circle）への球面距離二乗の平均を最小化する大円を学習し、
    各点をその大円へ球面距離で直交射影したときの角度座標 t を返す。

    モデル（大円の表現）
    -------------------
    U = [a, b] ∈ Stiefel(D,2)（列が正規直交）を学習する。
    大円は
      g(t) = cos(t) a + sin(t) b
    と表せる。

    射影（閉形式）
    -------------
    任意の x に対し
      r(x) = ||U^T x|| = sqrt((x·a)^2 + (x·b)^2) ∈ [0, 1]
      δ(x) = arccos(r(x))    # 大円への測地距離
      t_raw(x) = atan2(x·b, x·a)                  # 生の角度（-π, π]
      t0       = atan2((mu·b), (mu·a))            # mu: Fréchet mean の射影角
      t(x)     = wrap_to_pi(t_raw(x) - t0)        # 既定では mu の射影が 0
    で与えられる。

    学習目的関数
    -----------
      minimize_U  mean_i δ(x_i)^2 = mean_i arccos(||U^T x_i||)^2

    最適化（pymanopt）
    -----------------
    Stiefel(D,2) 上の多様体最適化として解く。
    本実装は `pymanopt` の **PyTorch backend** を使って自動微分を流す。

    座標の原点と符号について
    ----------------------
    - 本実装では、学習データの球面 Fréchet 平均 μ を大円へ射影した点を **t=0** に固定する
    （位相シフトの不定性を除去）。
    - ただし大円自体は同一でも、基底の取り方（例: b→-b）で **t→-t** が起こり得る。
    物理的な進行度（温度・時間など）に対して符号を揃えたい場合は、
    `corr(t, y)` が正になるように必要なら `t *= -1`（同時に b を反転）するのを推奨。

    Parameters
    ----------
    `eps_acos` : float, default=1e-7
        acos 安定化。
    `init_center_in_tangent` : bool, default=True
        初期値生成（tangent PCA init）で接空間中心化をするか。
    `init_mean_iters` : int, default=20
        初期値の Fréchet mean 反復回数。
    `init_mean_step` : float, default=1.0
        初期値の Fréchet mean 更新係数。
    `init_eps` : float, default=1e-8
        初期値の Log/Exp 安定化。

    Attributes
    ----------
    `U_` : torch.Tensor, shape (D, 2)
        学習済みの大円基底。
    `t0_` : torch.Tensor
        既定の原点（`t=0`）に対応する位相。学習データの Fréchet 平均を大円へ射影した点の角度。
    `is_fitted_` : bool
        学習済みフラグ。

    Examples
    --------
    >>> model = GreatCirclePGA1D()
    >>> model.fit(z_train, max_iterations=200)
    >>> t, z_hat, delta = model.transform(z_test, return_z_hat=True, return_delta=True)
    >>> z_recon = model.inverse_transform(t)
    """

    eps_acos: float = 1e-7

    init_center_in_tangent: bool = True
    init_mean_iters: int = 20
    init_mean_step: float = 1.0
    init_eps: float = 1e-8

    U_: Optional[torch.Tensor] = None
    t0_: Optional[torch.Tensor] = None
    is_fitted_: bool = False

    @staticmethod
    def _check_z(z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor, shape (N, D)

        Returns
        -------
        z_norm : torch.Tensor, shape (N, D)
            L2 正規化済み。
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor")
        if z.ndim != 2:
            raise ValueError(f"z must be (N, D), got shape={tuple(z.shape)}")
        if z.shape[0] < 2:
            raise ValueError("z must contain at least 2 samples")
        return F.normalize(z, dim=1)

    def _check_fitted(self) -> None:
        if (not self.is_fitted_) or (self.U_ is None) or (self.t0_ is None):
            raise RuntimeError("Call fit() first.")

    @torch.no_grad()
    def _init_U(self, z: torch.Tensor) -> torch.Tensor:
        """
        tangent PCA init から U0=[mu, p] を作り、正規直交化して返す。
        """
        mu, p = _tangent_pca_init(
            z,
            center_in_tangent=self.init_center_in_tangent,
            mean_iters=self.init_mean_iters,
            mean_step=self.init_mean_step,
            eps=self.init_eps,
        )
        U0 = torch.stack([mu, p], dim=1)
        return _orthonormalize_2col(U0)

    def fit(
        self,
        z: torch.Tensor,
        *,
        max_iterations: int = 200,
        optimizer: str = "cg",
        verbosity: int = 0,
        U0: Optional[torch.Tensor] = None,
    ) -> "GreatCirclePGA1D":
        r"""
        pymanopt（PyTorch backend）で大円を学習する。

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            入力点群（内部で L2 正規化）。
        max_iterations : int, default=200
            最適化反復回数（pymanopt 側）。
        optimizer : {"cg", "sd"}, default="cg"
            使用する最適化器。
            - "cg": ConjugateGradient（一般に収束が速い）
            - "sd": SteepestDescent（より単純）
        verbosity : int, default=0
            pymanopt の表示レベル（0=静か）。
        U0 : torch.Tensor, shape (D,2), optional
            初期値。None の場合は tangent init を用いる。

        Returns
        -------
        self : GreatCirclePGA1D

        Notes
        -----
        - cost は full-batch です。
        - データが極端に巨大な場合、full-batch の 1 反復が重くなるため、
          その場合は「ミニバッチ近似」を別途実装する方が現実的ですが、
          本メソッドは “pymanopt を使った厳密モデル” を優先しています。
        """
        z = self._check_z(z)
        device, dtype = z.device, z.dtype
        N, D = z.shape

        if U0 is None:
            U0 = self._init_U(z).to(device=device, dtype=dtype)
        else:
            if (not isinstance(U0, torch.Tensor)) or U0.shape != (D, 2):
                raise ValueError(f"U0 must be torch.Tensor with shape (D,2)={(D,2)}, got {None if not isinstance(U0, torch.Tensor) else tuple(U0.shape)}")
            U0 = _orthonormalize_2col(U0.to(device=device, dtype=dtype))

        # --- pymanopt setup (PyTorch backend) ---
        try:
            import pymanopt
            from pymanopt.manifolds import Stiefel
            from pymanopt.optimizers import ConjugateGradient, SteepestDescent
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "pymanopt is required for GreatCirclePGA1D.fit(). "
                "Install it with: pip install pymanopt"
            ) from e

        manifold = Stiefel(D, 2)

        @pymanopt.function.pytorch(manifold)
        def cost(U: torch.Tensor) -> torch.Tensor:
            # mean_i acos(||U^T x_i||)^2
            # NOTE: z は closure で参照。z と U の device が異なる場合はここで転送が発生する。
            x = z
            if x.device != U.device:
                x = x.to(U.device)
            proj = x @ U                      # (N,2)
            s = (proj * proj).sum(dim=1)     # (N,)
            r = torch.sqrt(torch.clamp(s, 0.0, 1.0))
            delta = _safe_acos(r, eps=self.eps_acos)
            return (delta * delta).mean()

        problem = pymanopt.Problem(manifold=manifold, cost=cost)

        if optimizer.lower() == "cg":
            opt = ConjugateGradient(max_iterations=max_iterations, verbosity=verbosity)
        elif optimizer.lower() == "sd":
            opt = SteepestDescent(max_iterations=max_iterations, verbosity=verbosity)
        else:
            raise ValueError("optimizer must be one of {'cg','sd'}")

        result = opt.run(problem, initial_point=U0)

        # result.point is already on the manifold; still enforce clean orthonormality
        U_star = _orthonormalize_2col(result.point)

        # --- default origin: projection of Fréchet mean onto the learned great circle ---
        # t0 = atan2(<mu,b>, <mu,a>) where mu is the spherical Fréchet mean of training data.
        mu = frechet_mean_sphere(z, iters=self.init_mean_iters, step=self.init_mean_step, eps=self.init_eps)
        mu = mu.to(device=U_star.device, dtype=U_star.dtype)
        ab = mu @ U_star  # (2,)
        t0 = torch.atan2(ab[1], ab[0])

        self.U_ = U_star.detach()
        self.t0_ = t0.detach()
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def transform(
        self,
        z: torch.Tensor,
        *,
        return_z_hat: bool = False,
        return_delta: bool = False,
    ):
        r"""
        点を学習済み大円へ射影し、角度座標 t を返す。

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            入力点群（内部で正規化）。
        return_z_hat : bool, default=False
            True の場合、射影点 z_hat（大円上の点）も返す。
        return_delta : bool, default=False
            True の場合、大円への測地距離 δ も返す。

        Returns
        -------
        t : torch.Tensor, shape (N,)
            大円上の角度座標（-π, π]）。既定では学習データの Fréchet 平均の射影が 0 になるようシフトされる。
        z_hat : torch.Tensor, shape (N, D), optional
            大円上の射影点（unit-norm）。`return_z_hat=True` のときのみ。
        delta : torch.Tensor, shape (N,), optional
            大円への測地距離（角度）。`return_delta=True` のときのみ。

        Notes
        -----
        t = wrap_to_pi(atan2(x·b, x·a) - t0_), δ = acos(||U^T x||)。
        """
        self._check_fitted()
        z = self._check_z(z)

        U = self.U_.to(device=z.device, dtype=z.dtype)  # (D,2)
        proj = z @ U
        ca = proj[:, 0]
        sb = proj[:, 1]
        s = ca * ca + sb * sb
        r = torch.sqrt(torch.clamp(s, 0.0, 1.0))
        t0 = self.t0_.to(device=z.device, dtype=z.dtype)
        t = _wrap_pi(torch.atan2(sb, ca) - t0)

        outs = [t]

        if return_z_hat:
            denom = torch.sqrt(torch.clamp(s, 1e-12, 1.0))
            z_hat = (ca[:, None] * U[:, 0] + sb[:, None] * U[:, 1]) / denom[:, None]
            z_hat = F.normalize(z_hat, dim=1)
            outs.append(z_hat)

        if return_delta:
            delta = _safe_acos(r, eps=self.eps_acos)
            outs.append(delta)

        return outs[0] if len(outs) == 1 else tuple(outs)

    @torch.no_grad()
    def inverse_transform(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        角度座標 t を大円上の点 g(t) に写像する。

        Parameters
        ----------
        t : torch.Tensor, shape (N,)
            角度座標（ラジアン）。

        Returns
        -------
        z : torch.Tensor, shape (N, D)
            大円上の点 g(t) = cos(t)a + sin(t)b（unit-norm）。
        """
        self._check_fitted()
        U = self.U_
        t0 = self.t0_.to(device=U.device, dtype=U.dtype)
        t = t.to(device=U.device, dtype=U.dtype)
        t = _wrap_pi(t + t0)
        z = torch.cos(t)[:, None] * U[:, 0] + torch.sin(t)[:, None] * U[:, 1]
        return F.normalize(z, dim=1)