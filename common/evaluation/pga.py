from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


PathLike = Union[str, Path]


def _safe_acos(x: torch.Tensor) -> torch.Tensor:
    r"""
    数値安定化付き arccos を計算する。

    概要
    ----
    球面幾何（S^{D-1}）では内積 c = <a, b> を arccos に通して角度を得ることが多い。
    しかし浮動小数誤差により c が [-1, 1] をわずかに外れると NaN が発生する。
    本関数は入力を [-1+ε, 1-ε] に clamp してから arccos を計算する。

    Parameters
    ----------
    x : torch.Tensor
        arccos の入力（通常は内積）。形状は任意。

    Returns
    -------
    theta : torch.Tensor
        `arccos(clamp(x))` の結果。形状は入力と同じ。

    Notes
    -----
    ε は固定で 1e-7 を用いる。勾配の厳密性よりも NaN 回避を優先する設計。
    """
    return torch.acos(torch.clamp(x, -1.0 + 1e-7, 1.0 - 1e-7))


def sphere_log(mu: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    球面上の Log 写像（S^{D-1} → T_μ S^{D-1}）を計算する。

    概要
    ----
    単位球面 S^{D-1} 上の基準点 μ と点 z に対して、μ から z へ向かう測地線の
    「接ベクトル」v を返す。これは μ における接空間 T_μ S^{D-1} の元であり、
    Spherical PGA の前処理（球面→接空間）として用いる。

    数式
    ----
    c = <μ, z>, θ = arccos(c),
    u = z - c μ,  v = (θ / ||u||) u.

    θ → 0 では v → 0 となる（極限）。

    Parameters
    ----------
    mu : torch.Tensor of shape (D,)
        基準点 μ（単位ベクトル）。内部で L2 正規化される。
    z : torch.Tensor of shape (..., D)
        球面上の点。内部で最後次元に対して L2 正規化される。
    eps : float, default=1e-8
        ||u|| のゼロ割回避用。`clamp_min(eps)` に用いる。

    Returns
    -------
    v : torch.Tensor of shape (..., D)
        μ における接空間ベクトル（Log 写像の結果）。

    Notes
    -----
    - 実装では θ < 1e-6 の領域で v を 0 に置換している（数値安定化）。
    - 入力が厳密な単位ベクトルでなくても内部で正規化するため、多少の誤差は許容する。
    """
    mu = F.normalize(mu, dim=0)
    z = F.normalize(z, dim=-1)

    c = (z * mu).sum(dim=-1, keepdim=True)  # (..., 1)
    theta = _safe_acos(c)                   # (..., 1)

    u = z - c * mu                          # (..., D)
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(eps)
    v = (theta / u_norm) * u
    # theta -> 0 limit
    v = torch.where(theta < 1e-6, torch.zeros_like(v), v)
    return v


def sphere_exp(mu: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""
    球面上の Exp 写像（T_μ S^{D-1} → S^{D-1}）を計算する。

    概要
    ----
    μ における接空間ベクトル v を、球面上の点 z = Exp_μ(v) に写像する。
    Spherical PGA では「1D 座標 t から主測地線上の点 z_hat を復元」する際に用いる。

    数式
    ----
    θ = ||v||,  z = cos(θ) μ + sin(θ) (v / θ).

    θ → 0 では z → μ（極限）となる。

    Parameters
    ----------
    mu : torch.Tensor of shape (D,)
        基準点 μ（単位ベクトル）。内部で L2 正規化される。
    v : torch.Tensor of shape (..., D)
        μ における接空間ベクトル。
    eps : float, default=1e-8
        θ のゼロ割回避用。`clamp_min(eps)` に用いる。

    Returns
    -------
    z : torch.Tensor of shape (..., D)
        球面上の点（単位ベクトル）。最後次元で L2 正規化して返す。

    Notes
    -----
    - 実装では θ < 1e-6 の領域で z を μ に置換している（数値安定化）。
    - 返り値は常に normalize される。
    """
    mu = F.normalize(mu, dim=0)
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)  # (..., 1)
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
    単位球面 S^{D-1} 上の Fréchet 平均（近似）を計算する。

    概要
    ----
    球面上の点群 z_i に対して、Fréchet 関数
    ∑ d(μ, z_i)^2 を最小化する μ を反復的に近似する。
    実装は Log/Exp を用いた勾配法（Riemannian gradient descent）に相当する簡易更新。

    Parameters
    ----------
    z : torch.Tensor of shape (N, D)
        点群（単位ベクトル）。内部で L2 正規化する。
    iters : int, default=10
        反復回数。0 の場合は `normalize(mean(z))` を返す。
    step : float, default=1.0
        更新ステップ係数（接空間平均ベクトルに乗じる）。
    eps : float, default=1e-8
        Log/Exp の数値安定化用。

    Returns
    -------
    mu : torch.Tensor of shape (D,)
        Fréchet 平均の近似（単位ベクトル）。

    Notes
    -----
    - 分散が小さい場合、`normalize(mean(z))` とほぼ一致する。
    - iters を増やすほど厳密に近づくが、通常は 5〜20 程度で十分。
    """
    z = F.normalize(z, dim=1)
    mu = F.normalize(z.mean(dim=0), dim=0)
    if iters <= 0:
        return mu
    for _ in range(iters):
        v = sphere_log(mu, z, eps=eps)  # (N, D)
        mu = sphere_exp(mu, step * v.mean(dim=0), eps=eps)
    return mu


class SphericalPGA1D:
    r"""
    球面PGAによる1次元座標推定と、 1 次元スカラー `t` のクリップ付き0-1スコア化。

    概要
    ----
    L2 正規化された潜在表現 `z ∈ S^{D-1}` に対して、球面上の主測地線（principal geodesic）を推定し、
    各サンプルを 1 次元スカラー `t` に射影する。さらに、`t` の外れ値（裾）で生成が不安定になりやすい点を踏まえ、
    学習時に推定した分位点レンジで t をクリップした上で [0,1] に min-max 正規化し、スコア `u` を提供する。

    目的（設計意図）
    --------------
    - `t` は連続量（実数）であり、端ではデータ密度が薄くなるため、逆変換（生成）でスペクトルが暴れやすい。
    - そこで `t` を `[t_lo_, t_hi_]` にクリップし、`u∈[0,1]` のスコアへ変換する。
    - クリップされた領域（外れ値側）については **厳密な逆変換を要求しない**（= 0/1 に飽和させる）。

    数学的定義
    ---------
    1. 球面平均 μ を推定し、接空間へ写像：

       v_i = log_μ(z_i)

    2. 必要なら接空間で中心化（`center_in_tangent=True`）：

       v'_i = v_i - mean(v)

    3. 接空間で PCA（SVD）を行い第一主成分方向 p を得る：

       t_i = <v'_i, p>

    4. 学習データ上の `t_train` から分位点でクリップ範囲を推定：

       t_lo = Q_q(t_train),  t_hi = Q_{1-q}(t_train)

    5. スコア化（0-1）：

       t_clip = clamp(t, t_lo, t_hi)  
       u = (t_clip - t_lo) / (t_hi - t_lo)

       端の外れ値は 0 / 1 に飽和する。

    Parameters
    ----------
    - `center_in_tangent` : bool, default=True
        Log 写像で得た接空間ベクトルを平均 0 に中心化してから PCA を行う。
        True の場合、`v_mean_` を保存し、transform/inverse_transform でも同じ補正を適用する。
    - `mean_iters` : int, default=20
        Fréchet 平均 μ の推定反復回数。0 の場合は `normalize(mean(z))` を用いる。
    - `mean_step` : float, default=1.0
        Fréchet 平均の更新ステップ係数。
    - `eps` : float, default=1e-8
        Log/Exp の数値安定化用。
    - `clip_quantile` : float, default=0.01
        クリップ範囲を決める分位点 q。
        `t_lo = Q_q`, `t_hi = Q_{1-q}` を用いる（例：0.01 は上下 1% を切る）。
        0 の場合はクリップなし（`t_lo=t_min, t_hi=t_max`）。
    - `range_floor` : float, default=1e-12
        `t_hi - t_lo` が極小になった場合の安全策用しきい値。
        分位点レンジが潰れた場合は `[t_min, t_max]` にフォールバックする。
        それでも潰れる場合は退避値として [0,1] を用いる。

    Attributes
    ----------
    - `mu_` : torch.Tensor of shape (D,)
        球面平均 μ（単位ベクトル）。
    - `p_` : torch.Tensor of shape (D,)
        接空間での主方向 p（単位ベクトル）。数値誤差対策として μ と直交化してから正規化する。
    - `v_mean_` : torch.Tensor of shape (D,)
        接空間ベクトルの平均（`center_in_tangent=True` の場合）。
    - `t_min_` : float
        学習データ上の t の最小値（診断用）。
    - `t_max_` : float
        学習データ上の t の最大値（診断用）。
    - `t_lo_` : float
        クリップ下限（分位点）。
    - `t_hi_` : float
        クリップ上限（分位点）。
    - `explained_var_ratio_` : float
        接空間 PCA における第 1 成分の分散説明率。
    - `is_fitted_` : bool
        学習済みフラグ。

    Notes
    -----
    - `normalize=True` のスコア `u` は **[0,1] の相対指標**であり、距離尺度ではない。
      0/1 は「分位点レンジ外へ出た（外れ値側）」の飽和を意味する。
    - `denormalize_t()` および `inverse_transform(normalized=True)` は
      `t ∈ [t_lo_, t_hi_]` の代表値へ戻すだけであり、外れ値側の厳密な逆写像は行わない設計。

    Examples
    --------
    >>> pga = SphericalPGA1D(clip_quantile=0.01)
    >>> pga.fit(z_train)  # z_train: (N, D), unit-norm preferred
    >>> u = pga.transform(z_test, normalize=True)  # u in [0,1]
    >>> z_hat = pga.inverse_transform(u, normalized=True)  # saturated inside [t_lo_, t_hi_]
    """

    def __init__(
        self,
        *,
        center_in_tangent: bool = True,
        mean_iters: int = 20,
        mean_step: float = 1.0,
        eps: float = 1e-8,
        clip_quantile: float = 0.01,
        range_floor: float = 1e-12,
    ) -> None:
        self.center_in_tangent = bool(center_in_tangent)
        self.mean_iters = int(mean_iters)
        self.mean_step = float(mean_step)
        self.eps = float(eps)

        q = float(clip_quantile)
        if not (0.0 <= q < 0.5):
            raise ValueError(f"clip_quantile must be in [0, 0.5), got {clip_quantile}")
        self.clip_quantile = q
        self.range_floor = float(range_floor)

        self.mu_: Optional[torch.Tensor] = None
        self.p_: Optional[torch.Tensor] = None
        self.v_mean_: Optional[torch.Tensor] = None

        self.t_min_: Optional[float] = None
        self.t_max_: Optional[float] = None
        self.t_lo_: Optional[float] = None
        self.t_hi_: Optional[float] = None

        self.explained_var_ratio_: float = 0.0
        self.is_fitted_: bool = False

    @staticmethod
    def _check_z(z: torch.Tensor) -> torch.Tensor:
        """
        入力 z の形状・型を検査し、最後次元で L2 正規化して返す。

        Parameters
        ----------
        z : torch.Tensor of shape (N, D)
            入力点群。

        Returns
        -------
        z_norm : torch.Tensor of shape (N, D)
            L2 正規化済み点群。

        Raises
        ------
        TypeError
            z が torch.Tensor でない場合。
        ValueError
            z が 2 次元でない、または N < 2 の場合。
        """
        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor")
        if z.ndim != 2:
            raise ValueError(f"z must be (N, D), got shape={tuple(z.shape)}")
        if z.size(0) < 2:
            raise ValueError("z must contain at least 2 samples")
        return F.normalize(z, dim=1)

    def _check_fitted(self) -> None:
        """
        学習済みかどうかを検査する。

        Raises
        ------
        RuntimeError
            fit() が呼ばれていない、または必要な属性が未設定の場合。
        """
        if (not self.is_fitted_) or self.mu_ is None or self.p_ is None or self.v_mean_ is None:
            raise RuntimeError("Call fit() before using this method.")
        if self.t_lo_ is None or self.t_hi_ is None:
            raise RuntimeError("t_lo_/t_hi_ are not set. Call fit() first.")

    @torch.no_grad()
    def fit(self, z: torch.Tensor, *, mu: Optional[torch.Tensor] = None) -> "SphericalPGA1D":
        """
        モデルを学習する（主測地線方向とクリップ範囲の推定）。

        Parameters
        ----------
        z : torch.Tensor of shape (N, D)
            学習用点群（単位球面上が望ましい）。
        mu : torch.Tensor of shape (D,), optional
            球面平均 μ を外部から与える場合に指定する。
            None の場合は `frechet_mean_sphere` により推定する。

        Returns
        -------
        self : SphericalPGA1D
            学習済みインスタンス。

        Notes
        -----
        fit() は以下を内部で行う：
        1) μ の推定
        2) v_i = log_μ(z_i) の計算
        3) （任意）中心化 v'_i = v_i - mean(v)
        4) 接空間 PCA（SVD）で主方向 p を取得
        5) t_train から分位点レンジ [t_lo_, t_hi_] を保存
        """
        z = self._check_z(z)
        device, dtype = z.device, z.dtype

        # mean direction mu
        if mu is None:
            mu_ = frechet_mean_sphere(z, iters=self.mean_iters, step=self.mean_step, eps=self.eps)
        else:
            if mu.ndim != 1 or mu.shape[0] != z.shape[1]:
                raise ValueError(f"mu must be (D,), got shape={tuple(mu.shape)}")
            mu_ = F.normalize(mu.to(device=device, dtype=dtype), dim=0)

        # tangent vectors
        V = sphere_log(mu_, z, eps=self.eps)  # (N, D)
        if self.center_in_tangent:
            v_mean_ = V.mean(dim=0)
            Vc = V - v_mean_[None, :]
        else:
            v_mean_ = torch.zeros_like(mu_)
            Vc = V

        # PCA via SVD
        _, S, Vt = torch.linalg.svd(Vc, full_matrices=False)
        p_ = Vt[0]
        p_ = p_ - (p_ @ mu_) * mu_
        p_ = F.normalize(p_, dim=0)

        var_total = (S**2).sum().item()
        var_1d = (S[0] ** 2).item() if S.numel() > 0 else 0.0
        self.explained_var_ratio_ = float(var_1d / var_total) if var_total > 0 else 0.0

        # training t and clipping bounds
        t_train = (Vc @ p_)  # (N,)
        t_min = float(t_train.min().item())
        t_max = float(t_train.max().item())

        if self.clip_quantile > 0.0:
            t_lo = float(torch.quantile(t_train, self.clip_quantile).item())
            t_hi = float(torch.quantile(t_train, 1.0 - self.clip_quantile).item())
        else:
            t_lo, t_hi = t_min, t_max

        if (t_hi - t_lo) <= self.range_floor:
            # fallback: no quantile clipping
            t_lo, t_hi = t_min, t_max
        if (t_hi - t_lo) <= self.range_floor:
            # degenerate
            t_lo, t_hi = 0.0, 1.0

        self.mu_ = mu_
        self.p_ = p_
        self.v_mean_ = v_mean_
        self.t_min_ = t_min
        self.t_max_ = t_max
        self.t_lo_ = t_lo
        self.t_hi_ = t_hi
        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        1D 座標 t を 0–1 スコア u に変換する（t クリップ + min–max）。

        Parameters
        ----------
        t : torch.Tensor
            1D 座標（形状は任意）。

        Returns
        -------
        u : torch.Tensor
            0–1 スコア。`t` と同形状。

        Notes
        -----
        - 内部で `t_clip = clamp(t, t_lo_, t_hi_)` を適用する。
        - したがって `t < t_lo_` は 0 に、`t > t_hi_` は 1 に飽和する。
        - `(t_hi_ - t_lo_)` が極小の場合はゼロ配列を返す（安全策）。
        """
        self._check_fitted()
        denom = float(self.t_hi_ - self.t_lo_)
        if denom <= self.range_floor:
            return torch.zeros_like(t)
        t_clip = torch.clamp(t, float(self.t_lo_), float(self.t_hi_))
        u = (t_clip - float(self.t_lo_)) / denom
        return torch.clamp(u, 0.0, 1.0)

    @torch.no_grad()
    def denormalize_t(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        0–1 スコア u を 1D 座標 t に戻す（代表値への復元）。

        Parameters
        ----------
        u : torch.Tensor
            0–1 スコア（形状は任意）。内部で [0,1] に clamp される。

        Returns
        -------
        t : torch.Tensor
            1D 座標（形状は入力と同じ）。値域は `[t_lo_, t_hi_]`。

        Notes
        -----
        `normalize_t()` が t をクリップしているため、外れ値側の情報は 0/1 に潰れている。
        よって本関数は **厳密な逆写像ではなく**、`[t_lo_, t_hi_]` 内の代表値へ戻すだけである。
        """
        self._check_fitted()
        u = torch.clamp(u, 0.0, 1.0)
        return u * float(self.t_hi_ - self.t_lo_) + float(self.t_lo_)

    @torch.no_grad()
    def transform(
        self,
        z: torch.Tensor,
        *,
        normalize: bool = False,
        return_delta: bool = False,
        return_z_hat: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        球面上の点 z を 1D 座標（t）または 0–1 スコア（u）へ変換する。

        Parameters
        ----------
        z : torch.Tensor of shape (N, D)
            入力点群。
        normalize : bool, default=False
            True の場合は 0–1 スコア u を返す。False の場合は生の t を返す。
        return_delta : bool, default=False
            True の場合、`delta = arccos(<z, z_hat>)` を追加で返す。
        return_z_hat : bool, default=False
            True の場合、主測地線上への再投影点 z_hat を追加で返す。

        Returns
        -------
        out : torch.Tensor or tuple of torch.Tensor
            - normalize=False: t (N,)
            - normalize=True : u (N,) in [0,1]
            オプションに応じて (z_hat), (delta) が後ろに付く。

        Notes
        -----
        return_z_hat / return_delta は内部で inverse_transform を呼ぶため追加コストがかかる。
        """
        self._check_fitted()
        z = self._check_z(z).to(device=self.mu_.device, dtype=self.mu_.dtype)

        V = sphere_log(self.mu_, z, eps=self.eps)
        Vc = V - self.v_mean_[None, :] if self.center_in_tangent else V
        t = (Vc @ self.p_)  # (N,)

        out0 = self.normalize_t(t) if normalize else t
        outs: Tuple[torch.Tensor, ...] = (out0,)

        if return_z_hat or return_delta:
            z_hat = self.inverse_transform(out0, normalized=normalize)
            if return_z_hat:
                outs = outs + (z_hat,)
            if return_delta:
                delta = _safe_acos((z * z_hat).sum(dim=1))
                outs = outs + (delta,)

        return outs[0] if len(outs) == 1 else outs

    @torch.no_grad()
    def inverse_transform(self, t_or_u: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
        """
        1D 座標（t）または 0–1 スコア（u）から、主測地線上の点 z_hat を復元する。

        Parameters
        ----------
        t_or_u : torch.Tensor of shape (N,)
            `normalized=False` の場合は t、`normalized=True` の場合は u と解釈する。
        normalized : bool, default=False
            True の場合、u を `denormalize_t()` で t に戻してから復元する。

        Returns
        -------
        z_hat : torch.Tensor of shape (N, D)
            主測地線上の復元点（単位ベクトル）。

        Notes
        -----
        normalized=True の場合、u は [0,1] に clamp され、t は `[t_lo_, t_hi_]` 内に復元される。
        つまり外れ値側（0/1 に飽和した部分）は代表値へ戻る（厳密可逆ではない）。
        """
        self._check_fitted()

        if not isinstance(t_or_u, torch.Tensor):
            t_or_u = torch.as_tensor(t_or_u, device=self.mu_.device, dtype=self.mu_.dtype)
        else:
            t_or_u = t_or_u.to(device=self.mu_.device, dtype=self.mu_.dtype)

        if t_or_u.ndim == 0:
            t_or_u = t_or_u[None]
        if t_or_u.ndim != 1:
            raise ValueError(f"t_or_u must be 1D (N,), got shape={tuple(t_or_u.shape)}")

        t = self.denormalize_t(t_or_u) if normalized else t_or_u

        if self.center_in_tangent:
            V_hat = self.v_mean_[None, :] + t[:, None] * self.p_[None, :]
        else:
            V_hat = t[:, None] * self.p_[None, :]

        return sphere_exp(self.mu_, V_hat, eps=self.eps)

    def state_dict(self) -> Dict[str, Any]:
        """
        学習済み状態を辞書として返す（torch.save 用）。

        Returns
        -------
        state : dict
            `torch.save(state, path)` で保存可能な辞書。

        Notes
        -----
        この state_dict は **当該実装形式に依存**する。互換性が必要ならバージョン管理を推奨。
        """
        self._check_fitted()
        return {
            "mu": self.mu_.detach().cpu(),
            "p": self.p_.detach().cpu(),
            "v_mean": self.v_mean_.detach().cpu(),
            "explained_var_ratio": float(self.explained_var_ratio_),
            "t_min": float(self.t_min_) if self.t_min_ is not None else None,
            "t_max": float(self.t_max_) if self.t_max_ is not None else None,
            "t_lo": float(self.t_lo_) if self.t_lo_ is not None else None,
            "t_hi": float(self.t_hi_) if self.t_hi_ is not None else None,
            "config": {
                "center_in_tangent": self.center_in_tangent,
                "mean_iters": self.mean_iters,
                "mean_step": self.mean_step,
                "eps": self.eps,
                "clip_quantile": self.clip_quantile,
                "range_floor": self.range_floor,
            },
        }

    def save(self, path: PathLike) -> None:
        """
        学習済みモデルを保存する（torch.save）。

        Parameters
        ----------
        path : str or pathlib.Path
            保存先パス。
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: PathLike, *, map_location: Optional[Union[str, torch.device]] = "cpu") -> "SphericalPGA1D":
        """
        保存済みモデルを読み込む（torch.load）。

        Parameters
        ----------
        path : str or pathlib.Path
            読み込み元パス。
        map_location : str or torch.device, default="cpu"
            torch.load の map_location。

        Returns
        -------
        model : SphericalPGA1D
            復元されたインスタンス。

        Notes
        -----
        - 本実装の state_dict 形式（quantile-clipped min–max）を前提とする。
        - 旧形式との後方互換を要求する場合は、別途ロード分岐を設けること。
        """
        obj = torch.load(Path(path), map_location=map_location)
        cfg = obj.get("config", {})

        model = cls(
            center_in_tangent=cfg.get("center_in_tangent", True),
            mean_iters=cfg.get("mean_iters", 10),
            mean_step=cfg.get("mean_step", 1.0),
            eps=cfg.get("eps", 1e-8),
            clip_quantile=cfg.get("clip_quantile", 0.01),
            range_floor=cfg.get("range_floor", 1e-12),
        )

        mu = obj["mu"]
        p = obj["p"]
        v_mean = obj["v_mean"]

        model.mu_ = F.normalize(mu.to(dtype=torch.float32), dim=0)
        p_ = p.to(dtype=torch.float32)
        p_ = p_ - (p_ @ model.mu_) * model.mu_
        model.p_ = F.normalize(p_, dim=0)
        model.v_mean_ = v_mean.to(dtype=torch.float32)

        model.explained_var_ratio_ = float(obj.get("explained_var_ratio", 0.0))
        model.t_min_ = obj.get("t_min", None)
        model.t_max_ = obj.get("t_max", None)
        model.t_lo_ = obj.get("t_lo", None)
        model.t_hi_ = obj.get("t_hi", None)

        if model.t_lo_ is None or model.t_hi_ is None:
            raise ValueError("Missing clipping bounds t_lo/t_hi in state_dict.")
        model.is_fitted_ = True
        return model