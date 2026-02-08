from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import torch
import torch.nn.functional as F


def _auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ExtrinsicPCAConfig:
    """
    ExtrinsicPCA の設定（ハイパーパラメータ）をまとめたデータクラス。

    Parameters
    ----------
    n_components : int, default=32
        射影後の次元数 ``k``。
    center : bool, default=False
        平均中心化（``X <- X - mean``）を行うかどうか。
        **下流が cosine 類似度（球面幾何）** を前提とする場合、中心化は幾何を崩しやすいので
        ``False`` が無難です。
    renorm : bool, default=True
        射影後の ``Z`` を L2 正規化し、低次元の単位球面（``S^{k-1}``）へ戻すかどうか。
        cosine 下流に渡すなら通常 ``True``。
    method : {"lowrank", "svd"}, default="lowrank"
        主成分方向の計算方法。
        - ``"lowrank"``: `torch.pca_lowrank` による randomized SVD（GPU でスケールしやすい）
        - ``"svd"``: `torch.linalg.svd` による厳密 SVD（高コストになり得る）
    q : int or None, default=None
        ``method="lowrank"`` の oversampling 次元。``None`` の場合は ``n_components + 8``。
        大きいほど近似精度は上がりやすい一方、計算量・メモリが増えます。
    niter : int, default=5
        ``method="lowrank"`` の power iteration 回数。大きいほど精度は上がりやすい一方、遅くなります。
    seed : int, default=42
        ``method="lowrank"`` の乱数シード。
    device : str or None, default=None
        計算デバイス。``None`` の場合は ``cuda -> mps -> cpu`` の順で自動選択します。
    dtype : {"float32", "float64"}, default="float32"
        計算 dtype。数値安定性を優先するなら ``float64`` も選択可能（遅くなる場合あり）。

    Notes
    -----
    - 本クラスは “外在的（extrinsic）” PCA であり、データが球面上にあっても
      **埋め込み先のユークリッド空間（R^D）で PCA を行ってから線形射影**します。
    """
    n_components: int = 32
    center: bool = False          # 球面/cosine下流なら False が無難
    renorm: bool = True           # 射影後に L2 正規化して低次元球面へ戻す
    method: Literal["lowrank", "svd"] = "lowrank"
    # lowrank params
    q: Optional[int] = None       # oversampling (default: n_components + 8)
    niter: int = 5                # power iterations
    seed: int = 42
    # compute
    device: Optional[str] = None  # None => auto
    dtype: Literal["float32", "float64"] = "float32"


class ExtrinsicPCA:
    """
    球面データのための外在的 PCA（Extrinsic PCA）。

    単位球面上のデータ ``X in R^{N×D}``（例：L2 正規化済み特徴）に対して、周辺ユークリッド空間 ``R^D`` で
    主成分方向 ``W in R^{D×k}`` を推定し、線形射影
    ``Z = X W`` により ``k`` 次元へ次元削減します。必要に応じて ``Z`` を L2 正規化し、
    低次元球面（cosine 幾何）へ戻します。

    Parameters
    ----------
    cfg : ExtrinsicPCAConfig
        ハイパーパラメータ一式。

    Attributes
    ----------
    cfg : ExtrinsicPCAConfig
        設定。
    device : torch.device
        計算に使用するデバイス。
    dtype : torch.dtype
        計算に使用する dtype。
    mean_ : torch.Tensor of shape (D,) or None
        ``center=True`` のときの特徴平均。``center=False`` の場合は ``None``。
    components_ : torch.Tensor of shape (D, k) or None
        主成分方向（列ベクトルが各主成分）。``fit`` 後にセットされます。
    singular_values_ : torch.Tensor of shape (k,) or None
        上位 ``k`` 個の特異値。
    explained_variance_ : torch.Tensor of shape (k,) or None
        ``center=True`` のときのみ、中心化済み共分散に対応する「説明分散」を格納します。
        実装上は ``S^2 / max(1, N-1)`` です。
    explained_moment_ : torch.Tensor of shape (k,) or None
        ``center=False`` のときのみ、原点周り二次モーメント（second moment）に対応する統計を格納します。
        実装上は ``S^2 / max(1, N)`` です（中心化していないため、一般的な“説明分散”とは別物）。

    Notes
    -----
    - **中心化（center）**:
      cosine 類似度を前提とする下流（球面クラスタリング等）では、中心化により方向情報が変わり得ます。
      そのためデフォルトは ``center=False`` です。
    - **正規化（renorm）**:
      射影 ``Z = XW`` の後に ``Z`` を L2 正規化することで、低次元でも cosine 幾何を保ちやすくなります。
    - **計算方法**:
      ``method="lowrank"`` は `torch.pca_lowrank`（randomized SVD）を使い、大規模データでスケールしやすいです。
      ``method="svd"`` は厳密ですが、``D`` や ``N`` が大きいと重くなります。
    - 本クラスは scikit-learn の PCA と同様の API 形状（``fit/transform/inverse_transform``）を意識していますが、
      **PyTorch Tensor 前提**であり、``torch.no_grad()`` 下で動作します。

    Examples
    --------
    >>> import torch
    >>> from extrinsic_pca import ExtrinsicPCA, ExtrinsicPCAConfig
    >>> X = torch.randn(1000, 256)
    >>> X = torch.nn.functional.normalize(X, dim=1)  # 球面データを仮定
    >>> pca = ExtrinsicPCA(ExtrinsicPCAConfig(n_components=32, center=False, renorm=True))
    >>> Z = pca.fit_transform(X)
    >>> Z.shape
    torch.Size([1000, 32])
    >>> # 近似的に元空間へ戻す（ユークリッド再構成）
    >>> Xhat = pca.inverse_transform(Z)
    >>> Xhat.shape
    torch.Size([1000, 256])
    """

    def __init__(self, cfg: ExtrinsicPCAConfig):
        self.cfg = cfg
        self.device = _auto_device(cfg.device)
        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

        self.mean_: Optional[torch.Tensor] = None          # (D,)
        self.components_: Optional[torch.Tensor] = None    # (D,k)
        self.singular_values_: Optional[torch.Tensor] = None  # (k,)
        self.explained_variance_: Optional[torch.Tensor] = None  # (k,) center=True only
        self.explained_moment_: Optional[torch.Tensor] = None    # (k,) center=False only

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "ExtrinsicPCA":
        """
        主成分方向（components_）を推定します。

        Parameters
        ----------
        X : torch.Tensor of shape (N, D)
            入力データ。``(N, D)`` の 2 次元テンソルである必要があります。
            ``cfg.device`` / ``cfg.dtype`` に従って内部で転送・型変換されます。

        Returns
        -------
        self : ExtrinsicPCA
            学習済みインスタンス（``components_`` 等がセットされます）。

        Raises
        ------
        TypeError
            ``X`` が ``torch.Tensor`` でない場合。
        ValueError
            ``X`` が 2 次元でない場合、または ``n_components`` が不正な場合。

        Notes
        -----
        - ``center=True`` の場合は平均との差分を取った ``Xc`` に対して分解します。
        - ``method="lowrank"`` の場合のみ ``seed`` が乱数性に影響します（再現性のため）。
        - ``center=True`` のときは ``explained_variance_ = S^2 / max(1, N-1)`` を格納します。
        - ``center=False`` のときは ``explained_variance_=None`` とし、代わりに ``explained_moment_ = S^2 / max(1, N)`` を格納します。
        """
        X = self._check_X(X)

        if self.cfg.center:
            self.mean_ = X.mean(dim=0)
            Xc = X - self.mean_
        else:
            self.mean_ = None
            Xc = X

        n, d = Xc.shape
        min_nd = min(int(n), int(d))

        k = int(self.cfg.n_components)
        if k <= 0 or k > min_nd:
            raise ValueError(
                f"n_components must be in [1, min(N,D)]. got {k}, N={int(n)}, D={int(d)}"
            )

        # compute principal directions in R^D
        if self.cfg.method == "lowrank":
            q = self.cfg.q if self.cfg.q is not None else (k + 8)
            q = int(q)
            # torch.pca_lowrank requires q <= min(N, D). Also keep q >= k.
            q = max(q, k)
            q = min(q, min_nd)

            self._set_seed(int(self.cfg.seed))
            # torch.pca_lowrank returns U, S, V where Xc ≈ U diag(S) V^T ; V is (D, q)
            U, S, V = torch.pca_lowrank(Xc, q=q, center=False, niter=int(self.cfg.niter))
            # principal components are first k columns of V
            W = V[:, :k].contiguous()
            Sk = S[:k].contiguous()
        else:
            # exact SVD: Xc = U S Vh, Vh is (D,D); principal directions are Vh[:k].T
            # full_matrices=False is generally what you want.
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            W = Vh[:k, :].T.contiguous()
            Sk = S[:k].contiguous()

        self.components_ = W  # (D,k)
        self.singular_values_ = Sk

        # --- output statistics ---
        # Centered data => covariance eigenvalues: S^2 / (N-1)
        # Uncentered data => second-moment eigenvalues: S^2 / N
        if self.cfg.center:
            denom = max(1, int(n) - 1)
            self.explained_variance_ = (Sk ** 2) / denom
            self.explained_moment_ = None
        else:
            denom = max(1, int(n))
            self.explained_variance_ = None
            self.explained_moment_ = (Sk ** 2) / denom
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        学習済み主成分方向へ射影して低次元表現を返します。

        Parameters
        ----------
        X : torch.Tensor of shape (N, D)
            入力データ。

        Returns
        -------
        Z : torch.Tensor of shape (N, k)
            射影後の表現。``cfg.renorm=True`` の場合は行方向に L2 正規化されます。

        Raises
        ------
        RuntimeError
            ``fit`` 前に呼び出された場合。
        """
        self._check_fitted()
        X = self._check_X(X)

        if self.mean_ is not None:
            X = X - self.mean_

        Z = X @ self.components_  # type: ignore # (N,k)

        if self.cfg.renorm:
            Z = F.normalize(Z, dim=1, eps=1e-12)
        return Z

    @torch.no_grad()
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        ``fit`` と ``transform`` を続けて実行します。

        Parameters
        ----------
        X : torch.Tensor of shape (N, D)
            入力データ。

        Returns
        -------
        Z : torch.Tensor of shape (N, k)
            射影後の表現。
        """
        return self.fit(X).transform(X)

    @torch.no_grad()
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        低次元表現から元の周辺空間（R^D）へ近似再構成します。

        ``Xhat = Z @ components_.T``（``center=True`` のときは ``+ mean_``）で計算します。

        Parameters
        ----------
        Z : torch.Tensor of shape (N, k)
            低次元表現。通常 ``transform`` の出力です。
            注意：``cfg.renorm=True`` の場合、``Z`` は単位ノルム化されているため、
            この再構成は **ユークリッド意味での近似** になり、元のスケール情報は失われます。

        Returns
        -------
        Xhat : torch.Tensor of shape (N, D)
            近似再構成（ユークリッド再構成）。多様体上の測地的再構成ではありません。

        Raises
        ------
        RuntimeError
            ``fit`` 前に呼び出された場合。
        """
        self._check_fitted()
        Z = Z.to(device=self.device, dtype=self.dtype)
        Xhat = Z @ self.components_.T # type: ignore
        if self.mean_ is not None:
            Xhat = Xhat + self.mean_
        return Xhat

    def save(self, path: str) -> None:
        """
        学習済み状態をファイルへ保存します（`torch.save`）。

        Parameters
        ----------
        path : str
            保存先パス。`torch.save` が書き込める場所を指定してください。

        Notes
        -----
        - 保存されるのは ``cfg`` と、``mean_ / components_ / singular_values_ / explained_variance_ / explained_moment_`` です。
        - Tensor は CPU に移して保存します。
        """
        self._check_fitted()
        payload = {
            "cfg": asdict(self.cfg),
            "mean": None if self.mean_ is None else self.mean_.detach().cpu(),
            "components": self.components_.detach().cpu(), # type: ignore
            "singular_values": None if self.singular_values_ is None else self.singular_values_.detach().cpu(),
            "explained_variance": None if self.explained_variance_ is None else self.explained_variance_.detach().cpu(),
            "explained_moment": None if self.explained_moment_ is None else self.explained_moment_.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ExtrinsicPCA":
        """
        保存済みファイルから復元します。

        Parameters
        ----------
        path : str
            読み込み元パス。
        device : str or None, default=None
            復元後に配置したいデバイス。指定した場合は ``cfg.device`` を上書きします。

        Returns
        -------
        obj : ExtrinsicPCA
            復元されたインスタンス（学習済み）。

        Notes
        -----
        - 読み込みは一旦 CPU に map してから、指定デバイスへ移します。
        """
        payload = torch.load(path, map_location="cpu")
        cfg = ExtrinsicPCAConfig(**payload["cfg"])
        if device is not None:
            cfg.device = device
        obj = cls(cfg)

        obj.mean_ = None if payload["mean"] is None else payload["mean"].to(obj.device, obj.dtype)
        obj.components_ = payload["components"].to(obj.device, obj.dtype)
        obj.singular_values_ = None if payload["singular_values"] is None else payload["singular_values"].to(obj.device, obj.dtype)
        obj.explained_variance_ = None if payload["explained_variance"] is None else payload["explained_variance"].to(obj.device, obj.dtype)
        em = payload.get("explained_moment", None)
        obj.explained_moment_ = None if em is None else em.to(obj.device, obj.dtype)
        return obj

    # -----------------
    # internal helpers
    # -----------------
    def _set_seed(self, seed: int) -> None:
        # CPU RNG
        torch.manual_seed(int(seed))
        # CUDA RNG (if available). This helps reproducibility of randomized algorithms.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _check_X(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D). got shape={tuple(X.shape)}")
        return X.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("ExtrinsicPCA is not fitted yet. Call fit() first.")