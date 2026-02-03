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
    Extrinsic PCA for spherical data: compute PCA in ambient Euclidean space (R^D).

    Fit on X (N,D), then transform by Z = X @ W where W=(D,k).
    Optionally re-normalize Z onto unit sphere in R^k.

    Notes:
      - center=False is often preferable when downstream uses cosine similarity.
      - method="lowrank" uses torch.pca_lowrank (randomized SVD), scalable on GPU.
      - method="svd" uses full SVD (exact but can be expensive).
    """

    def __init__(self, cfg: ExtrinsicPCAConfig):
        self.cfg = cfg
        self.device = _auto_device(cfg.device)
        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

        self.mean_: Optional[torch.Tensor] = None          # (D,)
        self.components_: Optional[torch.Tensor] = None    # (D,k)
        self.singular_values_: Optional[torch.Tensor] = None  # (k,)
        self.explained_variance_: Optional[torch.Tensor] = None  # (k,)

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "ExtrinsicPCA":
        X = self._check_X(X)

        if self.cfg.center:
            self.mean_ = X.mean(dim=0)
            Xc = X - self.mean_
        else:
            self.mean_ = None
            Xc = X

        k = int(self.cfg.n_components)
        if k <= 0 or k > Xc.shape[1]:
            raise ValueError(f"n_components must be in [1, D]. got {k}, D={Xc.shape[1]}")

        # compute principal directions in R^D
        if self.cfg.method == "lowrank":
            q = self.cfg.q if self.cfg.q is not None else (k + 8)
            torch.manual_seed(int(self.cfg.seed))
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

        # explained variance (sample) ~ S^2 / (N-1)
        n = Xc.shape[0]
        denom = max(1, n - 1)
        self.explained_variance_ = (Sk ** 2) / denom
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
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
        return self.fit(X).transform(X)

    @torch.no_grad()
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Approximate reconstruction in ambient space: Xhat = Z @ W^T (+ mean if centered).
        (This is Euclidean reconstruction, not manifold geodesic reconstruction.)
        """
        self._check_fitted()
        Z = Z.to(device=self.device, dtype=self.dtype)
        Xhat = Z @ self.components_.T # type: ignore
        if self.mean_ is not None:
            Xhat = Xhat + self.mean_
        return Xhat

    def save(self, path: str) -> None:
        self._check_fitted()
        payload = {
            "cfg": asdict(self.cfg),
            "mean": None if self.mean_ is None else self.mean_.detach().cpu(),
            "components": self.components_.detach().cpu(), # type: ignore
            "singular_values": None if self.singular_values_ is None else self.singular_values_.detach().cpu(),
            "explained_variance": None if self.explained_variance_ is None else self.explained_variance_.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ExtrinsicPCA":
        payload = torch.load(path, map_location="cpu")
        cfg = ExtrinsicPCAConfig(**payload["cfg"])
        if device is not None:
            cfg.device = device
        obj = cls(cfg)

        obj.mean_ = None if payload["mean"] is None else payload["mean"].to(obj.device, obj.dtype)
        obj.components_ = payload["components"].to(obj.device, obj.dtype)
        obj.singular_values_ = None if payload["singular_values"] is None else payload["singular_values"].to(obj.device, obj.dtype)
        obj.explained_variance_ = None if payload["explained_variance"] is None else payload["explained_variance"].to(obj.device, obj.dtype)
        return obj

    # -----------------
    # internal helpers
    # -----------------
    def _check_X(self, X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N,D). got shape={tuple(X.shape)}")
        return X.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _check_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("ExtrinsicPCA is not fitted yet. Call fit() first.")