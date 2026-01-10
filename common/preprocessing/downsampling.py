from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from chemomae.preprocessing import cosine_fps_downsample


def random_downsampling(
    data_nc: np.ndarray,
    ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (N, C) 行列を一様ランダムにダウンサンプリングする（重複なし）。

    入力 ``data_nc`` （shape ``(N, C)``）から、``ceil(N * ratio)`` 個のサンプルを
    **重複なし**で一様ランダムに抽出します。返り値には抽出後の行列と、
    元配列に対するインデックスを含めます。

    Parameters
    ----------
    data_nc : ndarray of shape (N, C)
        入力データ行列。N はサンプル数、C は特徴次元（例：波長チャネル数）。
    ratio : float
        抽出割合。``0 < ratio <= 1``。
    seed : int, default=42
        乱数シード（再現性のため）。

    Returns
    -------
    downsampled_nc : ndarray of shape (M, C)
        ダウンサンプリング後データ。ここで ``M = ceil(N * ratio)``。
    indices_n : ndarray of shape (M,), dtype=int64
        元データに対する抽出インデックス（昇順）。

    Notes
    -----
    - 抽出は **without replacement**（重複なし）。
    - ``indices_n`` を昇順に並べ替えるため、``downsampled_nc`` も元データ順に整列されます。
    """
    if not isinstance(data_nc, np.ndarray):
        raise TypeError(f"data_nc must be np.ndarray, got {type(data_nc)}")
    if data_nc.ndim != 2:
        raise ValueError(f"data_nc must have shape (N, C), got {data_nc.shape}")
    N, C = data_nc.shape
    if N <= 0 or C <= 0:
        raise ValueError(f"data_nc must have positive shape (N, C), got {data_nc.shape}")

    if not isinstance(ratio, (float, int, np.floating, np.integer)):
        raise TypeError(f"ratio must be float, got {type(ratio)}")
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be int, got {type(seed)}")

    M = int(np.ceil(N * ratio))
    rng = np.random.default_rng(int(seed))
    indices_n = rng.choice(N, size=M, replace=False)
    indices_n = np.sort(indices_n).astype(np.int64, copy=False)

    downsampled_nc = data_nc[indices_n]
    return downsampled_nc, indices_n


def cfp_downsampling(
    data_nc: np.ndarray,
    ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine FPS（cosine 距離に基づく Furthest Point Sampling）で (N, C) 行列をダウンサンプリングする。

    chemomae の ``cosine_fps_downsample`` を用いて、入力 ``data_nc`` （shape ``(N, C)``）から
    ``ratio`` に応じた代表点集合を抽出します。返り値のインデックスは **昇順にソート**して返します。

    Parameters
    ----------
    data_nc : ndarray of shape (N, C)
        入力データ行列。N はサンプル数、C は特徴次元（例：波長チャネル数）。
    ratio : float
        抽出割合。``0 < ratio <= 1``。
    seed : int, default=42
        FPS の初期点選択などに用いるシード（実装依存だが再現性のため）。

    Returns
    -------
    downsampled_nc : ndarray of shape (M, C)
        ダウンサンプリング後データ。ここで ``M`` は実装により概ね ``ceil(N * ratio)``。
    indices_n : ndarray of shape (M,), dtype=int64
        元データに対する選択インデックス（昇順）。

    Notes
    -----
    - 内部で ``torch.Tensor``（float32）に変換してから処理します。
    - 返り値は ``return_numpy=True`` のため ``np.ndarray`` です。
    - 本関数では ``indices_n`` を昇順にソートし、それに合わせて ``downsampled_nc`` も並べ替えます。
    """
    if not isinstance(data_nc, np.ndarray):
        raise TypeError(f"data_nc must be np.ndarray, got {type(data_nc)}")
    if data_nc.ndim != 2:
        raise ValueError(f"data_nc must have shape (N, C), got {data_nc.shape}")
    N, C = data_nc.shape
    if N <= 0 or C <= 0:
        raise ValueError(f"data_nc must have positive shape (N, C), got {data_nc.shape}")

    if not isinstance(ratio, (float, int, np.floating, np.integer)):
        raise TypeError(f"ratio must be float, got {type(ratio)}")
    ratio = float(ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be int, got {type(seed)}")

    data_nc_t = torch.as_tensor(data_nc, dtype=torch.float32)
    downsampled_nc, indices_n = cosine_fps_downsample(
        X=data_nc_t,
        ratio=ratio,
        seed=int(seed),
        return_numpy=True,
        return_indices=True,
    )

    indices_n = np.asarray(indices_n, dtype=np.int64)
    order_n = np.argsort(indices_n)
    indices_n = indices_n[order_n]
    downsampled_nc = np.asarray(downsampled_nc)[order_n]

    return downsampled_nc, indices_n