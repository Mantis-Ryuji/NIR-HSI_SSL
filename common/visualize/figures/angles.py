from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import colorcet as cc
from sklearn.manifold import MDS


def load_centroids(path: str | Path) -> np.ndarray:
    """
    保存済みクラスタ中心（centroids）をロードし、L2 正規化して返す。

    Parameters
    ----------
    path : str | Path
        centroid の保存先パス。`torch.load(path)` で読み込める形式を想定する。

        想定する保存形式は以下のいずれか：
        - dict 形式: {"centroids": Tensor | ndarray}
        - Tensor / ndarray そのもの（centroids のみを保存）

        centroids は shape (K, D) の 2 次元配列である必要がある。
        K はクラスタ数、D は特徴次元。

    Returns
    -------
    C : np.ndarray, shape (K, D), dtype float32
        行方向（クラスタごと）に L2 正規化した centroid 行列。

    Raises
    ------
    FileNotFoundError
        指定パスが存在しない場合。
    ValueError
        読み込んだ centroids が 2 次元でない場合。

    Notes
    -----
    - cosine 類似度を前提に centroid を扱うため、行ベクトルを L2 正規化して返す。
    - 0 除算回避のため、ノルムは `clip(norm, 1e-12, None)` を用いる。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")

    obj = torch.load(p, map_location="cpu")
    C = obj["centroids"] if isinstance(obj, dict) and "centroids" in obj else obj

    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()

    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"centroids must be 2D, got {C.shape}")

    n = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.clip(n, 1e-12, None)
    return C


def angle_matrix(C: np.ndarray) -> np.ndarray:
    """
    クラスタ中心行列 C から、クラスタ間角度行列 Θ（ラジアン）を計算する。

    `C` の各行が L2 正規化済み（単位ベクトル）であるとき、cosine 類似度は
    `S = C @ C.T` で与えられる。これを [-1, 1] にクリップした上で、

        Θ = arccos(S)

    により角度（radian）を得る。

    Parameters
    ----------
    C : np.ndarray, shape (K, D)
        クラスタ中心行列。行がクラスタ中心（ベクトル）を表す。

    Returns
    -------
    theta_rad : np.ndarray, shape (K, K)
        クラスタ間角度行列（単位: rad）。対角は 0。

    Raises
    ------
    ValueError
        C が 2 次元配列でない場合。

    Notes
    -----
    - `C` が厳密に正規化されていない場合でも計算は可能だが、
      cosine と角度の解釈が崩れるため、基本は正規化済みを推奨する。
    - 数値誤差で S が [-1, 1] を僅かに超えることがあるため、clip を入れている。
    """
    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError("C は 2 次元配列である必要があります。")

    S = C @ C.T
    S = np.clip(S, -1.0, 1.0)
    return np.arccos(S)


def plot_angle_kde_comparison(
    theta_ref_rad: np.ndarray,
    theta_lat_rad: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    クラスタ間角度（上三角成分）の分布を KDE で推定し、ref と latent を比較プロットして保存する。

    Parameters
    ----------
    theta_ref_rad : np.ndarray, shape (K, K)
        ref（例: Ref(SNV) 空間）における角度行列（radian）。
    theta_lat_rad : np.ndarray, shape (K, K)
        latent 空間における角度行列（radian）。
    out_path : str | Path
        保存先パス。

    Notes
    -----
    - 対角成分（0）を除いた上三角成分のみを取り出し、度数（°）に変換して KDE を推定する。
    - `bw_method=0.1` を固定しているため、K が小さい／分布が極端な場合に
      過度に平滑化・過小平滑化する可能性がある（必要なら引数化推奨）。
    - “Difference region” は2つの KDE 曲線の間を塗りつぶして差分の雰囲気を見せるためのもの。
    """
    from scipy.stats import gaussian_kde

    theta_ref_rad = np.asarray(theta_ref_rad, dtype=float)
    theta_lat_rad = np.asarray(theta_lat_rad, dtype=float)

    if theta_ref_rad.ndim != 2 or theta_ref_rad.shape[0] != theta_ref_rad.shape[1]:
        raise ValueError("theta_ref_rad は正方の 2 次元配列である必要があります。")
    if theta_lat_rad.ndim != 2 or theta_lat_rad.shape[0] != theta_lat_rad.shape[1]:
        raise ValueError("theta_lat_rad は正方の 2 次元配列である必要があります。")
    if theta_ref_rad.shape != theta_lat_rad.shape:
        raise ValueError(f"shape mismatch: ref={theta_ref_rad.shape} vs lat={theta_lat_rad.shape}")

    tri_ref = theta_ref_rad[np.triu_indices_from(theta_ref_rad, 1)]
    tri_lat = theta_lat_rad[np.triu_indices_from(theta_lat_rad, 1)]
    ref_deg, lat_deg = np.degrees(tri_ref), np.degrees(tri_lat)

    x = np.linspace(0, 180, 1000)
    kde_ref = gaussian_kde(ref_deg, bw_method=0.1)
    kde_lat = gaussian_kde(lat_deg, bw_method=0.1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(x, kde_ref(x), label=f"Ref(SNV) (μ={ref_deg.mean():.3g}°)", color="tab:blue", lw=2)
    ax.plot(x, kde_lat(x), label=f"Latent (μ={lat_deg.mean():.3g}°)", color="tab:orange", lw=2)
    ax.fill_between(x, kde_ref(x), kde_lat(x), color="gray", alpha=0.2, label="Difference region")

    ax.set_xlabel("Inter-cluster angle (°)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = Path(out_path)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mds_layout_from_angles(
    theta_rad: np.ndarray,
    out_path: str | Path,
    *,
    seed: Optional[int] = 42,
    show_ticks: bool = False,
) -> None:
    """
    角度行列（radian）を距離行列として MDS により 2D へ射影し、クラスタ配置を可視化して保存する。

    Parameters
    ----------
    theta_rad : np.ndarray, shape (K, K)
        角度行列（radian）。正方行列である必要がある。
        `angle_matrix(C)` の出力をそのまま渡す想定。
    out_path : str | Path
        保存先パス。
    seed : int | None, default 42
        MDS の乱数シード。None の場合は sklearn のデフォルト挙動に従う。
    show_ticks : bool, default False
        True の場合は軸ラベル・目盛り・グリッドを表示する。
        False の場合は目盛り等を消して見た目を簡潔にする（デフォルト）。

    Notes
    -----
    - `theta_rad` を「非ユークリッド距離」だとしても MDS は（最適化として）解を返すが、
      埋め込みの歪みは stress に反映される。図左上（axes 座標）に stress を表示する。
    - 色は `colorcet.glasbey_light` を用いてクラスタ番号ごとに割り当てる。
    - 点のサイズ・枠線・番号テキストは「クラスタ中心の配置図」を強調する設定。
    """
    D = np.asarray(theta_rad, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("theta_rad は正方の 2 次元配列である必要があります。")

    n_clusters = int(D.shape[0])

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=4,
        max_iter=300,
    )
    coords = mds.fit_transform(D)

    colors = list(cc.glasbey_light[:max(n_clusters, 1)])
    cmap = {i: colors[i % len(colors)] for i in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    for i, (x, y) in enumerate(coords):
        ax.scatter(
            x,
            y,
            s=250,
            color=cmap[i],
            label=f"{i}",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.text(
            x,
            y,
            str(i),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    ax.set_aspect("equal", adjustable="datalim")
    if show_ticks:
        ax.set_xlabel("MDS-1")
        ax.set_ylabel("MDS-2")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax.text(
        0.02,
        0.98,
        f"stress={mds.stress_:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    plt.tight_layout()
    p = Path(out_path)
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)