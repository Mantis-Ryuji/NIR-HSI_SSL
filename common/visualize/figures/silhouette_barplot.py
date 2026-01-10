from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[Sequence[float], np.ndarray]


def _to_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return x


def _to_1d_int(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(x.astype(float, copy=False))):
        raise ValueError(f"{name} contains NaN/Inf.")
    if not np.issubdtype(x.dtype, np.integer):
        x = x.astype(int, copy=False)
    return x


def _cluster_mean_and_std_of_means(scores: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, float]:
    """
    Compute:
      mean = mean(scores)
      std  = std( mean(scores[cluster==k]) over k ), ddof=1

    If the number of non-empty clusters < 2, std is returned as 0.0.
    """
    if scores.shape != cluster_ids.shape:
        raise ValueError(
            f"scores and cluster_ids must have same shape, got {scores.shape} vs {cluster_ids.shape}"
        )

    mean_all = float(np.mean(scores))

    uniq = np.unique(cluster_ids)
    per_cluster_means = []
    for k in uniq:
        mask = cluster_ids == k
        if np.any(mask):
            per_cluster_means.append(float(np.mean(scores[mask])))

    if len(per_cluster_means) < 2:
        std_means = 0.0
    else:
        std_means = float(np.std(np.asarray(per_cluster_means, dtype=float), ddof=1))

    return mean_all, std_means


def _validate_dict_inputs(
    scores: Mapping[str, ArrayLike],
    cluster_ids: Mapping[str, ArrayLike],
    splits: Tuple[str, ...],
    prefix: str,
) -> None:
    for sp in splits:
        if sp not in scores:
            raise KeyError(f"Missing split '{sp}' in {prefix}_scores")
        if sp not in cluster_ids:
            raise KeyError(f"Missing split '{sp}' in {prefix}_cluster_ids")


def plot_silhouette_bar(
    *,
    ref_scores: Mapping[str, ArrayLike],
    ref_cluster_ids: Mapping[str, ArrayLike],
    latent_scores: Mapping[str, ArrayLike],
    latent_cluster_ids: Mapping[str, ArrayLike],
    out_path: Optional[Union[Path, str]] = None,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    labels: Tuple[str, str] = ("ref_snv", "latent"),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ylabel: str = "Silhouette score (cosine)",
    dpi: int = 300,
    annotate: bool = True,
    annotation_fmt: str = "{m:.3g}±{s:.3g}",
    ylim: Tuple[float, float] = (-1.0, 1.0),
    bottom: float = -1.0,
    bar_width: float = 0.36,
    capsize: float = 4.0,
    grid: bool = True,
    major_yticks: Optional[float] = 0.5,
    minor_yticks: Optional[float] = 0.1,
) -> None:
    """
    Split-wise silhouette bar chart (cosine silhouette).

    Parameters
    ----------
    ref_scores, latent_scores : Mapping[str, ArrayLike]
        各 split の **各点シルエット** (s_i) の 1D 配列。
    ref_cluster_ids, latent_cluster_ids : Mapping[str, ArrayLike]
        各 split の **各点クラスタID** (1D int)。
        エラーバーは「クラスタ平均シルエット mean_k の split 内ばらつき」
        std({mean_k}) で計算する。
    out_path : Path | str | None, default None
        指定した場合は保存する。
    splits : tuple[str, ...], default ("train","val","test")
        描画順。
    labels : tuple[str, str], default ("ref_snv","latent")
        2系列の凡例名。
    colors : tuple[str, str], default ("tab:blue","tab:orange")
        2系列の色。
    ylabel : str
        y軸ラベル。
    dpi : int, default 300
        保存時のDPI。
    annotate : bool, default True
        棒の上に mean±std を表示する。
    annotation_fmt : str, default "{m:.3g}±{s:.3g}"
        注釈のフォーマット。
    ylim : tuple[float, float], default (-1, 1)
        y軸範囲。
    bottom : float, default -1
        棒のベースライン。silhouette の範囲 [-1,1] を前提に -1 を推奨。
    bar_width : float, default 0.36
        棒幅。
    capsize : float, default 4
        エラーバー capsize。
    grid : bool, default True
        グリッドを表示するか。
    major_yticks, minor_yticks : float | None
        y軸の tick 間隔。None の場合は設定しない。

    Notes
    -----
    - bar は `bottom` から `mean-bottom` の高さで描画する（mean が負でも見やすい）。
    - エラーバーは split 内クラスタ平均のばらつき（std of cluster means）であり、
      点ごとの標準偏差ではない。
    """
    # ---- validate keys ----
    _validate_dict_inputs(ref_scores, ref_cluster_ids, splits, prefix="ref")
    _validate_dict_inputs(latent_scores, latent_cluster_ids, splits, prefix="latent")

    if len(labels) != 2 or len(colors) != 2:
        raise ValueError("labels and colors must be length 2.")

    # ---- compute stats per split ----
    ref_means, ref_stds = [], []
    lat_means, lat_stds = [], []

    for sp in splits:
        rs = _to_1d_float(ref_scores[sp], f"ref_scores[{sp}]")
        rc = _to_1d_int(ref_cluster_ids[sp], f"ref_cluster_ids[{sp}]")
        ls = _to_1d_float(latent_scores[sp], f"latent_scores[{sp}]")
        lc = _to_1d_int(latent_cluster_ids[sp], f"latent_cluster_ids[{sp}]")

        if rs.shape[0] != rc.shape[0]:
            raise ValueError(f"ref split '{sp}': scores and cluster_ids length mismatch.")
        if ls.shape[0] != lc.shape[0]:
            raise ValueError(f"latent split '{sp}': scores and cluster_ids length mismatch.")

        rm, rstd = _cluster_mean_and_std_of_means(rs, rc)
        lm, lstd = _cluster_mean_and_std_of_means(ls, lc)

        ref_means.append(rm)
        ref_stds.append(rstd)
        lat_means.append(lm)
        lat_stds.append(lstd)

    ref_means = np.asarray(ref_means, dtype=float)
    ref_stds = np.asarray(ref_stds, dtype=float)
    lat_means = np.asarray(lat_means, dtype=float)
    lat_stds = np.asarray(lat_stds, dtype=float)

    # ---- sanity check for silhouette range (warn by clipping optional) ----
    # (silhouette ideally in [-1, 1], but numerical issues can slightly overshoot)
    # We don't clip; we just let the plot show it within ylim.
    y0, y1 = float(ylim[0]), float(ylim[1])
    if y0 >= y1:
        raise ValueError(f"ylim must satisfy ylim[0] < ylim[1], got {ylim}")

    # ---- plot ----
    n = len(splits)
    x = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=dpi)

    # Heights from bottom (for visibility of negative values)
    ref_heights = ref_means - bottom
    lat_heights = lat_means - bottom

    # Ensure non-negative heights (in case bottom is not <= min(mean))
    # This keeps matplotlib from drawing inverted bars unexpectedly.
    ref_heights = np.maximum(ref_heights, 0.0)
    lat_heights = np.maximum(lat_heights, 0.0)

    bars_ref = ax.bar(
        x - bar_width / 2,
        ref_heights,
        bottom=bottom,
        yerr=ref_stds,
        capsize=capsize,
        width=bar_width,
        color=colors[0],
        align="center",
        label=labels[0],
        linewidth=0.0,
    )
    bars_lat = ax.bar(
        x + bar_width / 2,
        lat_heights,
        bottom=bottom,
        yerr=lat_stds,
        capsize=capsize,
        width=bar_width,
        color=colors[1],
        align="center",
        label=labels[1],
        linewidth=0.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(splits))
    ax.set_ylabel(ylabel)
    ax.set_ylim(y0, y1)

    if major_yticks is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(major_yticks))
    if minor_yticks is not None:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_yticks))

    if grid:
        ax.grid(True, which="major", axis="y", alpha=0.25)
        ax.grid(True, which="minor", axis="y", alpha=0.10)

    ax.legend(loc="lower left", frameon=False)

    if annotate:
        # place above the error bar top; clamp to ylim for readability
        pad = 0.02 * (y1 - y0)
        for rect, m, s in zip(bars_ref, ref_means, ref_stds):
            y = m + s
            y_text = min(y + pad, y1 - 0.5 * pad)
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_text,
                annotation_fmt.format(m=m, s=s),
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )
        for rect, m, s in zip(bars_lat, lat_means, lat_stds):
            y = m + s
            y_text = min(y + pad, y1 - 0.5 * pad)
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_text,
                annotation_fmt.format(m=m, s=s),
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return None