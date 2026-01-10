from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


ArrayLike = Union[Sequence[float], np.ndarray]
SCSMetric = Literal["scs_intra", "scs_inter"]


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf.")
    return x


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    m = float(np.mean(x))
    if x.size < 2:
        return m, 0.0
    return m, float(np.std(x, ddof=1))


def _extract_metric_per_split(
    scores: Dict[str, Dict[str, list[Dict[str, Any]]]],
    *,
    space: Literal["ref_snv", "latent"],
    split: str,
    metric: SCSMetric,
) -> np.ndarray:
    if space not in scores:
        raise KeyError(f"Missing '{space}' in scores keys={list(scores.keys())}")
    if split not in scores[space]:
        raise KeyError(f"Missing split '{split}' in scores['{space}']")

    rows = scores[space][split]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"scores['{space}']['{split}'] must be non-empty list[dict]")

    vals: list[float] = []
    for r in rows:
        if not isinstance(r, dict):
            raise TypeError(f"Each row must be dict, got {type(r)}")
        if metric not in r:
            raise KeyError(f"Missing '{metric}' in a row: keys={list(r.keys())}")
        v = r[metric]
        if v is None:
            continue
        vals.append(float(v))

    return _as_1d_float(np.asarray(vals, dtype=float), f"{space}:{split}:{metric}")


def plot_scs_bar(
    *,
    scores: Dict[str, Dict[str, list[Dict[str, Any]]]],
    out_path: Union[str, Path],
    metric: SCSMetric = "scs_intra",
    splits: Tuple[str, str, str] = ("train", "val", "test"),
    labels: Tuple[str, str] = ("ref_snv", "latent"),
    colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
    ylabel: str | None = None,
    y_min: float = 0.0,
    y_max: float = 1.0,
    dpi: int = 300,
) -> None:
    """
    Split-wise SCS bar chart (ref_snv vs latent).

    Parameters
    ----------
    scores : dict
        **要求する形式**：

        `scores` は「空間 → split → サンプル行(list[dict])」のネスト構造を持つ辞書。

        - 第1階層キー（必須）: `"ref_snv"` と `"latent"`
        - 第2階層キー（必須）: `splits` で指定する split 名（例: `"train"`, `"val"`, `"test"`）
        - 値: 各 split に属する **サンプルごとのスコア行** の `list[dict]`

        各行(dict)は少なくとも以下のキーを持つ必要があります：
        - `"scs_intra"` : float | None
        - `"scs_inter"` : float | None

        （`"sample"` などのメタ情報キーは任意で追加してOK）

        例
        --
        scores = {
            "ref_snv": {
                "train": [
                    {"sample": "A001", "scs_intra": 0.82, "scs_inter": 0.91},
                    {"sample": "A002", "scs_intra": 0.79, "scs_inter": 0.88},
                ],
                "val":  [...],
                "test": [...],
            },
            "latent": {
                "train": [
                    {"sample": "A001", "scs_intra": 0.86, "scs_inter": 0.93},
                    {"sample": "A002", "scs_intra": 0.83, "scs_inter": 0.90},
                ],
                "val":  [...],
                "test": [...],
            },
        }

        備考:
        - `row[metric] is None` の行は欠損としてスキップします（その結果、全行 None だとエラー）。

    out_path : str | Path
        保存先パス。
    metric : {"scs_intra","scs_inter"}, default "scs_intra"
        描画する指標。
    splits : tuple[str, str, str], default ("train","val","test")
        描画順（このキーが scores 内に存在する必要がある）。
    labels : tuple[str, str], default ("ref_snv","latent")
        凡例ラベル（棒2系列）。
    colors : tuple[str, str], default ("tab:blue","tab:orange")
        棒2系列の色。
    ylabel : str | None
        y軸ラベル。None の場合は `"SCS (<metric>)"`。
    y_min, y_max : float
        y軸範囲。
    dpi : int
        保存時のDPI。

    Notes
    -----
    - エラーバーの std は「各サンプルの SCS の split 内ばらつき（サンプル標準偏差）」。
    - テキストはエラーバー上端に mean±std (.3g)。
    """
    if ylabel is None:
        ylabel = f"SCS ({metric})"

    # ---- compute mean/std per split ----
    ref_means, ref_stds = [], []
    lat_means, lat_stds = [], []
    for sp in splits:
        r = _extract_metric_per_split(scores, space="ref_snv", split=sp, metric=metric)
        l = _extract_metric_per_split(scores, space="latent", split=sp, metric=metric)

        rm, rs = _mean_std(r)
        lm, ls = _mean_std(l)

        ref_means.append(rm)
        ref_stds.append(rs)
        lat_means.append(lm)
        lat_stds.append(ls)

    ref_means = np.asarray(ref_means, dtype=float)
    ref_stds = np.asarray(ref_stds, dtype=float)
    lat_means = np.asarray(lat_means, dtype=float)
    lat_stds = np.asarray(lat_stds, dtype=float)

    # ---- plot ----
    n = len(splits)
    x = np.arange(n, dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=dpi)

    bars_ref = ax.bar(
        x - width / 2,
        ref_means,
        yerr=ref_stds,
        capsize=5,
        width=width,
        color=colors[0],
        align="center",
        label=labels[0],
    )
    bars_lat = ax.bar(
        x + width / 2,
        lat_means,
        yerr=lat_stds,
        capsize=5,
        width=width,
        color=colors[1],
        align="center",
        label=labels[1],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(list(splits))
    ax.set_ylabel(ylabel)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="lower left")

    # annotations at errorbar top: mean±std
    pad = 0.02 * (y_max - y_min + 1e-12)
    for rect, m, s in zip(bars_ref, ref_means, ref_stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            m + s + pad,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
    for rect, m, s in zip(bars_lat, lat_means, lat_stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            m + s + pad,
            f"{m:.3g}±{s:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return None