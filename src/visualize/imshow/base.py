from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_tight_image(
    value_map: np.ndarray,
    out_path: str,
    *,
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    width_inch: float = 8.0,
    add_cbar: bool = True,
    cbar_label: Optional[str] = None,
    cbar_fontsize: int = 20,
    cbar_ticks: Optional[List[float]] = None,
    cbar_fraction: float = 0.1,
    cbar_pad: float = 0.02,
    cbar_height: float = 0.8,
    cbar_ypos: float = 0.1,
) -> None:
    """
    2次元マップ（ヒートマップ等）を「余白最小（tight）」で保存する（任意でカラーバー付き）。

    本関数は **2次元配列** ``value_map`` （shape ``(H, W)``）のみを受け付けます。
    軸・余白を極力なくして保存し、必要に応じてカラーバーを付与します。

    Parameters
    ----------
    value_map : ndarray of shape (H, W)
        保存対象の2次元マップ（例：SNRマップ、ラベルマップ、距離マップ等）。
        NaN/Inf を含んでもよい（``vmin/vmax`` 推定時は NaN を無視）。
    out_path : str
        出力先パス（拡張子でフォーマットが決まる：.png, .jpg など）。
    cmap : str, default="jet"
        カラーマップ名。
    vmin : float, default=None
        表示下限。None の場合は ``nanmin(value_map)``。
    vmax : float, default=None
        表示上限。None の場合は ``nanmax(value_map)``。
    width_inch : float, default=8.0
        出力 figure の横幅（inch）。縦幅は ``H/W`` に合わせて自動決定。
    add_cbar : bool, default=True
        True の場合、カラーバーを描画する。
    cbar_label : str, default=None
        カラーバーのラベル。None の場合は付与しない。
    cbar_fontsize : int, default=20
        カラーバーのラベル・目盛りフォントサイズ。
    cbar_ticks : list of float, default=None
        カラーバーの ticks を明示指定する。None の場合は Matplotlib に任せる。
    cbar_fraction : float, default=0.1
        図全体に対するカラーバー領域の横幅（0〜1）。
    cbar_pad : float, default=0.02
        主画像領域とカラーバー領域の隙間（0〜1）。
    cbar_height : float, default=0.8
        カラーバー領域の高さ（0〜1）。
    cbar_ypos : float, default=0.1
        カラーバー領域の下端位置（0〜1）。

    Returns
    -------
    None

    Notes
    -----
    - マップは左右反転（``value_map[:, ::-1]``）して描画します。
    - 保存は ``dpi=300, bbox_inches="tight", pad_inches=0`` で行います。
    """
    # -------------------------
    # type / shape checks
    # -------------------------
    if not isinstance(value_map, np.ndarray):
        raise TypeError(f"value_map must be np.ndarray, got {type(value_map)}")
    if value_map.ndim != 2:
        raise ValueError(f"value_map must be 2D array with shape (H, W), got shape={value_map.shape}")

    H, W = value_map.shape
    if H <= 0 or W <= 0:
        raise ValueError(f"value_map must have positive shape (H, W), got {value_map.shape}")

    if not isinstance(out_path, str) or len(out_path) == 0:
        raise TypeError("out_path must be a non-empty str")

    if width_inch <= 0:
        raise ValueError("width_inch must be > 0")

    if not (0.0 < cbar_fraction < 1.0):
        raise ValueError("cbar_fraction must be in (0, 1)")
    if not (0.0 <= cbar_pad < 1.0):
        raise ValueError("cbar_pad must be in [0, 1)")
    if not (0.0 < cbar_height <= 1.0):
        raise ValueError("cbar_height must be in (0, 1]")
    if not (0.0 <= cbar_ypos <= 1.0):
        raise ValueError("cbar_ypos must be in [0, 1]")
    if cbar_ypos + cbar_height > 1.0 + 1e-12:
        raise ValueError("cbar_ypos + cbar_height must be <= 1")

    # -------------------------
    # figure size
    # -------------------------
    aspect = H / W
    figsize = (float(width_inch), float(width_inch) * float(aspect))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=figsize)

    if add_cbar:
        main_ax = fig.add_axes([0, 0, 1 - cbar_fraction - cbar_pad, 1]) # type: ignore
        cax = fig.add_axes([1 - cbar_fraction, cbar_ypos, cbar_fraction, cbar_height]) # type: ignore
    else:
        main_ax = fig.add_axes([0, 0, 1, 1]) # type: ignore
        cax = None

    # -------------------------
    # vmin/vmax
    # -------------------------
    if vmin is None:
        vmin = float(np.nanmin(value_map))
    if vmax is None:
        vmax = float(np.nanmax(value_map))

    im = main_ax.imshow(value_map[:, ::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    main_ax.set_axis_off()

    if add_cbar and cax is not None:
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=int(cbar_fontsize))
        cbar.ax.tick_params(labelsize=int(cbar_fontsize))
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)