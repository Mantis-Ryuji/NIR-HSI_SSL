from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_refl_norm_map(
    refl_cube: np.ndarray,
    binary_map: np.ndarray,
    indices_n: np.ndarray,
    out_path: str,
    *,
    show_downsampled_points: bool = True,
    cmap: str = "jet",
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
    反射率キューブからノルムマップを作成し、余白最小で保存する（ダウンサンプル点の可視化付き）。

    ``refl_cube`` （shape ``(H, W, C)``）から各画素の L2 ノルム

    ``norm_map[h, w] = || refl_cube[h, w, :] ||_2``

    を計算し、``binary_map==False`` の画素は ``NaN`` にして透明表示（背景抜き）します。
    表示レンジは ``vmin=0``、``vmax=sqrt(C)`` で固定します（反射率が概ね [0,1] を想定）。

    さらに、ダウンサンプリングで選ばれた点（``indices_n``）を黒点で重ね描きし、
    右上に凡例を表示します。

    Parameters
    ----------
    refl_cube : ndarray of shape (H, W, C)
        反射率データキューブ。
    binary_map : ndarray of shape (H, W), dtype=bool
        描画対象領域マップ。False の画素は ``NaN`` として透明表示する。
    indices_n : ndarray of shape (M,), dtype=int64
        ``(H, W)`` をフラット化（row-major）したときの選択インデックス。
        すなわち ``idx = h * W + w`` を想定する。
    out_path : str
        出力先パス（.png 等）。
    show_downsampled_points : bool, default=True
        True の場合、``indices_n`` の点を黒点で描画し、凡例も付ける。
    cmap : str, default="jet"
        カラーマップ名。
    width_inch : float, default=8.0
        出力 figure の横幅（inch）。縦幅は ``H/W`` に合わせて自動決定。
    add_cbar : bool, default=True
        True の場合、カラーバーを描画する。
    cbar_label : str, default=None
        カラーバーのラベル。
    cbar_fontsize : int, default=20
        カラーバーのラベル・目盛りフォントサイズ。
    cbar_ticks : list of float, default=None
        カラーバーの ticks を明示指定する。
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
    - マップは左右反転（``norm_map[:, ::-1]``）して描画します。
    - 保存は ``dpi=300, bbox_inches="tight", pad_inches=0`` で行います。
    """
    # -------------------------
    # type / shape checks
    # -------------------------
    if not isinstance(refl_cube, np.ndarray):
        raise TypeError(f"refl_cube must be np.ndarray, got {type(refl_cube)}")
    if refl_cube.ndim != 3:
        raise ValueError(f"refl_cube must have shape (H, W, C), got {refl_cube.shape}")

    H, W, C = refl_cube.shape
    if H <= 0 or W <= 0 or C <= 0:
        raise ValueError(f"refl_cube must have positive shape (H, W, C), got {refl_cube.shape}")

    if not isinstance(binary_map, np.ndarray):
        raise TypeError(f"binary_map must be np.ndarray, got {type(binary_map)}")
    if binary_map.shape != (H, W):
        raise ValueError(f"binary_map must have shape (H, W)={(H, W)}, got {binary_map.shape}")
    if binary_map.dtype != np.bool_ and binary_map.dtype != bool:
        raise TypeError(f"binary_map must be bool array, got dtype={binary_map.dtype}")

    if not isinstance(indices_n, np.ndarray):
        raise TypeError(f"indices_n must be np.ndarray, got {type(indices_n)}")
    if indices_n.ndim != 1:
        raise ValueError(f"indices_n must be 1D array with shape (M,), got {indices_n.shape}")
    if indices_n.size > 0 and not np.issubdtype(indices_n.dtype, np.integer):
        raise TypeError(f"indices_n must be integer array, got dtype={indices_n.dtype}")

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
    # make norm map (H, W)
    # -------------------------
    refl_cube = np.nan_to_num(refl_cube.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    norm_map = np.linalg.norm(refl_cube, axis=2).astype(np.float32, copy=False)  # (H, W)

    # mask out background as NaN (transparent)
    norm_map = norm_map.copy()
    norm_map[~binary_map] = np.nan

    # fixed range
    vmin = 0.0
    vmax = float(np.sqrt(C))

    # -------------------------
    # figure
    # -------------------------
    aspect = H / W
    figsize = (float(width_inch), float(width_inch) * float(aspect))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=figsize)

    if add_cbar:
        main_ax = fig.add_axes([0, 0, 1 - cbar_fraction - cbar_pad, 1])
        cax = fig.add_axes([1 - cbar_fraction, cbar_ypos, cbar_fraction, cbar_height])
    else:
        main_ax = fig.add_axes([0, 0, 1, 1])
        cax = None

    im = main_ax.imshow(norm_map[:, ::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    main_ax.set_axis_off()

    # -------------------------
    # overlay downsampled points
    # -------------------------
    if show_downsampled_points and indices_n.size > 0:
        idx = np.asarray(indices_n, dtype=np.int64)

        if (idx < 0).any() or (idx >= H * W).any():
            raise ValueError("indices_n contains out-of-range indices for flattened (H, W)")

        ys = idx // W
        xs = idx % W

        # because we draw norm_map[:, ::-1], x is flipped
        xs_plot = (W - 1) - xs

        main_ax.scatter(
            xs_plot,
            ys,
            s=8,
            c="k",
            marker="o",
            linewidths=0,
            label="downsampled",
        )
        main_ax.legend(loc="upper right", frameon=True)

    # -------------------------
    # colorbar
    # -------------------------
    if add_cbar and cax is not None:
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=int(cbar_fontsize))
        cbar.ax.tick_params(labelsize=int(cbar_fontsize))
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)