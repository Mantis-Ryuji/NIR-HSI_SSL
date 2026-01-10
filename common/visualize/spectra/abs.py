from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator


def plot_abs_spectra(
    wavenumber_nm: np.ndarray,
    abs_nc: np.ndarray,
    abs_snv_nc: np.ndarray,
    cmap: ListedColormap,
    *,
    out_path: Optional[str] = None,
    figure_size: Tuple[float, float] = (12, 5),
    show_legend: bool = True,
    legend_name: str = "cluster",
    dpi: int = 300,
) -> None:
    """
    疑似吸光度（Absorbance）スペクトルとSNV後スペクトルを、波数(cm^-1)軸で並べて可視化する。

    Parameters
    ----------
    wavenumber_nm : np.ndarray, shape (C,)
        波長[nm]。0より大きい値のみを想定。
    abs_nc : np.ndarray, shape (N, C)
        疑似吸光度スペクトル（例: 各クラスタの平均スペクトル）。
        一般に `abs = -log10(refl)` などで算出したものを想定（ここでは値域は固定しない）。
    abs_snv_nc : np.ndarray, shape (N, C)
        SNV後の疑似吸光度スペクトル。
    cmap : matplotlib.colors.ListedColormap
        N 本の線に割り当てる色（cmap.colors を使用）。
        N > len(cmap.colors) の場合は循環利用する。
    out_path : str | None, default None
        保存先パス。指定した場合は `fig.savefig(out_path, ...)` で保存する。
        None の場合は保存しない。
    figure_size : tuple of float, default (12, 5)
        図のサイズ（インチ）。
    show_legend : bool, default True
        凡例を表示するか。
    legend_name : str, default "cluster"
        凡例ラベルの接頭辞（例: "cluster0", "cluster1", ...）。
    dpi : int, default 300
        保存時のDPI。

    Notes
    -----
    - x 軸は nm -> cm^-1 に変換し、高波数→低波数（右向きに減少）で表示します。
    """
    # ---- shape / dtype checks ----
    wavenumber_nm = np.asarray(wavenumber_nm)
    abs_nc = np.asarray(abs_nc)
    abs_snv_nc = np.asarray(abs_snv_nc)

    if wavenumber_nm.ndim != 1:
        raise ValueError(f"wavenumber_nm must be 1D (C,), got shape={wavenumber_nm.shape}")
    if abs_nc.ndim != 2:
        raise ValueError(f"abs_nc must be 2D (N,C), got shape={abs_nc.shape}")
    if abs_snv_nc.ndim != 2:
        raise ValueError(f"abs_snv_nc must be 2D (N,C), got shape={abs_snv_nc.shape}")

    N, C = abs_nc.shape
    if abs_snv_nc.shape != (N, C):
        raise ValueError(
            f"abs_snv_nc must have same shape as abs_nc: expected {(N, C)}, got {abs_snv_nc.shape}"
        )
    if wavenumber_nm.shape[0] != C:
        raise ValueError(
            f"wavenumber_nm length must match C: expected C={C}, got {wavenumber_nm.shape[0]}"
        )
    if not np.all(np.isfinite(wavenumber_nm)):
        raise ValueError("wavenumber_nm contains NaN/Inf.")
    if np.any(wavenumber_nm <= 0):
        raise ValueError("wavenumber_nm must be > 0 to convert to wavenumber (cm^-1).")

    # ---- nm -> cm^-1 ----
    wavenumber_cm = 1e7 / wavenumber_nm  # [cm^-1]
    x_min, x_max = float(np.min(wavenumber_cm)), float(np.max(wavenumber_cm))

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    ax11, ax12 = axes[0], axes[1]

    # axis styling (do once)
    for ax in (ax11, ax12):
        ax.set_xlim(x_max, x_min)  # reverse axis
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.10)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")

    ax11.set_ylabel("Absorbance")
    ax11.set_ylim(0.0, 1.0)
    ax12.set_ylabel("SNV(Absorbance)")
    ax12.set_ylim(-3.0, 3.0)

    colors = getattr(cmap, "colors", None)
    if colors is None or len(colors) == 0:
        raise ValueError("cmap.colors is empty. Please pass a ListedColormap with colors.")

    n_colors = len(colors)

    for idx in range(N):
        color = colors[idx % n_colors]
        label = f"{legend_name}{int(idx)}"
        ax11.plot(wavenumber_cm, abs_nc[idx, :], c=color, lw=1.0, label=label)
        ax12.plot(wavenumber_cm, abs_snv_nc[idx, :], c=color, lw=1.0, label=label)

    if show_legend:
        ax12.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return None