from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


def plot_recon_spectra(
    x_origin: torch.Tensor,
    x_recon: torch.Tensor,
    visible_mask: torch.Tensor,
    *,
    n_blocks: int = 16,
    wavenumber_nm: np.ndarray,
    out_path: Optional[Union[Path, str]] = None,
    max_plots: int = 10,
    seed: int = 42,
    y_min: float = -3.0,
    y_max: float = 3.0,
    figure_size: Tuple[float, float] = (12.0, 15.0),
    x_major: float = 100.0,
    x_minor: float = 20.0,
    show_legend: bool = True,
    dpi: int = 300,
) -> None:
    """
    マスク付き再構成スペクトルをグリッド表示する（入力は (B, C) の Tensor を想定）。

    Parameters
    ----------
    x_origin : torch.Tensor, shape (B, C)
        元スペクトル。
    x_recon : torch.Tensor, shape (B, C)
        再構成スペクトル。
    visible_mask : torch.Tensor, shape (B, C)
        可視マスク。True/1 が可視、False/0 がマスク領域を想定。
    n_blocks : int, default 16
        ブロック（パッチ）数。C が n_blocks で割り切れる必要がある。
    wavenumber_nm : np.ndarray, shape (C,)
        波長[nm] の 1D 配列。
    out_path : Path | str | None, default None
        指定した場合は保存する。
    max_plots : int, default 10
        表示する最大サンプル数。
    seed : int, default 42
        サンプル抽出の乱数シード。
    y_min, y_max : float, default (-3, 3)
        y 軸範囲（SNVを想定）。
    figure_size : tuple of float, default (12, 15)
        図のサイズ（インチ）。
    x_major, x_minor : float, default (100, 20)
        x 軸 major/minor locator（nm単位）。
    show_legend : bool, default True
        凡例を表示するか。
    dpi : int, default 300
        保存時のDPI。
    """
    rng = random.Random(seed)

    # ---- checks ----
    if x_origin.ndim != 2 or x_recon.ndim != 2 or visible_mask.ndim != 2:
        raise ValueError("x_origin/x_recon/visible_mask must be 2D (B, C).")
    if x_origin.shape != x_recon.shape or x_origin.shape != visible_mask.shape:
        raise ValueError(
            f"shape mismatch: x_origin={tuple(x_origin.shape)}, "
            f"x_recon={tuple(x_recon.shape)}, visible_mask={tuple(visible_mask.shape)}"
        )

    B, C = x_origin.shape
    if B == 0:
        raise ValueError("No samples to plot (B=0).")

    wavenumber_nm = np.asarray(wavenumber_nm, dtype=float)
    if wavenumber_nm.ndim != 1 or wavenumber_nm.shape[0] != C:
        raise ValueError(f"wavenumber_nm must be 1D length C={C}, got shape={wavenumber_nm.shape}")
    if not np.all(np.isfinite(wavenumber_nm)):
        raise ValueError("wavenumber_nm contains NaN/Inf.")

    if C % n_blocks != 0:
        raise ValueError(f"seq_len C={C} must be divisible by n_blocks={n_blocks}.")
    block_size = C // n_blocks

    # ---- to numpy ----
    x_origin_np = x_origin.detach().cpu().numpy()
    x_recon_np = x_recon.detach().cpu().numpy()
    vmask_np = visible_mask.detach().cpu().numpy().astype(bool)

    n_sel = min(max_plots, B)
    idxs = rng.sample(range(B), n_sel)

    # 10枚を 5x2（踏襲）。max_plots < 10 でもOK
    n_rows, n_cols = 5, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size, sharex=True)
    axes = axes.ravel()

    proxy_mask = Rectangle(
        (0, 0),
        1,
        1,
        facecolor="lightgray",
        alpha=0.5,
        edgecolor="dimgray",
        linewidth=1.0,
        label="Masked patch",
    )

    x_min, x_max = float(np.min(wavenumber_nm)), float(np.max(wavenumber_nm))

    # ---- common axis styling (plot_refl_spectra に寄せる) ----
    for ax in axes:
        ax.set_xlim(x_min, x_max)  # nmは反転しない
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.10)
        ax.set_xlabel("Wavenumber (nm)")
        ax.set_ylabel("Reflectance (SNV)")

    # ---- draw ----
    for ax_idx in range(n_rows * n_cols):
        ax = axes[ax_idx]
        if ax_idx >= n_sel:
            ax.axis("off")
            continue

        i = idxs[ax_idx]
        x0 = x_origin_np[i]
        xr = x_recon_np[i]
        vm = vmask_np[i]

        masked_flat = ~vm
        block_mask = masked_flat.reshape(n_blocks, block_size).any(axis=1)

        # masked blocks shading
        for b, is_masked in enumerate(block_mask):
            if not is_masked:
                continue

            i0, i1 = b * block_size, (b + 1) * block_size
            xL = float(min(wavenumber_nm[i0], wavenumber_nm[i1 - 1]))
            xR = float(max(wavenumber_nm[i0], wavenumber_nm[i1 - 1]))

            ax.add_patch(
                Rectangle(
                    (xL, y_min),
                    (xR - xL),
                    (y_max - y_min),
                    facecolor="lightgray",
                    alpha=0.5,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (xL, y_min),
                    (xR - xL),
                    (y_max - y_min),
                    facecolor="none",
                    edgecolor="dimgray",
                    linewidth=1.0,
                    zorder=1,
                    antialiased=True,
                )
            )
            ax.vlines([xL, xR], y_min, y_max, colors="dimgray", linewidth=1.0, zorder=2)

        # spectra
        ax.plot(wavenumber_nm, x0, c="k", lw=1.0, label="Original", zorder=3)
        ax.plot(wavenumber_nm, xr, c="r", lw=1.0, label="Reconstructed", zorder=4)

        if show_legend:
            ax.legend(
                handles=[ax.lines[0], ax.lines[1], proxy_mask],
                loc="upper right",
                frameon=False,
            )

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return None