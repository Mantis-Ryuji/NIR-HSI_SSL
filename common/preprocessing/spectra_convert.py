from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def raw_to_refl(
    intensity_cube: np.ndarray,
    white_cube: np.ndarray,
    dark_cube: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    生強度データキューブを白基準・暗基準で補正して反射率キューブに変換する。

    各画素（H, W）で波長方向（C）に対し、次の白暗補正を適用します。

    ``refl_cube = (intensity_cube - dark_cube) / (white_cube - dark_cube)``

    Parameters
    ----------
    intensity_cube : ndarray of shape (H, W, C)
        生強度キューブ ``I``。
    white_cube : ndarray of shape (H, W, C)
        白基準キューブ ``W``。
    dark_cube : ndarray of shape (H, W, C)
        暗基準キューブ ``D``。
    eps : float, default=1e-12
        分母安定化用の微小量。``white_cube - dark_cube`` が 0 に近い波長での
        0 除算を避ける目的で分母に加算します。

    Returns
    -------
    refl_cube : ndarray of shape (H, W, C), dtype=float32
        反射率キューブ。

    Notes
    -----
    - 入力 ``*_cube`` の NaN/Inf は 0 に置換します。
    - 分母は ``(white_cube - dark_cube + eps)`` として 0 除算を回避します（簡易版）。
    - 出力の NaN/Inf も 0 に置換します。
    """
    # -------------------------
    # type / shape checks
    # -------------------------
    if not isinstance(intensity_cube, np.ndarray):
        raise TypeError(f"intensity_cube must be np.ndarray, got {type(intensity_cube)}")
    if not isinstance(white_cube, np.ndarray):
        raise TypeError(f"white_cube must be np.ndarray, got {type(white_cube)}")
    if not isinstance(dark_cube, np.ndarray):
        raise TypeError(f"dark_cube must be np.ndarray, got {type(dark_cube)}")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    if intensity_cube.ndim != 3:
        raise ValueError(f"intensity_cube must have shape (H, W, C), got {intensity_cube.shape}")
    if white_cube.ndim != 3:
        raise ValueError(f"white_cube must have shape (H, W, C), got {white_cube.shape}")
    if dark_cube.ndim != 3:
        raise ValueError(f"dark_cube must have shape (H, W, C), got {dark_cube.shape}")

    if white_cube.shape != intensity_cube.shape:
        raise ValueError(f"white_cube shape must match intensity_cube, got {white_cube.shape} vs {intensity_cube.shape}")
    if dark_cube.shape != intensity_cube.shape:
        raise ValueError(f"dark_cube shape must match intensity_cube, got {dark_cube.shape} vs {intensity_cube.shape}")

    H, W, C = intensity_cube.shape
    if H <= 0 or W <= 0 or C <= 0:
        raise ValueError(f"invalid shape (H, W, C)={intensity_cube.shape}")

    # -------------------------
    # compute reflectance cube
    # -------------------------
    intensity_cube = np.nan_to_num(
        intensity_cube.astype(np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    white_cube = np.nan_to_num(
        white_cube.astype(np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    dark_cube = np.nan_to_num(
        dark_cube.astype(np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    denom_cube = white_cube - dark_cube
    refl_cube = (intensity_cube - dark_cube) / (denom_cube + float(eps))
    refl_cube = np.nan_to_num(refl_cube, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return refl_cube


def _check_nc(x_nc: np.ndarray, *, name: str) -> tuple[int, int]:
    """shape (N, C) を厳密チェックする。"""
    if not isinstance(x_nc, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(x_nc)}")
    if x_nc.ndim != 2:
        raise ValueError(f"{name} must have shape (N, C), got {x_nc.shape}")
    N, C = x_nc.shape
    if N <= 0 or C <= 0:
        raise ValueError(f"{name} must have positive shape (N, C), got {x_nc.shape}")
    return N, C


def refl2abs_log10(refl_nc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    反射率行列を擬似吸光度（pseudo-absorbance）に変換する（log10）。

    ``abs_nc = -log10(refl_nc)``

    Parameters
    ----------
    refl_nc : ndarray of shape (N, C)
        反射率 ``R``（サンプル, 波長）。
    eps : float, default=1e-12
        下限クランプ値。``refl_nc <= 0`` を ``eps`` に置換して対数発散を防ぎます。

    Returns
    -------
    abs_nc : ndarray of shape (N, C), dtype=float32
        擬似吸光度。

    Notes
    -----
    - 入力の NaN/Inf は 0 に置換します。
    - ``refl_nc`` は ``max(refl_nc, eps)`` にクランプしてから ``-log10`` を取ります。
    - 出力の NaN/Inf も 0 に置換します。
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    _check_nc(refl_nc, name="refl_nc")

    refl_nc = np.nan_to_num(refl_nc.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    refl_nc = np.maximum(refl_nc, float(eps))

    abs_nc = -np.log10(refl_nc)
    abs_nc = np.nan_to_num(abs_nc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return abs_nc


def refl2abs_km(refl_nc: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    反射率行列を Kubelka-Munk 変換（K/S 近似）に変換する。

    ``km_nc = (1 - refl_nc)^2 / (2 * refl_nc)``

    Parameters
    ----------
    refl_nc : ndarray of shape (N, C)
        反射率 ``R``（サンプル, 波長）。
    eps : float, default=1e-12
        下限クランプ値。``refl_nc <= 0`` を ``eps`` に置換して発散を防ぎます。

    Returns
    -------
    km_nc : ndarray of shape (N, C), dtype=float32
        Kubelka-Munk 変換結果。

    Notes
    -----
    - 入力の NaN/Inf は 0 に置換します。
    - ``refl_nc`` は ``max(refl_nc, eps)`` にクランプしてから計算します。
    - 出力の NaN/Inf も 0 に置換します。
    """
    if eps <= 0:
        raise ValueError("eps must be > 0")
    _check_nc(refl_nc, name="refl_nc")

    refl_nc = np.nan_to_num(refl_nc.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    refl_nc = np.maximum(refl_nc, float(eps))

    km_nc = (1.0 - refl_nc) ** 2 / (2.0 * refl_nc)
    km_nc = np.nan_to_num(km_nc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return km_nc


def abs_to_deriv_sg(
    abs_nc: np.ndarray,
    *,
    window_length: int = 7,
    polyorder: int = 3,
    deriv: int = 2,
    delta: float = 1.0,
    mode: str = "interp",
) -> np.ndarray:
    """
    擬似吸光度行列に Savitzky-Golay フィルタを適用して微分スペクトルを得る。

    Parameters
    ----------
    abs_nc : ndarray of shape (N, C)
        入力スペクトル（擬似吸光度など、サンプル, 波長）。
    window_length : int, default=7
        SG フィルタの窓長（奇数）。``window_length >= polyorder + 2`` が必要。
    polyorder : int, default=3
        SG フィルタの多項式次数（``polyorder < window_length``）。
    deriv : int, default=2
        微分次数（0: 平滑化、1: 一次、2: 二次, ...）。
    delta : float, default=1.0
        波長刻み。微分スケールに影響するため、等間隔で ``Δλ`` が分かるなら
        ``delta=Δλ`` を推奨。
    mode : str, default="interp"
        端点処理。``scipy.signal.savgol_filter`` の ``mode`` をそのまま渡す。

    Returns
    -------
    deriv_nc : ndarray of shape (N, C), dtype=float32
        SG による微分後スペクトル。

    Notes
    -----
    - 入力の NaN/Inf は 0 に置換してから処理します。
    - 出力の NaN/Inf も 0 に置換します。
    """
    _check_nc(abs_nc, name="abs_nc")

    if not isinstance(window_length, (int, np.integer)):
        raise TypeError(f"window_length must be int, got {type(window_length)}")
    if not isinstance(polyorder, (int, np.integer)):
        raise TypeError(f"polyorder must be int, got {type(polyorder)}")
    if not isinstance(deriv, (int, np.integer)):
        raise TypeError(f"deriv must be int, got {type(deriv)}")
    if window_length <= 0:
        raise ValueError("window_length must be > 0")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if polyorder >= window_length:
        raise ValueError("polyorder must be < window_length")
    if window_length < polyorder + 2:
        raise ValueError("window_length must be >= polyorder + 2")
    if deriv < 0:
        raise ValueError("deriv must be >= 0")
    if delta <= 0:
        raise ValueError("delta must be > 0")

    abs_nc = np.nan_to_num(abs_nc.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    deriv_nc = savgol_filter(
        abs_nc,
        window_length=int(window_length),
        polyorder=int(polyorder),
        deriv=int(deriv),
        delta=float(delta),
        axis=1,  # (N, C) の C 方向
        mode=mode,
    )

    deriv_nc = np.nan_to_num(deriv_nc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return deriv_nc