from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu


def binarization(
    intensity_cube: np.ndarray,
    *,
    margin_px: int = 1,
) -> Dict[str, Any]:
    """
    強度キューブから木材領域マップを推定する（L2ノルム + Otsu + 内側マージン）。

    本関数は、強度データキューブ ``intensity_cube`` （shape ``(H, W, C)``）から、
    各画素のスペクトル強度の大きさを表す L2 ノルムマップ

    ``norm_map[h, w] = || intensity_cube[h, w, :] ||_2``

    を計算し、そのノルムマップに対して Otsu の自動閾値を適用して木材領域を粗く二値化します。
    さらに、境界近傍ではスペクトル混合が起きやすいため、推定マップの
    「境界から ``margin_px`` ピクセル以上内側」に限定したマップを返します。

    Parameters
    ----------
    intensity_cube : ndarray of shape (H, W, C)
        強度データキューブ（生強度や補正前後の強度など）。
        NaN/Inf を含んでもよい（内部で 0 に置換する）。
    margin_px : int, default=1
        境界から内側へ削る幅（ピクセル）。0 の場合は境界削除を行わない。
        値が大きいほど境界混合の影響は減るが、採用領域が小さくなる。

    Returns
    -------
    out : dict
        結果を格納した辞書。

        - ``"norm_map"`` : ndarray of shape (H, W), dtype=float32
            L2 ノルムマップ。NaN/Inf は 0 に置換済み。
        - ``"binary_map"`` : ndarray of shape (H, W), dtype=bool
            内側マージン適用後の木材領域マップ。True が採用画素。

    Notes
    -----
    - 二値化はあくまで粗推定（背景/試料がノルムで分離できることを仮定）であり、
      照明ムラや測定条件に依存して閾値が不安定になる場合があります。
    - ``distance_transform_edt`` は「True 領域内の各画素について、最寄りの False までの距離」を返します。
      そのため ``dist_map >= margin_px`` とすることで、境界から指定距離以上内側の画素だけを残せます。
    - ``norm_map`` が定数（``max == min``）の場合、Otsu が無意味なので ``binary_map`` は全 False を返します。
    """
    # -------------------------
    # type / shape checks
    # -------------------------
    if not isinstance(intensity_cube, np.ndarray):
        raise TypeError(f"intensity_cube must be np.ndarray, got {type(intensity_cube)}")
    if intensity_cube.ndim != 3:
        raise ValueError(f"intensity_cube must have shape (H, W, C), got {intensity_cube.shape}")

    H, W, C = intensity_cube.shape
    if H <= 0 or W <= 0 or C <= 0:
        raise ValueError(f"invalid shape (H, W, C)={intensity_cube.shape}")

    if not isinstance(margin_px, (int, np.integer)):
        raise TypeError(f"margin_px must be int, got {type(margin_px)}")
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")

    # -------------------------
    # compute norm map (H, W)
    # -------------------------
    intensity_cube = intensity_cube.astype(np.float32, copy=False)
    norm_map = np.linalg.norm(intensity_cube, axis=2)  # (H, W)
    norm_map = np.nan_to_num(norm_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # -------------------------
    # Otsu -> binary map (H, W)
    # -------------------------
    vmin = float(norm_map.min())
    vmax = float(norm_map.max())
    if vmax == vmin:
        binary_map = np.zeros((H, W), dtype=bool)
    else:
        thresh = float(threshold_otsu(norm_map))
        binary_map = norm_map > thresh

    # -------------------------
    # margin (distance map in True region)
    # -------------------------
    if margin_px > 0:
        dist_map = distance_transform_edt(binary_map)
        binary_map = dist_map >= float(margin_px) # type: ignore

    return {
        "norm_map": norm_map,
        "binary_map": binary_map,
    }