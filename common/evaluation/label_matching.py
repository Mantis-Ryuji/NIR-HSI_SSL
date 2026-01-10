from __future__ import annotations

from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def compute_label_map(y_ref: np.ndarray, y_lat: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    """
    ref_snv 側クラスタラベルを latent 側ラベルに揃えるための置換（対応付け）を Hungarian 法で求める。

    目的は「同一サンプルに対して ref と latent で付いたクラスタIDの対応」を最も一致するように
    置換すること。すなわち、混同行列（一致数行列）C を作り、

        C[r, c] = |{ i : y_ref[i] = r かつ y_lat[i] = c }|

    を最大化する割当 (r -> c) を Hungarian 法で解く。

    Parameters
    ----------
    y_ref : np.ndarray, shape (N,)
        ref_snv 空間でのクラスタラベル（各サンプルのクラスタID）。整数を想定。
        典型的には 0..K-1 の連番。
    y_lat : np.ndarray, shape (N,)
        latent 空間でのクラスタラベル（各サンプルのクラスタID）。整数を想定。
        典型的には 0..K-1 の連番。

    Returns
    -------
    label_map : dict[int, int]
        ref ラベル -> latent ラベル の対応付け。
        例: {0: 2, 1: 0, 2: 1} は ref=0 を latent=2 として扱う、という意味。
    C : np.ndarray, shape (K, K)
        一致数行列（混同行列）。

    Notes
    -----
    - K（クラスタ数）は ref と latent で同一である前提。
    - 本実装は `np.unique` で得たクラス集合を用いるが、混同行列の添字に
      直接ラベル値を使っているため、ラベルは 0..K-1 の連番であることを強く推奨する。
      連番でないラベル（例: {1,3,7}）を渡すと C[r, col] のインデックスが破綻する。
      必要なら事前にラベルを 0..K-1 に再符号化してから渡すこと。
    """
    y_ref = np.asarray(y_ref)
    y_lat = np.asarray(y_lat)

    if y_ref.ndim != 1 or y_lat.ndim != 1:
        raise ValueError(f"y_ref and y_lat must be 1D, got {y_ref.shape} and {y_lat.shape}")
    if y_ref.shape[0] != y_lat.shape[0]:
        raise ValueError(f"Length mismatch: y_ref={y_ref.shape[0]} vs y_lat={y_lat.shape[0]}")

    ref_classes = np.unique(y_ref)
    lat_classes = np.unique(y_lat)
    assert ref_classes.size == lat_classes.size, "Kは共通前提（サイズ不一致）"
    K = int(ref_classes.size)

    # 混同行列 C[r, c] = count( y_ref == r and y_lat == c )
    C = np.zeros((K, K), dtype=np.int64)
    for r in ref_classes:
        mask_r = (y_ref == r)
        col, cnt = np.unique(y_lat[mask_r], return_counts=True)
        C[int(r), col.astype(int)] = cnt

    ridx, cidx = linear_sum_assignment(-C)  # 一致数を最大化
    label_map = {int(r): int(c) for r, c in zip(ridx, cidx)}
    return label_map, C


def apply_map(y_ref: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    """
    ref ラベル配列に対して label_map(ref->latent) を適用し、latent 側ラベルに揃えた配列を返す。

    Parameters
    ----------
    y_ref : np.ndarray, shape (N,)
        ref 側クラスタラベル（整数）。
    label_map : dict[int, int]
        ref -> latent の対応付け（compute_label_map の戻り値）。

    Returns
    -------
    y_new : np.ndarray, shape (N,)
        対応付け後の ref ラベル（latent と比較可能なIDに置換済み）。
    """
    y_ref = np.asarray(y_ref)
    if y_ref.ndim != 1:
        raise ValueError(f"y_ref must be 1D, got {y_ref.shape}")

    y_new = y_ref.copy()
    for r, c in label_map.items():
        y_new[y_ref == r] = c
    return y_new


def save_aligned_centroids(
    path_unmatched: Path,
    path_out: Path,
    label_map: Dict[int, int],
) -> None:
    """
    ref 側の centroid（行=refラベル順）を、latent ラベル順 (0..K-1) に並べ替えて保存する。

    典型用途：
    - ref_snv で学習した CosineKMeans の centroid を保存してあるが、
      latent 空間のクラスタIDと「同じ番号として比較・可視化」したい。
    - compute_label_map で得た `label_map(ref->latent)` を用いて並べ替える。

    Parameters
    ----------
    path_unmatched : Path
        ref 側 centroid の保存ファイル。
        `torch.load` で読み込み、以下のどちらかの形式を想定する：
        - dict 形式: {"centroids": Tensor or ndarray}
        - Tensor / ndarray そのもの
    path_out : Path
        並べ替え後 centroid の保存先。
        常に dict 形式 {"centroids": torch.Tensor} で torch.save する。
    label_map : dict[int, int]
        ref -> latent の対応付け。

    Notes
    -----
    - centroid は行方向に L2 正規化して保存する（cosine 用の前提）。
    - 並べ替えは「latent の行インデックス」を 0..K-1 として、その順に
      対応する ref の centroid を並べる（latent->ref の逆写像を作って使用）。
    """
    path_unmatched = Path(path_unmatched)
    path_out = Path(path_out)

    obj = torch.load(path_unmatched, map_location="cpu")
    C = obj["centroids"] if isinstance(obj, dict) and "centroids" in obj else obj
    if isinstance(C, torch.Tensor):
        C = C.detach().cpu().numpy()
    C = np.asarray(C, dtype=np.float32)
    if C.ndim != 2:
        raise ValueError(f"centroids must be 2D, got {C.shape}")

    # 念のため行方向に正規化
    n = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.clip(n, 1e-12, None)

    K = C.shape[0]

    # inverse: latent -> ref を作り latent 順に並べ替え
    lat2ref = {int(lat): int(ref) for ref, lat in label_map.items()}
    C_aligned = np.empty_like(C)

    for lat_idx in range(K):
        if lat_idx not in lat2ref:
            raise KeyError(f"latent index {lat_idx} is missing in label_map")
        ref_idx = lat2ref[lat_idx]
        if not (0 <= ref_idx < K):
            raise IndexError(f"ref index out of range: {ref_idx} (K={K})")
        C_aligned[lat_idx] = C[ref_idx]

    # 再正規化して保存（dict形式で揃える）
    n = np.linalg.norm(C_aligned, axis=1, keepdims=True)
    C_aligned = C_aligned / np.clip(n, 1e-12, None)

    path_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": torch.from_numpy(C_aligned)}, path_out)


def verify_label_matching(y_ref_matched: np.ndarray, y_latent: np.ndarray) -> None:
    """
    ハンガリアン整合後に、ラベル集合が一致しているか（K と ID 集合）を検証する。

    Parameters
    ----------
    y_ref_matched : np.ndarray, shape (N,)
        apply_map 等で latent 側ラベルに揃えた ref ラベル配列。
    y_latent : np.ndarray, shape (N,)
        latent 側ラベル配列。

    Notes
    -----
    - ここでは「ユニークラベル数が一致」かつ「ラベル集合が完全一致」をチェックする。
      連番（0..K-1）であることまでは強制しないが、運用上は連番が望ましい。
    """
    y_ref_matched = np.asarray(y_ref_matched)
    y_latent = np.asarray(y_latent)

    if y_ref_matched.ndim != 1 or y_latent.ndim != 1:
        raise ValueError(
            f"y_ref_matched and y_latent must be 1D, got {y_ref_matched.shape} and {y_latent.shape}"
        )
    if y_ref_matched.shape[0] != y_latent.shape[0]:
        raise ValueError(
            f"Length mismatch: y_ref_matched={y_ref_matched.shape[0]} vs y_latent={y_latent.shape[0]}"
        )

    U_ref = np.unique(y_ref_matched)
    U_lat = np.unique(y_latent)
    assert U_ref.size == U_lat.size, f"K mismatch: ref={U_ref.size}, latent={U_lat.size}"
    assert np.array_equal(np.sort(U_ref), np.sort(U_lat)), "Label sets differ; not matched?"
    print(f"[OK] Labels matched. K={U_ref.size}")