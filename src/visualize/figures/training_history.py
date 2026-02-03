from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plot_training_history(
    training_history: Iterable[Mapping[str, Any]],
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    学習履歴（train/val loss と learning rate）を可視化して保存する。

    (1, 2) の図を作成し、左に損失曲線（train/val）と best epoch のマーカー、
    右に学習率（lr）の推移を描画する。best epoch は val_loss が最小となる epoch。

    Parameters
    ----------
    training_history : Iterable[Mapping[str, Any]]
        学習履歴レコード列（通常は list[dict]）。各レコードは辞書型で、少なくとも以下のキーを持つ。

        必須キー（各 epoch レコード）
        - epoch : int
            エポック番号（昇順である必要はない。内部で epoch でソートする）。
        - train_loss : float
            訓練損失（例: masked SSE の batch mean）。
        - val_loss : float
            検証損失（train_loss と同じ定義）。
        - lr : float
            当該 epoch の学習率。

        任意キー
        - test_loss : float | None
            テスト損失。本関数では **training_history の末尾レコード**に含まれている場合のみ参照し、
            best epoch の位置に “test@best(val)” として追加描画する。
            （test_loss を epoch ごとに持っていても、末尾以外は無視する）

        取り扱い
        - training_history に「辞書でない要素」や「必須キーが揃っていない要素」が混在していても良いが、
          それらは描画対象から除外する（= dict かつ必須キーを満たすレコードだけを使用）。
        - 描画対象レコードが 1 件も無い場合は例外を送出する。

    out_path : str | Path | None, default None
        保存先パス。None の場合は保存せず、描画してすぐに close する。
        指定する場合、親ディレクトリは事前に作成しておくこと（本関数では mkdir しない）。

    Raises
    ------
    ValueError
        - training_history が空、または有効なレコードが存在しない場合
        - 数値変換に失敗する値が含まれる場合
    TypeError
        - training_history が反復可能でない場合

    Notes
    -----
    - best epoch は `argmin(val_loss)` で決定する。
    - x 軸の主目盛りは 20 epoch 間隔に固定する（MultipleLocator(20)）。
    - y 軸ラベルは現状 "Masked SSE (batch mean)" 固定（用途に応じて変更可）。
    """
    # out_path は str でも Path でも受け取れるようにする
    if out_path is not None:
        out_path = Path(out_path)

    # iterable を一度 list 化して末尾アクセス可能にする
    training_history_list = list(training_history)
    if len(training_history_list) == 0:
        raise ValueError("training_history must be non-empty.")

    # --- 末尾の test レコードを抽出 ---
    last = training_history_list[-1]
    test_loss = None
    if isinstance(last, Mapping) and "test_loss" in last and last["test_loss"] is not None:
        test_loss = float(last["test_loss"])

    # --- 学習履歴レコード抽出 ---
    recs = [
        r for r in training_history_list
        if isinstance(r, Mapping) and all(k in r for k in ("epoch", "train_loss", "val_loss", "lr"))
    ]
    if len(recs) == 0:
        raise ValueError(
            "No valid records found in training_history. "
            "Each record must contain keys: epoch, train_loss, val_loss, lr."
        )

    recs.sort(key=lambda r: int(r["epoch"]))

    epochs = np.array([int(r["epoch"]) for r in recs], dtype=int)
    tr = np.array([float(r["train_loss"]) for r in recs], dtype=float)
    va = np.array([float(r["val_loss"]) for r in recs], dtype=float)
    lr = np.array([float(r["lr"]) for r in recs], dtype=float)

    # --- best epoch by val(min) ---
    best_idx = int(np.argmin(va))
    best_epoch = int(epochs[best_idx])
    best_train = float(tr[best_idx])
    best_val = float(va[best_idx])

    # --- 目盛り用の範囲を決定 ---
    step = 20
    xmin = int(np.floor(epochs.min() / step) * step)
    xmax = int(np.ceil(epochs.max() / step) * step)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    # --- (Left) losses ---
    ax = axes[0]
    ax.plot(epochs, tr, label="train_loss")
    ax.plot(epochs, va, label="val_loss")
    ax.scatter(
        [best_epoch],
        [best_train],
        s=60,
        label=f"train@best(val) = {best_train:.3g} (ep {best_epoch})",
    )
    ax.scatter(
        [best_epoch],
        [best_val],
        s=60,
        label=f"val@best(val) = {best_val:.3g} (ep {best_epoch})",
    )
    if test_loss is not None:
        ax.scatter(
            [best_epoch],
            [float(test_loss)],
            s=60,
            marker="x",
            label=f"test@best(val) = {float(test_loss):.3g} (ep {best_epoch})",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("Masked SSE (batch mean)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- (Right) learning rate ---
    ax = axes[1]
    ax.plot(epochs, lr, label="learning rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- 両サブプロットの x 軸を step の倍数目盛りに統一 ---
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(step))

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")

    plt.close(fig)
    return None