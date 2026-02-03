from __future__ import annotations

import os
import time
import copy
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


TaskType = Literal["classification", "regression"]
ActType = Literal["relu", "gelu", "silu"]
NormType = Literal["layernorm", "batchnorm", "none"]
OptType  = Literal["adamw", "adam"]


def _infer_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_activation(name: ActType) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.GELU()


def _make_norm(name: NormType, dim: int) -> nn.Module:
    if name == "layernorm":
        return nn.LayerNorm(dim)
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class ResidualFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_mul: float = 2.0,
        dropout: float = 0.0,
        activation: ActType = "gelu",
        norm: NormType = "layernorm",
    ):
        super().__init__()
        h = max(1, int(round(dim * hidden_mul)))
        self.n1 = _make_norm(norm, dim)
        self.fc1 = nn.Linear(dim, h)
        self.act = _make_activation(activation)
        self.drop1 = nn.Dropout(dropout)

        self.n2 = _make_norm(norm, h)
        self.fc2 = nn.Linear(h, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.n1(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.n2(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class MLPNet(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.0,
        activation: ActType = "gelu",
        norm: NormType = "layernorm",
        residual: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.stem = nn.Linear(in_dim, width)

        blocks: List[nn.Module] = []
        for _ in range(depth):
            if residual:
                blocks.append(
                    ResidualFFN(
                        width,
                        hidden_mul=2.0,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                    )
                )
            else:
                blocks.append(
                    nn.Sequential(
                        _make_norm(norm, width),
                        nn.Linear(width, width),
                        _make_activation(activation),
                        nn.Dropout(dropout),
                    )
                )
        self.blocks = nn.Sequential(*blocks)
        self.head_norm = _make_norm(norm, width)
        self.head = nn.Linear(width, out_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_norm(x)
        x = self.head(x)
        return x


@dataclass
class MLPConfig:
    """
    `MLPEstimator` の設定（モデル構造・学習・計算設定）をまとめたデータクラス。

    Parameters
    ----------
    task : {"classification", "regression"}, default="classification"
        タスク種別。
        - ``"classification"``: 分類（2値/多クラス）
        - ``"regression"``: 回帰

    in_dim : int, default=0
        入力特徴次元 ``D``。`fit` 前に **必ず** 正の値を設定します。

    num_classes : int or None, default=None
        分類時のクラス数 ``K``。
        ``task="classification"`` の場合に必須（``K>=2``）。
        ``K==2`` のときは内部の出力次元は 1（BCEWithLogits）になります。

    out_dim : int or None, default=None
        回帰時の出力次元。``task="regression"`` の場合に使用します。
        ``None`` のときは 1 とみなします。

    width : int, default=256
        MLP の中間幅（隠れ次元）。

    depth : int, default=4
        ブロック段数（`ResidualFFN` または単純 MLP ブロックを何個積むか）。

    dropout : float, default=0.1
        Dropout 率。

    activation : {"relu", "gelu", "silu"}, default="gelu"
        活性化関数。

    norm : {"layernorm", "batchnorm", "none"}, default="layernorm"
        正規化層の種類。

    residual : bool, default=True
        True の場合、残差付き FFN (`ResidualFFN`) を使用します。

    epochs : int, default=50
        学習エポック数。

    lr : float, default=3e-4
        学習率。

    weight_decay : float, default=1e-4
        weight decay（L2 正則化）。AdamW/Adam の引数として渡されます。

    optimizer : {"adamw", "adam"}, default="adamw"
        最適化手法。

    grad_clip_norm : float or None, default=1.0
        勾配ノルムクリップ。`None` で無効。

    device : str or None, default=None
        計算デバイス。`None` の場合は ``cuda -> mps -> cpu`` の順で自動選択。

    amp : bool, default=True
        Mixed Precision を有効にするかどうか。
        実装では CUDA のときのみ有効化されます。

    amp_dtype : {"bf16", "fp16"}, default="bf16"
        AMP の dtype。fp16 の場合のみ GradScaler を使用します。

    early_stopping : bool, default=True
        Early stopping を使うかどうか（`val_loader` がある場合のみ有効）。

    patience : int, default=10
        改善が無いエポック数がこの回数続いたら打ち切り。

    min_delta : float, default=0.0
        改善判定の最小差分（``best_val - current_val > min_delta`` を改善とみなす）。

    restore_best : bool, default=True
        学習終了後に、検証損失が最良だった重みへ戻すかどうか。

    seed : int or None, default=42
        乱数シード。`None` の場合は設定しません。

    verbose : int, default=1
        ログ出力レベル。
        - 0: 出力なし
        - 1: epoch ごとの train/val loss を出力
        - 2 以上: 追加で step ログも出力

    log_every : int, default=50
        ``verbose>=2`` かつ学習時に、何 step ごとに loss を表示するか。

    Notes
    -----
    - 分類の損失は、2値なら BCEWithLogits、多クラスなら CrossEntropy。
    - 回帰の損失は MSE。
    """
    task: TaskType = "classification"

    # I/O
    in_dim: int = 0
    # classification:
    num_classes: Optional[int] = None  # required if task=classification
    # regression:
    out_dim: Optional[int] = None       # default=1 if task=regression

    # model
    width: int = 256
    depth: int = 4
    dropout: float = 0.1
    activation: ActType = "gelu"
    norm: NormType = "layernorm"
    residual: bool = True

    # training
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: OptType = "adamw"
    grad_clip_norm: Optional[float] = 1.0

    # compute
    device: Optional[str] = None  # None => auto
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    # early stopping (val_loader がある時だけ効く)
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0
    restore_best: bool = True

    # misc
    seed: Optional[int] = 42
    verbose: int = 1
    log_every: int = 50  # steps


class MLPEstimator:
    """
    DataLoader 入出力の簡易 MLP 推定器（分類/回帰）。

    PyTorch の `DataLoader` を入力として、MLP の学習・推論・評価を行います。
    scikit-learn の推定器 API（`fit` / `predict` / `evaluate` / `save` / `load`）に近い形で
    最低限の実験を回せるようにしたユーティリティです。

    Parameters
    ----------
    cfg : MLPConfig
        モデル構造・学習設定。

    Attributes
    ----------
    cfg : MLPConfig
        設定。
    device : torch.device
        計算デバイス（auto 推論または `cfg.device` 指定）。
    model : MLPNet
        学習対象の MLP 本体。
    opt : torch.optim.Optimizer
        最適化器（AdamW/Adam）。
    history : dict of list
        学習履歴。少なくとも ``{"train_loss": [...], "val_loss": [...]}`` を保持します。
        `val_loader` が無い場合、`val_loss` は空のままです。
    best_state : dict or None
        Early stopping における最良検証損失時の `state_dict`（`restore_best=True` なら復元に使用）。
    best_val : float or None
        最良の検証損失（`val_loader` がある場合のみ）。

    Notes
    -----
    入力データ形式
        `DataLoader` が返す `batch` は以下を想定します。

        - ``(x, y)`` または ``{"x": x, "y": y}``
        - 推論系（`predict` / `predict_proba`）では `y` が無くても可

        ここで `x` は shape ``(B, D)`` を想定し、内部で ``float32`` に変換して device へ転送します。
        回帰の `y` は内部で 2 次元 ``(B, out_dim)`` へ整形されます（``(B,)`` の場合は ``(B,1)``）。

    分類タスクの扱い
        - ``num_classes == 2`` の場合: 出力は 1 次元 logit（BCEWithLogits）
        - ``num_classes >= 3`` の場合: 出力は ``K`` 次元 logit（CrossEntropy）

        `predict` は class index（long）を返し、`predict_proba` は確率を ``(N, K)`` で返します
        （2値でも ``K=2`` に整形）。

    AMP（Mixed Precision）
        `cfg.amp=True` でも、実装上は CUDA の場合のみ AMP を有効化します。
        `amp_dtype="fp16"` のときのみ GradScaler を使用します。

    Examples
    --------
    分類（多クラス）
    >>> cfg = MLPConfig(task="classification", in_dim=128, num_classes=10)
    >>> est = MLPEstimator(cfg).fit(train_loader, val_loader)
    >>> y_pred = est.predict(test_loader)          # (N,)
    >>> y_prob = est.predict_proba(test_loader)    # (N, 10)
    >>> metrics = est.evaluate(test_loader)        # {"loss": ..., "acc": ...}

    回帰
    >>> cfg = MLPConfig(task="regression", in_dim=32, out_dim=1)
    >>> est = MLPEstimator(cfg).fit(train_loader, val_loader)
    >>> y_hat = est.predict(test_loader)           # (N,)
    >>> metrics = est.evaluate(test_loader)        # {"loss": ..., "mse": ..., "mae": ..., "r2": ...}
    """

    def __init__(self, cfg: MLPConfig):
        self.cfg = cfg
        self.device = _infer_device(cfg.device)
        _set_seed(cfg.seed)

        self.model = self._build_model().to(self.device)
        self.opt = self._build_optimizer(self.model)

        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val: Optional[float] = None

        self._use_amp = bool(cfg.amp and self.device.type == "cuda")
        self._amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self._use_amp and self._amp_dtype == torch.float16))

    # ----------------
    # public methods
    # ----------------
    def fit(self, train_loader, val_loader=None) -> "MLPEstimator":
        """
        モデルを学習します（オプションで early stopping）。

        Parameters
        ----------
        train_loader : iterable
            学習用 DataLoader。各バッチは ``(x, y)`` または ``{"x": x, "y": y}`` を想定します。
        val_loader : iterable or None, default=None
            検証用 DataLoader。指定された場合、検証損失を監視して early stopping を行います。

        Returns
        -------
        self : MLPEstimator
            学習済みの推定器（チェーン可能）。

        Notes
        -----
        - `cfg.early_stopping=True` かつ `val_loader` がある場合のみ early stopping が有効です。
        - `cfg.restore_best=True` の場合、学習終了後に最良検証損失の重みへ復元します。
        """
        cfg = self.cfg
        best_val = float("inf")
        best_state = None
        patience_left = int(cfg.patience)

        for epoch in range(1, int(cfg.epochs) + 1):
            t0 = time.time()

            tr_loss = self._run_epoch(train_loader, train=True)
            va_loss = None
            if val_loader is not None:
                va_loss = self._run_epoch(val_loader, train=False)

            self.history["train_loss"].append(float(tr_loss))
            if va_loss is not None:
                self.history["val_loss"].append(float(va_loss))

            if cfg.verbose:
                msg = f"[epoch {epoch:03d}] train_loss={tr_loss:.6f}"
                if va_loss is not None:
                    msg += f"  val_loss={va_loss:.6f}"
                msg += f"  time={time.time()-t0:.1f}s"
                print(msg)

            if cfg.early_stopping and (val_loader is not None):
                improved = (best_val - float(va_loss)) > float(cfg.min_delta) # type: ignore
                if improved:
                    best_val = float(va_loss) # type: ignore
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_left = int(cfg.patience)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if cfg.verbose:
                            print(f"EarlyStopping: no improvement for {cfg.patience} epochs.")
                        break

        self.best_val = best_val if val_loader is not None else None
        self.best_state = best_state

        if cfg.restore_best and (self.best_state is not None):
            self.model.load_state_dict(self.best_state)

        return self

    @torch.no_grad()
    def predict(self, loader) -> torch.Tensor:
        """
        推論を行い、ラベル（分類）または予測値（回帰）を返します。

        Parameters
        ----------
        loader : iterable
            推論用 DataLoader。各バッチは ``(x, y)`` / ``{"x": x, "y": y}`` / ``x`` を許容します。

        Returns
        -------
        y_hat : torch.Tensor
            - 分類: shape ``(N,)`` の class index（dtype long）
            - 回帰: shape ``(N, out_dim)``、ただし ``out_dim==1`` のときは ``(N,)``

        Notes
        -----
        分類の 2値判定は `sigmoid(logit) >= 0.5` により行います。
        """
        self.model.eval()
        preds: List[torch.Tensor] = []

        for batch in loader:
            x, _ = self._unpack_batch(batch, need_y=False)
            logits = self.model(x)

            if self.cfg.task == "classification":
                if self._is_binary():
                    p1 = torch.sigmoid(logits).squeeze(1)
                    yhat = (p1 >= 0.5).long()
                else:
                    yhat = torch.argmax(logits, dim=1)
                preds.append(yhat.cpu())
            else:
                out = logits
                if out.shape[1] == 1:
                    out = out[:, 0]
                preds.append(out.cpu())

        return torch.cat(preds, dim=0)

    @torch.no_grad()
    def predict_proba(self, loader) -> torch.Tensor:
        """
        分類の確率予測を返します（classification のみ）。

        Parameters
        ----------
        loader : iterable
            推論用 DataLoader。

        Returns
        -------
        proba : torch.Tensor of shape (N, K)
            クラス確率。2値分類でも ``K=2`` に整形して返します。

        Raises
        ------
        RuntimeError
            `task != "classification"` の場合。

        Notes
        -----
        - 2値分類: ``[p0, p1]`` を返します（`p1 = sigmoid(logit)`）。
        - 多クラス: `softmax` を返します。
        """
        if self.cfg.task != "classification":
            raise RuntimeError("predict_proba is only available for classification.")

        self.model.eval()
        probs: List[torch.Tensor] = []

        for batch in loader:
            x, _ = self._unpack_batch(batch, need_y=False)
            logits = self.model(x)

            if self._is_binary():
                p1 = torch.sigmoid(logits).squeeze(1)
                p0 = 1.0 - p1
                p = torch.stack([p0, p1], dim=1)
            else:
                p = torch.softmax(logits, dim=1)

            probs.append(p.cpu())

        return torch.cat(probs, dim=0)

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """
        損失と基本指標を計算します。

        Parameters
        ----------
        loader : iterable
            評価用 DataLoader。

        Returns
        -------
        metrics : dict
            - 分類: ``{"loss": float, "acc": float}``
            - 回帰: ``{"loss": float, "mse": float, "mae": float, "r2": float}``

        Notes
        -----
        回帰の指標は CPU 側へ集約して double で計算します（数値安定性のため）。
        """
        self.model.eval()
        total_loss = 0.0
        total_n = 0

        if self.cfg.task == "classification":
            correct = 0
        else:
            # accumulate on CPU for stability
            y_all: List[torch.Tensor] = []
            p_all: List[torch.Tensor] = []

        for batch in loader:
            x, y = self._unpack_batch(batch, need_y=True)
            logits = self.model(x)
            loss = self._loss(logits, y) # type: ignore

            n = x.shape[0]
            total_loss += float(loss.detach().cpu()) * n
            total_n += int(n)

            if self.cfg.task == "classification":
                if self._is_binary():
                    p1 = torch.sigmoid(logits).squeeze(1)
                    yhat = (p1 >= 0.5).long()
                else:
                    yhat = torch.argmax(logits, dim=1)
                correct += int((yhat == y).sum().detach().cpu())
            else:
                pred = logits.detach().cpu()
                yt = y.detach().cpu() # type: ignore
                if pred.shape[1] == 1:
                    pred = pred[:, 0]
                if yt.ndim == 2 and yt.shape[1] == 1:
                    yt = yt[:, 0]
                y_all.append(yt)
                p_all.append(pred)

        out: Dict[str, float] = {"loss": total_loss / max(1, total_n)}

        if self.cfg.task == "classification":
            out["acc"] = correct / max(1, total_n)
            return out

        yv = torch.cat(y_all, dim=0).double()
        pv = torch.cat(p_all, dim=0).double()

        mse = torch.mean((pv - yv) ** 2).item()
        mae = torch.mean(torch.abs(pv - yv)).item()

        # R^2
        y_mean = torch.mean(yv, dim=0, keepdim=True)
        ss_res = torch.sum((yv - pv) ** 2).item()
        ss_tot = torch.sum((yv - y_mean) ** 2).item()
        r2 = 0.0 if ss_tot <= 0 else (1.0 - ss_res / ss_tot)

        out.update({"mse": mse, "mae": mae, "r2": float(r2)})
        return out

    def save(self, path: str) -> None:
        """
        推定器を保存します（`torch.save`）。

        保存内容は ``cfg``（dict 化）、`model.state_dict()`、`best_val`、`history` です。

        Parameters
        ----------
        path : str
            保存先パス。
        """
        payload = {
            "cfg": asdict(self.cfg),
            "state_dict": self.model.state_dict(),
            "best_val": self.best_val,
            "history": self.history,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "MLPEstimator":
        """
        保存済みファイルから推定器を復元します。

        Parameters
        ----------
        path : str
            読み込み元パス。
        map_location : str or None, default=None
            `torch.load` の `map_location`。指定しない場合は CPU に読み込みます。

        Returns
        -------
        estimator : MLPEstimator
            復元された推定器。
        """
        loc = torch.device(map_location) if map_location is not None else torch.device("cpu")
        payload = torch.load(path, map_location=loc)

        cfg = MLPConfig(**payload["cfg"])
        obj = cls(cfg)
        obj.model.load_state_dict(payload["state_dict"])
        obj.best_val = payload.get("best_val", None)
        obj.history = payload.get("history", {"train_loss": [], "val_loss": []})
        obj.model.to(obj.device)
        obj.model.eval()
        return obj

    # ----------------
    # internals
    # ----------------
    def _is_binary(self) -> bool:
        return (self.cfg.task == "classification") and (int(self.cfg.num_classes) == 2) # type: ignore

    def _build_model(self) -> MLPNet:
        if self.cfg.in_dim <= 0:
            raise ValueError("cfg.in_dim must be set (>0).")

        if self.cfg.task == "classification":
            if self.cfg.num_classes is None or int(self.cfg.num_classes) <= 1:
                raise ValueError("cfg.num_classes must be set (>=2) for classification.")
            out_dim = 1 if int(self.cfg.num_classes) == 2 else int(self.cfg.num_classes)
        else:
            out_dim = 1 if (self.cfg.out_dim is None) else int(self.cfg.out_dim)
            if out_dim <= 0:
                raise ValueError("cfg.out_dim must be >=1 for regression.")

        return MLPNet(
            in_dim=int(self.cfg.in_dim),
            out_dim=int(out_dim),
            width=int(self.cfg.width),
            depth=int(self.cfg.depth),
            dropout=float(self.cfg.dropout),
            activation=self.cfg.activation,
            norm=self.cfg.norm,
            residual=bool(self.cfg.residual),
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.cfg.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))
        return torch.optim.AdamW(model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))

    def _unpack_batch(self, batch: Any, *, need_y: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        batch:
          - (x, y) or {"x": x, "y": y} を想定
          - predict系では y が無くても良い
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            y = batch[1] if (need_y and len(batch) > 1) else None
        elif isinstance(batch, dict):
            x = batch["x"]
            y = batch.get("y", None) if need_y else None
        else:
            # DataLoader が x のみ返すケース
            x, y = batch, None

        x = x.to(self.device, non_blocking=True).float()

        if need_y:
            if y is None:
                raise ValueError("This loader must yield y for training/evaluation.")
            if self.cfg.task == "classification":
                y = y.to(self.device, non_blocking=True).long()
            else:
                y = y.to(self.device, non_blocking=True).float()
                # ensure 2D for regression loss
                if y.ndim == 1:
                    y = y[:, None]
        return x, y

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cfg.task == "classification":
            if self._is_binary():
                # logits: (B,1), y: (B,)
                return F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float())
            return F.cross_entropy(logits, y)
        # regression: logits: (B, out_dim), y: (B, out_dim)
        return F.mse_loss(logits, y)

    def _run_epoch(self, loader, *, train: bool) -> float:
        self.model.train(train)

        total_loss = 0.0
        total_n = 0

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=bool(self._use_amp),
        )

        for step, batch in enumerate(loader, start=1):
            x, y = self._unpack_batch(batch, need_y=True)

            if train:
                self.opt.zero_grad(set_to_none=True)

            with autocast_ctx:
                logits = self.model(x)
                loss = self._loss(logits, y) # type: ignore

            if train:
                if self._scaler.is_enabled():
                    self._scaler.scale(loss).backward()
                    if self.cfg.grad_clip_norm is not None:
                        self._scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                    self._scaler.step(self.opt)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                    self.opt.step()

            n = x.shape[0]
            total_loss += float(loss.detach().cpu()) * n
            total_n += int(n)

            if self.cfg.verbose >= 2 and train and (step % int(self.cfg.log_every) == 0):
                print(f"  step {step:05d} loss={float(loss.detach().cpu()):.6f}")

        return total_loss / max(1, total_n)