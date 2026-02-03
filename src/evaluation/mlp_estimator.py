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
    # task
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
    DataLoader 入出力の MLP 推定器（classification / regression）。

    - train_loader は (x, y) を返すこと
    - x は shape (B, D)
    - classification:
        - num_classes==2 のとき out_dim=1 (BCEWithLogits)
        - num_classes>=3 のとき out_dim=num_classes (CrossEntropy)
    - regression:
        - out_dim = cfg.out_dim (Noneなら1)
        - y は (B,) or (B, out_dim)
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
        - classification: (N,) の class index (long)
        - regression: (N, out_dim) or out_dim==1なら (N,)
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
        classification のみ。返り値は (N, K) の確率（binaryでもK=2に整形）
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
        - classification: {"loss", "acc"}
        - regression: {"loss", "mse", "mae", "r2"}
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