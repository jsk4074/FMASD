# sans/aefc_detached.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import torch
import torch.nn as nn

@dataclass
class DetachedAEFCConfig:
    in_dim: int
    hidden: int = 512
    depth: int = 2
    dropout: float = 0.1
    activation: str = "gelu"   # "relu" | "silu" | "gelu"
    layernorm: bool = True
    # mix: y = ref + alpha * (AEFC(x_detached) - ref)
    alpha: float = 1.0

def _act(name: str) -> nn.Module:
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "silu": return nn.SiLU(inplace=True)
    return nn.GELU()

def _repeat_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    d = int(x.shape[-1])
    if d == target_dim: return x
    if d > target_dim:  return x[..., :target_dim].contiguous()
    reps = (target_dim + d - 1) // d
    xr = x.repeat_interleave(reps, dim=-1)
    return xr[..., :target_dim].contiguous()

def _flatten_last(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 0: x = x.view(1)
    lead = tuple(x.shape[:-1]) if x.ndim >= 1 else (1,)
    D = int(x.shape[-1]) if x.ndim >= 1 else 1
    N = 1
    for a in lead: N *= int(a)
    return x.reshape(N, D), lead, D

def _unflatten_last(x2d: torch.Tensor, lead: Sequence[int]) -> torch.Tensor:
    return x2d.view(*lead, x2d.shape[-1])

class DetachedAEFC(nn.Module):
    """
    Simple MLP that operates along the LAST dim (shape-agnostic).
    Forward mixes a DETACHED reference latent to prevent trivial collapse:
        out = ref + alpha * ( f(x_detached) - ref )
    """
    def __init__(self, cfg: DetachedAEFCConfig):
        super().__init__()
        self.cfg = cfg
        act = _act(cfg.activation)
        layers = []
        D = cfg.in_dim
        H = max(16, int(cfg.hidden))
        # in -> H
        layers += [nn.Linear(D, H)]
        if cfg.layernorm: layers += [nn.LayerNorm(H)]
        layers += [act, nn.Dropout(cfg.dropout)]
        # hidden
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(H, H)]
            if cfg.layernorm: layers += [nn.LayerNorm(H)]
            layers += [act, nn.Dropout(cfg.dropout)]
        # H -> out_dim (=in_dim)
        layers += [nn.Linear(H, D)]
        self.mlp = nn.Sequential(*layers)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x2d, lead, _ = _flatten_last(x)
        y2d = self.mlp(x2d)
        return _unflatten_last(y2d, lead)

    def forward(self, x: torch.Tensor, ref: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        # Align dims
        D = int(x.shape[-1])
        y_in = _repeat_to_dim(x.detach(), self.cfg.in_dim)
        r_in = _repeat_to_dim(ref.detach(), self.cfg.in_dim)
        y_hat = self.transform(y_in)
        # bring back to original last-dim
        y_hat = _repeat_to_dim(y_hat, D)
        r_in  = _repeat_to_dim(r_in,  D)
        a = self.cfg.alpha if alpha is None else float(alpha)
        return r_in + a * (y_hat - r_in)

    # I/O
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg.__dict__}, path)

    @staticmethod
    def load(path: str, map_location=None) -> "DetachedAEFC":
        chk = torch.load(path, map_location=map_location)
        cfg = DetachedAEFCConfig(**chk["cfg"])
        m = DetachedAEFC(cfg)
        m.load_state_dict(chk["state_dict"])
        return m
