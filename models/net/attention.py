# net/attention.py
"""
Lightweight attention modules for image restoration.

Exports (drop-in compatible):
    - SEBlock(channels, reduction=16, gate='sigmoid', residual=True)
    - CBAM(channels, reduction=16, spatial_kernel=7, residual=True)

Optional extras you can import later:
    - ECA(channels, k=3)  # Efficient Channel Attention (ultra-light)
    - FiLM1d(channels, cond_dim=1)  # tiny FiLM for scalar conditioning (e.g., JPEG QF)

Notes:
- AMP-safe (no dtype tricks).
- Channels-last friendly (ops are layout-agnostic).
- Final gates are zero-initialized where useful for safer training.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utils
# ---------------------------
def _make_gate(gate: str = "sigmoid") -> nn.Module:
    gate = (gate or "sigmoid").lower()
    if gate in ("sigmoid", "sig"):
        return nn.Sigmoid()
    if gate in ("hsigmoid", "hard_sigmoid", "hard-sigmoid"):
        return nn.Hardsigmoid()
    if gate in ("tanh",):
        return nn.Tanh()
    # default
    return nn.Sigmoid()


def _init_last_bn_or_conv_zero(m: nn.Module):
    """Zero-init the last conv/bn to start as identity (stabilizes early training)."""
    if isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if m.weight is not None:
            nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------
# SE: Squeeze-and-Excitation
# ---------------------------
class SEBlock(nn.Module):
    """
    Classic channel attention.
    - residual=True keeps SE as a modulation (x * w); False returns only modulated branch.
    - 'gate' can be 'sigmoid' (default) or 'hsigmoid' etc.
    """
    def __init__(self, channels: int, reduction: int = 16, gate: str = "sigmoid", residual: bool = True):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(channels, mid, 1, bias=True)
        self.act  = nn.ReLU(inplace=True)
        self.fc2  = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = _make_gate(gate)
        self.residual = residual

        # Good default init: final conv bias to zero
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc2(self.act(self.fc1(w)))
        w = self.gate(w)
        out = x * w
        return x + out if not self.residual else out  # residual=True = x*w (standard); False = x + x*w


# ---------------------------
# CBAM: Channel + Spatial
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, gate: str = "sigmoid"):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 1x1 "MLP"
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = _make_gate(gate)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        b = self.fc2(self.act(self.fc1(self.max_pool(x))))
        w = self.gate(a + b)
        return x * w


class SpatialAttention(nn.Module):
    """
    Spatial attention using depthwise-separable conv for efficiency.
    """
    def __init__(self, kernel_size: int = 7, gate: str = "sigmoid"):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.reduce = nn.Conv2d(2, 2, 1, bias=False)  # light pre-mix
        self.dw = nn.Conv2d(2, 2, kernel_size, padding=padding, groups=2, bias=False)
        self.proj = nn.Conv2d(2, 1, 1, bias=True)
        self.gate = _make_gate(gate)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxv], dim=1)        # (B,2,H,W)
        h = self.reduce(cat)
        h = self.dw(h)
        w = self.gate(self.proj(h))                # (B,1,H,W)
        return x * w


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    residual=True returns x + spatial(channel(x)) for extra stability;
    residual=False returns purely modulated output (classic CBAM style tends to be residual).
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7, gate: str = "sigmoid", residual: bool = True):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction, gate)
        self.sa = SpatialAttention(spatial_kernel, gate)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ca(x)
        h = self.sa(h)
        return x + h if self.residual else h


# ---------------------------
# ECA: Efficient Channel Attention (optional)
# ---------------------------
class ECA(nn.Module):
    """
    ECA (CVPR'20): avoids the MLP, uses 1D conv over channel descriptors.
    Very cheap; often similar to SE for restoration tasks.
    """
    def __init__(self, channels: int, k: int = 3, gate: str = "sigmoid"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k // 2), bias=False)
        self.gate = _make_gate(gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y: (B,C,1,1) -> (B,1,C)
        y = self.pool(x).squeeze(-1).transpose(1, 2)  # (B,C,1,1) -> (B,C,1) -> (B,1,C)
        y = self.conv1d(y).transpose(1, 2).unsqueeze(-1)  # (B,1,C)->(B,C,1)->(B,C,1,1)
        w = self.gate(y)
        return x * w


# ---------------------------
# FiLM for scalar conditioning (optional)
# ---------------------------
class FiLM1d(nn.Module):
    """
    Tiny FiLM layer to modulate feature maps from a scalar condition (e.g., JPEG QF/100).
    - cond: (B,1) or (B,) tensor in [0,1] (or any range; MLP will learn the map)
    - returns: x * (1 + gamma) + beta
    Use inside residual blocks; zero-init last layer keeps start near identity.
    """
    def __init__(self, channels: int, cond_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        # zero-init the last linear for identity start (gamma=0, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.view(-1, 1)
        gb = self.mlp(cond)                          # (B, 2C)
        gamma, beta = torch.chunk(gb, 2, dim=1)      # (B, C), (B, C)
        gamma = gamma.view(x.size(0), -1, 1, 1)
        beta  = beta.view(x.size(0), -1, 1, 1)
        return x * (1 + gamma) + beta
