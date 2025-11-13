# net/sfp_lite.py
"""
A simplified SFP-style model, tuned for JPEG restoration experiments.

Key improvements:
- Supports quality-aware concat: in_ch can be 4 (RGB + Q-map), while out_ch remains 3 (RGB).
- Safer residual blocks (second conv zero-initialized).
- Drop-in attention: SE / CBAM toggles preserved.
- Optional FiLM hook (disabled by default) for scalar conditioning inside blocks.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SEBlock, CBAM, FiLM1d  # FiLM is optional; unused unless enabled


# ---------------------------
# Building blocks
# ---------------------------
class ResidualConvBlock(nn.Module):
    """
    Conv -> GELU -> Conv (+ residual)
    Second conv is zero-initialized to start near identity (stabilizes training).
    """
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=True)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=pad, bias=True)
        # zero-init conv2 for stable residual learning
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h


class BasicPB(nn.Module):
    """
    Basic processing block:
      - residual conv block
      - optional FiLM modulation (scalar cond)
      - optional attention (SE or CBAM)
    """
    def __init__(self, channels: int, use_se: bool = False, use_cbam: bool = False,
                 use_film: bool = False, film_hidden: int = 32):
        super().__init__()
        self.body = ResidualConvBlock(channels)
        self.use_se = bool(use_se)
        self.use_cbam = bool(use_cbam)
        self.use_film = bool(use_film)

        if self.use_film:
            self.film = FiLM1d(channels, cond_dim=1, hidden=film_hidden)  # scalar cond (e.g., q/100)

        if self.use_se:
            self.se = SEBlock(channels)
        if self.use_cbam:
            self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        out = self.body(x)
        if self.use_film and cond is not None:
            # cond shape: (B,) or (B,1) in any range; FiLM learns its mapping
            out = self.film(out, cond)
        if self.use_se:
            out = self.se(out)
        if self.use_cbam:
            out = self.cbam(out)
        return out


# ---------------------------
# Model
# ---------------------------
class SFP_Lite(nn.Module):
    """
    A compact SFP-like network.

    Args:
        in_ch:   number of input channels (3 for RGB, 4 for RGB+Q-map)
        base_channels: feature width
        n_parts: number of processing parts (e.g., 4–6)
        use_se / use_cbam: enable attention in each part
        out_ch:  number of output channels (keep 3 for RGB even if in_ch=4)
        residual: add RGB skip connection (output = clamp(entry→parts→exit + input[:,:out_ch]))
        use_film: enable FiLM inside parts (for scalar conditioning passed at forward)
        film_hidden: FiLM MLP hidden size
    """
    def __init__(self,
                 in_ch: int = 3,
                 base_channels: int = 48,
                 n_parts: int = 4,
                 use_se: bool = False,
                 use_cbam: bool = False,
                 out_ch: int = 3,
                 residual: bool = True,
                 use_film: bool = False,
                 film_hidden: int = 32):
        super().__init__()

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.base_channels = int(base_channels)
        self.n_parts = int(n_parts)
        self.use_se = bool(use_se)
        self.use_cbam = bool(use_cbam)
        self.residual = bool(residual)
        self.use_film = bool(use_film)
        self.film_hidden = int(film_hidden)

        self.entry = nn.Conv2d(self.in_ch, self.base_channels, 3, padding=1, bias=True)

        self.parts = nn.ModuleList([
            BasicPB(self.base_channels, use_se=self.use_se, use_cbam=self.use_cbam,
                    use_film=self.use_film, film_hidden=self.film_hidden)
            for _ in range(self.n_parts)
        ])

        self.exit = nn.Conv2d(self.base_channels, self.out_ch, 3, padding=1, bias=True)

        # small init on exit for gentle early updates
        nn.init.zeros_(self.exit.bias)

    # EMA helper uses this to reconstruct identical topology
    def _get_name_kwargs(self):
        return dict(
            in_ch=self.in_ch,
            base_channels=self.base_channels,
            n_parts=self.n_parts,
            use_se=self.use_se,
            use_cbam=self.use_cbam,
            out_ch=self.out_ch,
            residual=self.residual,
            use_film=self.use_film,
            film_hidden=self.film_hidden,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B,in_ch,H,W). If in_ch==4 (RGB+Q), network still outputs 3ch RGB.
        cond: optional scalar conditioning (B,) or (B,1); only used if use_film=True.
        """
        h = self.entry(x)
        for p in self.parts:
            # if FiLM is disabled or cond is None, the block will ignore cond
            h = p(h, cond=cond)
        out = self.exit(h)

        if self.residual:
            # add RGB skip from input (first 3 channels if input has 4)
            skip = x[:, :self.out_ch, :, :]
            out = out + skip

        return torch.clamp(out, 0.0, 1.0)
