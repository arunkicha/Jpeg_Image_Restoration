# net/naf.py
import torch, torch.nn as nn, torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):  # x split on channel
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    NAFNet-style block (ECCV'22) simplified:
      Conv(1x1) -> DWConv(3x3) -> SimpleGate -> Conv(1x1) with LayerNorm2d,
      plus lightweight channel attention (squeeze 1x1) and residual with beta/gamma.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dwc = c * dw_expand
        ffnc = c * ffn_expand

        self.norm1 = nn.LayerNorm([c, 1, 1], elementwise_affine=True)
        self.pw1 = nn.Conv2d(c, dwc, 1, 1, 0)
        self.dw = nn.Conv2d(dwc, dwc, 3, 1, 1, groups=dwc)
        self.sg = SimpleGate()
        self.se = nn.Conv2d(dwc // 2, c, 1, 1, 0)  # cheap channel weighting

        self.norm2 = nn.LayerNorm([c, 1, 1], elementwise_affine=True)
        self.pw2 = nn.Conv2d(c, ffnc, 1, 1, 0)
        self.pw3 = nn.Conv2d(ffnc, c, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        h = self.norm1(x)
        h = self.pw1(h)
        h = self.dw(h)
        h = self.sg(h)
        h = self.se(h)
        y = x + h * self.beta

        h2 = self.norm2(y)
        h2 = self.pw3(F.gelu(self.pw2(h2)))
        return y + h2 * self.gamma
