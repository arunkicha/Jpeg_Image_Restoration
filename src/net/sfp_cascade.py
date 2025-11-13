# net/sfp_cascade.py
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from .naf import NAFBlock

# ---- GlobalPB (ViT-lite at 1/8) ----
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d); self.act = nn.GELU(); self.fc2 = nn.Linear(4*d, d)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d); self.mlp = MLP(d)
    def forward(self, x):
        h = self.norm1(x); a,_ = self.attn(h,h,h); x = x + a
        h = self.norm2(x); x = x + self.mlp(h)
        return x

class GlobalPB(nn.Module):
    """ Operates only at 1/8 scale (cheap, global). """
    def __init__(self, c, heads=4):
        super().__init__()
        self.proj_in = nn.Conv2d(c, c, 1,1,0)
        self.tb = TransformerBlock(c, heads=heads)
        self.proj_out = nn.Conv2d(c, c, 1,1,0)
    def forward(self, x):
        b,c,h,w = x.shape
        z = self.proj_in(x).permute(0,2,3,1).contiguous().view(b, h*w, c)
        z = self.tb(z)
        z = z.view(b,h,w,c).permute(0,3,1,2).contiguous()
        return self.proj_out(z) + x

# ---- BasicPB with QE gating ----
class BasicPB(nn.Module):
    def __init__(self, c, use_se=False, use_cbam=False):
        super().__init__()
        self.body = NAFBlock(c)
        self.q_fc = nn.Sequential(nn.Linear(1, c), nn.Sigmoid())  # channel gate from scalar q~
        self.se = nn.Identity(); self.cbam = nn.Identity()
        if use_se:
            from .attention import SEBlock
            self.se = SEBlock(c)
        if use_cbam:
            from .attention import CBAM
            self.cbam = CBAM(c)
    def forward(self, x, q_scalar):
        g = self.q_fc(q_scalar.view(-1,1)).view(x.size(0), -1, 1, 1)
        x = self.body(x) * g + x
        x = self.se(x); x = self.cbam(x)
        return x

# ---- U-Net-ish IP Part ----
class IPPart(nn.Module):
    def __init__(self, in_c, c=64, use_se=False, use_cbam=False, add_global=True):
        super().__init__()
        # enc
        self.in_conv = nn.Conv2d(in_c, c, 3,1,1)
        self.enc1 = BasicPB(c,use_se,use_cbam)
        self.down1 = nn.Conv2d(c,c,3,2,1)
        self.enc2 = BasicPB(c,use_se,use_cbam)
        self.down2 = nn.Conv2d(c,c,3,2,1)   # -> 1/4
        self.enc3 = BasicPB(c,use_se,use_cbam)
        self.down3 = nn.Conv2d(c,c,3,2,1)   # -> 1/8
        self.enc4 = BasicPB(c,use_se,use_cbam)
        self.globalpb = GlobalPB(c) if add_global else nn.Identity()

        # dec
        self.up3 = nn.PixelShuffle(1)  # no change; we’ll up by convt
        self.deconv3 = nn.ConvTranspose2d(c, c, 4, 2, 1)
        self.dec3 = BasicPB(c,use_se,use_cbam)
        self.deconv2 = nn.ConvTranspose2d(c, c, 4, 2, 1)
        self.dec2 = BasicPB(c,use_se,use_cbam)
        self.deconv1 = nn.ConvTranspose2d(c, c, 4, 2, 1)
        self.dec1 = BasicPB(c,use_se,use_cbam)

        self.out_conv = nn.Conv2d(c, 3, 3,1,1)

    def forward(self, x, q, prev_feats=None):
        h0 = self.in_conv(x)
        e1 = self.enc1(h0, q)                   # 1
        e2 = self.enc2(self.down1(e1), q)       # 1/2
        e3 = self.enc3(self.down2(e2), q)       # 1/4
        e4 = self.enc4(self.down3(e3), q)       # 1/8
        e4 = self.globalpb(e4)

        d3 = self.dec3(self.deconv3(e4), q) + e3
        d2 = self.dec2(self.deconv2(d3), q) + e2
        d1 = self.dec1(self.deconv1(d2), q) + e1

        y = x[:, :3] + self.out_conv(d1)        # residual to RGB input
        # features for next part (decoder outputs)
        feats = (d1.detach(), d2.detach(), d3.detach(), e4.detach())
        return torch.clamp(y,0,1), feats

# ---- QE Part ----
class QEPart(nn.Module):
    def __init__(self, in_c=3, c=64):
        super().__init__()
        self.head = nn.Conv2d(in_c, c, 3,1,1)
        self.b1 = NAFBlock(c); self.b2 = NAFBlock(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c, c//2), nn.ReLU(True), nn.Linear(c//2, 1))
    def forward(self, x):
        h = self.b2(self.b1(self.head(x)))
        q = torch.sigmoid(self.fc(self.pool(h)))  # in [0,1]
        return q

# ---- Full model ----
class SFP_Cascade(nn.Module):
    def __init__(self, in_ch=3, base_channels=64, n_parts=6, use_se=False, use_cbam=False):
        super().__init__()
        self.n_parts = n_parts
        self.qe = QEPart(in_c=in_ch, c=base_channels)
        self.parts = nn.ModuleList([
            IPPart(in_c=in_ch, c=base_channels, use_se=use_se, use_cbam=use_cbam, add_global=True if i<=2 else False)
            for i in range(n_parts)
        ])

    def _get_name_kwargs(self):
        return dict(in_ch=3, base_channels=64, n_parts=self.n_parts)

    def forward(self, x):
        q = self.qe(x)                    # (B,1) quality representation (low q ⇒ more degradation)
        outs = []
        feats = None
        y = x
        for p in self.parts:
            y, feats = p(y, q, prev_feats=feats)
            outs.append(y)
        return outs, q
