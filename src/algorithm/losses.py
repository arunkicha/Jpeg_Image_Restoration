# algorithm/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Iterable, Tuple

from torchvision.models import vgg16, VGG16_Weights
from pytorch_msssim import ssim as ssim_pt, ms_ssim as ms_ssim_pt


# -----------------------------
# Utility helpers
# -----------------------------
def _to_3ch(x: torch.Tensor) -> torch.Tensor:
    """Ensure 3 channels for perceptual loss / SSIM."""
    if x.size(1) == 3:
        return x
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3, ...]


# -----------------------------
# Charbonnier (robust L1)
# -----------------------------
class CharbonnierLoss(nn.Module):
    """Smooth L1 loss (Charbonnier)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps2))


# -----------------------------
# Perceptual (VGG16)
# -----------------------------
class VGGPerceptual(nn.Module):
    """
    VGG16 perceptual features in fp32 (safe under AMP).
    """
    _LAYER_IDX = {
        'relu1_2': 3,
        'relu2_2': 8,
        'relu3_3': 15,
        'relu4_3': 22,
    }

    def __init__(self, layers: Iterable[str] = ('relu2_2', 'relu3_3')):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        cut_ids = [self._LAYER_IDX[k] for k in layers]
        self.slices = nn.ModuleList()
        start = 0
        for cid in cut_ids:
            self.slices.append(vgg[start:cid + 1])
            start = cid + 1
        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = _to_3ch(x).float()
        x = (x.clamp(0, 1) - self.mean) / self.std
        feats = []
        h = x
        for s in self.slices:
            h = s(h)
            feats.append(h)
        return tuple(feats)


# -----------------------------
# DCT helpers (8x8 block DCT-II)
# -----------------------------
_DCT_CACHE = {}

def _get_dct_matrix(N: int, device, dtype):
    key = (device, dtype, N)
    if key not in _DCT_CACHE:
        k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
        n = torch.arange(N, device=device, dtype=dtype).view(1, -1)
        D = torch.cos((torch.pi * (2 * n + 1) * k) / (2 * N))
        D[0] = D[0] / (N ** 0.5)
        D[1:] = D[1:] * (2 / N) ** 0.5
        _DCT_CACHE[key] = D
    return _DCT_CACHE[key]

def dct2_8x8(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    D = _get_dct_matrix(8, x.device, x.dtype)
    patches = F.unfold(x, kernel_size=8, stride=8)
    patches = patches.transpose(1, 2).contiguous().view(B, -1, C, 8, 8)
    X = torch.einsum('ij,blcjk->blcik', D, patches)
    X = torch.einsum('blcik,kj->blcij', X, D.t())
    return X


# -----------------------------
# Hybrid Loss
# -----------------------------
class HybridLoss(nn.Module):
    """
    total = alpha * L1/Charbonnier
          + beta * Perceptual(VGG)
          + gamma * (1 - {MS-}SSIM)
          + dct_weight * L1(DCT)
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 0.004,
                 gamma: float = 0.3,
                 use_msssim: bool = True,
                 dct_weight: float = 0.03,
                 use_charbonnier: bool = False,
                 use_y_for_ssim: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.use_msssim = use_msssim
        self.dct_weight = float(dct_weight)
        self.use_y_for_ssim = use_y_for_ssim

        self.perc = VGGPerceptual(('relu2_2', 'relu3_3'))
        self.l1_like = CharbonnierLoss(1e-3) if use_charbonnier else nn.L1Loss()

        # fixed radial mask for high-frequency DCT emphasis
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
        freq = (xx.float() ** 2 + yy.float() ** 2)
        mask = freq / max(1.0, float(freq.max()))
        self.register_buffer('dct_mask', mask.view(1, 1, 1, 8, 8), persistent=False)

    def perceptual_loss(self, pred, gt):
        pf = self.perc(pred)
        gf = self.perc(gt)
        loss = pred.new_tensor(0.0, dtype=torch.float32)
        for a, b in zip(pf, gf):
            loss += F.l1_loss(a, b)
        return loss

    def _rgb_to_y(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def ssim_term(self, pred, gt):
        x = pred.clamp(0, 1)
        y = gt.clamp(0, 1)
        if self.use_y_for_ssim:
            x = _to_3ch(x)
            y = _to_3ch(y)
            x = self._rgb_to_y(x)
            y = self._rgb_to_y(y)
        val = ms_ssim_pt(x, y, data_range=1.0, size_average=True) if self.use_msssim else ssim_pt(x, y, data_range=1.0, size_average=True)
        return 1.0 - val

    def dct_loss(self, pred, gt):
        if self.dct_weight <= 0:
            return pred.new_tensor(0.0)
        P = dct2_8x8(pred.clamp(0, 1))
        G = dct2_8x8(gt.clamp(0, 1))
        return F.l1_loss(P * self.dct_mask, G * self.dct_mask)

    def forward(self, pred, gt):
        loss_l1 = self.l1_like(pred, gt) * self.alpha
        loss_perc = self.perceptual_loss(pred, gt) * self.beta if self.beta > 0 else pred.new_tensor(0.0)
        loss_struct = self.ssim_term(pred, gt) * self.gamma if self.gamma > 0 else pred.new_tensor(0.0)
        loss_dct = self.dct_loss(pred, gt) * self.dct_weight if self.dct_weight > 0 else pred.new_tensor(0.0)
        return loss_l1 + loss_perc + loss_struct + loss_dct


# Expose plain SSIM for external calls
def ssim(pred, gt):
    return ssim_pt(pred.clamp(0, 1), gt.clamp(0, 1), data_range=1.0, size_average=True)

# ---- Multi-Part IR loss + QE loss (paper) ----
class MPIRLoss(nn.Module):
    """
    L = sum_k λ_p(k) * [ |Ik - GT|_1 + λ_FD |FFT(Ik) - FFT(GT)|_1 ] + λ_Q |Q_hat - Q_gt|
    Options: add tiny SSIM and/or YCbCr term (togglable).
    """
    def __init__(self, lambda_fd=0.005, lambda_q=0.05, lambda_ssim=0.0, lambda_cbcr=0.0, use_msssim=False):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_fd = float(lambda_fd)
        self.lambda_q = float(lambda_q)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_cbcr = float(lambda_cbcr)
        self.use_msssim = use_msssim

    def _fft_l1(self, a, b):
        A = torch.fft.rfftn(a, dim=(-2,-1)); B = torch.fft.rfftn(b, dim=(-2,-1))
        return (A - B).abs().mean()

    def _ycbcr(self, x):
        r,g,b = x[:,0:1], x[:,1:2], x[:,2:3]
        y  = 0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
        cr = 0.5*r - 0.418688*g - 0.081312*b + 0.5
        return y, cb, cr

    def _ssim_term(self, x, y):
        if self.lambda_ssim <= 0: return x.new_tensor(0.0)
        if self.use_msssim:
            return (1.0 - ms_ssim_pt(x, y, data_range=1.0, size_average=True))
        else:
            return (1.0 - ssim_pt(x, y, data_range=1.0, size_average=True))

    def forward(self, outs, q_hat, gt, qf_gt=None):
        """
        outs: list of Bx3HxW images (per part)
        q_hat: Bx1 in [0,1]
        gt: Bx3HxW
        qf_gt: B (0..100); if None, weights fallback to uniform
        """
        N = len(outs)
        if qf_gt is None:
            lambdas = [1.0/N]*N
            q_loss = gt.new_tensor(0.0)
        else:
            Qgt = 1.0 - (qf_gt.float().view(-1,1)/100.0)  # 0..1
            # per-batch λ_p(k)
            lam = []
            for k in range(1, N+1):
                num = N - (N-1) * (Qgt - (k/N)).abs()
                lam.append(num)
            lam = torch.stack(lam, dim=1)  # BxN
            lam = lam / lam.sum(dim=1, keepdim=True)
            lambdas = [lam[:,k-1].mean().item() for k in range(1,N+1)]  # scalar weights averaged over batch
            q_loss = (q_hat.view(-1,1) - Qgt).abs().mean() * self.lambda_q

        total = 0.0
        for k, Ik in enumerate(outs, start=1):
            w = float(lambdas[k-1])
            l_sp = self.l1(Ik, gt)
            l_fd = self._fft_l1(Ik, gt) if self.lambda_fd>0 else 0.0
            l = l_sp + self.lambda_fd * l_fd

            if self.lambda_ssim > 0:
                yx, yg = Ik[:, :1], gt[:, :1]
                l += self.lambda_ssim * self._ssim_term(yx, yg)

            if self.lambda_cbcr > 0:
                _, cb1, cr1 = self._ycbcr(Ik); _, cb2, cr2 = self._ycbcr(gt)
                l += self.lambda_cbcr * (self.l1(cb1, cb2) + self.l1(cr1, cr2)) * 0.5

            total += w * l
        return total + q_loss
