import os
import argparse
import yaml
import math
import random
import csv
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from contextlib import contextmanager, nullcontext

from dataset.synth import SynthJPEGDataset
from net.sfp_lite import SFP_Lite
from algorithm.losses import HybridLoss
from net.sfp_cascade import SFP_Cascade
from algorithm.losses import MPIRLoss
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ---------------------------
# Global setup
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ---------------------------
# EMA Helper
# ---------------------------
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.ema.to(device)

    def _clone_model(self, model):
        ema_model = type(model)(**model._get_name_kwargs()) if hasattr(model, '_get_name_kwargs') else type(model)()
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
        return ema_model

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

    def state_dict(self):
        return self.ema.state_dict()


# ---------------------------
# Deterministic Validation Context
# ---------------------------
@contextmanager
def deterministic_eval():
    orig_bench = cudnn.benchmark
    orig_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        cudnn.benchmark = orig_bench
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = orig_allow_tf32


# ---------------------------
# Utilities
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--opt', type=str, default='opts/opts_se_hybrid.yml')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def load_opt(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def group_params(model, new_keywords=('se', 'cbam', 'attention')):
    pretrained, new = [], []
    for n, p in model.named_parameters():
        (new if any(k in n.lower() for k in new_keywords) else pretrained).append(p)
    return pretrained, new

def tensor_to_np(t):
    t = t.detach().clamp(0, 1).cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    return (t * 255.0).round().astype(np.uint8)

def crop_border(im, b=0):
    if b <= 0:
        return im
    return im[b:-b, b:-b] if im.shape[0] > 2*b and im.shape[1] > 2*b else im

def make_qmap_like(x: torch.Tensor, q_scalar: torch.Tensor | float):
    """
    x: (B,C,H,W) tensor on device; q_scalar: float or (B,) tensor in [1..100]
    returns: (B,1,H,W) quality map = q/100
    """
    if isinstance(q_scalar, torch.Tensor):
        q = q_scalar.float().view(-1, 1, 1, 1)
    else:
        q = torch.full((x.size(0), 1, 1, 1), float(q_scalar), device=x.device, dtype=x.dtype)
    q = q.clamp(1.0, 100.0) / 100.0
    return q.expand(-1, 1, x.size(-2), x.size(-1))


# ---------------------------
# Validation helpers
# ---------------------------
def build_val_loaders(opt, dataset_cls, device_pin=True, num_workers=0):
    crop = opt.get('crop', 128)
    val_root = opt.get('valid_roots', opt.get('valid_root', opt.get('train_roots')))
    use_fixed = bool(opt.get('val_use_fixed_qfs', False))
    if not use_fixed:
        ds = dataset_cls(
            val_root, crop_size=crop, augment=False,
            quality_min=opt.get('val_quality_min', 10),
            quality_max=opt.get('val_quality_max', 40),
        )
        return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                          pin_memory=device_pin, persistent_workers=(num_workers > 0))

    loaders = {}
    for q in opt.get('val_fixed_qfs', [10, 20, 30, 40]):
        ds = dataset_cls(val_root, crop_size=crop, augment=False, quality_min=q, quality_max=q)
        loaders[q] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                                pin_memory=device_pin, persistent_workers=(num_workers > 0))
    return loaders

def validate_one_loader(val_loader, model, device, crop_b=0, quality_cond='none'):
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='[VAL]', leave=False):
            # batch: (inp, gt, q)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inp, gt = batch[0], batch[1]
                q = batch[2] if len(batch) >= 3 else None
            elif isinstance(batch, dict):
                inp, gt = batch['inp'], batch['gt']
                q = batch.get('q', None)
            else:
                raise RuntimeError("Validation batch format unexpected.")

            inp, gt = inp.to(device), gt.to(device)

            # quality-aware concat if enabled
            if quality_cond == 'concat':
                if q is None:
                    # fallback to mid quality if dataset didn't return q
                    q = torch.full((inp.size(0),), 75.0, device=inp.device)
                qmap = make_qmap_like(inp, q)  # (B,1,H,W)
                inp = torch.cat([inp, qmap], dim=1)  # (B,4,H,W)

            pred = model(inp)

            pred_np = tensor_to_np(pred[0])
            gt_np = tensor_to_np(gt[0])
            pred_y = cv2.cvtColor(pred_np, cv2.COLOR_RGB2YCrCb)[:,:,0]
            gt_y   = cv2.cvtColor(gt_np,   cv2.COLOR_RGB2YCrCb)[:,:,0]
            if crop_b > 0:
                pred_y = crop_border(pred_y, crop_b)
                gt_y   = crop_border(gt_y,   crop_b)
            psnr_sum += compare_psnr(gt_y, pred_y, data_range=255)
            ssim_sum += compare_ssim(gt_y, pred_y, data_range=255)
            n += 1
    model.train()
    return psnr_sum/n, ssim_sum/n

def validate_multiqf(val_loaders, model, device, crop_b=0, quality_cond='none'):
    if isinstance(val_loaders, dict):
        metrics = {}
        for qf, loader in val_loaders.items():
            psnr, ssim = validate_one_loader(loader, model, device, crop_b, quality_cond)
            metrics[qf] = {'psnr': psnr, 'ssim': ssim}
        avg_psnr = sum(m['psnr'] for m in metrics.values()) / len(metrics)
        avg_ssim = sum(m['ssim'] for m in metrics.values()) / len(metrics)
        return metrics, avg_psnr, avg_ssim
    else:
        psnr, ssim = validate_one_loader(val_loaders, model, device, crop_b, quality_cond)
        return { -1: {'psnr': psnr, 'ssim': ssim} }, psnr, ssim


# ---------------------------
# Curriculum helper
# ---------------------------
def get_qf_stage_for_epoch(epoch, qf_stages, default=(5, 95)):
    if not qf_stages:
        return default
    for s in qf_stages:
        se = int(s.get('start_epoch', 0))
        ee = int(s.get('end_epoch', 1 << 30))
        if se <= epoch < ee:
            return int(s.get('qmin', default[0])), int(s.get('qmax', default[1]))
    return default


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    opt = load_opt(args.opt)

    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    pin_memory = bool(use_cuda)

    cpu_count = os.cpu_count() or 4
    num_workers = opt.get('workers', min(8, cpu_count))

    quality_cond = opt.get('quality_cond', 'none')  # 'none' or 'concat'
    in_ch = 4 if quality_cond == 'concat' else 3

    # ---- Train dataset ----
    train_root = opt.get('train_roots', opt.get('train_root'))
    init_qmin, init_qmax = get_qf_stage_for_epoch(0, opt.get('qf_stages', []))
    train_ds = SynthJPEGDataset(train_root, crop_size=opt.get('crop', 128),
                                augment=True, quality_min=init_qmin, quality_max=init_qmax)
    train_loader = DataLoader(train_ds, batch_size=opt.get('batch_size', 12),
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0),
                              prefetch_factor=(min(8, int(opt.get('prefetch_factor', 4))) if num_workers > 0 else None))
    print(f"[CURRICULUM] Initial train QF: {init_qmin}-{init_qmax} | Dataset size: {len(train_ds)}")

    # ---- Val loaders ----
    val_loaders = build_val_loaders(opt, SynthJPEGDataset, True, num_workers)

    # ---- Model ----
    model = SFP_Lite(in_ch=in_ch,
                     base_channels=opt.get('base_channels', 48),
                     n_parts=opt.get('n_parts', 4),
                     use_se=opt.get('use_se', True),
                     use_cbam=opt.get('use_cbam', False)).to(device)
    try: model = model.to(memory_format=torch.channels_last)
    except: pass

    ema = ModelEMA(model, decay=opt.get('ema_decay', 0.9999), device=device)

    # ---- Optimizer ----
    pretrained_params, new_params = group_params(model)
    opt_groups = [
        {'params': pretrained_params, 'lr': opt.get('lr_pretrained', 1e-5)},
        {'params': new_params, 'lr': opt.get('lr_new', 1e-4)}
    ]
    optimizer = torch.optim.Adam(opt_groups, fused=True) if hasattr(torch.optim.Adam, 'fused') else torch.optim.Adam(opt_groups)

    # ---- Scheduler ----
    total_epochs = int(opt.get('epochs', 50))
    total_steps = len(train_loader) * total_epochs
    warmup_steps = int(opt.get('warmup_steps', 1000))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Loss ----
    loss_fn = HybridLoss(alpha=opt.get('alpha', 1.0),
                         beta=opt.get('beta', 0.005),
                         gamma=opt.get('gamma', 0.3),
                         use_msssim=opt.get('use_msssim', True),
                         dct_weight=opt.get('dct_weight', 0.05)).to(device)

    # ---- AMP ----
    use_amp = bool(opt.get('use_amp', True)) and use_cuda
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    best_psnr = -1.0
    save_dir = opt.get('save_dir', 'exp/se_hybrid')
    os.makedirs(save_dir, exist_ok=True)

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        state = ck.get('model', ck)
        model.load_state_dict(state, strict=False)
        if 'optimizer' in ck:
            optimizer.load_state_dict(ck['optimizer'])
        if use_amp and 'scaler' in ck and scaler is not None:
            scaler.load_state_dict(ck['scaler'])
        start_epoch = int(ck.get('epoch', -1)) + 1
        global_step = int(ck.get('global_step', 0))
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- Dry run ----
    if args.dry_run:
        model.eval()
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inp, gt = batch[0], batch[1]
            q = batch[2] if len(batch) >= 3 else None
        else:
            inp, gt, q = batch['inp'], batch['gt'], batch.get('q', None)
        inp = inp.to(device, memory_format=torch.channels_last)
        gt  = gt.to(device,  memory_format=torch.channels_last)
        if quality_cond == 'concat':
            if q is None:
                q = torch.full((inp.size(0),), 75.0, device=inp.device)
            qmap = make_qmap_like(inp, q)
            inp = torch.cat([inp, qmap], dim=1)
        with torch.no_grad(), amp_ctx:
            pred = model(inp)
            _ = loss_fn(pred, gt)
        print(f"✅ Dry run OK | inp {tuple(inp.shape)} pred {tuple(pred.shape)}")
        return

    # ---- Training loop ----
    log_interval = int(opt.get('log_interval', 500))
    sample_interval = int(opt.get('sample_interval', 5000))
    ckpt_every = int(opt.get('ckpt_every', 1))
    accum_steps = max(1, int(opt.get('accum_steps', 1)))
    val_freq = opt.get('val_freq', 1)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, total_epochs):
        epoch_loss = 0.0
        qmin, qmax = get_qf_stage_for_epoch(epoch, opt.get('qf_stages', []))
        train_ds.quality_min, train_ds.quality_max = qmin, qmax
        print(f"[CURRICULUM] Epoch {epoch}: QF range {qmin}-{qmax}")

        for i, batch in enumerate(train_loader):
            # unpack batch (inp, gt, q)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inp, gt = batch[0], batch[1]
                q = batch[2] if len(batch) >= 3 else None
            elif isinstance(batch, dict):
                inp, gt = batch['inp'], batch['gt']
                q = batch.get('q', None)
            else:
                raise RuntimeError("Training batch format unexpected.")

            inp = inp.to(device, non_blocking=True, memory_format=torch.channels_last)
            gt  = gt.to(device,  non_blocking=True, memory_format=torch.channels_last)

            # --- Quality conditioning (concat q-map) ---
            if quality_cond == 'concat':
                if q is None:
                    q = torch.full((inp.size(0),), 75.0, device=inp.device)
                else:
                    q = q.to(device, non_blocking=True)
                qmap = make_qmap_like(inp, q)  # (B,1,H,W)
                inp = torch.cat([inp, qmap], dim=1)  # (B,4,H,W)

            with amp_ctx:
                pred = model(inp)
                loss = loss_fn(pred, gt) / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            epoch_loss += loss.item() * accum_steps
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = epoch_loss / (i + 1)
                current_lr = scheduler.get_last_lr()
                print(f"Epoch {epoch} | Step {global_step} | Loss {avg_loss:.6f} | LR {current_lr}")

            if global_step % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    # for visual, rebuild q-map for first sample
                    vis_inp = inp[0:1].detach().cpu().clamp(0,1) if inp.size(1)==3 else inp[0:1, :3].detach().cpu().clamp(0,1)
                    grid = torch.cat([vis_inp, pred[0:1].detach().cpu().clamp(0,1), gt[0:1].detach().cpu().clamp(0,1)], dim=0)
                    save_image(grid, os.path.join(save_dir, f'sample_{global_step}.png'), nrow=3)
                model.train()

        # ---- Validation ----
        if (epoch + 1) % val_freq == 0:
            with deterministic_eval():
                metrics, avg_psnr, avg_ssim = validate_multiqf(
                    val_loaders, ema.ema, device, crop_b=opt.get('val_crop_border', 0), quality_cond=quality_cond
                )

            if isinstance(val_loaders, dict):
                qf_line = " | ".join([f"QF{q}: {m['psnr']:.2f}/{m['ssim']:.4f}" for q, m in sorted(metrics.items())])
                print(f"[VAL] Epoch {epoch+1} | {qf_line} || AVG PSNR: {avg_psnr:.3f} | AVG SSIM: {avg_ssim:.4f}")
            else:
                m = metrics[-1]
                print(f"[VAL] Epoch {epoch+1} | PSNR: {m['psnr']:.3f} | SSIM: {m['ssim']:.4f}")

            # CSV log
            csv_path = os.path.join(save_dir, 'val_log.csv')
            header_needed = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                if header_needed:
                    if isinstance(val_loaders, dict):
                        hdr = ['epoch', 'avg_psnr', 'avg_ssim'] + [f'psnr_qf{q}' for q in sorted(metrics.keys())] + [f'ssim_qf{q}' for q in sorted(metrics.keys())]
                    else:
                        hdr = ['epoch', 'psnr', 'ssim']
                    w.writerow(hdr)
                if isinstance(val_loaders, dict):
                    row = [epoch+1, avg_psnr, avg_ssim] + [metrics[q]['psnr'] for q in sorted(metrics.keys())] + [metrics[q]['ssim'] for q in sorted(metrics.keys())]
                else:
                    row = [epoch+1, metrics[-1]['psnr'], metrics[-1]['ssim']]
                w.writerow(row)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(ema.state_dict(), os.path.join(save_dir, 'best_model_ema.pth'))
                print(f"✅ Best EMA updated on AVG PSNR={best_psnr:.3f}")

        # ---- Checkpoint ----
        if epoch % ckpt_every == 0:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': (scaler.state_dict() if (use_amp and scaler is not None) else None),
                'opt': opt,
                'epoch': epoch,
                'global_step': global_step
            }
            torch.save(ckpt, os.path.join(save_dir, f'ckpt_epoch_{epoch}.pth'))
            print(f"Saved checkpoint at epoch {epoch}")

if __name__ == '__main__':
    main()
