import os, glob, argparse, csv, yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from net.sfp_lite import SFP_Lite
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ---- speed/precision toggles (PyTorch 2.x safe) ----
cudnn.benchmark = True  # pick best algo for fixed shapes
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision('high')  # 'highest'|'high'|'medium'
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
# ----------------------------------------------------

# --------------- I/O helpers ----------------
def load_image(path):
    img = Image.open(path).convert('RGB')
    t = to_tensor(img)  # (C,H,W) in [0,1]
    return t

def save_image_tensor(t, path):
    t = t.detach().clamp(0,1).cpu()
    img = to_pil_image(t)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def to_np_uint8_rgb(t):
    # t: (1,C,H,W) [0,1] -> RGB uint8 (H,W,3)
    a = t[0].detach().clamp(0,1).cpu().numpy()
    a = np.transpose(a, (1,2,0))
    a = (a * 255.0).round().astype(np.uint8)
    return a

def to_np_uint8_y(t):
    # t: (1,C,H,W) [0,1] -> Y channel uint8
    a = to_np_uint8_rgb(t)
    y = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)[:,:,0]
    return y

def crop_border(im, b):
    if b <= 0: return im
    if im.shape[0] <= 2*b or im.shape[1] <= 2*b:
        return im
    return im[b:-b, b:-b]

def make_qmap_like(x, q_scalar):
    """
    x: (1,3,H,W) or (1,4,H,W)
    q_scalar: float in [1..100]
    returns (1,1,H,W) map with q/100
    """
    q = float(q_scalar)
    q = max(1.0, min(100.0, q))
    return torch.full((x.size(0), 1, x.size(2), x.size(3)), q/100.0,
                      device=x.device, dtype=x.dtype)

# ----------------- TTA (x8) -----------------
def _aug_list(x):
    augs = [
        x,
        torch.flip(x, dims=[-1]),
        torch.flip(x, dims=[-2]),
        torch.rot90(x, k=1, dims=(-2,-1)),
        torch.rot90(x, k=2, dims=(-2,-1)),
        torch.rot90(x, k=3, dims=(-2,-1)),
        torch.flip(torch.rot90(x, k=1, dims=(-2,-1)), dims=[-1]),
        torch.flip(torch.rot90(x, k=1, dims=(-2,-1)), dims=[-2]),
    ]
    return augs

def _deaug(y, idx):
    if idx == 0:  return y
    if idx == 1:  return torch.flip(y, dims=[-1])
    if idx == 2:  return torch.flip(y, dims=[-2])
    if idx == 3:  return torch.rot90(y, k=3, dims=(-2,-1))
    if idx == 4:  return torch.rot90(y, k=2, dims=(-2,-1))
    if idx == 5:  return torch.rot90(y, k=1, dims=(-2,-1))
    if idx == 6:  return torch.rot90(torch.flip(y, dims=[-1]), k=3, dims=(-2,-1))
    if idx == 7:  return torch.rot90(torch.flip(y, dims=[-2]), k=3, dims=(-2,-1))
    return y

def forward_with_tta(model, inp):
    outs = []
    for i, aug in enumerate(_aug_list(inp)):
        pred = model(aug)
        outs.append(_deaug(pred, i))
    out = torch.mean(torch.stack(outs, dim=0), dim=0)
    return out

# ----------------- Main -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help='Folder with input JPEGs')
    p.add_argument('--output', type=str, required=True, help='Folder to save restored images')
    p.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint (.pth)')
    p.add_argument('--ema', action='store_true', help='Checkpoint contains EMA weights only (or prefer EMA block)')
    p.add_argument('--gt', type=str, default=None, help='(Optional) GT folder for metrics (filenames must match)')
    p.add_argument('--crop-border', type=int, default=0, help='Crop border (pixels) before PSNR/SSIM')
    p.add_argument('--tta', action='store_true', help='Enable x8 Test-Time Augmentation')
    p.add_argument('--save-ext', type=str, default='.png', help='Output image extension (.png recommended)')
    p.add_argument('--metric', type=str, default='rgb', choices=['rgb','y'],
                   help='Metric space: rgb (paper) or y (luma-only)')

    # model config (defaults; may be overridden by YAML)
    p.add_argument('--base-ch', type=int, default=48)
    p.add_argument('--n-parts', type=int, default=4)
    p.add_argument('--use-se', action='store_true', default=True)
    p.add_argument('--use-cbam', action='store_true', default=False)

    # quality-aware conditioning (must match training)
    p.add_argument('--quality-cond', type=str, default='concat', choices=['none','concat'])
    p.add_argument('--q', type=float, default=None, help='Assumed JPEG quality (5..95). Overrides opts.')
    p.add_argument('--q-sweep', type=str, default=None,
                   help='Comma list of Q values to average, e.g. "10,20,30,40,60,80"')
    p.add_argument('--opts', type=str, default=None, help='Optional YAML to read defaults like test_q_default/test_q_sweep')
    return p.parse_args()

def _resolve_state(ck, prefer_ema=False):
    # return a state_dict from common checkpoint formats
    if isinstance(ck, dict):
        if prefer_ema:
            for k in ['model_ema', 'ema', 'state_dict_ema', 'ema_state_dict']:
                if k in ck and isinstance(ck[k], dict):
                    return ck[k]
        for k in ['state_dict', 'model', 'net', 'weights']:
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        some_keys = list(ck.keys())
        if some_keys and any(k.endswith('.weight') or k.endswith('.bias') for k in some_keys):
            return ck
    return ck

def _detect_n_parts_from_state(state):
    # count consecutive parts.k.* groups
    part_idxs = set()
    for k in state.keys():
        if k.startswith('parts.'):
            try:
                idx = int(k.split('.')[1])
                part_idxs.add(idx)
            except:  # ignore parsing errors
                pass
    return (max(part_idxs) + 1) if part_idxs else None

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load optional opts
    opt = {}
    if args.opts and os.path.isfile(args.opts):
        with open(args.opts, 'r') as f:
            opt = yaml.safe_load(f) or {}

    # ----- pull model hyperparams from opts if present (map common aliases) -----
    mopt = opt.get('model', {})
    args.base_ch      = int(mopt.get('base_channels', mopt.get('base_ch', args.base_ch)))
    args.n_parts      = int(mopt.get('n_parts', mopt.get('parts', args.n_parts)))
    args.use_se       = bool(mopt.get('use_se', args.use_se))
    args.use_cbam     = bool(mopt.get('use_cbam', args.use_cbam))
    args.quality_cond = str(mopt.get('quality_cond', args.quality_cond)).lower()
    # ---------------------------------------------------------------------------

    # defaults from opts or fallbacks
    q_default = float(opt.get('test_q_default', 75))
    q_sweep_default = list(map(float, opt.get('test_q_sweep', [10,20,30,40,60,80])))

    # infer in_ch from (possibly overridden) quality-cond
    in_ch = 4 if args.quality_cond == 'concat' else 3

    # ---------------- load weights (robust) ----------------
    ck = torch.load(args.ckpt, map_location=device)
    state = _resolve_state(ck, prefer_ema=args.ema)

    # auto-detect n_parts if mismatch
    auto_n_parts = _detect_n_parts_from_state(state)
    if auto_n_parts is not None and auto_n_parts != args.n_parts:
        print(f"[AUTO] Detected n_parts={auto_n_parts} from checkpoint (was {args.n_parts}). Using detected value.")
        args.n_parts = auto_n_parts

    print(f"[CFG] base_ch={args.base_ch}, n_parts={args.n_parts}, use_se={args.use_se}, use_cbam={args.use_cbam}, quality_cond={args.quality_cond}")
    if isinstance(ck, dict):
        print(f"[CKPT] top-level keys: {list(ck.keys())[:6]}")
    else:
        print("[CKPT] non-dict checkpoint (unexpected)")

    # model
    model = SFP_Lite(
        in_ch=in_ch,
        base_channels=args.base_ch,
        n_parts=args.n_parts,
        use_se=args.use_se,
        use_cbam=args.use_cbam
    ).to(device)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print("\n[WARN] State dict mismatch detected.")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    model.eval()

    # collect files
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(args.input, e)))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError("No input images found.")

    os.makedirs(args.output, exist_ok=True)

    have_gt = args.gt is not None and os.path.isdir(args.gt)
    total_psnr, total_ssim, count = 0.0, 0.0, 0

    # CSV log
    csv_path = os.path.join(args.output, 'eval_log.csv')
    csv_header_written = False
    csv_f = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_f)

    # helper to run once with a specific Q
    def run_with_q(inp3, qval):
        if args.quality_cond == 'concat':
            qmap = make_qmap_like(inp3, qval)
            inp = torch.cat([inp3, qmap], dim=1)  # (1,4,H,W)
        else:
            inp = inp3
        with torch.no_grad():
            if args.tta:
                return forward_with_tta(model, inp)
            else:
                return model(inp)

    for fp in tqdm(files, desc='[TEST]'):
        name = os.path.splitext(os.path.basename(fp))[0]
        inp3 = load_image(fp).unsqueeze(0).to(device, memory_format=torch.channels_last)  # (1,3,H,W)

        # choose Q strategy
        if args.q is not None:
            preds = [run_with_q(inp3, args.q)]
        else:
            qs = q_sweep_default
            if args.q_sweep:
                qs = [float(v) for v in args.q_sweep.split(',') if v.strip()]
            if not qs:
                qs = [q_default]
            preds = [run_with_q(inp3, qv) for qv in qs]

        # average if multiple
        pred = torch.mean(torch.stack(preds, dim=0), dim=0) if len(preds) > 1 else preds[0]

        # save image
        out_path = os.path.join(args.output, name + args.save_ext)
        save_image_tensor(pred[0], out_path)

        if have_gt:
            # find GT by same name with common extensions
            gt = None
            for ge in ('.png', '.jpg', '.jpeg', '.bmp'):
                gpath = os.path.join(args.gt, name + ge)
                if os.path.exists(gpath):
                    gt = load_image(gpath).unsqueeze(0).to(device)
                    break

            if gt is not None:
                cb = max(0, int(args.crop_border))

                if args.metric == 'rgb':
                    pred_rgb = to_np_uint8_rgb(pred)
                    gt_rgb   = to_np_uint8_rgb(gt)
                    if cb > 0:
                        pred_rgb = crop_border(pred_rgb, cb)
                        gt_rgb   = crop_border(gt_rgb, cb)
                    psnr = compare_psnr(gt_rgb, pred_rgb, data_range=255)
                    # skimage SSIM expects channel_axis for color
                    ssim = compare_ssim(gt_rgb, pred_rgb, data_range=255, channel_axis=2)
                else:
                    pred_y = to_np_uint8_y(pred)
                    gt_y   = to_np_uint8_y(gt)
                    if cb > 0:
                        pred_y = crop_border(pred_y, cb)
                        gt_y   = crop_border(gt_y, cb)
                    psnr = compare_psnr(gt_y, pred_y, data_range=255)
                    ssim = compare_ssim(gt_y, pred_y, data_range=255)

                total_psnr += psnr
                total_ssim += ssim
                count += 1

                if not csv_header_written:
                    writer.writerow(['name', 'psnr', 'ssim', 'metric', 'tta', 'q_used'])
                    csv_header_written = True
                q_used = args.q if args.q is not None else (args.q_sweep if args.q_sweep else q_sweep_default)
                writer.writerow([name, f'{psnr:.4f}', f'{ssim:.6f}', args.metric, int(args.tta), q_used])

    csv_f.close()

    if have_gt and count > 0:
        print(f"\n== Averages ({args.metric.upper()}, crop={args.crop_border}, TTA={args.tta}) ==")
        print(f"PSNR: {total_psnr / count:.3f}  |  SSIM: {total_ssim / count:.4f}")
        print(f"Per-image CSV saved to: {csv_path}")
    else:
        print("\nDone. (No GT folder provided; skipped metrics.)")

if __name__ == '__main__':
    main()
