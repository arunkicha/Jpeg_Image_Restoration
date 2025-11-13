import torch

ckpt = torch.load("exp/se_hybrid_b12_e120_psnr/best_model_ema.pth", map_location="cpu")

if isinstance(ckpt, dict):
    print("Top-level keys:", ckpt.keys())
    for k in ['epoch', 'global_step', 'best_psnr', 'val_psnr']:
        if k in ckpt:
            print(f"{k}:", ckpt[k])
else:
    print("This is a raw state_dict (EMA weights only). Num keys:", len(ckpt))
    keys = list(ckpt.keys())
    print("First 10 keys:", keys[:10])
