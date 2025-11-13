# dataset/synth.py
"""
Synthesized dataset loader:
- Loads high-quality images from a folder and generates compressed input
  by applying PIL JPEG with controllable or random quality.
- Use for DIV2K / Flickr2K base images. Outputs tensors in [0,1].
"""
import os
import glob
import random
from PIL import Image
import io
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch

class SynthJPEGDataset(Dataset):
    def __init__(self, root, crop_size=128, split='train', augment=True,
                 quality_min=5, quality_max=95):
        """
        Args:
            root: dataset root or list of roots
            crop_size: random crop size
            split: train / val (unused but kept for compatibility)
            augment: whether to apply random flips/rotations
            quality_min, quality_max: JPEG quality factor range (inclusive)
        """
        super().__init__()
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        if isinstance(root, (list, tuple)):
            files = []
            for r in root:
                cand = glob.glob(os.path.join(r, '**', '*.*'), recursive=True)
                cand = [p for p in cand if p.lower().endswith(exts)]
                files.extend(cand)
            # deduplicate while keeping order
            self.files = list(dict.fromkeys(sorted(files)))
        else:
            cand = glob.glob(os.path.join(root, '**', '*.*'), recursive=True)
            self.files = [p for p in cand if p.lower().endswith(exts)]

        self.crop_size = crop_size
        self.augment = augment
        self.qf_min = int(quality_min)
        self.qf_max = int(quality_max)

    def __len__(self):
        return len(self.files)

    def _random_jpeg(self, pil_img):
        q = random.randint(self.qf_min, self.qf_max)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img, q

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')

        # random crop
        if self.crop_size is not None:
            w, h = img.size
            if w < self.crop_size or h < self.crop_size:
                img = img.resize(
                    (max(w, self.crop_size), max(h, self.crop_size)),
                    Image.BICUBIC
                )
                w, h = img.size
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        # augmentation (flip + 90deg rotations using transpose to avoid resampling issues)
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            rot = random.choice([0, 1, 2, 3])  # 0=0°,1=90°,2=180°,3=270°
            if rot == 1:
                img = img.transpose(Image.ROTATE_90)
            elif rot == 2:
                img = img.transpose(Image.ROTATE_180)
            elif rot == 3:
                img = img.transpose(Image.ROTATE_270)

        # synth JPEG (create degraded input from GT)
        noisy_img, q = self._random_jpeg(img)

        # to tensor [0,1]
        gt = TF.to_tensor(img)         # GT in [0,1]
        inp = TF.to_tensor(noisy_img)  # Degraded input in [0,1]

        return inp, gt, q
