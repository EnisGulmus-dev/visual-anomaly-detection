from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def img_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    if mask.mode != "L":
        mask = mask.convert("L")
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


class MVTecBottle(Dataset):
    
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        self.root = Path(root) / "bottle"
        self.split = split
        self.image_size = image_size
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict]:
        samples = []

        if self.split == "train":
            img_dir = self.root / "train" / "good"
            for img_path in sorted(img_dir.glob("*.png")):
                samples.append({
                    "img": img_path,
                    "mask": None,
                    "label": 0,
                    "defect": "good"
                })
            return samples

        test_dir = self.root / "test"
        for defect_dir in test_dir.iterdir():
            if not defect_dir.is_dir():
                continue

            defect = defect_dir.name
            for img_path in sorted(defect_dir.glob("*.png")):
                if defect == "good":
                    samples.append({
                        "img": img_path,
                        "mask": None,
                        "label": 0,
                        "defect": "good"
                    })
                else:
                    mask_path = self.root / "ground_truth" / defect / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Mask not found for {img_path}")
                    samples.append({
                        "img": img_path,
                        "mask": mask_path,
                        "label": 1,
                        "defect": defect
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = Image.open(s["img"]).resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )
        img = img_to_tensor(img)

        if s["mask"] is None:
            mask = torch.zeros((1, self.image_size, self.image_size))
        else:
            m = Image.open(s["mask"]).resize(
                (self.image_size, self.image_size),
                Image.NEAREST
            )
            mask = mask_to_tensor(m)

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(s["label"]),
            "defect_type": s["defect"],
            "path": str(s["img"])
        }
