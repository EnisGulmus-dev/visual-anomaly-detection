import os
import torch
import numpy as np
import cv2

from src.datasets.mvtec_bottle import MVTecBottle
from src.models.autoencoder import ConvAutoencoder

# -------- config --------
ROOT = "data/mvtec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "runs/autoencoder_bottle.pth"
OUT_DIR = "runs/infer"
IMAGE_SIZE = 256
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("Using device:", DEVICE)

# Dataset
test_ds = MVTecBottle(ROOT, split="test", image_size=IMAGE_SIZE)

# Model
model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def to_uint8(x):
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

with torch.no_grad():
    for i in range(len(test_ds)):
        sample = test_ds[i]

        img = sample["image"].unsqueeze(0).to(DEVICE)  # (1,3,H,W)
        recon = model(img)

        # pixel-wise reconstruction error
        err_map = torch.abs(recon - img).mean(dim=1, keepdim=True)  # (1,1,H,W)

        heat = err_map.squeeze().cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

        heat_color = cv2.applyColorMap(to_uint8(heat), cv2.COLORMAP_JET)

        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        img_np = to_uint8(img_np)

        overlay = cv2.addWeighted(img_np, 0.6, heat_color, 0.4, 0)

        fname = f"{i:03d}_{sample['defect_type']}.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), overlay)

print("Inference completed.")
print("Results saved to:", OUT_DIR)
