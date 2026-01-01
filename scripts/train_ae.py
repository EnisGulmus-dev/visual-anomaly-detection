import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.mvtec_bottle import MVTecBottle
from src.models.autoencoder import ConvAutoencoder

# -------- config --------
ROOT = "data/mvtec"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
# ------------------------

print("Using device:", DEVICE)

# Dataset & loader
train_ds = MVTecBottle(ROOT, split="train", image_size=256)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Model
model = ConvAutoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs = batch["image"].to(DEVICE)

        recon = model(imgs)
        loss = F.l1_loss(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: L1 loss = {avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), "runs/autoencoder_bottle.pth")
print("Model saved to runs/autoencoder_bottle.pth")
