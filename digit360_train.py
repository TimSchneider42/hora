import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Dataset (concatenate all .npz files fully into memory)
# ============================================================


class Digit360Dataset(Dataset):
    def __init__(self, npz_paths):
        all_imgs = []
        all_dirs = []

        for p in npz_paths:
            data = np.load(p)
            imgs = data["imgs"]  # (N, H, W, 3)
            dirs = data["directions"]  # (N, 3)

            all_imgs.append(imgs)
            all_dirs.append(dirs)

            print(f"Loaded {p.name}: {len(imgs)} samples")

        imgs = np.concatenate(all_imgs, axis=0)
        dirs = np.concatenate(all_dirs, axis=0)

        self.imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
        self.dirs = torch.from_numpy(dirs).float()

        print(f"Total samples: {len(self.imgs)}")

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], self.dirs[idx]


# ============================================================
# Small CNN
# ============================================================


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


# ============================================================
# Hybrid loss (zero = MSE, non-zero = cosine)
# ============================================================


def hybrid_direction_loss(pred, target, eps=1e-6):
    target_norm = torch.norm(target, dim=1)
    zero_mask = target_norm < eps
    nonzero_mask = ~zero_mask

    loss = torch.tensor(0.0, device=pred.device)

    if zero_mask.any():
        loss_zero = F.mse_loss(
            pred[zero_mask],
            torch.zeros_like(pred[zero_mask]),
            reduction="mean",
        )
        loss = loss + loss_zero

    if nonzero_mask.any():
        pred_n = F.normalize(pred[nonzero_mask], dim=1)
        target_n = F.normalize(target[nonzero_mask], dim=1)
        loss_cos = 1.0 - F.cosine_similarity(pred_n, target_n).mean()
        loss = loss + loss_cos

    return loss


# ============================================================
# Angular error metric (degrees, non-zero only)
# ============================================================


def angular_error_deg(pred, target, eps=1e-6):
    target_norm = torch.norm(target, dim=1)
    mask = target_norm >= eps

    if not mask.any():
        return None

    pred_n = F.normalize(pred[mask], dim=1)
    target_n = F.normalize(target[mask], dim=1)

    cos_sim = F.cosine_similarity(pred_n, target_n, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    angles = torch.acos(cos_sim) * (180.0 / np.pi)
    return angles.mean().item()


# ============================================================
# Training loop
# ============================================================


def train(
    npz_paths,
    batch_size=32,
    epochs=10000,
    lr=1e-3,
    val_split=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = Digit360Dataset(npz_paths)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_ang = []

        for imgs, dirs in train_loader:
            imgs = imgs.to(device)
            dirs = dirs.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = hybrid_direction_loss(preds, dirs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

            ang = angular_error_deg(preds.detach(), dirs)
            if ang is not None:
                train_ang.append(ang)

        train_loss /= n_train
        train_ang = np.mean(train_ang) if train_ang else float("nan")

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_ang = []

        with torch.no_grad():
            for imgs, dirs in val_loader:
                imgs = imgs.to(device)
                dirs = dirs.to(device)

                preds = model(imgs)
                loss = hybrid_direction_loss(preds, dirs)
                val_loss += loss.item() * imgs.size(0)

                ang = angular_error_deg(preds, dirs)
                if ang is not None:
                    val_ang.append(ang)

        val_loss /= max(1, n_val)
        val_ang = np.mean(val_ang) if val_ang else float("nan")

        print(
            f"Epoch {epoch:05d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train ang: {train_ang:.2f}° | "
            f"Val loss: {val_loss:.4f} | "
            f"Val ang: {val_ang:.2f}°"
        )

    return model


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    data_dir = Path(__file__).parent / "digit360_data"
    npz_files = sorted(data_dir.glob("*.npz"))

    if not npz_files:
        raise RuntimeError("No .npz files found in digit360_data/")

    print(f"Found {len(npz_files)} datasets")

    model = train(npz_files)

    out_path = Path("digit360_cnn_hybrid.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved model to:", out_path)
