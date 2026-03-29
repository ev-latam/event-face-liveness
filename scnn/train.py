import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from spikingjelly.activation_based import functional

from dataset import NpyEventDataset, build_samples_from_folder
from model import EventSCNN


def compute_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy().astype(np.int32)

    preds = (probs >= threshold).astype(np.int32)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    apcer = fp / (fp + tn + 1e-8)   # fake → real
    bpcer = fn / (fn + tp + 1e-8)   # real → fake
    acer = 0.5 * (apcer + bpcer)

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return accuracy, apcer, bpcer, acer, auc


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
        desc = "Train"
    else:
        model.eval()
        desc = "Val"

    total_loss = 0
    total_samples = 0

    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for x, y in pbar:
        x = x.to(device).permute(1, 0, 2, 3, 4).contiguous()
        y = y.to(device).float()

        if train:
            optimizer.zero_grad()

        logits = model(x).squeeze(1)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        functional.reset_net(model)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    acc, apcer, bpcer, acer, auc = compute_metrics(all_logits, all_labels)

    return total_loss / total_samples, acc, apcer, bpcer, acer, auc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = "data/train"
    val_dir = "data/val"

    T = 32
    H = 346
    W = 346
    batch_size = 8
    epochs = 20
    lr = 1e-3

    # ========================
    # LOAD DATA
    # ========================
    train_samples, train_map = build_samples_from_folder(train_dir)
    val_samples, val_map = build_samples_from_folder(val_dir)

    print("Train classes:", train_map)
    print("Val classes:", val_map)

    # IMPORTANT: ensure mapping is consistent
    # expected:
    # {'fake': 0, 'real': 1}

    dataset_train = NpyEventDataset(train_samples, T=T, H=H, W=W)
    dataset_val = NpyEventDataset(val_samples, T=T, H=H, W=W)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # ========================
    # MODEL
    # ========================
    model = EventSCNN(in_channels=2, num_classes=1).to(device)

    # ========================
    # LOSS (with imbalance)
    # ========================
    labels = np.array([label for _, label in train_samples])
    num_neg = np.sum(labels == 0)  # fake
    num_pos = np.sum(labels == 1)  # real

    pos_weight = torch.tensor([num_neg / (num_pos + 1e-8)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acer = float("inf")

    # ========================
    # TRAIN LOOP
    # ========================
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1:02d}/{epochs}")

        train_loss, train_acc, train_apcer, train_bpcer, train_acer, train_auc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )

        val_loss, val_acc, val_apcer, val_bpcer, val_acer, val_auc = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        print(
            f"TRAIN | loss={train_loss:.4f} acc={train_acc:.4f} "
            f"ACER={train_acer:.4f} AUC={train_auc:.4f}"
        )

        print(
            f"VAL   | loss={val_loss:.4f} acc={val_acc:.4f} "
            f"ACER={val_acer:.4f} AUC={val_auc:.4f}"
        )

        # Save best model
        if val_acer < best_acer:
            best_acer = val_acer
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

    print(f"\nBest ACER: {best_acer:.4f}")


if __name__ == "__main__":
    main()