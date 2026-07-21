import argparse
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("../..")
sys.path.append("../")
sys.path.append("./")
sys.path.append("../utils")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from event_reps import time_surface
from iou_matching_evaluation import evaluate_iou_detection
from tcnn import SaccadeModel
from train_utils import load_dense_gt, remove_short_events, resize_with_padding


def evaluate_model_dense(
    model,
    val_loader,
    centers_val,
    seq_ids_val,
    y_val,
    T,
    dt_us,
    threshold=0.5,
    device="cuda",
):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())

    probs = torch.cat(all_probs).numpy()
    lengths = [len(seq) for seq in y_val]
    pred_masks = [np.zeros(L, dtype=np.float32) for L in lengths]
    counts = [np.zeros(L, dtype=np.float32) for L in lengths]

    for i in range(len(probs)):
        s = seq_ids_val[i]
        center = centers_val[i]

        start = center - T // 2
        end = center + T // 2

        left_clip = max(0, -start)
        right_clip = max(0, end - lengths[s])

        start = max(0, start)
        end = min(lengths[s], end)

        win = probs[i][left_clip:T - right_clip]
        pred_masks[s][start:end] += win
        counts[s][start:end] += 1

    for s in range(len(pred_masks)):
        mask = counts[s] > 0
        pred_masks[s][mask] /= counts[s][mask]

    tp = 0
    fp = 0
    fn = 0
    for s in range(len(pred_masks)):
        binary_pred = pred_masks[s] > threshold
        binary_pred = remove_short_events(binary_pred, 20)

        metrics = evaluate_iou_detection(
            binary_pred.astype(np.uint8),
            y_val[s],
            dt_us,
            iou_thresh=0.5,
        )

        tp += metrics["tp"]
        fp += metrics["fp"]
        fn += metrics["fn"]

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


def chunk_time_series_dense(
    X,
    y_dense,
    window_us,
    dt_us,
    stride_us=None,
    return_seq_id=False,
):
    if stride_us is None:
        stride_us = window_us

    T = int(window_us / dt_us)
    stride = int(stride_us / dt_us)

    X_out = []
    y_out = []
    centers_out = []
    seq_ids = []

    samples, length, _channels, _height, _width = X.shape

    for s in range(samples):
        for start in range(0, length - T + 1, stride):
            end = start + T
            x_win = X[s, start:end]
            y_win = y_dense[s, start:end]
            X_out.append(x_win)
            y_out.append(y_win)
            centers_out.append(start + T // 2)

            if return_seq_id:
                seq_ids.append(s)

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.float32)
    centers_out = np.array(centers_out, dtype=np.int32)

    if return_seq_id:
        seq_ids = np.array(seq_ids)
        return X_out, y_out, centers_out, seq_ids

    return X_out, y_out, centers_out


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    dt_us,
    window_us=200e3,
    stride_us=40e3,
    epochs=30,
    batch_size=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    p = y_train.mean()
    pos_weight = torch.tensor([(1 - p) / p]).to(device)
    criterion = FocalLoss(pos_weight, gamma=2.0)
    T = int(window_us / dt_us)

    (
        X_train_chunks,
        y_train_chunks,
        centers_train,
        seq_ids_train,
    ) = chunk_time_series_dense(
        X_train,
        y_train,
        window_us,
        dt_us,
        stride_us,
        return_seq_id=True,
    )

    (
        X_val_chunks,
        y_val_chunks,
        centers_val,
        seq_ids_val,
    ) = chunk_time_series_dense(
        X_val,
        y_val,
        window_us,
        dt_us,
        stride_us,
        return_seq_id=True,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train_chunks, dtype=torch.float32),
        torch.tensor(y_train_chunks, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_chunks, dtype=torch.float32),
        torch.tensor(y_val_chunks, dtype=torch.float32),
    )

    train_loader_val = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
    )

    best_f1_val = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        for _, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        scheduler.step()

        if epoch % 5 == 0:
            precisions_train = []
            recalls_train = []
            f1s_train = []
            ths_train = []
            thresholds = np.linspace(0.15, 0.9, 10)

            for thr in thresholds:
                metrics = evaluate_model_dense(
                    model,
                    train_loader_val,
                    centers_train,
                    seq_ids_train,
                    y_train,
                    T,
                    dt_us,
                    threshold=thr,
                )
                precisions_train.append(metrics["precision"])
                recalls_train.append(metrics["recall"])
                f1s_train.append(metrics["f1"])
                ths_train.append(thr)

            max_f1_idx = np.argmax(f1s_train)
            best_thr = ths_train[max_f1_idx]
            best_f1 = f1s_train[max_f1_idx]
            precision = precisions_train[max_f1_idx]
            recall = recalls_train[max_f1_idx]

            print("Best metrics train:")
            print(
                f"thr={best_thr:.3f}",
                "P:", precision,
                "R:", recall,
                "F1:", best_f1,
            )

            metrics = evaluate_model_dense(
                model,
                val_loader,
                centers_val,
                seq_ids_val,
                y_val,
                T,
                dt_us,
                threshold=best_thr,
            )
            print("Best metrics validation:")
            print(
                f"thr={best_thr:.3f}",
                "P:", metrics["precision"],
                "R:", metrics["recall"],
                "F1:", metrics["f1"],
            )
            if metrics["f1"] > best_f1_val:
                best_f1_val = metrics["f1"]
                torch.save(model.state_dict(), "best_model.pth")
                print("New best model saved with F1:", best_f1)

    return best_f1_val


def prepare_data(USERS, ex_time, base_path, dt_us, tau, TARGET_SIZE):
    Xs = []
    ys = []
    ts = []

    for user_name in USERS:
        print("User", user_name)

        root_event_folder = f"{base_path}raw_data/user_{user_name}/exp{ex_time}/prophesee"
        events = np.load(root_event_folder + "/left_eye.npy", allow_pickle=True)

        ts_ev = events[:, 0]
        xs_ev = events[:, 1].astype(np.int32)
        ys_ev = events[:, 2].astype(np.int32)
        ps_ev = events[:, 3]

        t_min = ts_ev.min()
        t_max = ts_ev.max()
        t_uniform = np.arange(t_min, t_max, dt_us)

        sae_sequence = []
        img_size = np.max(ys_ev) + 1, np.max(xs_ev) + 1

        for t_ref in t_uniform:
            mask = (ts_ev >= t_ref - 3 * tau) & (ts_ev <= t_ref)
            ev_slice = (
                ts_ev[mask],
                xs_ev[mask],
                ys_ev[mask],
                ps_ev[mask],
            )
            sae = time_surface(ev_slice, img_size, t_ref, tau)

            sae_on = resize_with_padding(sae[0], TARGET_SIZE)
            sae_off = resize_with_padding(sae[1], TARGET_SIZE)
            sae = np.stack([sae_on, sae_off])
            sae_sequence.append(sae)

        Xs.append(np.stack(sae_sequence))

        image_align_event_time = (
            f"{base_path}processed_data/frame_align_event_timestamp/"
            f"user_{user_name}_exp_{ex_time}.txt"
        )
        with open(image_align_event_time) as f:
            timestamp = [float(i) for i in f.readlines()]

        y = load_dense_gt(
            user_name,
            t_uniform,
            timestamp,
            "../annotations/saccades.csv",
        )

        ys.append(y)
        ts.append(t_uniform)

    X_all = np.array(Xs)
    y_all = np.array(ys)
    t_all = np.array(ts)

    return X_all, y_all, t_all


def main(epochs=15, base_path="/data2/nico/RGBE_Gaze/"):

    # Same subjects and parameters as in the original paper
    #USERS = [1,2,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    USERS = [1,2]
   
    ex_time = 1
    dt_us = 2e3
    tau = 6e3
    TARGET_SIZE = (32, 32)

    X_all, y_all, t_all = prepare_data(USERS, ex_time, base_path, dt_us, tau, TARGET_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1s = []
    for i in range(len(USERS)):
        X_train = np.concatenate([X_all[:i], X_all[i + 1 :]])
        y_train = np.concatenate([y_all[:i], y_all[i + 1 :]])
        X_val = X_all[i : i + 1]
        y_val = y_all[i : i + 1]

        mean = X_train.mean()
        std = X_train.std() + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        model = SaccadeModel().to(device)
        best_f1 = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            dt_us,
            window_us=400e3,
            stride_us=125e3,
            epochs=epochs,
            batch_size=64,
        )
        f1s.append(best_f1)
        print(f"User {USERS[i]} - F1: {best_f1:.4f}")

    print("Average F1 across users:", np.mean(f1s))
    print("F1 scores per user:", f1s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--base_path", type=str, default="/data2/nico/RGBE_Gaze/")
    args = parser.parse_args()

    main(epochs=args.epochs, base_path=args.base_path)