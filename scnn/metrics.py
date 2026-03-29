import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(logits, labels, threshold=0.5):
    """
    logits: torch tensor [N]
    labels: torch tensor [N] (0=fake, 1=real)
    """

    probs = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy()

    preds = (probs >= threshold).astype(int)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    apcer = FP / (FP + TN + 1e-8)
    bpcer = FN / (TP + FN + 1e-8)
    acer = (apcer + bpcer) / 2

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "AUC": auc
    }