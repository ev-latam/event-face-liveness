
import numpy as np
from scipy.ndimage import label

def evaluate_iou_detection(pred_mask, y, dt_ms, iou_thresh=0.3):
    """
    Evaluate saccade/blink detection using IoU matching.
    
    Args:
        pred_mask: boolean array of predicted saccades
        y: boolean array of ground truth saccades
        dt_ms: timestep in milliseconds
        iou_thresh: IoU threshold for matching (default 0.3)
    
    Returns:
        Dict with keys: 'tp', 'fp', 'fn', 'precision', 'recall', 'durations_ms'
    """
    labels_pred, n_pred = label(pred_mask)
    labels_gt, n_gt = label(y)

    pred_ids = np.unique(labels_pred)
    pred_ids = pred_ids[pred_ids != 0]

    gt_ids = np.unique(labels_gt)
    gt_ids = gt_ids[gt_ids != 0]
    
    # greedy IoU matching
    matches = []

    for pid in pred_ids:
        idx_p = np.where(labels_pred == pid)[0]
        for gid in gt_ids:
            idx_g = np.where(labels_gt == gid)[0]
            inter = np.intersect1d(idx_p, idx_g).size
            union = np.union1d(idx_p, idx_g).size

            if union == 0:
                continue

            iou = inter / union
            matches.append((iou, pid, gid))

    matches.sort(reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0
    for iou, pid, gid in matches:
        if iou < iou_thresh:
            break
        if pid in matched_pred:
            continue
        if gid in matched_gt:
            continue

        matched_pred.add(pid)
        matched_gt.add(gid)
        tp += 1

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("Pred events:", len(pred_ids),"GT events:", len(gt_ids))
    print("Precision:", precision)
    print("Recall:", recall)

    durations_ms = [
        len(np.where(labels_pred == k)[0]) * dt_ms
        for k in pred_ids
    ]

    if len(durations_ms) > 0:
        print("Durations ms:",np.min(durations_ms),
              np.median(durations_ms),
              np.max(durations_ms))

    return {'tp': tp,'fp': fp,'fn': fn,
            'precision': precision,
            'recall': recall,
            'durations_ms': durations_ms,
            'pred_ids': pred_ids,
            'gt_ids': gt_ids
    }
