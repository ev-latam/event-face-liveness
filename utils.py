
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

highlight_color_1 = '#d01c8b'
highlight_color_2 = '#4dac26'
light_color_1 = '#fc8d59'
light_color_2 = '#91bfdb'
color_1 = '#0571b0'
color_2 = '#d73027'

def plot_features_with_gt_and_pred(t,X,y,states,dt,
                                   user,t_start_sec=None,
                                   t_end_sec=None,
                                   type="Blinks"):
    if t_start_sec is not None:
        i_start = np.searchsorted(t, t_start_sec)
    else:
        i_start = 0

    if t_end_sec is not None:
        i_end = np.searchsorted(t, t_end_sec)
    else:
        i_end = len(t)

    t = t[i_start:i_end]
    X = X[i_start:i_end]
    y = y[i_start:i_end]
    states = states[i_start:i_end]

    pred_mask = (states == 1)

    ymin = np.min(X)
    ymax = np.max(X)

    plt.figure(figsize=(12, 2))
    plt.plot(t, X, label="$A(t)$", color=color_2, alpha=0.6)
    plt.fill_between(
        t,
        ymin,
        ymax,
        where=y.astype(bool),
        alpha=0.5,
        label=f"GT {type}",
        color=light_color_1
    )

    plt.fill_between(
        t,
        ymin,
        ymax,
        where=pred_mask,
        alpha=0.5,
        label=f"Predicted {type}",
        color=light_color_2
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Filtered $A(t)$")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{type}_detection_user_{user}.png", dpi=300)


def find_first_greater_index(seq, target):
    left, right = 0, len(seq) - 1
    result = -1  

    while left <= right:
        mid = (left + right) // 2
        if seq[mid] > target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result

def load_dense_gt(user_name, ts, timestamp, csv_file):
        
    df = pd.read_csv(csv_file)
    y = np.zeros_like(ts)

    user_saccades = df[df['user'] == user_name]
    for _, row in user_saccades.iterrows():
        frame_ix_start = row['start']
        frame_ix_end = row['end']
        t_start_index = find_first_greater_index(ts, timestamp[frame_ix_start])
        t_end_index = find_first_greater_index(ts, timestamp[frame_ix_end])
        y[t_start_index:t_end_index] = 1

    return y