import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import label

def load_dense_gt(user_name, ts, timestamp, csv_file):
        
    df = pd.read_csv(csv_file)
    y = np.zeros_like(ts)

    user_saccades = df[df['user'] == user_name]
    for _, row in user_saccades.iterrows():
        frame_ix_start = row['start']
        frame_ix_end = row['end']
        t_start_index = find_first_greater_index(ts, timestamp[frame_ix_start])
        t_end_index = find_first_greater_index(ts, timestamp[frame_ix_end])
        #print("Duration:", timestamp[frame_ix_end] - timestamp[frame_ix_start], "ms")
        y[t_start_index:t_end_index] = 1

    return y


def remove_short_events(binary, min_len=20):

    labels, n = label(binary)
    for i in range(1, n + 1):
        idx = labels == i
        if idx.sum() < min_len:
            binary[idx] = 0

    return binary


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


def resize_with_padding(img, target_size):

    H, W = img.shape

    scale = min(
        target_size[0] / H,
        target_size[1] / W
    )

    new_H = int(H * scale)
    new_W = int(W * scale)

    img_resized = cv2.resize(img, (new_W, new_H))

    canvas = np.zeros(target_size, dtype=img.dtype)

    y0 = (target_size[0] - new_H) // 2
    x0 = (target_size[1] - new_W) // 2

    canvas[y0:y0+new_H, x0:x0+new_W] = img_resized

    return canvas


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

