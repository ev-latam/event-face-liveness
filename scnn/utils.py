import numpy as np
import torch


def events_to_tensor(events, T=8, H=128, W=128):
    """
    Convert raw event stream [N, 4] = (x, y, t, p)
    into dense tensor [T, 2, H, W].
    """

    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    t = events[:, 2]
    p = events[:, 3]

    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / (t_max - t_min + 1e-8)
    t_bin = (t_norm * (T - 1)).astype(np.int32)

    tensor = np.zeros((T, 2, H, W), dtype=np.float32)

    for i in range(len(events)):
        xi, yi = x[i], y[i]
        ti = t_bin[i]
        pi = p[i]

        if 0 <= xi < W and 0 <= yi < H:
            if pi > 0:
                tensor[ti, 1, yi, xi] += 1.0
            else:
                tensor[ti, 0, yi, xi] += 1.0

    tensor = tensor / (tensor.max() + 1e-6)
    return torch.from_numpy(tensor)

    