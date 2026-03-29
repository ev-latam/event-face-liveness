import os
import numpy as np
from torch.utils.data import Dataset
from utils import events_to_tensor


class NpyEventDataset(Dataset):
    """
    Dataset for raw event .npy files.
    Each sample must contain [N, 4] = (x, y, t, p).
    """

    def __init__(self, samples, T=8, H=128, W=128):
        """
        samples: list of tuples -> [(file_path, label), ...]
        """
        self.samples = samples
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        events = np.load(file_path)  # shape [N, 4]
        x = events_to_tensor(events, T=self.T, H=self.H, W=self.W)

        return x, label


def build_samples_from_folder(root_dir):
    """
    Expected structure:
    root_dir/
        class_0/
            a.npy
            b.npy
        class_1/
            c.npy
            d.npy
    """
    class_names = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                fpath = os.path.join(class_dir, fname)
                samples.append((fpath, class_to_idx[class_name]))

    return samples, class_to_idx

    