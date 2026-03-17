"""
PTB-XL PyTorch Dataset
- Loads raw 12-lead ECG signals using wfdb
- Applies bandpass filter (0.5-40 Hz) via neurokit2
- Normalizes per lead: zero mean, unit variance
- Returns tensor (12, 1000) + integer label
"""

import os
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Constants ──────────────────────────────────────────────
SAMPLING_RATE  = 100          # Hz (using records100 / filename_lr)
SIGNAL_LENGTH  = 1000         # 10s x 100Hz
NUM_LEADS      = 12
LABEL_MAP      = {"Low": 0, "Medium": 1, "High": 2}

class PTBXLDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str, augment: bool = False):
        """
        Args:
            csv_path : path to train.csv / val.csv / test.csv
            data_dir : root path to ptbxl records folder
            augment  : apply light augmentation (train only)
        """
        self.df       = pd.read_csv(csv_path, index_col="ecg_id")
        self.data_dir = data_dir
        self.augment  = augment
        self.records  = self.df.index.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = int(row["label"])

        # ── Load signal ────────────────────────────────────
        rec_path = os.path.join(
            self.data_dir,
            row["filename_lr"].replace(".hea", "")
        )
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal.astype(np.float32)  # (1000, 12)

        # ── Bandpass filter per lead (0.5–40 Hz) ──────────
        filtered = np.zeros_like(signal)
        for i in range(NUM_LEADS):
            try:
                filtered[:, i] = nk.signal_filter(
                    signal[:, i],
                    sampling_rate=SAMPLING_RATE,
                    lowcut=0.5, highcut=40.0,
                    method="butterworth", order=4
                )
            except Exception:
                filtered[:, i] = signal[:, i]  # fallback: no filter

        # ── Normalize per lead (zero mean, unit variance) ──
        mean = filtered.mean(axis=0, keepdims=True)
        std  = filtered.std(axis=0, keepdims=True) + 1e-8
        normalized = (filtered - mean) / std          # (1000, 12)

        # ── Augmentation (train only) ──────────────────────
        if self.augment:
            # 1. Gaussian noise
            if np.random.rand() < 0.3:
                normalized += np.random.normal(0, 0.05, normalized.shape)
            # 2. Random amplitude scale
            if np.random.rand() < 0.3:
                normalized *= np.random.uniform(0.9, 1.1)

        # ── Transpose to (12, 1000) for Conv1d ────────────
        x = torch.tensor(normalized.T, dtype=torch.float32)  # (12, 1000)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


def get_class_weights(csv_path: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted loss."""
    df     = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def get_sampler(csv_path: str) -> WeightedRandomSampler:
    """Weighted sampler so each batch has balanced classes."""
    df        = pd.read_csv(csv_path)
    counts    = df["label"].value_counts().sort_index().values
    class_w   = 1.0 / counts
    sample_w  = [class_w[l] for l in df["label"].values]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def get_dataloaders(data_dir: str, split_dir: str, batch_size: int = 32):
    """
    Returns train, val, test DataLoaders.
    Train uses WeightedRandomSampler for class balance.
    """
    train_ds = PTBXLDataset(
        os.path.join(split_dir, "train.csv"), data_dir, augment=True
    )
    val_ds = PTBXLDataset(
        os.path.join(split_dir, "val.csv"), data_dir, augment=False
    )
    test_ds = PTBXLDataset(
        os.path.join(split_dir, "test.csv"), data_dir, augment=False
    )

    sampler = get_sampler(os.path.join(split_dir, "train.csv"))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Sanity check ───────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR  = "data/ptbxl/records/"
    SPLIT_DIR = "data/"

    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, SPLIT_DIR)

    # Check one batch
    x, y = next(iter(train_loader))
    print(f"Batch shape  : {x.shape}")   # expect (32, 12, 1000)
    print(f"Labels       : {y}")
    print(f"Signal range : {x.min():.2f} to {x.max():.2f}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")
    print("Dataset class weights:", get_class_weights(SPLIT_DIR + "train.csv"))
    print("\nSanity check passed!")