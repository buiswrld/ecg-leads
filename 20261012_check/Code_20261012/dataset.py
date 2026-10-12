"""
PTB-XL dataset.

CHANGED vs. previous version:
  - CRITICAL: corruption is applied to the full 12-lead signal BEFORE the lead
    subset is extracted (Methods 3.3). Previously the subset was sliced first
    and the corruption applied to the reduced tensor, which silently no-opped
    whenever a swap partner was absent.
  - Empty lead list is handled with `is None`, not truthiness, so leads=[] no
    longer silently means "all 12 leads".
  - Corruption target lead is a global lead NAME, not a position in the subset.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

from ecg import LEAD_ORDER, apply_corruption


class PTBXLDataset(Dataset):
    def __init__(self, csv_path, data_root, split="train", leads=None, mlb=None,
                 corruption=None, corruption_lead=None, X_array=None):
        self.lead_map = list(LEAD_ORDER)

        # `is None` -- an empty list is a real (degenerate) request, not "all"
        if leads is None:
            self.leads = list(self.lead_map)
            self.lead_indices = None
        else:
            if len(leads) == 0:
                raise ValueError("leads=[] is not a valid subset; pass None for all 12 leads.")
            self.leads = list(leads)
            self.lead_indices = [self.lead_map.index(l) for l in leads]

        self.corruption = corruption
        self.corruption_lead = corruption_lead

        df = pd.read_csv(csv_path)
        df["diagnostic_superclass"] = df["diagnostic_superclass"].apply(eval)

        if split == "train":
            mask = df.strat_fold <= 8
        elif split == "val":
            mask = df.strat_fold == 9
        elif split == "test":
            mask = df.strat_fold == 10
        else:
            raise ValueError(f"Unknown split: {split}")

        full_X = np.load(f"{data_root}/X_numpy_ndarray.npy", mmap_mode="r") if X_array is None else X_array
        self.X = np.asarray(full_X[mask.values], dtype=np.float32)
        self.df = df[mask].reset_index(drop=True)

        if mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.y = self.mlb.fit_transform(self.df["diagnostic_superclass"])
        else:
            self.mlb = mlb
            self.y = self.mlb.transform(self.df["diagnostic_superclass"])

        # Alphabetical from MultiLabelBinarizer: CD, HYP, MI, NORM, STTC.
        self.class_names = list(self.mlb.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # (T, 12) -> (12, T). Keep all 12 leads at this point.
        ecg = torch.from_numpy(np.asarray(self.X[index], dtype=np.float32)).transpose(0, 1)

        # 1. Corrupt the FULL 12-lead signal.
        if self.corruption is not None:
            ecg = apply_corruption(ecg, self.corruption, corruption_lead=self.corruption_lead)

        # 2. THEN extract the subset.
        if self.lead_indices is not None:
            ecg = ecg[self.lead_indices]

        label = torch.from_numpy(self.y[index].astype(np.float32))
        return ecg, label
