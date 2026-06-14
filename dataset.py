import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from ecg import ECGCorruptions


class PTBXLDataset(Dataset):
    def __init__(
        self,
        csv_path="processed_ptbxl_metadata.csv",
        data_root=".",
        split="train",
        leads=None,
        mlb=None,
        # ── corruption knobs ──────────────────────────────────────────────
        # Name of any ECGCorruptions static method, or None for clean data.
        # e.g. "ra_la_swap", "v1_v2_swap", "single_lead_polarity_inversion"
        corruption=None,
        corruption_lead_idx=0,  # only used by single_lead_polarity_inversion
    ):
        self.split = split
        self.corruption = corruption
        self.corruption_lead_idx = corruption_lead_idx

        df = pd.read_csv(csv_path, index_col='ecg_id')
        df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(
            lambda x: ast.literal_eval(x)
        )

        if split == "train":
            mask = (df.strat_fold <= 8).values
        elif split == "val":
            mask = (df.strat_fold == 9).values
        elif split == "test":
            mask = (df.strat_fold == 10).values
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.df = df[mask].reset_index(drop=True)
        full_X = np.load(os.path.join(data_root, "X_numpy_ndarray.npy"))
        self.X = full_X[mask]

        self.lead_map = ["I", "II", "III", "aVR", "aVL", "aVF",
                         "V1", "V2", "V3", "V4", "V5", "V6"]

        # Keep track of requested leads, but do not slice early —
        # corruption must run on the full 12-lead tensor first.
        self.lead_indices = (
            [self.lead_map.index(l) for l in leads] if leads else None
        )

        if mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.encoded_labels = self.mlb.fit_transform(
                self.df['diagnostic_superclass']
            )
        else:
            self.mlb = mlb
            self.encoded_labels = self.mlb.transform(
                self.df['diagnostic_superclass']
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ecg_signal = self.X[index]

        # 1. Convert full 12-lead numpy array to tensor.
        #    numpy shape: (num_samples, 12) → transpose → (12, num_samples)
        ecg = torch.tensor(ecg_signal, dtype=torch.float32).transpose(0, 1)

        # 2. Apply corruption on the full 12-lead tensor so that lead index
        #    constants (_I=0, _II=1, ...) always point to the correct lead.
        if self.corruption is not None:
            corruption_fn = getattr(ECGCorruptions, self.corruption, None)
            if corruption_fn is None:
                raise ValueError(
                    f"Corruption '{self.corruption}' not found in "
                    "ECGCorruptions. Valid options: "
                    "ra_la_swap, ra_ll_swap, la_ll_swap, "
                    "v1_v2_swap, v2_v3_swap, single_lead_polarity_inversion"
                )
            if self.corruption == "single_lead_polarity_inversion":
                ecg = corruption_fn(ecg, lead_idx=self.corruption_lead_idx)
            else:
                ecg = corruption_fn(ecg)

        # 3. Slice down to the requested leads AFTER corruption is applied.
        #    Shape: (n_leads, num_samples) where n_leads <= 12
        if self.lead_indices is not None:
            ecg = ecg[self.lead_indices, :]

        label = torch.tensor(self.encoded_labels[index], dtype=torch.float32)
        return {"ecg": ecg, "label": label}