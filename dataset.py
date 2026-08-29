import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from ecg import ECGCorruptions 

class PTBXLDataset(Dataset):
    def __init__(self, csv_path="processed_ptbxl_metadata.csv", data_root=".", split="train", leads=None, mlb=None, corruption=None, corruption_lead_idx=0):
        self.split = split
        self.corruption = corruption
        self.corruption_lead_idx = corruption_lead_idx
        self.leads = leads if leads is not None else ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: ast.literal_eval(x))

        if split == "train":
            mask = (df.strat_fold <= 8).values
        elif split == "val":
            mask = (df.strat_fold == 9).values
        elif split == "test":
            mask = (df.strat_fold == 10).values
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
        
        # self.sample_ids = df[mask].index.to_numpy()
        self.df = df[mask].reset_index(drop=True)
        self.sample_ids = np.arange(len(self.df))
        full_X = np.load(os.path.join(data_root, "X_numpy_ndarray.npy"))
        self.X = full_X[mask]
        self.lead_map = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_indices = [self.lead_map.index(l) for l in leads] if leads else None

        if mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.encoded_labels = self.mlb.fit_transform(self.df['diagnostic_superclass'])
        else:
            self.mlb = mlb
            self.encoded_labels = self.mlb.transform(self.df['diagnostic_superclass'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ecg_signal = self.X[index]

        # Changed: Apply corruption on the full 12 leads before selecting leads
        ecg = torch.tensor(ecg_signal, dtype=torch.float32).transpose(0, 1)
        if self.corruption is not None:
            corruption_fn = getattr(ECGCorruptions, self.corruption, None)
            if corruption_fn is not None:
                if self.corruption == "single_lead_polarity_inversion":
                    ecg = corruption_fn(ecg, lead_idx=self.corruption_lead_idx)
                else:
                    #ecg = corruption_fn(ecg)
                    ecg = corruption_fn(ecg, leads=self.lead_map)
            else:
                raise ValueError(f"Corruption '{self.corruption}' not found in ECGCorruptions class.")

        if self.lead_indices is not None:
            ecg = ecg[self.lead_indices]

        label = torch.tensor(self.encoded_labels[index], dtype=torch.float32)

        return {"ecg": ecg, "label": label, "sample_id": self.sample_ids[index]}