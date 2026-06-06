import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class PTBXLDataset(Dataset):
    def __init__(self, csv_path="processed_ptbxl_metadata.csv", data_root=".", split="train", leads=None, mlb=None):
        self.split = split
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
            
        self.df = df[mask].reset_index(drop=True)
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
        if self.lead_indices is not None:
            ecg_signal = ecg_signal[:, self.lead_indices]
            
        ecg = torch.tensor(ecg_signal, dtype=torch.float32).transpose(0, 1)
        label = torch.tensor(self.encoded_labels[index], dtype=torch.float32)

        return {"ecg": ecg, "label": label}