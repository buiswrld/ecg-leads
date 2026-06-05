import pandas as pd
import numpy as np
import ast
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class PTBXLDataset(Dataset):
    def __init__(self, csv_path, data_root, split="train", leads=None, mlb=None):
        """
        csv_path: path to 'processed_ptbxl_metadata.csv'
        data_root: folder directory containing 'X_numpy_ndarray.npy'
        split: "train", "val", or "test"
        leads: optional list of lead names, e.g. ["I", "II", "V1"]
        """
        self.split = split
        
        # 1. Load the pre-processed metadata CSV
        df = pd.read_csv(csv_path, index_col='ecg_id')
        
        # Convert string representations of lists back to actual Python lists
        df['diagnostic_superclass'] = df['diagnostic_superclass'].apply(lambda x: ast.literal_eval(x))

        # 2. Filter the index using the recommended stratification folds
        if split == "train":
            mask = (df.strat_fold <= 8).values
        elif split == "val":
            mask = (df.strat_fold == 9).values
        elif split == "test":
            mask = (df.strat_fold == 10).values
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        # Keep only the rows for this specific split
        self.df = df[mask].reset_index(drop=True)
        
        # 3. Load the pre-saved numpy matrix from Step 3 and slice it with the exact same mask
        full_X = np.load(os.path.join(data_root, "X_numpy_ndarray.npy"))
        self.X = full_X[mask]
        
        # 4. Handle optional lead selection filtering
        self.lead_map = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_indices = [self.lead_map.index(l) for l in leads] if leads else None

        # 5. Set up the Label Encoder to convert text to numeric vectors
        # If classes are provided (for val/test), use them. Otherwise, fit to the training labels.
        if mlb is None:
            # If no mlb is provided (Training set), create a new one and fit it
            self.mlb = MultiLabelBinarizer()
            self.encoded_labels = self.mlb.fit_transform(self.df['diagnostic_superclass'])
        else:
            # If a pre-fitted mlb is passed (Val/Test sets), reuse it exactly
            self.mlb = mlb
            self.encoded_labels = self.mlb.transform(self.df['diagnostic_superclass'])

    def __len__(self):
        # Return the exact number of rows in this data split
        return len(self.df)

    def __getitem__(self, index):
        # 1. Grab the matrix slice from memory
        ecg_signal = self.X[index]  # Shape: (1000, 12)
        
        # 2. Filter out specific leads if requested
        if self.lead_indices is not None:
            ecg_signal = ecg_signal[:, self.lead_indices]
            
        # 3. Convert to tensor and flip shape to match PyTorch expectations: (channels, time_steps)
        ecg = torch.tensor(ecg_signal, dtype=torch.float32).transpose(0, 1)
        
        # 4. Extract the numeric binary tensor label instead of a text list
        label = torch.tensor(self.encoded_labels[index], dtype=torch.float32)

        return {
            "ecg": ecg,
            "label": label
        }