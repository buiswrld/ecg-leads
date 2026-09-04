import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import PTBXLDataset


class PTBXLDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, data_root, leads=None, corruption=None,
                 corruption_lead=None, batch_size=32, num_workers=0):
        super().__init__()
        self.csv_path, self.data_root = csv_path, data_root
        self.leads = leads
        self.corruption, self.corruption_lead = corruption, corruption_lead
        self.batch_size, self.num_workers = batch_size, num_workers
        self._X = None
        self.class_names = None

    def setup(self, stage=None):
        if self._X is None:
            self._X = np.load(f"{self.data_root}/X_numpy_ndarray.npy", mmap_mode="r")

        common = dict(csv_path=self.csv_path, data_root=self.data_root,
                      leads=self.leads, X_array=self._X)
        # Fit the binarizer on train, reuse for val/test.
        self.train_ds = PTBXLDataset(split="train", **common)
        mlb = self.train_ds.mlb
        self.class_names = self.train_ds.class_names
        self.val_ds = PTBXLDataset(split="val", mlb=mlb, corruption=self.corruption,
                                   corruption_lead=self.corruption_lead, **common)
        self.test_ds = PTBXLDataset(split="test", mlb=mlb, corruption=self.corruption,
                                    corruption_lead=self.corruption_lead, **common)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
