import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PTBXLDataset

class PTBXLDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        csv_path="processed_ptbxl_metadata.csv", 
        data_root=".", 
        batch_size=32, 
        leads=None,                  # Added so it's accessible by the dataset
        num_workers=2, #4               # Added for DataLoader optimization
        corruption=None,             # The "turning on" switch string
        corruption_lead_idx=0        # Target index for single-lead inversions
    ):
        super().__init__()
        # self.save_hyperparameters() saves everything above to self.hparams
        self.save_hyperparameters()
        
        # We instantiate a clean training dataset here just to extract a single 
        # fitted MultiLabelBinarizer instance for all evaluation sets to share.
        dummy_train = PTBXLDataset(
            csv_path=self.hparams.csv_path, 
            data_root=self.hparams.data_root, 
            split="train"
        )
        self.fitted_mlb = dummy_train.mlb

    def train_dataloader(self):
        dataset = PTBXLDataset(
            csv_path=self.hparams.csv_path,
            data_root=self.hparams.data_root,
            split="train",
            leads=self.hparams.leads,
            mlb=self.fitted_mlb,
            corruption=None  # Keep training completely clean
        )
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        dataset = PTBXLDataset(
            csv_path=self.hparams.csv_path,
            data_root=self.hparams.data_root,
            split="val",
            leads=self.hparams.leads,
            mlb=self.fitted_mlb,
            corruption=None  # Keep validation completely clean
        )
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        dataset = PTBXLDataset(
            csv_path=self.hparams.csv_path,
            data_root=self.hparams.data_root,
            split="test",
            leads=self.hparams.leads,
            mlb=self.fitted_mlb,
            corruption=self.hparams.corruption,                  # Switch activated here!
            corruption_lead_idx=self.hparams.corruption_lead_idx  # Lead index passed here!
        )
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers
        )
