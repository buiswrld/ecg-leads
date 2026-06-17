import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import PTBXLDataset
from metrics import compute_classification_metrics

try:
    from resnet1d import ResNet1D
except ImportError:
    ResNet1D = None


class ECGClassificationTask(pl.LightningModule):
    def __init__(
        self,
        num_leads=12,
        num_classes=5,
        learning_rate=1e-3,
        # ── dataset knobs ─────────────────────────────────────────────────
        csv_path="processed_ptbxl_metadata.csv",
        data_root=".",
        leads=None,
        batch_size=32,
        num_workers=0,
        # ── corruption switch ─────────────────────────────────────────────
        # This is the "turning on" switch. Set to any ECGCorruptions method
        # name to corrupt the test set, or leave as None for clean data.
        # e.g. "ra_la_swap", "v1_v2_swap", "single_lead_polarity_inversion"
        corruption=None,
        corruption_lead_idx=0,  # only used by single_lead_polarity_inversion
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.csv_path = csv_path
        self.data_root = data_root
        self.leads = leads
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.corruption = corruption
        self.corruption_lead_idx = corruption_lead_idx

        if ResNet1D is not None:
            self.model = ResNet1D(
                in_channels=num_leads,
                num_classes=num_classes,
            )
        else:
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_leads * 1000, num_classes),
            )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.validation_outputs = []
        self.test_outputs = []

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, ecg):
        return self.model(ecg)

    # ── Steps ─────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        ecgs = batch["ecg"]      
        labels = batch["label"]  
        logits = self.forward(ecgs)  
        loss = self.loss_fn(logits, labels)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ecgs = batch["ecg"]
        labels = batch["label"]
        logits = self.forward(ecgs)
        loss = self.loss_fn(logits, labels)
        self.validation_outputs.append((logits.detach(), labels.detach()))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        ecgs = batch["ecg"]
        labels = batch["label"]
        logits = self.forward(ecgs)
        loss = self.loss_fn(logits, labels)
        self.test_outputs.append((logits.detach(), labels.detach()))
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return

        logits = torch.cat([item[0] for item in self.validation_outputs], dim=0)
        labels = torch.cat([item[1] for item in self.validation_outputs], dim=0)
        metrics = compute_classification_metrics(logits, labels)
        self.log("val_auroc", metrics["auroc"], prog_bar=True, logger=True)
        self.log("val_auprc", metrics["auprc"], prog_bar=True, logger=True)
        self.log("val_f1", metrics["f1"], prog_bar=True, logger=True)
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        logits = torch.cat([item[0] for item in self.test_outputs], dim=0)
        labels = torch.cat([item[1] for item in self.test_outputs], dim=0)
        metrics = compute_classification_metrics(logits, labels)
        self.log("test_auroc", metrics["auroc"], prog_bar=True, logger=True)
        self.log("test_auprc", metrics["auprc"], prog_bar=True, logger=True)
        self.log("test_f1", metrics["f1"], prog_bar=True, logger=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # ── DataLoaders (corruption switch is fed into dataset here) ──────────────

    def setup(self, stage=None):
        # Training set — always clean, fits the MLB encoder.
        self._train_ds = PTBXLDataset(
            csv_path=self.csv_path,
            data_root=self.data_root,
            split="train",
            leads=self.leads,
            mlb=None,
            corruption=None,        # training is always clean
            corruption_lead_idx=0,
        )
        shared_mlb = self._train_ds.mlb

        # Validation set — always clean, reuses MLB encoder.
        self._val_ds = PTBXLDataset(
            csv_path=self.csv_path,
            data_root=self.data_root,
            split="val",
            leads=self.leads,
            mlb=shared_mlb,
            corruption=None,        # validation is always clean
            corruption_lead_idx=0,
        )

        # Test set — corruption switch is fed in here.
        self._test_ds = PTBXLDataset(
            csv_path=self.csv_path,
            data_root=self.data_root,
            split="test",
            leads=self.leads,
            mlb=shared_mlb,
            corruption=self.corruption,              # ← the switch
            corruption_lead_idx=self.corruption_lead_idx,
        )

    def train_dataloader(self):
        return DataLoader(self._train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)