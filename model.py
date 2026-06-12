import torch
import torch.nn as nn
import pytorch_lightning as pl
from metrics import compute_classification_metrics

# Try to import ResNet1D from your local directory
try:
    from resnet1d import ResNet1D
except ImportError:
    # Fallback placeholder in case resnet1d.py isn't found right away
    ResNet1D = None

class ECGClassificationTask(pl.LightningModule):
    def __init__(self, num_leads=12, num_classes=5, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        if ResNet1D is not None:
            self.model = ResNet1D(
                in_channels=num_leads,
                num_classes=num_classes
            )
        else:
            # Fallback simple network if resnet1d file is missing
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_leads * 1000, num_classes)
            )

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, ecg):
        return self.model(ecg)

    def training_step(self, batch, batch_idx):
        ecgs = batch["ecg"]      
        labels = batch["label"]  
        logits = self.forward(ecgs)  
        loss = self.loss_fn(logits, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   
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