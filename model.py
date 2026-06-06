import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        ecgs = batch["ecg"]
        labels = batch["label"]
        logits = self.forward(ecgs)
        loss = self.loss_fn(logits, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)