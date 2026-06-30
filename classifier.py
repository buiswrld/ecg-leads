import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from metrics import ECGMetrics

try:
    from resnet1d import ResNet1D
except ImportError:
    ResNet1D = None

class ECGClassificationTask(pl.LightningModule):
    def __init__(self, num_leads=12, num_classes=5, learning_rate=1e-3, max_epochs=10, warmup_epochs=1, corruption=None, corruption_lead_idx=0):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        
        if ResNet1D is not None:
            self.model = ResNet1D(
                in_channels=num_leads, num_classes=num_classes
            )
        else:
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_leads * 1000, num_classes)
            )
            
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Keep metrics isolated across evaluation steps
        self.val_metrics = ECGMetrics(num_classes=num_classes)
        self.test_metrics = ECGMetrics(num_classes=num_classes)

        self.val_outputs = []
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
        
        self.val_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": labels.detach().cpu()
        })
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        ecgs = batch["ecg"]
        labels = batch["label"]
        logits = self.forward(ecgs)
        loss = self.loss_fn(logits, labels)

        self.test_outputs.append({
            "logits": logits.detach().cpu(),
            "labels": labels.detach().cpu()
        })

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss
    
    def on_validation_epoch_end(self):
        all_logits = torch.cat([x["logits"] for x in self.val_outputs], dim=0)

        all_labels = torch.cat([x["labels"] for x in self.val_outputs], dim=0)

        scores = self.val_metrics.compute_all(all_logits, all_labels)

        self.log("val_auroc", scores["auroc"], prog_bar=False)
        self.log("val_auprc", scores["auprc"], prog_bar=False)
        self.log("val_f1", scores["f1"], prog_bar=False)

        self.val_outputs.clear()
    
    def on_test_epoch_end(self):
        all_logits = torch.cat([x["logits"] for x in self.test_outputs], dim=0)

        all_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0)

        # Calculate metrics over entire test set
        scores = self.test_metrics.compute_all(all_logits, all_labels)

        self.log("test_auroc", scores["auroc"], prog_bar=True, logger=True)
        self.log("test_auprc", scores["auprc"], prog_bar=True, logger=True)
        self.log("test_f1", scores["f1"], prog_bar=True, logger=True)

        self.test_epoch_results = scores

        self.test_outputs.clear()

    def configure_optimizers(self):
        # Using AdamW optimizer which applies weight decay regularization correctly
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        warmup_iters = self.warmup_epochs
        cosine_iters = max(1, self.max_epochs - self.warmup_epochs) 
                
        # Calculate linear warmup schedule parameters
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters= warmup_iters
        )
        
        # Calculate main cosine annealing schedule parameters
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cosine_iters, 
            eta_min=1e-6
        )
        
        # Combine them sequentially: Warmup first, then Cosine decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Step the scheduler once every epoch completion
                "frequency": 1
            }
        }
