"""
Lightning task.

  - Stores full per-class + ALL results in self.last_results after each eval.
  - Also stores raw probs/labels, so paired clean-vs-corrupted instability
    (probability shift, flip rate) can be computed without re-running.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from resnet1d import ResNet1D
from metrics import ECGMetrics


class ECGClassificationTask(pl.LightningModule):
    def __init__(self, num_leads=12, num_classes=5, learning_rate=1e-3,
                 weight_decay=1e-4, max_epochs=10, warmup_epochs=1,
                 class_names=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet1D(in_channels=num_leads, num_classes=num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = ECGMetrics(num_classes=num_classes, class_names=class_names)
        self._buf = []
        self.last_results = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _eval_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self._buf.append((torch.sigmoid(logits).detach().cpu(), y.detach().cpu()))
        return loss

    def validation_step(self, batch, _):
        return self._eval_step(batch, "val")

    def test_step(self, batch, _):
        return self._eval_step(batch, "test")

    def _finalize(self):
        if not self._buf:
            return
        probs = torch.cat([p for p, _ in self._buf])
        labels = torch.cat([l for _, l in self._buf])
        res = self.metrics.compute_all(probs, labels)
        self.last_results = {
            **res["ALL"],                 # auroc/auprc/f1/sensitivity/specificity/fnr
            "per_class": res["per_class"],
            "probs": probs, "labels": labels,
        }
        self._buf = []

    def on_validation_epoch_end(self):
        self._finalize()
        if self.last_results:
            self.log("val_f1", self.last_results["f1"], prog_bar=True)

    def on_test_epoch_end(self):
        self._finalize()

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay)
        warm = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                        total_iters=self.hparams.warmup_epochs)
        cos = CosineAnnealingLR(opt, T_max=max(1, self.hparams.max_epochs -
                                               self.hparams.warmup_epochs), eta_min=1e-6)
        sched = SequentialLR(opt, [warm, cos], milestones=[self.hparams.warmup_epochs])
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
