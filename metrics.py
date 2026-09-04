"""
Diagnostic and robustness metrics.

  - Computes per-class metrics for each diagnostic superclass plus an aggregate
    ALL row (Methods 4.2).
  - Class names are taken from mlb.classes_ to preserve the dataset's class order.
"""

import numpy as np
import torch
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score,
    MultilabelStatScores,
)


class ECGMetrics:
    def __init__(self, num_classes=5, class_names=None, threshold=0.5, device="cpu"):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.threshold = threshold
        k = dict(num_labels=num_classes)
        # average=None -> per-class vectors
        self.auroc = MultilabelAUROC(average=None, **k).to(device)
        self.auprc = MultilabelAveragePrecision(average=None, **k).to(device)
        self.f1 = MultilabelF1Score(average=None, threshold=threshold, **k).to(device)
        self.stats = MultilabelStatScores(average=None, threshold=threshold, **k).to(device)

    def reset(self):
        for m in (self.auroc, self.auprc, self.f1, self.stats):
            m.reset()

    def compute_all(self, probs, labels):
        """
        probs:  (N, C) float in [0,1]
        labels: (N, C) binary
        Returns {"per_class": {name: {...}}, "ALL": {...}}
        """
        self.reset()
        labels_int = labels.int()
        auroc = self.auroc(probs, labels_int).cpu().numpy()
        auprc = self.auprc(probs, labels_int).cpu().numpy()
        f1 = self.f1(probs, labels_int).cpu().numpy()
        sc = self.stats(probs, labels_int).cpu().numpy()  # (C, 5): tp, fp, tn, fn, sup

        per_class, acc = {}, {k: [] for k in
                              ["auroc", "auprc", "f1", "sensitivity", "specificity", "fnr"]}
        for i, name in enumerate(self.class_names):
            tp, fp, tn, fn = float(sc[i][0]), float(sc[i][1]), float(sc[i][2]), float(sc[i][3])
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            row = {
                "auroc": float(auroc[i]), "auprc": float(auprc[i]), "f1": float(f1[i]),
                "sensitivity": sens, "specificity": spec, "fnr": 1.0 - sens,
            }
            per_class[name] = row
            for k in acc:
                acc[k].append(row[k])

        # ALL = macro average across classes, consistent for every metric.
        all_row = {k: float(np.mean(v)) for k, v in acc.items()}
        return {"per_class": per_class, "ALL": all_row}

    @staticmethod
    def instability(clean_probs, corr_probs, threshold=0.5):
        """Probability shift, label-wise flip rate, and any-class flip rate."""
        clean_pred = (clean_probs >= threshold).int()
        corr_pred = (corr_probs >= threshold).int()
        prob_shift = torch.abs(corr_probs - clean_probs).mean().item()
        flip_rate = (clean_pred != corr_pred).float().mean().item()
        any_flip = ((clean_pred != corr_pred).any(dim=1)).float().mean().item()
        return {"probability_shift": prob_shift,
                "prediction_flip_rate": flip_rate,
                "any_class_flip_rate": any_flip}
