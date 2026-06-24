import torch
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score,
    MultilabelStatScores
)

class ECGMetrics:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.auroc_metric = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.auprc_metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.f1_metric = MultilabelF1Score(num_labels=num_classes, average="macro")
        # Stat scores give us True Positives, False Positives, True Negatives, False Negatives
        self.stats_metric = MultilabelStatScores(num_labels=num_classes, average="micro")

    def compute_all(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        probs = torch.sigmoid(logits)
        labels_int = labels.to(torch.long)
        device = logits.device

        auroc = self.auroc_metric.to(device)(probs, labels_int)
        auprc = self.auprc_metric.to(device)(probs, labels_int)
        f1 = self.f1_metric.to(device)(probs, labels_int)
        
        # Calculate raw stat counts for clinical ratios
        tp, fp, tn, fn, _ = self.stats_metric.to(device)(probs, labels_int)
        
        # Avoid division by zero bugs
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        fnr = fn / (tp + fn + 1e-6)

        return {
            "auroc": auroc.item(),
            "auprc": auprc.item(),
            "f1": f1.item(),
            "sensitivity": sensitivity.item(),
            "specificity": specificity.item(),
            "fnr": fnr.item(),
            "raw_probs": probs.detach().cpu(),
            "raw_preds": (probs > 0.5).to(torch.int).detach().cpu()
        }

    @staticmethod
    def compute_instability(clean_res: dict, corr_res: dict) -> dict:
        """
        Computes the novelty metrics: Diagnosis Flip Rate and Probability Shift
        by comparing clean predictions against corrupted predictions.
        """
        clean_preds = clean_res["raw_preds"]
        corr_preds = corr_res["raw_preds"]
        clean_probs = clean_res["raw_probs"]
        corr_probs = corr_res["raw_probs"]

        # Diagnosis Flip Rate: How often did a 0 become a 1, or a 1 become a 0?
        flips = (clean_preds != corr_preds).float().mean().item()

        # Probability Shift: Mean absolute difference in output confidence strings
        prob_shift = torch.abs(clean_probs - corr_probs).mean().item()

        # False Negative Amplification (Clean was 1, Corrupted became 0)
        fn_amp = ((clean_preds == 1) & (corr_preds == 0)).float().mean().item()

        # False Positive Amplification (Clean was 0, Corrupted became 1)
        fp_amp = ((clean_preds == 0) & (corr_preds == 1)).float().mean().item()

        return {
            "flip_rate": flips,
            "prob_shift": prob_shift,
            "fn_amp": fn_amp,
            "fp_amp": fp_amp,
            "perf_drop_auroc": max(0.0, clean_res["auroc"] - corr_res["auroc"])
        }
