import torch
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
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

    def compute_per_class(self, logits: torch.Tensor, labels: torch.Tensor, class_names: list) -> list:
        """
        Computes clinical and performance metrics for each individual class.
        
        Args:
            logits: torch.Tensor of shape (num_samples, num_classes)
            labels: torch.Tensor of shape (num_samples, num_classes)
            class_names: list of strings containing class names in column order
            
        Returns:
            list of dicts, where each dict contains metrics for a single class.
        """
        from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(torch.int)
        labels_int = labels.to(torch.long)
        device = logits.device
        
        # Instantiate standalone binary metrics on the correct device
        auroc_fn = BinaryAUROC().to(device)
        auprc_fn = BinaryAveragePrecision().to(device)
        f1_fn = BinaryF1Score().to(device)
        
        eps = 1e-6
        per_class_results = []
        
        for c_idx, c_name in enumerate(class_names):
            c_probs = probs[:, c_idx]
            c_preds = preds[:, c_idx]
            c_labels = labels_int[:, c_idx]
            
            # Raw true/false counts for clinical ratios
            tp = float(((c_preds == 1) & (c_labels == 1)).sum())
            fp = float(((c_preds == 1) & (c_labels == 0)).sum())
            tn = float(((c_preds == 0) & (c_labels == 0)).sum())
            fn = float(((c_preds == 0) & (c_labels == 1)).sum())
            
            sensitivity = tp / (tp + fn + eps)
            specificity = tn / (tn + fp + eps)
            fnr = fn / (tp + fn + eps)
            
            # Binary metric scores
            auroc = auroc_fn(c_probs, c_labels).item()
            auprc = auprc_fn(c_probs, c_labels).item()
            f1 = f1_fn(c_probs, c_labels).item()
            
            per_class_results.append({
                "class": c_name,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "fnr": fnr
            })
            
        return per_class_results

    @staticmethod
    def compute_instability(clean_res: dict, corr_res: dict) -> dict:
        """
        Computes the novelty metrics: Diagnosis Flip Rate and Probability Shift by comparing 
        clean predictions against corrupted predictions.
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
