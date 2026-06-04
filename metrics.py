from __future__ import annotations

from typing import Dict

import torch


def _validate_inputs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Metrics expect one score and one binary target per class.'''
    if logits.ndim != 2 or labels.ndim != 2:
        raise ValueError("Expected logits and labels with shape (batch, num_classes).")
    if logits.shape != labels.shape:
        raise ValueError(f"Shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}.")

    return logits.detach().float(), labels.detach().float()


def _nanmean(values: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(values)
    if not valid.any():
        return values.new_tensor(float("nan"))
    return values[valid].mean()


def _macro_auroc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    '''Compute each class AUROC by ranking predictions, then average valid classes.'''
    positives = labels.sum(dim=0)
    negatives = labels.size(0) - positives
    valid_classes = (positives > 0) & (negatives > 0)
    if not valid_classes.any():
        return probabilities.new_tensor(float("nan"))

    scores = probabilities[:, valid_classes]
    targets = labels[:, valid_classes]
    order = torch.argsort(scores, dim=0, descending=True)
    sorted_targets = torch.gather(targets, dim=0, index=order)

    tp = sorted_targets.cumsum(dim=0)
    fp = (1.0 - sorted_targets).cumsum(dim=0)
    pos = targets.sum(dim=0).clamp_min(1.0)
    neg = (targets.size(0) - targets.sum(dim=0)).clamp_min(1.0)

    tpr = tp / pos
    fpr = fp / neg
    zero = torch.zeros((1, tpr.size(1)), dtype=tpr.dtype, device=tpr.device)
    tpr = torch.cat((zero, tpr), dim=0)
    fpr = torch.cat((zero, fpr), dim=0)

    return torch.trapz(tpr, fpr, dim=0).mean()


def _macro_auprc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    '''Average precision is computed from precision at each positive-ranked example.'''
    positives = labels.sum(dim=0)
    valid_classes = positives > 0
    if not valid_classes.any():
        return probabilities.new_tensor(float("nan"))

    scores = probabilities[:, valid_classes]
    targets = labels[:, valid_classes]
    order = torch.argsort(scores, dim=0, descending=True)
    sorted_targets = torch.gather(targets, dim=0, index=order)

    tp = sorted_targets.cumsum(dim=0)
    ranks = torch.arange(1, targets.size(0) + 1, dtype=targets.dtype, device=targets.device).unsqueeze(1)
    precision_at_k = tp / ranks
    average_precision = (precision_at_k * sorted_targets).sum(dim=0) / targets.sum(dim=0).clamp_min(1.0)

    return average_precision.mean()


def _macro_f1(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> torch.Tensor:
    '''Convert probabilities to binary predictions, then compute macro F1.'''
    predictions = (probabilities >= threshold).to(labels.dtype)

    tp = (predictions * labels).sum(dim=0)
    fp = (predictions * (1.0 - labels)).sum(dim=0)
    fn = ((1.0 - predictions) * labels).sum(dim=0)

    denominator = (2.0 * tp) + fp + fn
    per_class_f1 = torch.where(
        denominator > 0,
        (2.0 * tp) / denominator.clamp_min(1.0),
        torch.full_like(denominator, float("nan")),
    )
    return _nanmean(per_class_f1)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute macro AUROC, macro AUPRC, and macro F1 for multi-label logits."""
    '''Public metric entry point used by validation and test logging.'''
    logits, labels = _validate_inputs(logits, labels)

    with torch.no_grad():
        probabilities = torch.sigmoid(logits)
        return {
            "auroc": _macro_auroc(probabilities, labels),
            "auprc": _macro_auprc(probabilities, labels),
            "f1": _macro_f1(probabilities, labels, threshold),
        }


compute_metrics = compute_classification_metrics
